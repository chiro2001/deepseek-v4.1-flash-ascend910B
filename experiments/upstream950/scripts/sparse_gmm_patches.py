"""Use native routing type 2 once, shared by both unquantized GMM calls."""
import os
from dataclasses import replace

import torch
import torch_npu

ARM = os.getenv('UP950_ARM', 'baseline')
SPARSE_DISPATCH = False
INSTALLED = False
ROUTES = {}
GMMS = {}
DISPATCHES = {}


def capturing():
    from vllm.forward_context import get_forward_context
    return get_forward_context().capturing


def install(model):
    global INSTALLED, ORIGINAL_ROUTING
    if INSTALLED:
        return
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    ORIGINAL_ROUTING = DeviceOperator.npu_moe_init_routing
    original_dispatch = TokenDispatcherWithAllGather.token_dispatch

    def routing(x, ids, **kwargs):
        sparse = SPARSE_DISPATCH
        actual_kwargs = dict(kwargs)
        if sparse:
            assert kwargs['expert_num'] == 8 and kwargs['quant_mode'] == -1, kwargs
            actual_kwargs['expert_tokens_num_type'] = 2
        output = ORIGINAL_ROUTING(x, ids, **actual_kwargs)
        if capturing() and x.shape == (1, 5120):
            ROUTES.setdefault(ARM, []).append((x, ids, dict(kwargs), output, sparse))
        return output

    def dispatch(self, token_dispatch_input):
        global SPARSE_DISPATCH
        data = token_dispatch_input
        eligible = (data.hidden_states.shape == (1, 5120)
                    and data.hidden_states.dtype == torch.bfloat16
                    and not data.quant.is_quant and self.num_experts_local == 8)
        selected = eligible and ARM == 'sparse'
        previous = SPARSE_DISPATCH
        SPARSE_DISPATCH = selected
        try:
            output = original_dispatch(self, data)
        finally:
            SPARSE_DISPATCH = previous
        if selected:
            assert output.group_list.shape == (8, 2), output.group_list.shape
            output = replace(output, group_list_type=2)
        if capturing() and eligible:
            DISPATCHES.setdefault(ARM, []).append(output)
        return output

    DeviceOperator.npu_moe_init_routing = staticmethod(routing)
    TokenDispatcherWithAllGather.token_dispatch = dispatch
    for kind in ['gmm1', 'gmm2']:
        original = getattr(AscendUnquantizedFusedMoEMethod, 'apply_' + kind)

        def bind(method, label):
            def wrapped(self, data, *args):
                output = method(self, data, *args)
                x = data.hidden_states if label == 'gmm1' else args[0]
                if capturing() and x.shape[0] == 2:
                    w1, w2 = self.get_mlp_weights(data.layer)
                    weight = self._maybe_transpose(w1 if label == 'gmm1' else w2, data.need_trans)
                    GMMS.setdefault(ARM, {}).setdefault(label, []).append(
                        (x, weight, data.group_list, data.group_list_type, output, data.swiglu_limit)
                    )
                return output
            return wrapped

        setattr(AscendUnquantizedFusedMoEMethod, 'apply_' + kind, bind(original, kind))
    INSTALLED = True


def coverage(name):
    routes, dispatches, gmms = ROUTES[name], DISPATCHES[name], GMMS[name]
    assert len(routes) == len(dispatches) == len(gmms['gmm1']) == len(gmms['gmm2']) == 40
    expected = name == 'sparse'
    assert all(item[4] == expected for item in routes)
    assert all(item.group_list_type == (2 if expected else 1) for item in dispatches)
    return {'route_calls': 40, 'gmm1_calls': 40, 'gmm2_calls': 40,
            'sparse_route_calls': 40 if expected else 0, 'extra_list_build_kernels': 0}


def set_arm(name):
    global ARM
    ARM = name


def reset(name):
    ROUTES[name] = []
    GMMS[name] = {}
    DISPATCHES[name] = []


def prepare(name, baseline_name):
    if name != 'sparse':
        return
    pairs = []
    for x, ids, kwargs, _, _ in ROUTES[baseline_name]:
        sparse_kwargs = dict(kwargs)
        sparse_kwargs['expert_tokens_num_type'] = 2
        pairs.append(ORIGINAL_ROUTING(x, ids, **sparse_kwargs)[2].to(torch.int64))
    for kind, refs in GMMS[baseline_name].items():
        for pair, (x, weight, _, _, _, _) in zip(pairs, refs):
            torch_npu.npu_grouped_matmul([x], weight if isinstance(weight, list) else [weight],
                                        group_list=pair, group_list_type=2,
                                        group_type=0, split_item=2)


def audit(worker, name):
    """Compare captured real routing buffers against native on current inputs."""
    torch.npu.synchronize()
    unique = set()
    max_error = 0.0
    for x, ids, kwargs, output, sparse in ROUTES[name]:
        native = ORIGINAL_ROUTING(x, ids, **kwargs)
        assert torch.equal(output[0], native[0]), 'Expanded token mismatch'
        assert torch.equal(output[1], native[1]), 'Row-index mismatch'
        counts = native[2].to(torch.int64)
        actual_counts = torch.zeros_like(counts)
        if sparse:
            pairs = output[2].to(torch.int64)
            actual_counts.scatter_add_(0, pairs[:, 0], pairs[:, 1])
            nonzero = pairs[pairs[:, 1] != 0]
            assert torch.all(nonzero[1:, 0] > nonzero[:-1, 0]), nonzero
        else:
            actual_counts = output[2]
        assert torch.equal(counts, actual_counts), 'Expert-count mismatch'
        unique.update(ids.cpu().flatten().tolist())
        assert x.float().std().item() > 1e-4, 'Constant validation input'
    for kind, refs in GMMS[name].items():
        for x, weight, group, group_type, output, limit in refs:
            assert group_type == (2 if name == 'sparse' else 1)
            counts = group
            if group_type == 2:
                counts = torch.zeros(8, dtype=torch.int64, device=x.device)
                counts.scatter_add_(0, group[:, 0], group[:, 1])
            expected = torch_npu.npu_grouped_matmul([x], weight if isinstance(weight, list) else [weight],
                                                   group_list=counts, group_list_type=1,
                                                   group_type=0, split_item=2)[0]
            if kind == 'gmm1' and limit > 0:
                gate, up = expected.chunk(2, dim=-1)
                gate.clamp_(max=limit)
                up.clamp_(min=-limit, max=limit)
            assert torch.equal(output, expected), ('GMM output mismatch', kind, group_type)
            max_error = max(max_error, (output.float()-expected.float()).abs().max().item())
    return {**coverage(name), 'expanded_x_bitwise_equal': True,
            'row_idx_bitwise_equal': True, 'expert_counts_equal': True,
            'unique_experts': sorted(unique), 'max_error': max_error}
