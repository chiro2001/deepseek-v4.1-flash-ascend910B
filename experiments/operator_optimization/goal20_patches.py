"""Independent candidate graph banks on the proven activation+overlap baseline."""
import os

import torch
import torch_npu

import activation_patches as activation
import runtime_patches as baseline
from hc_static import hc_static, round_hf32
from selected_gemv import gemv
from short_ops import hc_post, route_init, route_combine

ARM = os.getenv('GOAL20_ARM', 'baseline')
CONFIGS = {
    'baseline': set(), 'hcstatic': {'hcstatic'}, 'hcpost': {'hcpost'},
    'route': {'route'}, 'gmm1': {'gmm1'}, 'gmm2': {'gmm2'},
    'all_candidates': {'hcstatic', 'hcpost', 'route', 'gmm1', 'gmm2'},
    'combo': {'hcstatic', 'hcpost', 'route', 'gmm1'},
}
REFS = {}; BANKS = {}; WEIGHTS = {}; MODELS = None
ORIGINAL_GMM = torch_npu.npu_grouped_matmul
ORIGINAL_HC = baseline.compute_hc_candidate
ORIGINAL_INIT = None; ORIGINAL_COMBINE = None


def enabled(kind):
    return kind in CONFIGS[ARM]


def cached_weight(weight, layout):
    key = (id(weight), weight.data_ptr(), weight._version, tuple(weight.shape), tuple(weight.stride()), layout)
    if key not in WEIGHTS:
        from vllm.forward_context import is_forward_context_available, get_forward_context
        assert not (is_forward_context_available() and get_forward_context().capturing), ('uncached weight in capture', key)
        value = torch_npu.npu_format_cast(weight, 2) if torch_npu.get_npu_format(weight) != 2 else weight
        if layout == 'hf32':
            value = round_hf32(value.contiguous())
        elif layout == 'nk':
            value = value.transpose(-1, -2).contiguous()
        WEIGHTS[key] = (weight, value)
    return WEIGHTS[key][1]


def remember(kind, args, output, selected):
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing:
        REFS.setdefault(ARM, {}).setdefault(kind, []).append((args, output, selected))


def compute_hc(x, fn, scale, base, pmix, iters=20, norm_eps=1e-20, hc_eps=1e-6):
    if not enabled('hcstatic'):
        return ORIGINAL_HC(x, fn, scale, base, pmix, iters, norm_eps, hc_eps)
    shape = x.shape[:-2]
    values = hc_static(x.reshape(1, 4, 5120), cached_weight(fn, 'hf32'), scale, base,
                pmix.reshape(1, 4) if pmix is not None else None,
                hc_sinkhorn_iters=iters, norm_eps=norm_eps, hc_eps=hc_eps)
    return (values[0].reshape(*shape, 5120), values[1].reshape(*shape, 4),
            values[2].reshape(*shape, 4, 4), values[3].reshape(*shape, 4))


@torch.library.custom_op('goal20::hc_post', mutates_args=())
def post_op(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    eligible = x.numel() == 5120 and residual.numel() == 20480
    selected = eligible and enabled('hcpost')
    if selected:
        result = hc_post(x, residual, post, comb, block=512, fma=False)
    else:
        result = torch.ops._C_ascend.npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)).squeeze(0)
    if eligible:
        remember('hcpost', (x, residual, post, comb), result, selected)
    return result


@post_op.register_fake
def post_fake(x, residual, post, comb):
    return torch.empty_like(residual)


@torch.library.custom_op('goal20::route_init', mutates_args=())
def init_op(x: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = enabled('route')
    if selected:
        result = route_init(x, ids)[:3]
    else:
        result = ORIGINAL_INIT(x, ids, active_num=2, expert_num=8,
                 expert_tokens_num_type=1, expert_tokens_num_flag=True,
                 active_expert_range=[0, 8], quant_mode=-1)[:3]
        result = (result[0], result[1], result[2].to(torch.int64))
    remember('route_init', (x, ids), result, selected)
    return result


@init_op.register_fake
def init_fake(x, ids):
    return (torch.empty((2, 5120), dtype=x.dtype, device=x.device),
            torch.empty((2,), dtype=torch.int32, device=x.device),
            torch.empty((8,), dtype=torch.int64, device=x.device))


@torch.library.custom_op('goal20::route_combine', mutates_args=())
def combine_op(x: torch.Tensor, reverse: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    selected = enabled('route')
    result = route_combine(x, reverse, probs) if selected else ORIGINAL_COMBINE(x, reverse, probs)
    remember('route_combine', (x, reverse, probs), result, selected)
    return result


@combine_op.register_fake
def combine_fake(x, reverse, probs):
    return torch.empty((1, 5120), dtype=x.dtype, device=x.device)


@torch.library.custom_op('goal20::gmm', mutates_args=())
def gmm_op(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor, count_type: int) -> torch.Tensor:
    kind = 'gmm1' if x.shape[-1] == 5120 else 'gmm2'
    selected = enabled(kind)
    if selected:
        result = gemv(x, cached_weight(w, 'nk'), counts=counts, count_type=count_type,
                      kind='vector', bn=8, bk=512, nk_layout=True)
    else:
        result = ORIGINAL_GMM([x], [w], group_list=counts, split_item=2,
                              group_type=0, group_list_type=count_type)[0]
    remember(kind, (x, w, counts, count_type), result, selected)
    return result


@gmm_op.register_fake
def gmm_fake(x, w, counts, count_type):
    return torch.empty((2, w.shape[-1]), dtype=x.dtype, device=x.device)


def install(model):
    global ORIGINAL_INIT, ORIGINAL_COMBINE, MODELS
    MODELS = model
    baseline.compute_hc_candidate = compute_hc
    # Cache both original and transpose views; GMM obtains a new transpose view
    # each call, so dispatch uses the persistent prepared view below.
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
    persistent = {}
    original_transpose = AscendUnquantizedFusedMoEMethod._maybe_transpose
    def transpose(self, weight, need_trans):
        if isinstance(weight, torch.Tensor) and weight.ndim == 3 and weight.shape[0] == 8:
            key = (id(weight), weight.data_ptr(), weight._version, need_trans)
            if key not in persistent:
                persistent[key] = original_transpose(self, weight, need_trans)
                cached_weight(persistent[key], 'nk')
            return persistent[key]
        return original_transpose(self, weight, need_trans)
    AscendUnquantizedFusedMoEMethod._maybe_transpose = transpose
    for name, param in model.named_parameters():
        if param.shape == (24, 20480) and 'hc' in name:
            cached_weight(param, 'hf32')
        if param.ndim == 3 and param.shape[0] == 8 and name.endswith(('w13_weight', 'w2_weight')):
            # Unquantized routed method uses need_trans=True for loaded layout.
            transpose(None, param, True)
            transpose(None, param, False)
    def gmm_dispatch(x, weight, *args, **kwargs):
        if (not args and len(x) == len(weight) == 1
                and x[0].dtype == weight[0].dtype == torch.bfloat16
                and x[0].shape in [(2, 5120), (2, 256)]
                and tuple(weight[0].shape) in [(8, 5120, 512), (8, 256, 5120)]
                and kwargs.get('bias') is None and kwargs.get('split_item') == 2
                and kwargs.get('group_type') == 0 and kwargs.get('group_list_type') in [0, 1]
                and kwargs.get('group_list') is not None and kwargs['group_list'].shape == (8,)):
            return [gmm_op(x[0], weight[0], kwargs['group_list'], kwargs['group_list_type'])]
        return ORIGINAL_GMM(x, weight, *args, **kwargs)
    torch_npu.npu_grouped_matmul = gmm_dispatch
    classes = {type(m) for m in model.modules() if hasattr(m, 'hc_post') and hasattr(m, 'hc_mult')}
    for cls in classes:
        original = cls.hc_post
        def bind(fallback):
            def dispatch(self, x, residual, post, comb):
                # Batch eligibility belongs inside the opaque op. A numel
                # branch here was frozen while tracing the prefill example.
                if x.shape[-1] == 5120 and residual.shape[-2] == 4 and x.dtype == torch.bfloat16:
                    return post_op(x, residual, post, comb)
                return fallback(self, x, residual, post, comb)
            return dispatch
        cls.hc_post = bind(original)
    from vllm_ascend.device.device_op import DeviceOperator
    ORIGINAL_INIT = DeviceOperator.npu_moe_init_routing
    ORIGINAL_COMBINE = DeviceOperator.npu_moe_token_unpermute
    def init_dispatch(x, ids, **kw):
        if (x.shape == (1, 5120) and ids.shape == (1, 2) and x.dtype == torch.bfloat16
                and kw.get('expert_num') == 8 and kw.get('active_num') == 2
                and kw.get('expert_tokens_num_type', 1) == 1
                and kw.get('expert_tokens_num_flag', True)
                and kw.get('active_expert_range') == [0, 8]
                and kw.get('quant_mode', -1) == -1 and kw.get('scale') is None
                and kw.get('act_quant_type') is None):
            return (*init_op(x, ids), None)
        return ORIGINAL_INIT(x, ids, **kw)
    def combine_dispatch(permuted_tokens, sorted_indices, probs):
        if (permuted_tokens.shape == (2, 5120) and sorted_indices.shape == (2,)
                and probs.shape == (1, 2) and probs.dtype == permuted_tokens.dtype == torch.bfloat16):
            return combine_op(permuted_tokens, sorted_indices, probs)
        return ORIGINAL_COMBINE(permuted_tokens, sorted_indices, probs)
    DeviceOperator.npu_moe_init_routing = staticmethod(init_dispatch)
    DeviceOperator.npu_moe_token_unpermute = staticmethod(combine_dispatch)


def save_bank(worker, name):
    activation.save_bank(worker, 'overlap')
    refs = REFS[name]
    expected = {'hcpost': 80, 'route_init': 40, 'route_combine': 40, 'gmm1': 40, 'gmm2': 40}
    assert {k: len(v) for k, v in refs.items()} == expected, (name, {k: len(v) for k,v in refs.items()})
    selected = {k: sum(r[2] for r in rows) for k, rows in refs.items()}
    assert all(selected[k] == (n if enabled('route' if k.startswith('route') else k) else 0)
               for k, n in expected.items()), (name, selected)
    BANKS[name] = (activation.BANKS['overlap'], activation.REFS['overlap'])
    return {'arm': name, 'refs': expected, 'selected': selected, 'hcstatic': enabled('hcstatic')}


def warm(worker):
    # Compile kernels outside capture using safe dummy routing metadata.
    groups = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], device='npu', dtype=torch.int64)
    ids = torch.tensor([[0, 1]], device='npu', dtype=torch.int32)
    for x, fn, scale, base, pmix, _ in baseline.HC_REFS['both']:
        hc_static(x.reshape(1,4,5120), cached_weight(fn,'hf32'), scale, base,
                  pmix.reshape(1,4) if pmix is not None else None)
    for kind, rows in REFS['baseline'].items():
        for args, _, _ in rows:
            if kind == 'hcpost': hc_post(*args, block=512, fma=False)
            elif kind == 'route_init': route_init(args[0], ids)
            elif kind == 'route_combine': route_combine(args[0], torch.tensor([0,1],device='npu',dtype=torch.int32), args[2])
            else: gemv(args[0], cached_weight(args[1],'nk'), counts=groups, kind='vector',bn=8,bk=512,nk_layout=True)
    torch.npu.synchronize()
    return {'cached_weights': len(WEIGHTS)}


def create_bank(worker, name):
    global ARM
    from vllm_ascend.compilation import acl_graph as ag
    worker = baseline.actual(worker); torch.npu.synchronize(); ARM = name
    REFS[name] = {}; activation.REFS['overlap'] = {'routed': [], 'shared': []}
    baseline.ROUTER_REFS['both'] = []; baseline.HC_REFS['both'] = []
    for wrapper in ag._acl_graph_wrappers: wrapper.concrete_aclgraph_entries = {}
    ag._graph_params = None; ag.set_graph_params([1]); worker.model_runner.capture_model()
    return save_bank(worker, name)


def switch_bank(worker, name):
    global ARM
    activation.BANKS['overlap'], activation.REFS['overlap'] = BANKS[name]
    ARM = name
    return activation.switch_bank(worker, 'overlap')


def audit(worker, name):
    result = {'activation': activation.audit(worker, 'overlap'), 'hc': baseline.audit_hc(worker, 'both'),
              'router': baseline.audit_router(worker, 'both')}
    for kind, rows in REFS[name].items():
        worst = 0.; exact = 0; min_std = float('inf')
        for index,(args, stored, selected) in enumerate(rows):
            if kind == 'hcpost':
                x, residual, post, comb = args
                expected = torch.ops._C_ascend.npu_hc_post(x.unsqueeze(0),residual.unsqueeze(0),post.unsqueeze(0),comb.unsqueeze(0)).squeeze(0)
            elif kind == 'route_init':
                x, ids = args
                expected = ORIGINAL_INIT(x,ids,active_num=2,expert_num=8,expert_tokens_num_type=1,
                           expert_tokens_num_flag=True,active_expert_range=[0,8],quant_mode=-1)[:3]
                for a, b in zip(stored, expected): torch.testing.assert_close(a,b.to(a.dtype),rtol=0,atol=0)
                exact += 1; continue
            elif kind == 'route_combine': expected = ORIGINAL_COMBINE(*args)
            else:
                x,w,counts,count_type = args
                expected = ORIGINAL_GMM([x],[w],group_list=counts,split_item=2,group_type=0,group_list_type=count_type)[0]
                if kind=='gmm1':
                    # The routed activation mutates this intermediate in place.
                    limit=activation.REFS['overlap']['routed'][index][1][0]
                    gate,up=expected.chunk(2,-1)
                    gate.clamp_(max=limit);up.clamp_(min=-limit,max=limit)
                min_std = min(min_std,float(w.float().std()))
            torch.testing.assert_close(stored,expected,rtol=1/64,atol=1/64)
            worst=max(worst,float((stored-expected).abs().max())); exact+=int(torch.equal(stored,expected))
        if kind in ['gmm1','gmm2'] and os.getenv('TINY_PERF_RANDOM_VALIDATION')=='1': assert min_std>1e-4
        result[kind]={'calls':len(rows),'selected':sum(r[2] for r in rows),'max_abs':worst,'bit_equal_calls':exact}
    return result
