"""Experimental formal W4A8 prefix banks, with captured consumer snapshots.

Only the expert boundary representation changes. Weights, dynamic quantization,
expert ordering, activation, arithmetic and communication retain native code.
"""
import dataclasses
import hashlib
import inspect
import os
import textwrap

import torch

ARM = 'tp8base'
PREFIX_ARMS = ('tp8prefix', 'tp8prefixroute', 'tp8prefixup', 'tp8prefixuproute')
UP_ARMS = ('tp8prefixup', 'tp8prefixuproute')
REFS = {}
COVERAGE = {}
SOURCE = {}
FORCE_NATIVE = False
CURRENT_STAGE = None
INSTALLED = False


def set_arm(name):
    global ARM
    ARM = name


def capturing():
    from vllm.forward_context import get_forward_context
    return get_forward_context().capturing


def install():
    global INSTALLED
    if INSTALLED or os.getenv('STACK_W4A8_PREFIX_ENABLED') != '1':
        return
    assert os.getenv('STACK_REAL_WEIGHTS') == '1'
    from vllm_ascend.ops.fused_moe import token_dispatcher as td
    from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod as W4A8
    cls = td.TokenDispatcherWithAllGather
    original_dispatch = cls.token_dispatch
    source = textwrap.dedent(inspect.getsource(original_dispatch))
    # Freeze the two representation changes; reject an incompatible image.
    assert source.count('expert_tokens_num_type=1,') == 1
    assert source.count('group_list_type = 1') == 1
    direct_source = source.replace('expert_tokens_num_type=1,', 'expert_tokens_num_type=0,')
    direct_source = direct_source.replace('group_list_type = 1', 'group_list_type = 0')
    namespace = dict(original_dispatch.__globals__)
    exec(compile(direct_source, '<formal_w4a8_prefix_dispatch>', 'exec'), namespace)
    direct_dispatch = namespace['token_dispatch']
    SOURCE['dispatch_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    SOURCE['direct_dispatch_sha256'] = hashlib.sha256(direct_source.encode()).hexdigest()
    original_convert = W4A8._maybe_convert_group_list
    SOURCE['convert_sha256'] = hashlib.sha256(inspect.getsource(original_convert).encode()).hexdigest()

    def dispatch(self, token_dispatch_input):
        inp = token_dispatch_input
        eligible = inp.quant.quant_type == td.QuantType.W4A8 and ARM in PREFIX_ARMS
        if not eligible:
            return original_dispatch(self, inp)
        assert self.num_experts_local == 48 and self.top_k == 6
        if ARM == 'tp8prefixup':
            # Preserve the native counts for down-GMM; create the up prefix
            # inside that hook without mutating the shared compute input.
            return original_dispatch(self, inp)
        if ARM in ('tp8prefixroute', 'tp8prefixuproute'):
            output = direct_dispatch(self, inp)
        else:
            output = original_dispatch(self, inp)
            assert output.group_list_type == 1
            output = dataclasses.replace(output, group_list=output.group_list.cumsum(0), group_list_type=0)
        assert output.group_list_type == 0 and output.group_list.shape == (48,)
        assert output.group_list.dtype == torch.int64
        return output

    def convert(self, mlp_compute_input):
        inp = mlp_compute_input
        if ARM not in PREFIX_ARMS or FORCE_NATIVE or (ARM in UP_ARMS and CURRENT_STAGE == 'apply_gmm2'):
            return original_convert(self, inp)
        # The dispatcher computes one dynamic prefix reused by both GMMs.
        assert inp.group_list_type == 0 and inp.group_list.shape == (48,)
        assert inp.group_list.dtype == torch.int64
        return inp.group_list, 0

    cls.token_dispatch = dispatch
    W4A8._maybe_convert_group_list = convert
    for stage in ('apply_gmm1_act_quant', 'apply_gmm2'):
        original = getattr(W4A8, stage)
        SOURCE[stage + '_sha256'] = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()

        def bind(method, kind):
            def wrapped(self, mlp_compute_input, *args, **kwargs):
                global CURRENT_STAGE
                inp = mlp_compute_input
                selected = ARM in PREFIX_ARMS and not FORCE_NATIVE
                if selected and ARM == 'tp8prefixup' and kind == 'apply_gmm1_act_quant':
                    assert inp.group_list_type == 1
                    inp = dataclasses.replace(inp, group_list=inp.group_list.cumsum(0), group_list_type=0)
                snapshot = (selected or ARM == 'tp8base') and capturing() and os.getenv('STACK_REAL_AUDIT') == '1'
                if selected and capturing():
                    counts = COVERAGE.setdefault(ARM, {})
                    counts[kind] = counts.get(kind, 0) + 1
                if snapshot:
                    saved = dataclasses.replace(inp, group_list=inp.group_list.clone(),
                        hidden_states=inp.hidden_states.clone() if kind == 'apply_gmm1_act_quant' else None,
                        dynamic_scale=inp.dynamic_scale.clone() if inp.dynamic_scale is not None else None)
                    saved_args = tuple(value.clone() for value in args)
                    saved_kwargs = {key: value.clone() for key, value in kwargs.items()}
                previous_stage = CURRENT_STAGE
                CURRENT_STAGE = kind
                try:
                    output = method(self, inp, *args, **kwargs)
                finally:
                    CURRENT_STAGE = previous_stage
                if snapshot:
                    outputs = output if isinstance(output, tuple) else (output,)
                    REFS.setdefault(ARM, []).append((self, method, kind, saved, saved_args, saved_kwargs,
                        tuple(value.clone() for value in outputs)))
                return output
            return wrapped
        setattr(W4A8, stage, bind(original, stage))
    # Upstream calls these hooks with keyword arguments. Check this before
    # spending time loading the formal checkpoint, including the down-GMM args.
    sentinel = object()
    inspect.signature(cls.token_dispatch).bind(sentinel, token_dispatch_input=sentinel)
    inspect.signature(W4A8._maybe_convert_group_list).bind(sentinel, mlp_compute_input=sentinel)
    inspect.signature(W4A8.apply_gmm1_act_quant).bind(sentinel, mlp_compute_input=sentinel)
    inspect.signature(W4A8.apply_gmm2).bind(sentinel, mlp_compute_input=sentinel,
                                        hidden_states=sentinel, act_out_scale=sentinel)
    SOURCE['keyword_contract_checked'] = True
    INSTALLED = True


def reset(name):
    REFS[name] = []
    COVERAGE[name] = {}


def status(name):
    return {'enabled': INSTALLED, 'selected': name in PREFIX_ARMS,
            'representation': 'native routing counts; prefix for up-GMM only' if name == 'tp8prefixup' else
                'routing prefix; restore down-GMM counts' if name == 'tp8prefixuproute' else
                'routing prefix' if name == 'tp8prefixroute' else
                'one cumsum prefix' if name == 'tp8prefix' else 'native counts',
            'down_gmm_prefix_selected': name in ('tp8prefix', 'tp8prefixroute'),
            'captured_gmm_calls': COVERAGE.get(name, {}),
            'consumer_snapshots': len(REFS.get(name, [])), 'source': SOURCE}


@torch.inference_mode()
def audit(name):
    global FORCE_NATIVE
    if name not in PREFIX_ARMS:
        return status(name)
    rows = REFS.get(name, [])
    counts = COVERAGE.get(name, {})
    assert counts.get('apply_gmm1_act_quant') == 40 and counts.get('apply_gmm2') == 40, counts
    assert len(rows) == 80, len(rows)
    result = []
    for owner, method, stage, inp, args, kwargs, outputs in rows:
        group = inp.group_list.cpu()
        prefix = group if inp.group_list_type == 0 else group.cumsum(0)
        count = torch.cat([prefix[:1], torch.diff(prefix)])
        assert prefix.shape == (48,) and (count >= 0).all()
        saved = dataclasses.replace(inp,
            hidden_states=inp.hidden_states.clone() if inp.hidden_states is not None else None)
        FORCE_NATIVE = True
        try:
            reference = method(owner, saved, *(value.clone() for value in args),
                               **{key: value.clone() for key, value in kwargs.items()})
        finally:
            FORCE_NATIVE = False
        reference = reference if isinstance(reference, tuple) else (reference,)
        assert len(reference) == len(outputs)
        valid_rows = int(prefix[-1])
        for actual, expected in zip(outputs, reference):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.ndim >= 1 and actual.shape[0] >= valid_rows
            # Routing reserves capacity for all global top-k assignments. Only
            # [0, prefix[-1]) belongs to this rank's experts; GMM's unwritten
            # tail is not read by the masked unpermute consumer.
            a, b = actual[:valid_rows].cpu(), expected[:valid_rows].cpu()
            if not torch.equal(a, b):
                from pathlib import Path
                from vllm.distributed import get_tensor_model_parallel_rank
                import json
                root = Path(os.environ['VLLM_CACHE_ROOT']).parent
                stem = root / f'prefix_failure_{name}_{stage}_rank{get_tensor_model_parallel_rank()}'
                metric = {'arm': name, 'stage': stage, 'dtype': str(a.dtype),
                    'shape': list(actual.shape), 'valid_rows': valid_rows,
                    'differing_elements': int((a != b).sum()),
                    'max_abs': float((a.float()-b.float()).abs().max())}
                stem.with_suffix('.json').write_text(json.dumps(metric, indent=2)+'\n')
                torch.save({'actual': a, 'reference': b, 'prefix': prefix}, stem.with_suffix('.pt'))
                raise AssertionError(metric)
        result.append({'stage': stage, 'local_experts': 48, 'active_experts': int((count > 0).sum()),
            'local_tokens': int(prefix[-1]), 'output_shapes': [list(t.shape) for t in outputs],
            'output_dtypes': [str(t.dtype) for t in outputs], 'bitwise_equal': True,
            'compared_rows': valid_rows, 'scope': 'all active consumer rows; routing capacity tail excluded'})
    return {**status(name), 'native_counts_consumer_comparisons': result, 'passed': True}


@torch.inference_mode()
def probe_native_up(worker):
    """Use live formal weights and actual last-decode inputs before bank capture."""
    global ARM
    from vllm.distributed import get_tensor_model_parallel_rank
    rows = REFS.get('tp8base', [])
    up = [row for row in rows if row[2] == 'apply_gmm1_act_quant']
    assert len(up) == 40, len(up)
    previous = ARM
    result = []
    try:
        ARM = 'tp8prefixup'
        for owner, method, stage, inp, args, kwargs, outputs in up:
            assert inp.group_list_type == 1
            counts = inp.group_list.cpu()
            prefix = counts.cumsum(0)
            candidate_input = dataclasses.replace(inp, hidden_states=inp.hidden_states.clone(),
                group_list=inp.group_list.cumsum(0), group_list_type=0)
            actual = method(owner, candidate_input, *(value.clone() for value in args),
                            **{key: value.clone() for key, value in kwargs.items()})
            actual = actual if isinstance(actual, tuple) else (actual,)
            valid = int(prefix[-1])
            assert len(actual) == len(outputs)
            for value, expected in zip(actual, outputs):
                assert value.dtype == expected.dtype and value.shape == expected.shape
                assert torch.equal(value[:valid].cpu(), expected[:valid].cpu()), (stage, valid)
            result.append({'local_tokens': valid, 'active_experts': int((counts > 0).sum()),
                'output_dtypes': [str(value.dtype) for value in actual], 'bitwise_equal': True})
    finally:
        ARM = previous
    return {'rank': get_tensor_model_parallel_rank(), 'passed': True, 'layers': len(result),
            'scope': 'actual formal up-GMM INT8 and scale consumer inputs/weights, outside capture', 'rows': result}
