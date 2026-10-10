"""Eager layer-0 first-decode boundaries; preserve native computations."""
from types import MethodType
import torch

ENABLED = False
CURRENT = {}
SNAPSHOTS = {}
HANDLES = []
STAGES = ('norm_input', 'norm_output', 'project_q', 'project_qr', 'project_kv',
          'core_q', 'core_cache_visible', 'core_seq_lens', 'core_query_start',
          'core_metadata', 'core_output', 'oproj_input', 'wo_b_input',
          'wo_b_local', 'wo_b_output', 'oproj_output')


def eligible(x):
    if not ENABLED or not isinstance(x, torch.Tensor) or x.shape[0] != 1:
        return False
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    return not context.capturing and context.attn_metadata is not None


def save(name, value):
    if name not in CURRENT:
        assert isinstance(value, torch.Tensor), (name, type(value))
        CURRENT[name] = value.clone()


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def install(worker):
    worker = getattr(worker, 'worker', worker)
    assert worker.vllm_config.model_config.enforce_eager
    layers = [m for name, m in worker.model_runner.model.named_modules()
              if name.endswith('model.layers.0') and hasattr(m, 'hc_pre')]
    assert len(layers) == 1
    layer = layers[0]
    attn = layer.self_attn
    impl = attn.v41_impl
    v1_impl = attn.dsa_attn.dsa_attn.impl
    assert not v1_impl.multistream_dsv4_dsa_overlap
    assert not impl.role.has_long_context, 'This probe normalizes SWA-only layer0 cache'
    assert attn.wo_b.reduce_results and attn.wo_b.tp_size == 8

    def norm_before(module, inputs):
        if eligible(inputs[0]): save('norm_input', inputs[0])

    def norm_after(module, inputs, output):
        if eligible(inputs[0]): save('norm_output', first_tensor(output))

    HANDLES.extend([layer.input_layernorm.register_forward_pre_hook(norm_before),
                    layer.input_layernorm.register_forward_hook(norm_after)])
    original_project = impl._project_q_kv

    def project(self, owner, hidden_states, cos, sin):
        output = original_project(owner, hidden_states, cos, sin)
        if owner is attn and eligible(hidden_states):
            for name, value in zip(('project_q', 'project_qr', 'project_kv'), output):
                save(name, value)
        return output

    impl._project_q_kv = MethodType(project, impl)
    original_core = impl._native_attention

    def core(self, owner, q, metadata, *, source_cache, compressed_indices):
        selected = owner is attn and eligible(q) and 'core_q' not in CURRENT
        if selected:
            save('core_q', q)
            # Compare visible KV in logical token order, not raw physical page
            # IDs. Only the 128 visible rows are read; unused cache is excluded.
            cache = owner.dsa_attn.swa_cache_layer.kv_cache[0]
            sequence = metadata.swa.seq_lens[:1]
            offsets = torch.arange(-owner.window_size, 0, device=q.device)
            positions = (sequence[0] + offsets).clamp_min(0).long()
            blocks = metadata.swa.block_table[0].index_select(
                0, torch.div(positions, cache.shape[1], rounding_mode='floor')).long()
            visible = cache[blocks, positions.remainder(cache.shape[1])]
            save('core_cache_visible', visible)
            save('core_seq_lens', sequence)
            save('core_query_start', metadata.swa.query_start_loc[:2])
            save('core_metadata', metadata.swa.smla_metadata)
        output = original_core(owner, q, metadata, source_cache=source_cache,
                               compressed_indices=compressed_indices)
        if selected: save('core_output', output)
        return output

    impl._native_attention = MethodType(core, impl)
    original_oproj = v1_impl._forward_o_proj

    def oproj(self, x, output):
        selected = eligible(x) and 'oproj_input' not in CURRENT
        if selected: save('oproj_input', x)
        result = original_oproj(x, output)
        if selected: save('oproj_output', result)
        return result

    v1_impl._forward_o_proj = MethodType(oproj, v1_impl)
    original_apply = attn.wo_b.quant_method.apply

    def apply(self, module, x, *args, **kwargs):
        selected = module is attn.wo_b and eligible(x) and 'wo_b_local' not in CURRENT
        if selected: save('wo_b_input', x)
        result = original_apply(module, x, *args, **kwargs)
        if selected: save('wo_b_local', first_tensor(result))
        return result

    attn.wo_b.quant_method.apply = MethodType(apply, attn.wo_b.quant_method)

    def wo_b_after(module, inputs, output):
        if eligible(inputs[0]): save('wo_b_output', first_tensor(output))

    HANDLES.append(attn.wo_b.register_forward_hook(wo_b_after))
    return {'layer': 0, 'stages': list(STAGES), 'native_computations_preserved': True,
            'scope': 'First real eager decode; NPU snapshots perturb timing; no performance claim'}


def begin(worker, tag):
    global ENABLED
    torch.npu.synchronize()
    assert tag not in SNAPSHOTS
    CURRENT.clear()
    ENABLED = True
    return {'tag': tag}


def finish(worker, tag):
    global ENABLED
    ENABLED = False
    torch.npu.synchronize()
    assert set(CURRENT) == set(STAGES), {'missing': sorted(set(STAGES) - set(CURRENT))}
    SNAPSHOTS[tag] = dict(CURRENT)
    return {'tag': tag, 'coverage': len(CURRENT)}


def compare_tensor(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype
    return {'equal': torch.equal(x, y), 'shape': list(x.shape), 'dtype': str(x.dtype),
            'differing_elements': int((x != y).sum().item()),
            'max_abs': float((x.float() - y.float()).abs().max().item())}


@torch.inference_mode()
def compare(worker, reference, actual):
    from vllm.distributed import get_tensor_model_parallel_rank
    rows = [dict(stage=name, **compare_tensor(SNAPSHOTS[reference][name], SNAPSHOTS[actual][name]))
            for name in STAGES]
    return {'rank': get_tensor_model_parallel_rank(), 'boundaries': rows,
            'first_difference': next((r for r in rows if not r['equal']), None),
            'scope': 'Cross-request boundaries; metadata is compared raw, cache in visible logical order'}


@torch.inference_mode()
def repeat_allreduce(worker, tag, repeats=10):
    from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
    assert not ENABLED and repeats >= 2
    data = SNAPSHOTS[tag]
    reference, results = None, []
    for index in range(repeats):
        # Each rank replays its own saved local vector, with every collective
        # participating on all 8 ranks in the same order.
        output = get_tp_group().all_reduce(data['wo_b_local'].clone())
        torch.npu.synchronize()
        if reference is None: reference = output.clone()
        results.append({'repeat': index, 'vs_first': compare_tensor(reference, output),
                        'vs_forward': compare_tensor(data['wo_b_output'], output)})
    return {'rank': get_tensor_model_parallel_rank(), 'repeats': results,
            'scope': 'Native TP allreduce on identical saved rank-local wo_b vectors; collective timing perturbed'}
