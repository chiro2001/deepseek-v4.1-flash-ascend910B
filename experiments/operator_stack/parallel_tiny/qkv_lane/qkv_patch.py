"""BF16 TP1 synthetic tiny: merge native Q_a/KV, retain every native tail op."""
import functools
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import torch
import torch_npu

ARM = 'native_a'
LAYERS = {}
REFS = {}
BANKS = {}
SOURCE = {}


@torch.library.custom_op('tiny_qkv_lane::project', mutates_args=())
def project(x: torch.Tensor, weight: torch.Tensor, layer_id: int) -> torch.Tensor:
    output = torch.ops.vllm.unquantized_gemm(x, weight, None)
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing:
        audit = os.getenv('QKV_LANE_AUDIT') == '1'
        REFS.setdefault(ARM, []).append({'x': x.clone() if audit else x,
                                        'output': output.clone() if audit else output,
                                        'layer_id': layer_id})
    return output


@project.register_fake
def project_fake(x, weight, layer_id):
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)


def merged(impl, x, token_scale):
    entry = LAYERS[id(impl)]
    assert ARM == 'merged' and x.shape == (1, 5120) and x.dtype == torch.bfloat16
    assert token_scale is None
    output = project(x, entry['packed_weight'], id(impl))
    return output[:, :512], output[:, 512:]


@torch.no_grad()
def install(model):
    assert not LAYERS
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
    from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl
    original = DeepseekV41EagerAttentionImpl.multistream_preprocess
    raw = inspect.getsource(original)
    source = textwrap.dedent(raw)
    qa = 'q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)'
    kv = 'kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)'
    assert source.count(qa) == source.count(kv) == 1
    updated = source.replace(qa, 'q_a, merged_kv = tiny_qkv_project(v1_impl, q_quant, q_scale)')
    updated = updated.replace(kv, 'kv = merged_kv')
    namespace = {**original.__globals__, 'tiny_qkv_project': merged}
    exec(compile(updated, '<tiny_qkv_lane_prolog>', 'exec'), namespace)
    candidate = namespace['multistream_preprocess']
    for name, module in model.named_modules():
        wrapper = getattr(module, 'dsa_attn', None)
        attention = getattr(wrapper, 'dsa_attn', None)
        impl = getattr(attention, 'impl', None)
        if impl is None or not isinstance(impl, AscendDSAImpl) or id(impl) in LAYERS:
            continue
        assert isinstance(getattr(module, 'v41_impl', None), DeepseekV41EagerAttentionImpl), name
        assert not impl.cv_wq_a._has_communication and not impl.cv_wkv._has_communication
        assert not impl.cv_wq_a._is_w8a8_dynamic and not impl.cv_wkv._is_w8a8_dynamic
        assert type(impl.cv_wq_a._quant_method) is type(impl.cv_wkv._quant_method)
        qa_weight, kv_weight = impl.wq_a.weight, impl.wkv.weight
        assert qa_weight.shape == kv_weight.shape == (512, 5120)
        assert qa_weight.dtype == kv_weight.dtype == torch.bfloat16
        assert impl.wq_a.bias is None and impl.wkv.bias is None
        nd = torch.cat([torch_npu.npu_format_cast(w, 2) for w in (qa_weight, kv_weight)], dim=0).contiguous()
        requested_format = int(os.getenv('QKV_LANE_PACK_FORMAT', '2'))
        assert requested_format in (2, 29)
        packed = torch_npu.npu_format_cast(nd, requested_format)
        packed_format = int(torch_npu.get_npu_format(packed))
        assert packed_format == requested_format, ('Requested QKV layout was not materialized', requested_format, packed_format)
        assert torch.equal(torch_npu.npu_format_cast(packed, 2), nd)
        LAYERS[id(impl)] = {'impl': impl, 'name': name, 'packed_weight': packed, 'packed_format':packed_format,
                            'native_formats': [torch_npu.get_npu_format(w) for w in (qa_weight, kv_weight)]}
    assert len(LAYERS) == 40, len(LAYERS)

    @functools.wraps(original)
    def dispatch(self, attn, hidden_states, *args, **kwargs):
        impl = attn.dsa_attn.dsa_attn.impl
        if ARM == 'merged' and id(impl) in LAYERS and hidden_states.shape == (1, 5120):
            return candidate(self, attn, hidden_states, *args, **kwargs)
        return original(self, attn, hidden_states, *args, **kwargs)
    DeepseekV41EagerAttentionImpl.multistream_preprocess = dispatch
    SOURCE.update(native_file_sha256=hashlib.sha256(Path(inspect.getfile(DeepseekV41EagerAttentionImpl)).read_bytes()).hexdigest(),
                  native_method_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                  candidate_method_sha256=hashlib.sha256(updated.encode()).hexdigest(),
                  adapter_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  actual_boundary='DeepseekV41EagerAttentionImpl.multistream_preprocess',
                  preserved_tail='Native floating qr, RMSNorm, Q_b, RoPE, SWA cache, compressor, stream/events')


def set_arm(name):
    global ARM
    assert name in ('native_a', 'native_b', 'merged')
    ARM = name
    # Match formal staging: initial cold/static compilation is single-stream;
    # every measured native/candidate bank is subsequently multistream.
    for entry in LAYERS.values():
        entry['impl'].multistream_dsv4_dsa_overlap = True


def status(worker=None):
    rows = REFS.get(ARM, [])
    return {'arm': ARM, 'registered_layers': len(LAYERS), 'captured_calls': len(rows),
            'multistream_layers': sum(e['impl'].multistream_dsv4_dsa_overlap for e in LAYERS.values()),
            'unique_layers': len({r['layer_id'] for r in rows}), 'source': SOURCE,
            'layout': {'native': sorted({tuple(e['native_formats']) for e in LAYERS.values()}),
                       'merged': sorted({e['packed_format'] for e in LAYERS.values()})}}


def save(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    entries = [(wrapper, wrapper.concrete_aclgraph_entries, wrapper.graph_pool)
               for wrapper in ag._acl_graph_wrappers]
    assert sum(len(value) for _, value, _ in entries) == 1
    BANKS[name] = (entries, ag._graph_params)
    return status(worker)


@torch.inference_mode()
def create(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize()
    set_arm(name)
    if name == 'merged':
        for entry in LAYERS.values():
            x = torch.zeros((1, 5120), device=entry['packed_weight'].device, dtype=torch.bfloat16)
            torch.ops.vllm.unquantized_gemm(x, entry['packed_weight'], None)
        torch.npu.synchronize()
    REFS[name] = []
    pool = torch.npu.graph_pool_handle()
    for wrapper in ag._acl_graph_wrappers:
        wrapper.concrete_aclgraph_entries = {}
        wrapper.graph_pool = pool
    ag._graph_params = None
    ag.set_graph_params([1])
    worker.model_runner.capture_model()
    receipt = save(worker, name)
    if name == 'merged':
        assert receipt['captured_calls'] == receipt['unique_layers'] == 40, receipt
    return receipt


def switch(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize()
    set_arm(name)
    entries, params = BANKS[name]
    for wrapper, values, pool in entries:
        wrapper.concrete_aclgraph_entries = values
        wrapper.graph_pool = pool
    ag._graph_params = params
    return {'arm': name}


def _ordered(tensor):
    bits = tensor.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -bits - 32768, bits)


@torch.inference_mode()
def audit(worker):
    assert os.getenv('QKV_LANE_AUDIT') == '1'
    rows = REFS['merged']
    assert len(rows) == 40 and len({r['layer_id'] for r in rows}) == 40
    reports = []
    for row in rows:
        entry = LAYERS[row['layer_id']]
        impl = entry['impl']
        expected = [wrapper.matmul(row['x'], None) for wrapper in (impl.cv_wq_a, impl.cv_wkv)]
        actual = (row['output'][:, :512], row['output'][:, 512:])
        for kind, a, b in zip(('q_a', 'kv'), actual, expected):
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), entry['name']
            ulp = int((_ordered(a) - _ordered(b)).abs().max().item())
            exact = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            reports.append({'layer': entry['name'], 'projection': kind, 'max_bf16_ulp': ulp,
                            'bitwise_equal': exact, 'max_abs': float((a.float()-b.float()).abs().max().item()),
                            'input_absmax': float(row['x'].float().abs().max().item())})
    result = {'scope': 'Full synthetic BF16 TP1 tiny graph actual captured Q_a/KV consumers',
              'consumer_calls': 40, 'projection_outputs': 80, 'max_bf16_ulp': max(r['max_bf16_ulp'] for r in reports),
              'all_bitwise_equal': all(r['bitwise_equal'] for r in reports), 'rows': reports}
    result['passed'] = result['max_bf16_ulp'] <= 1
    return result
