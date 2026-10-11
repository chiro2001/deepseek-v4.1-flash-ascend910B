"""Merge native replicated Q_a/KV for decode, retaining the native tail."""
import functools
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

import torch
import torch_npu

ARM = 'tp8base'
INSTALLED = False
LAYERS = {}
REFS = {}
SOURCE = {}


def set_arm(name):
    global ARM
    ARM = name


@torch.library.custom_op('formal_qkv::project', mutates_args=())
def project(x: torch.Tensor, token_scale: torch.Tensor, weight: torch.Tensor,
            weight_scale: torch.Tensor, layer_id: int) -> torch.Tensor:
    output = torch_npu.npu_quant_matmul(x, weight, weight_scale,
                                      pertoken_scale=token_scale, output_dtype=torch.bfloat16)
    from vllm.forward_context import get_forward_context
    if get_forward_context().capturing:
        audit = os.getenv('STACK_REAL_AUDIT') == '1'
        REFS.setdefault(ARM, []).append({'x': x.clone() if audit else x,
                                        'token_scale': token_scale.clone() if audit else token_scale,
                                        'output': output.clone() if audit else output,
                                        'layer_id': layer_id})
    return output


@project.register_fake
def project_fake(x, token_scale, weight, weight_scale, layer_id):
    return torch.empty((x.shape[0], 1792), device=x.device, dtype=torch.bfloat16)


def merged(impl, x, token_scale):
    entry = LAYERS[id(impl)]
    assert ARM == 'tp8qkv' and x.shape == (1, 5120) and x.dtype == torch.int8
    assert token_scale.shape == (1,) and token_scale.dtype == torch.float32
    assert all(w._version == v for w, v in entry['versioned_weights'])
    output = project(x, token_scale, entry['packed_weight'], entry['packed_scale'], id(impl))
    return output[:, :1280], output[:, 1280:]


@torch.inference_mode()
def install(model):
    global INSTALLED
    if INSTALLED or os.getenv('STACK_QKV_MERGE_ENABLED') != '1':
        return
    assert os.getenv('STACK_REAL_WEIGHTS') == '1'
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, _is_w8a8_dynamic
    original = AscendDSAImpl._mla_prolog_multistream
    source_path = Path(inspect.getfile(AscendDSAImpl))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == os.environ['STACK_QKV_NATIVE_FILE_SHA256']
    raw = inspect.getsource(original)
    tree = textwrap.dedent(raw)
    qa = 'wq_a_result = self.cv_wq_a.matmul(q_quant, q_pertoken_scale)'
    kv = 'kv = self.cv_wkv.matmul(kv_quant, kv_pertoken_scale)'
    assert tree.count(qa) == tree.count(kv) == 1
    updated = tree.replace(qa, 'wq_a_result, formal_kv = formal_qkv_project(self, q_quant, q_pertoken_scale)')
    updated = updated.replace(kv, 'kv = formal_kv')
    namespace = {**original.__globals__, 'formal_qkv_project': merged}
    exec(compile(updated, '<formal_qkv_merge_prolog>', 'exec'), namespace)
    candidate = namespace['_mla_prolog_multistream']
    signature = inspect.signature(original)
    for name, module in model.named_modules():
        wrapper = getattr(module, 'dsa_attn', None)
        attention = getattr(wrapper, 'dsa_attn', None)
        impl = getattr(attention, 'impl', None)
        if impl is None or not isinstance(impl, AscendDSAImpl) or id(impl) in LAYERS:
            continue
        assert _is_w8a8_dynamic(impl.wq_a) and _is_w8a8_dynamic(impl.wkv)
        assert not impl.cv_wq_a._has_communication and not impl.cv_wkv._has_communication
        qa_weight, kv_weight = impl.wq_a.weight, impl.wkv.weight
        qa_scale, kv_scale = impl.wq_a.weight_scale, impl.wkv.weight_scale
        assert qa_weight.shape == (5120, 1280) and kv_weight.shape == (5120, 512)
        assert qa_weight.dtype == kv_weight.dtype == torch.int8
        assert qa_scale.shape == (1280,) and kv_scale.shape == (512,)
        assert qa_scale.dtype == kv_scale.dtype == torch.bfloat16
        assert impl.wq_a.bias is None and impl.wkv.bias is None
        assert all(torch_npu.get_npu_format(w) == 29 for w in (qa_weight, kv_weight))
        nd_weights = [torch_npu.npu_format_cast(w, 2) for w in (qa_weight, kv_weight)]
        packed_nd = torch.cat(nd_weights, dim=1).contiguous()
        packed = torch_npu.npu_format_cast(packed_nd, 29)
        assert torch.equal(torch_npu.npu_format_cast(packed, 2), packed_nd)
        LAYERS[id(impl)] = {'impl': impl, 'name': name, 'packed_weight': packed,
                            'packed_scale': torch.cat((qa_scale, kv_scale)),
                            'versioned_weights': [(w, w._version) for w in
                                                  (qa_weight, kv_weight, qa_scale, kv_scale)]}
    assert len(LAYERS) == 40, len(LAYERS)

    @functools.wraps(original)
    def dispatch(self, hidden_states, *args, **kwargs):
        if ARM == 'tp8qkv' and id(self) in LAYERS and hidden_states.shape == (1, 5120):
            assert hidden_states.dtype == torch.bfloat16
            bound = signature.bind(self, hidden_states, *args, **kwargs)
            assert not bound.arguments.get('is_prefill', False)
            return candidate(self, hidden_states, *args, **kwargs)
        return original(self, hidden_states, *args, **kwargs)
    AscendDSAImpl._mla_prolog_multistream = dispatch
    SOURCE.update(native_file_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  native_method_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                  patched_method_sha256=hashlib.sha256(updated.encode()).hexdigest(),
                  patch_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  preserved_tail='Native stream/events, RMSNorm, RoPE, SWA cache, Q_b and compressor')
    INSTALLED = True


def reset(name):
    REFS[name] = []


@torch.inference_mode()
def warm():
    for entry in LAYERS.values():
        weight = entry['packed_weight']
        x = torch.zeros((1, 5120), dtype=torch.int8, device=weight.device)
        scale = torch.ones((1,), dtype=torch.float32, device=weight.device)
        torch_npu.npu_quant_matmul(x, weight, entry['packed_scale'], pertoken_scale=scale,
                                 output_dtype=torch.bfloat16)
    torch.npu.synchronize()


def status(name):
    rows = REFS.get(name, [])
    return {'enabled': INSTALLED, 'registered_layers': len(LAYERS), 'captured_calls': len(rows),
            'selected_calls': len(rows), 'unique_layers': len({r['layer_id'] for r in rows}),
            'source': SOURCE}


@torch.inference_mode()
def audit(name):
    rows = REFS.get(name, [])
    assert name == 'tp8qkv' and len(rows) == 40 and status(name)['unique_layers'] == 40, status(name)
    for row in rows:
        entry = LAYERS[row['layer_id']]
        impl = entry['impl']
        expected = [torch_npu.npu_quant_matmul(row['x'], layer.weight, layer.weight_scale,
                                             pertoken_scale=row['token_scale'], output_dtype=torch.bfloat16)
                    for layer in (impl.wq_a, impl.wkv)]
        actual = (row['output'][:, :1280], row['output'][:, 1280:])
        for a, b in zip(actual, expected):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), entry['name']
    return {'passed': True, 'consumer_calls': len(rows), 'unique_layers': 40,
            'all_bitwise_equal': True,
            'scope': 'Actual captured forty-layer Q_a and KV projections; model/cache gates are separate'}
