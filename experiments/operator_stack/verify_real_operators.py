"""Selective formal weights, synthetic inputs, original numerical gates.

This is an operator compatibility check, not full-model TP8 validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from safetensors import safe_open

from formal_model_contract import inspect_checkpoint
from hc_static import hc_static, round_hf32
from indexer_post import indexer_post
from indexer_patches import native_post


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--section', choices=['both', 'hc', 'indexer'], default='both')
    parser.add_argument('--hc-candidate', choices=['static', 'sequential', 'divrn'], default='static')
    parser.add_argument('--cache-block', type=int, choices=[0, 1], default=0)
    parser.add_argument('--indexer-candidate', choices=['fused', 'scalar'], default='fused')
    args = parser.parse_args()
    contract = inspect_checkpoint(args.model)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {'checkpoint_config_sha256': contract['config_sha256'],
              'model': args.model, 'weights': 'unaltered formal checkpoint tensors',
              'inputs': 'synthetic; not observed full-model activations',
              'device': 0, 'visible_physical_chips': __import__('os').environ.get('STACK_PHYSICAL_CHIPS', __import__('os').environ['ASCEND_RT_VISIBLE_DEVICES']),
              'section': args.section, 'hc_candidate': args.hc_candidate, 'cache_block': args.cache_block,
              'indexer_candidate': args.indexer_candidate,
              'status': 'started', 'hc_cases': [], 'indexer_cases': [], 'tensor_sha256': {}}
    target.write_text(json.dumps(result, indent=2) + '\n')
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op
    bootstrap_custom_op_env()
    assert enable_custom_op(), 'Vendor custom operators must be available for the native reference'
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    torch.manual_seed(20261010)
    config = json.loads((Path(args.model) / 'config.json').read_text())
    text = config.get('text_config', config)
    tensors = {}
    candidate_post = indexer_post
    if args.indexer_candidate == 'scalar':
        from indexer_post_scalar import indexer_post_scalar
        candidate_post = indexer_post_scalar

    def load(name):
        if name not in tensors:
            info = contract['operator_tensors'][name]
            with safe_open(str(Path(args.model) / info['shard']), framework='pt', device='cpu') as archive:
                tensor = archive.get_tensor(name).contiguous()
            result['tensor_sha256'][name] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
            dtype = torch.bfloat16 if info.get('runtime_dtype') == 'BF16' else tensor.dtype
            tensors[name] = tensor.to(device='npu', dtype=dtype)
        return tensors[name]

    try:
        for name in sorted(k for k in contract['operator_tensors'] if k.endswith('_fn') and args.section in ['both', 'hc']):
            fn = load(name)
            prefix = name[:-3]
            scale, bias = load(prefix + '_scale'), load(prefix + '_base')
            rounded = round_hf32(fn)
            for magnitude in [.001, 1., 100.]:
                for use_mix in [False, True]:
                    x = (torch.randn((1, 4, 5120), device='npu') * magnitude).to(torch.bfloat16)
                    mix = torch.ones((1, 4), device='npu') if use_mix else None
                    kwargs = dict(hc_sinkhorn_iters=int(text.get('hc_sinkhorn_iters', 20)),
                                  norm_eps=1e-20, hc_eps=1e-6)
                    reference = torch.ops._C_ascend.npu_hc_pre_v2(x, fn, scale, bias, mix,
                                                                               hc_mult=4, **kwargs)
                    if args.hc_candidate == 'divrn':
                        from real_hc_divrn import hc_divrn
                        candidate = hc_divrn(x, rounded, scale, bias, mix, **kwargs)
                    else:
                        candidate = hc_static(x, rounded, scale, bias, mix, **kwargs)
                    if args.hc_candidate == 'sequential':
                        from real_hc_candidate import replace_collapse
                        candidate = replace_collapse(x, candidate, mix)
                    result['current_case'] = {'tensor': name, 'magnitude': magnitude, 'pre_mix': use_mix}
                    errors = []
                    for i, (actual, expected) in enumerate(zip(candidate, reference)):
                        torch.testing.assert_close(actual.float(), expected.float(),
                                                   rtol=4e-3 if i == 0 else 1e-4,
                                                   atol=4e-3 if i == 0 else 1e-4)
                        errors.append(float((actual.float() - expected.float()).abs().max().item()))
                    result['hc_cases'].append({'tensor': name, 'magnitude': magnitude,
                                               'pre_mix': use_mix, 'max_abs_errors': errors})
            print('REAL_HC_WEIGHT_PASSED', name, flush=True)
        result.pop('current_case', None)
        for name in sorted(k for k in contract['operator_tensors'] if k.endswith('wk.weight') and args.section in ['both', 'indexer']):
            weight = load(name)
            gamma = load(name.replace('wk.weight', 'k_norm.weight'))
            eps = float(text.get('rms_norm_eps', 1e-6))
            module = SimpleNamespace(width=128, rope_width=64,
                                     k_norm=lambda value: torch_npu.npu_rms_norm(value, gamma, epsilon=eps)[0])
            for rows in [1, 4]:
                for magnitude in [0., .001, 1., 100.]:
                    for rope_dtype in [torch.bfloat16, torch.float32]:
                        latent = (torch.randn((rows, 512), device='npu') * magnitude).to(torch.bfloat16)
                        projected = torch.nn.functional.linear(latent, weight)
                        angle = torch.randn((rows, 32), device='npu').repeat_interleave(2, dim=-1)
                        cos = angle.cos().to(rope_dtype).view(rows, 1, 1, 64)
                        sin = angle.sin().to(rope_dtype).view(rows, 1, 1, 64)
                        coordinates = torch.tensor([[args.cache_block, i] for i in range(rows)], device='npu', dtype=torch.int64)
                        if rows == 4:
                            coordinates[-1] = -1
                        # Strided hybrid-like layout, nonzero unwritten sentinels.
                        # Hybrid caches pad between pages, while tokens within
                        # a page remain dense. A gap between token rows is not
                        # the native scatter operator's layout contract.
                        key_storage = torch.full((2, 256, 1, 128), 19, device='npu', dtype=torch.int8)
                        scale_storage = torch.full((2, 256, 1, 1), 1.5, device='npu', dtype=torch.float16)
                        reference_key, reference_scale = key_storage[:, :128], scale_storage[:, :128]
                        candidate_key = key_storage.clone()[:, :128]
                        candidate_scale = scale_storage.clone()[:, :128]
                        quant, scale = native_post(module, projected, coordinates, cos, sin,
                                                   reference_key, reference_scale)
                        debug = candidate_post(projected, gamma, cos.reshape(-1,64), sin.reshape(-1,64), coordinates,
                                             candidate_key, candidate_scale, eps, debug=True)
                        assert torch.equal(debug[2], quant), (name, rows, magnitude, rope_dtype, 'functional INT8')
                        assert torch.equal(debug[3], scale), (name, rows, magnitude, rope_dtype, 'functional scale')
                        assert torch.equal(candidate_key, reference_key), (name, 'cache INT8')
                        assert torch.equal(candidate_scale, reference_scale), (name, 'cache FP16 scale')
                        result['indexer_cases'].append({'tensor': name, 'rows': rows, 'magnitude': magnitude,
                                                       'rope_dtype': str(rope_dtype), 'bitwise_equal': True,
                                                       'masked_rows': int(rows == 4)})
            print('REAL_INDEXER_WEIGHT_PASSED', name, flush=True)
        result['status'] = 'passed'
        result.pop('current_case', None)
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = str(exc)
        if args.section in ['both', 'hc'] and 'current_case' in result:
            torch.save({'case': result['current_case'], 'x': x.cpu(),
                        'pre_mix': mix.cpu() if mix is not None else None,
                        'native': [value.cpu() for value in reference],
                        'candidate': [value.cpu() for value in candidate]},
                       target.with_suffix('.failure.pt'))
        raise
    finally:
        result['hc_case_count'] = len(result['hc_cases'])
        result['indexer_case_count'] = len(result['indexer_cases'])
        target.write_text(json.dumps(result, indent=2) + '\n')
        print('REAL_OPERATOR_PRECISION', json.dumps({k: v for k, v in result.items()
                                                    if k not in ['hc_cases', 'indexer_cases', 'tensor_sha256']}), flush=True)


if __name__ == '__main__':
    main()
