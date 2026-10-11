"""Probe merged native INT8 Q_a/KV projection and a bounded MLA prolog.

This uses unaltered replicated weights and the existing native quantized
matmul. Synthetic inputs and a prolog without RoPE/cache do not prove TP8
model precision or an end-to-end gain.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
from safetensors import safe_open


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    equal = bool(torch.equal(actual, expected))
    record = {'dtype': str(actual.dtype), 'shape': list(actual.shape), 'exact_equal': equal}
    if actual.dtype == torch.bfloat16:
        def ordered(x):
            bits = x.view(torch.int16).to(torch.int32) & 65535
            return torch.where(bits & 32768 != 0, 32768 - (bits & 32767), 32768 + bits)
        ulp = (ordered(actual) - ordered(expected)).abs()
        record.update(finite=bool(torch.isfinite(actual).all()), max_bf16_ulp=int(ulp.max()),
                      elements_above_one_ulp=int((ulp > 1).sum()),
                      bitwise_equal=bool(torch.equal(actual.view(torch.int16), expected.view(torch.int16))))
        record['passed'] = record['finite'] and record['elements_above_one_ulp'] == 0
    else:
        record['passed'] = equal
    return record


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-chips', required=True)
    args = parser.parse_args()
    assert args.physical_chips == os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '8,9,10,11,12,13,14,15'
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op
    bootstrap_custom_op_env()
    assert enable_custom_op(), 'Native RMSNormDynamicQuant reference must be available'
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    torch_npu.npu.config.allow_internal_format = True
    args.output.mkdir(parents=True, exist_ok=True)
    config_bytes = (args.model / 'config.json').read_bytes()
    cfg = json.loads(config_bytes)['text_config']
    assert (cfg['hidden_size'], cfg['q_lora_rank'], cfg['num_hidden_layers']) == (5120, 1280, 40)
    wm = json.loads((args.model / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    ascend_root = Path(importlib.util.find_spec('vllm_ascend').origin).parent
    source_path = ascend_root / 'attention/dsa_v1.py'
    source_file = source_path.read_text()
    impl = next(node for node in ast.parse(source_file).body
                if isinstance(node, ast.ClassDef) and node.name == 'AscendDSAImpl')
    method = next(node for node in impl.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_mla_prolog_multistream')
    source = ast.get_source_segment(source_file, method)
    assert 'share_quant' in source and 'main_stream.wait_event(e_kv_matmul_done)' in source
    result = {'completed': False, 'precision_passed': False, 'eligible_for_model_trial': False,
              'scope': 'Independent real replicated Q_a/KV and rank0 Q_b weights; synthetic inputs; '
                       'bounded prolog excludes RoPE, cache, Q-head normalization, and compressor',
              'executing_chip': 8, 'physical_chips': args.physical_chips, 'model': str(args.model),
              'checkpoint_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
              'native_multistream_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'native_source_file_sha256': hashlib.sha256(source_file.encode()).hexdigest(),
              'native_source_extraction': 'AST method segment; attention module not imported',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'precision_gate': 'Original one-BF16-ULP gate plus exact downstream quantized Q/scale',
              'profiler_during_timing': 'OFF', 'weights': [], 'cases': [], 'pairs': []}
    def save():
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def load(key, rows=None):
        with safe_open(str(args.model / wm[key]), framework='pt', device='cpu') as f:
            return (f.get_slice(key)[:rows] if rows is not None else f.get_tensor(key)).contiguous()
    save()
    for layer in (0, 20):
        prefix = f'layers.{layer}.attn.'
        cpu_weights = [load(prefix + part + '.weight', n) for part, n in
                       [('wq_a', 1280), ('wkv', 512), ('wq_b', 4096)]]
        assert [tuple(w.shape) for w in cpu_weights] == [(1280, 5120), (512, 5120), (4096, 1280)]
        assert all(w.dtype == torch.int8 for w in cpu_weights)
        cpu_scales = [load(prefix + part + '.weight_scale', n).flatten().to(torch.bfloat16)
                      for part, n in [('wq_a', 1280), ('wkv', 512), ('wq_b', 4096)]]
        weights = [torch_npu.npu_format_cast(w.t().contiguous().to('npu'), 29) for w in cpu_weights]
        scales = [s.to('npu') for s in cpu_scales]
        packed_cpu = torch.cat(cpu_weights[:2], dim=0).t().contiguous()
        packed = torch_npu.npu_format_cast(packed_cpu.to('npu'), 29)
        packed_scale = torch.cat(cpu_scales[:2]).to('npu')
        assert torch_npu.get_npu_format(packed) == 29
        assert torch.equal(torch_npu.npu_format_cast(packed, 2).cpu(), packed_cpu)
        q_norm = load(prefix + 'q_norm.weight').to(dtype=torch.bfloat16, device='npu')
        kv_norm = load(prefix + 'kv_norm.weight').to(dtype=torch.bfloat16, device='npu')
        assert tuple(q_norm.shape) == (1280,) and tuple(kv_norm.shape) == (512,)
        result['weights'].append({'layer': layer, 'q_b_rank_shard': 0,
                                 'weight_sha256': [digest(w) for w in cpu_weights],
                                 'loaded_bf16_scale_sha256': [digest(s) for s in cpu_scales],
                                 'packed_weight_sha256': digest(packed_cpu),
                                 'packed_shape': list(packed.shape), 'packed_format': 29,
                                 'packed_roundtrip_bitwise_equal': True})
        def matmul(x, weight, scale, token_scale):
            return torch_npu.npu_quant_matmul(x, weight, scale, pertoken_scale=token_scale,
                                            output_dtype=torch.bfloat16)
        def projections(hidden, merge):
            x, token_scale = torch_npu.npu_dynamic_quant(hidden)
            if merge:
                output = matmul(x, packed, packed_scale, token_scale)
                return output[:, :1280], output[:, 1280:]
            return matmul(x, weights[0], scales[0], token_scale), matmul(x, weights[1], scales[1], token_scale)
        aux = torch.npu.Stream()
        def prolog(hidden, merge):
            main_stream = torch.npu.current_stream()
            x, token_scale = torch_npu.npu_dynamic_quant(hidden)
            if merge:
                output = matmul(x, packed, packed_scale, token_scale)
                qa, kv = output[:, :1280], output[:, 1280:]
                start = main_stream.record_event()
                with torch.npu.stream(aux):
                    aux.wait_event(start)
                    kv_out = torch_npu.npu_rms_norm(kv, kv_norm, epsilon=cfg['rms_norm_eps'])[0]
                qr, qs = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(qa, q_norm, epsilon=cfg['rms_norm_eps'])
                q = matmul(qr, weights[2], scales[2], qs)
            else:
                qa = matmul(x, weights[0], scales[0], token_scale)
                start = main_stream.record_event()
                with torch.npu.stream(aux):
                    aux.wait_event(start)
                    kv = matmul(x, weights[1], scales[1], token_scale)
                    kv_done = aux.record_event()
                qr, qs = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(qa, q_norm, epsilon=cfg['rms_norm_eps'])
                norm_done = main_stream.record_event()
                main_stream.wait_event(kv_done)
                with torch.npu.stream(aux):
                    aux.wait_event(norm_done)
                    kv_out = torch_npu.npu_rms_norm(kv, kv_norm, epsilon=cfg['rms_norm_eps'])[0]
                q = matmul(qr, weights[2], scales[2], qs)
            main_stream.wait_stream(aux)
            return qa, kv, qr, qs, q, kv_out
        names = ['q_a', 'kv_raw', 'q_norm_quant', 'q_norm_scale', 'q_b', 'kv_norm']
        for magnitude in (.001, 1., 100.):
            for style in ('random', 'constant', 'alternating', 'outlier'):
                torch.manual_seed(20261011 + layer)
                hidden = torch.randn((1, 5120), device='npu') * magnitude
                if style == 'constant': hidden.fill_(magnitude)
                elif style == 'alternating': hidden = torch.where(torch.arange(5120, device='npu') % 2 == 0, magnitude, -magnitude).reshape(1, -1)
                elif style == 'outlier': hidden[0, 0] = magnitude * 1000
                hidden = hidden.to(torch.bfloat16)
                expected, actual = prolog(hidden, False), prolog(hidden, True)
                row = {'layer': layer, 'magnitude': magnitude, 'style': style,
                       'outputs': {name: compare(a, b) for name, a, b in zip(names, actual, expected)}}
                row['passed'] = all(v['passed'] for v in row['outputs'].values())
                result['cases'].append(row); save()
                print('QKV_MERGE_PRECISION', json.dumps(row), flush=True)
                if not row['passed']:
                    result.update(completed=True, rejection='Independent downstream precision gate failed')
                    save(); return
        functions = {'separate_projection': lambda x: projections(x, False),
                     'merged_projection': lambda x: projections(x, True),
                     'native_multistream_prolog': lambda x: prolog(x, False),
                     'merged_multistream_prolog': lambda x: prolog(x, True)}
        calls, replays = 128, 8
        inputs = [torch.randn((1, 5120), device='npu', dtype=torch.bfloat16) for _ in range(calls)]
        graphs, references = {}, {}
        for name, fn in functions.items():
            for _ in range(3): fn(inputs[0])
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                refs = [fn(x) for x in inputs]
            graphs[name], references[name] = graph, refs
        result['timing_contract'] = {'calls_per_graph': calls, 'replays': replays,
                                    'distinct_input_buffers': calls, 'minimum_device_to_submission_ratio': 2.}
        for pair in range(8):
            times, submission = {}, {}
            order = list(functions) if pair % 2 == 0 else list(reversed(functions))
            for name in order:
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record(); started = time.perf_counter()
                for _ in range(replays): graphs[name].replay()
                submission[name] = (time.perf_counter() - started) * 1e6 / replays
                end.record(); end.synchronize()
                times[name] = begin.elapsed_time(end) * 1000 / (calls * replays)
            result['pairs'].append({'layer': layer, 'pair': pair, 'microseconds': times,
                                    'device_to_submission_ratio': {n: times[n] * calls / submission[n] for n in times}})
            save()
        del graphs, references, functions, inputs
    assert len(result['cases']) == 24 and len(result['pairs']) == 16
    result['summary'] = {}
    for label, native, merged in [('projection', 'separate_projection', 'merged_projection'),
                                  ('bounded_prolog', 'native_multistream_prolog', 'merged_multistream_prolog')]:
        rows = result['pairs']
        ratios = [r['microseconds'][native] / r['microseconds'][merged] for r in rows]
        result['summary'][label] = {'native_us': statistics.median(r['microseconds'][native] for r in rows),
                                    'merged_us': statistics.median(r['microseconds'][merged] for r in rows),
                                    'paired_speedup_median': statistics.median(ratios),
                                    'faster_pairs': sum(v > 1 for v in ratios), 'pairs': len(rows)}
    result['timing_validated'] = all(v >= 2 for r in result['pairs'] for v in r['device_to_submission_ratio'].values())
    result.update(completed=True, precision_passed=True)
    result['eligible_for_model_trial'] = result['timing_validated'] and result['summary']['bounded_prolog']['paired_speedup_median'] > 1.01
    save(); print('QKV_MERGE_COMPLETE', json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
