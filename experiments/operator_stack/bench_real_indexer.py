"""Formal Indexer postprocessing weights; synthetic projected inputs.

Paired graph-event operator timing is separate from TP8 end-to-end timing.
"""
import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import torch_npu
from safetensors import safe_open

from formal_model_contract import inspect_checkpoint
from indexer_post import indexer_post
from indexer_patches import native_post


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--replay-only', action='store_true')
    parser.add_argument('--compare-scalar', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    contract = inspect_checkpoint(args.model)
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op
    bootstrap_custom_op_env()
    assert enable_custom_op(), 'Native vendor reference must be available'
    torch.npu.set_device(0)
    torch.manual_seed(20261010)
    def weight(name):
        with safe_open(str(Path(args.model) / contract['operator_tensors'][name]['shard']),
                       framework='pt', device='cpu') as archive:
            return archive.get_tensor(name).to(device='npu', dtype=torch.bfloat16)
    name = sorted(k for k in contract['operator_tensors'] if k.endswith('wk.weight'))[0]
    wk, gamma = weight(name), weight(name.replace('wk.weight', 'k_norm.weight'))
    config = json.loads((Path(args.model)/'config.json').read_text())
    eps = float(config.get('text_config', config)['rms_norm_eps'])
    module = SimpleNamespace(width=128, rope_width=64,
                             k_norm=lambda value: torch_npu.npu_rms_norm(value, gamma, epsilon=eps)[0])
    latent = torch.randn((1, 512), device='npu', dtype=torch.bfloat16)
    projected = torch.nn.functional.linear(latent, wk)
    angle = torch.randn((1, 32), device='npu').repeat_interleave(2, dim=-1)
    cos = angle.cos().to(torch.bfloat16).view(1, 1, 1, 64)
    sin = angle.sin().to(torch.bfloat16).view(1, 1, 1, 64)
    slots = torch.tensor([[0, 3]], device='npu', dtype=torch.int64)
    key_store = torch.zeros((2, 256, 1, 128), device='npu', dtype=torch.int8)
    scale_store = torch.zeros((2, 256, 1, 1), device='npu', dtype=torch.float16)
    key_cache, scale_cache = key_store[:, :128], scale_store[:, :128]
    funcs = {'native': lambda: native_post(module, projected, slots, cos, sin, key_cache, scale_cache),
             'fused': lambda: indexer_post(projected, gamma, cos.reshape(-1,64), sin.reshape(-1,64), slots, key_cache, scale_cache, eps)}
    if args.compare_scalar:
        from indexer_post_scalar import indexer_post_scalar
        funcs['scalar'] = lambda: indexer_post_scalar(projected, gamma, cos.reshape(-1,64), sin.reshape(-1,64), slots, key_cache, scale_cache, eps)
    if args.replay_only:
        # Independent 128-case precision receipts gate this profiling replay.
        # Keep the debug-store kernel out of the selected msprof op window.
        for _ in range(30):
            funcs['fused']()
        torch.npu.synchronize()
        print('REAL_INDEXER_REPLAY_COMPLETE', name, tuple(projected.shape), flush=True)
        return
    quant, scale = funcs['native']()
    reference_key, reference_scale = key_cache.clone(), scale_cache.clone()
    debug = indexer_post(projected, gamma, cos.reshape(-1,64), sin.reshape(-1,64), slots, key_cache, scale_cache, eps, debug=True)
    assert torch.equal(debug[2], quant) and torch.equal(debug[3], scale)
    assert torch.equal(key_cache, reference_key) and torch.equal(scale_cache, reference_scale)
    if args.compare_scalar:
        scalar_debug = indexer_post_scalar(projected, gamma, cos.reshape(-1,64), sin.reshape(-1,64), slots, key_cache, scale_cache, eps, debug=True)
        assert torch.equal(scalar_debug[2], quant) and torch.equal(scalar_debug[3], scale)
        assert torch.equal(key_cache, reference_key) and torch.equal(scale_cache, reference_scale)
    graphs = {}
    calls = 20
    replays = 50
    for kind, function in funcs.items():
        for _ in range(10):
            function()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(calls):
                function()
        graphs[kind] = graph
    pairs = []
    for pair in range(8):
        row = {'pair': pair}
        for kind in (list(funcs) if pair % 2 == 0 else list(reversed(funcs))):
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            for _ in range(replays):
                graphs[kind].replay()
            end.record()
            end.synchronize()
            row[kind + '_us'] = begin.elapsed_time(end) * 1000 / (calls * replays)
        row['speedup'] = row['native_us'] / row['fused_us']
        if args.compare_scalar:
            row['scalar_vs_fused_speedup'] = row['fused_us'] / row['scalar_us']
        pairs.append(row)
    result = {'weight': name, 'checkpoint_config_sha256': contract['config_sha256'],
              'inputs': 'synthetic latent projected by unaltered formal BF16 weight',
              'scope': 'single Indexer postprocessing graph events; excludes projection and full TP8',
              'shape': [1, 128], 'dtype': 'bfloat16', 'precision': 'functional/cache bitwise equal',
              'profiler_during_timing': 'OFF', 'pairs': pairs,
              'native_median_us': statistics.median(p['native_us'] for p in pairs),
              'fused_median_us': statistics.median(p['fused_us'] for p in pairs),
              'paired_speedup_median': statistics.median(p['speedup'] for p in pairs)}
    if args.compare_scalar:
        result['scalar_median_us'] = statistics.median(p['scalar_us'] for p in pairs)
        result['scalar_vs_fused_paired_speedup_median'] = statistics.median(p['scalar_vs_fused_speedup'] for p in pairs)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('REAL_INDEXER_BENCHMARK', json.dumps(result), flush=True)
    if args.profile:
        for kind, function in funcs.items():
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
                    record_shapes=True,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profile'/kind))) as prof:
                for _ in range(10):
                    function()
                    prof.step()


if __name__ == '__main__':
    main()
