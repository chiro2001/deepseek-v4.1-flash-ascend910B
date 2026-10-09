"""An isolated SIMD GEMV experiment for the measured narrow FP32 router."""
import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def tiny_router_gemv(x, w, out, K: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)
    xv = tl.load(x+k, k<K, 0).to(tl.float32)
    wv = tl.load(w+n*K+k, k<K, 0).to(tl.float32)
    value = tl.sum(xv*wv, axis=0)
    tl.store(out+n, value)


def vector(x, w):
    assert x.shape == (1,5120) and w.shape[1] == 5120
    assert x.is_contiguous() and w.is_contiguous()
    output = torch.empty((1, w.shape[0]), dtype=torch.float32, device=x.device)
    tiny_router_gemv[(w.shape[0],)](x, w, output, 5120, 8192)
    return output


def narrow_router(x, w):
    """Candidate dispatcher: use the measured fast case; retain native otherwise."""
    if (x.shape == (1,5120) and w.shape == (8,5120)
            and x.dtype == w.dtype == torch.float32
            and x.is_contiguous() and w.is_contiguous()):
        return vector(x,w)
    return torch.nn.functional.linear(x,w)


def graph_time(func, x, w):
    for _ in range(5):
        func(x, w)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [func(x, w) for _ in range(20)]
    samples = []
    for _ in range(7):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end)*1000/1000)
    return statistics.median(samples), samples


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='/work/results/router_gemv.json')
    p.add_argument('--profile-dir')
    args = p.parse_args()
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    torch.manual_seed(20261009)
    results = []
    for n in [8, 128]:
        weight = (torch.randn((n, 5120), dtype=torch.float32)/5120**0.5).npu()
        x = torch.randn((1,5120), dtype=torch.float32, device='npu')
        native = lambda x, w: torch.nn.functional.linear(x, w)
        max_abs, native_max_abs, top2_agree = 0, 0, 0
        for _ in range(64):
            x.copy_(torch.randn_like(x))
            actual, reference = vector(x, weight), native(x, weight)
            cpu_reference = torch.nn.functional.linear(x.cpu().double(), weight.cpu().double()).float()
            max_abs = max(max_abs, (actual.cpu()-cpu_reference).abs().max().item())
            native_max_abs = max(native_max_abs, (reference.cpu()-cpu_reference).abs().max().item())
            torch.testing.assert_close(actual.cpu(), cpu_reference, rtol=1e-4, atol=2e-5)
            top2_agree += int(torch.equal(actual.topk(2).indices.cpu(), reference.topk(2).indices.cpu()))
        native_us, native_samples = graph_time(native, x, weight)
        vector_us, vector_samples = graph_time(vector, x, weight)
        results.append({'N':n, 'K':5120, 'dtype':'float32', 'max_abs_error_vs_fp64':max_abs,
                        'native_max_abs_error_vs_fp64':native_max_abs,
                        'top2_agreement':top2_agree, 'precision_cases':64,
                        'native_graph_us':native_us, 'vector_graph_us':vector_us,
                        'speedup':native_us/vector_us, 'native_samples_us':native_samples,
                        'vector_samples_us':vector_samples})
        if args.profile_dir:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=10,repeat=1),
                record_shapes=True,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f'{args.profile_dir}/N{n}')) as prof:
                for _ in range(10):
                    native(x,weight)
                    vector(x,weight)
                    prof.step()
    report={'physical_chip':4,'method':'20 kernels per graph, 50 replays, 7 repeats, profiler OFF',
            'hf32_allowed':torch.npu.matmul.allow_hf32, 'results':results,
            'deployment':'isolated experiment only; no model or production patch'}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
