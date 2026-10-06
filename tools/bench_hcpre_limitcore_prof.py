#!/usr/bin/env python3
"""HcPre 核数预算：用 torch_npu profiler 取**设备内核时长**（不受 host 派发影响）。"""
import glob, os, statistics as st, csv

import torch
import torch_npu  # noqa

torch.ops.load_library("/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so")

dev = torch.device("npu:0")
torch.npu.set_device(dev)
M, HC, H = 6, 4, 5120
x = torch.randn(M, HC, H, dtype=torch.bfloat16, device=dev) * 0.1
hc_fn = torch.randn(24, HC * H, dtype=torch.float32, device=dev) * 0.01
hc_scale = torch.randn(3, dtype=torch.float32, device=dev) * 0.01
hc_base = torch.randn(24, dtype=torch.float32, device=dev) * 0.01
pre_mix = torch.rand(M, HC, dtype=torch.float32, device=dev)
OP = torch.ops._C_ascend.npu_hc_pre_v2
KW = dict(hc_mult=HC, hc_sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6)

REPS = 60


def body(aic):
    if aic <= 0:
        for _ in range(REPS):
            OP(x, hc_fn, hc_scale, hc_base, pre_mix, **KW)
        return
    from torch.npu.npugraph_ex.scope import limit_core_num
    for _ in range(REPS):
        with limit_core_num(aic, 48):
            OP(x, hc_fn, hc_scale, hc_base, pre_mix, **KW)


def measure(aic, tag):
    for _ in range(30):
        OP(x, hc_fn, hc_scale, hc_base, pre_mix, **KW)
    torch.npu.synchronize()
    d = f"/tmp/hcprof_{tag}"
    os.makedirs(d, exist_ok=True)
    exp = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False, data_simplification=False)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=exp, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(d)) as p:
        body(aic)
        torch.npu.synchronize()
    fs = glob.glob(d + "/**/kernel_details.csv", recursive=True)
    if not fs:
        return None, 0
    vals = []
    with open(fs[0], newline="") as fh:
        for r in csv.DictReader(fh):
            if "HcPre" not in (r.get("Name") or ""): continue
            try: vals.append(float(r["Duration(us)"]))
            except Exception: pass
    return (st.median(vals) if vals else None), len(vals)


print(f"{'AIC':>6}{'设备中位us':>12}{'样本':>8}{'vs 不限':>10}")
base = None
for aic in [0, 24, 16, 8, 4]:
    med, n = measure(aic, f"a{aic}")
    if med is None:
        print(f"{aic:>6}   no data"); continue
    if base is None: base = med
    lbl = "不限" if aic == 0 else str(aic)
    print(f"{lbl:>6}{med:>12.2f}{n:>8}{med/base:>9.2f}x")
