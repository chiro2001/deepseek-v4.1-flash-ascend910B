#!/usr/bin/env python3
"""构建期冒烟：本层携带的 15 个文件必须都能 py_compile 通过。

为什么放构建期而不是起服期：语法错的补丁件一旦烘进镜像，就会在**起服时**
才炸，而那时排查成本高得多（还要区分是"我的补丁坏了"还是"基底不兼容"）。
不需要 NPU，所以无卡机器上也能跑。
"""
import py_compile
import pathlib
import sys

BASE = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend")
FILES = [
    "models/deepseek_v41/engram_hbm.py",
    "models/deepseek_v41/engram_hash.py",
    "models/deepseek_v41/engram_gate.py",
    "models/deepseek_v41/engram_jit_kernel.py",
    "models/deepseek_v41/engram_plan_kernel.py",
    "models/deepseek_v41/engram_device_index.py",
    "models/deepseek_v41/engram_graph.py",
    "models/deepseek_v41/model.py",
    "models/deepseek_v41/indexer.py",
    "ascend_forward_context.py",
    "ops/rope_dsv4.py",
    "worker/block_table.py",
    "ops/fused_moe/token_dispatcher.py",
    "attention/dsa_v41.py",
    "distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py",
]

bad = []
for f in FILES:
    p = BASE / f
    if not p.is_file():
        bad.append(f"{f}: MISSING")
        continue
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as exc:  # noqa: BLE001
        bad.append(f"{f}: {exc}")

print(f"[build-smoke] py_compile {len(FILES) - len(bad)}/{len(FILES)} ok")
if bad:
    print("\n".join(bad))
    sys.exit(1)
