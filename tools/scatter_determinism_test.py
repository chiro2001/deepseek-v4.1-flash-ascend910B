#!/usr/bin/env python3
"""单算子确定性测试：`npu_scatter_nd_update_sk` 在逐位相同输入下是否确定？

动机：官方"确定性计算 API 清单"把 `npu_scatter_nd_update` / `npu_scatter_nd_update_` 列在
**Ascend 950DT 的非确定 API** 里，而我们的 KV cache 写入用的正是同族的
`torch.ops._C_ascend.npu_scatter_nd_update_sk`（镜像版 dsa_v41.py 里 3 处调用点）。

做法：固定输入（含 -1 跳过行，复刻 builder 的真实用法），连续调用 N 次，
每次与首次结果**逐位**比较（`torch.equal` + 最大绝对差）。
同时覆盖两种索引特征：唯一索引 / 含重复索引。

用法: scatter_det.py [calls] [T]
"""
import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEV = "npu:0"
D = 512
PAGE = 128
NB = 512
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
T = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def run_case(name, idx, upd):
    var0 = torch.zeros((NB * PAGE, D), dtype=torch.bfloat16, device=DEV)
    first = None
    worst = 0.0
    n_diff = 0
    for i in range(N):
        var = var0.clone()
        torch.ops._C_ascend.npu_scatter_nd_update_sk(var, idx, upd)
        torch.npu.synchronize()
        if first is None:
            first = var.clone()
            continue
        eq = bool(torch.equal(var, first))
        if not eq:
            n_diff += 1
            d = float((var.float() - first.float()).abs().max())
            worst = max(worst, d)
    print("  %-28s 逐位相同=%s  不同轮数=%d/%d  最大绝对差=%.3e"
          % (name, n_diff == 0, n_diff, N - 1, worst), flush=True)
    return n_diff


print("=== npu_scatter_nd_update_sk 确定性（%d 次调用，T=%d）===" % (N, T))
bad = 0

# 1) 唯一索引，无 -1
idx1 = torch.zeros((T, 2), dtype=torch.int32, device=DEV)
idx1[:, 0] = torch.arange(T, dtype=torch.int32, device=DEV) * 3
idx1[:, 1] = 0
upd1 = torch.randn((T, D), dtype=torch.bfloat16, device=DEV)
bad += run_case("唯一索引/无 -1", idx1, upd1)

# 2) 含 -1 跳过行（builder 的真实用法）
idx2 = idx1.clone()
if T >= 4:
    idx2[1, 0] = -1
    idx2[1, 1] = -1
    idx2[3, 0] = -1
    idx2[3, 1] = -1
bad += run_case("含 -1 跳过行", idx2, upd1)

# 3) 重复索引（同一行被写两次 —— 排序/tie-break 敏感）
idx3 = torch.zeros((T, 2), dtype=torch.int32, device=DEV)
idx3[:, 0] = 5
idx3[:, 1] = 7
bad += run_case("重复索引(全部同一行)", idx3, upd1)

# 4) 半重复
idx4 = idx1.clone()
if T >= 4:
    idx4[T - 1] = idx4[0]
bad += run_case("部分重复索引", idx4, upd1)

print()
if bad == 0:
    print("⇒ **该算子在上述用法下逐位确定**（N=%d 次全部相同）" % N)
    print("   ⇒ 官方清单里的随机性（若存在）不在这里触发；需继续查 decode 专属环节")
else:
    print("⇒ **该算子存在非确定性**（有 %d 种用法出现不同结果）" % bad)
    print("   ⇒ 与官方 Ascend 950DT 非确定清单一致；这是 KV 写入路径的抖动源")
