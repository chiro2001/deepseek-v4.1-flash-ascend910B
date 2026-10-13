#!/usr/bin/env python3
"""[DBO-ALLYIELD] 把 ubatching 的 yield 钩子挂到**唯一的 allreduce 咽喉点**上。

事实（本轮实测）：
  · 每步 91 次集合通信，全部是 allReduce，全部经过
    `torch.ops.vllm.all_reduce` → `GroupCoordinator._all_reduce_out_place`；
  · 通信窗口内 AIC = 0.000 ms（100% 暴露）；
  · vllm_ascend 全仓 `dbo_yield*` 零使用 ⇒ DBO 两路 ubatch 只在同一 compute 流上串行
    （实测 0.52×）。通信本该在 comm_stream 上跑，把 CPU/计算流让给兄弟 ubatch。

改法：在 `all_reduce()` 里按开关 `V41_DBO_ALLYIELD=1` 插入
`dbo_yield_and_switch_from_compute_to_comm()` / `..._from_comm_to_compute()`。
未开 ubatching 时这两个函数是 no-op（`_register_ubatch_function` 自带判空）。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm/vllm/distributed/parallel_state.py")
s = P.read_text()
if "DBO-ALLYIELD" in s:
    print("已打过 DBO-ALLYIELD，跳过")
    raise SystemExit(0)

old = (
    "def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:\n"
    "    assert group_name in _groups, f\"Group {group_name} is not found.\"\n"
    "    group = _groups[group_name]()\n"
    "    if group is None:\n"
    "        raise ValueError(f\"Group {group_name} is destroyed.\")\n"
    "    return group._all_reduce_out_place(tensor)\n"
)
new = (
    "# [DBO-ALLYIELD 2026-10-06] ubatching 的 yield 开关（默认关，A/B 用）。\n"
    "import os as _dbo_os\n"
    "\n"
    "_DBO_ALLYIELD = _dbo_os.environ.get(\"V41_DBO_ALLYIELD\", \"0\") == \"1\"\n"
    "_DBO_AY_N = [0]\n"
    "\n"
    "\n"
    "def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:\n"
    "    assert group_name in _groups, f\"Group {group_name} is not found.\"\n"
    "    group = _groups[group_name]()\n"
    "    if group is None:\n"
    "        raise ValueError(f\"Group {group_name} is destroyed.\")\n"
    "    if _DBO_ALLYIELD:\n"
    "        from vllm.v1.worker.ubatching import (\n"
    "            dbo_yield_and_switch_from_comm_to_compute,\n"
    "            dbo_yield_and_switch_from_compute_to_comm,\n"
    "        )\n"
    "        dbo_yield_and_switch_from_compute_to_comm()\n"
    "        try:\n"
    "            out = group._all_reduce_out_place(tensor)\n"
    "        finally:\n"
    "            dbo_yield_and_switch_from_comm_to_compute()\n"
    "        _DBO_AY_N[0] += 1\n"
    "        if _DBO_AY_N[0] % 200 == 0:\n"
    "            print(\"[DBO-ALLYIELD] allreduce-yields=%d\" % _DBO_AY_N[0], flush=True)\n"
    "        return out\n"
    "    return group._all_reduce_out_place(tensor)\n"
)
assert old in s, "all_reduce 锚点未找到"
s = s.replace(old, new, 1)
bak = P.with_suffix(".py.bak_allyield_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
