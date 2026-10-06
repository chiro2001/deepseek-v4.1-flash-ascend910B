#!/usr/bin/env python3
"""[DBO-GRAPH-SERIAL] 判别实验：把图内两个 ubatch 改成**同流串行**。

背景：图模式 DBO 捕获成功（nub=2，5 档 FULL 全捕获），但 replay 时
  · 多流开 → 挂死（等事件不来）；
  · 多流关 → `MTE accesses an invalid GM address`（fftsplus aivector）。
两种表现都指向"replay 时图里某处读错了内存"，但原因有两类：
  (A) **双流并发**：两个 ubatch 在同一张图里同时在跑，撞了模块级共享 workspace
      / 非流安全的中间缓冲；
  (B) **地址生命周期**：捕获期写进图的某批 tensor 地址在 replay 时不再有效。

本开关把两半批放在**同一条流上顺序执行**（仍在捕获区内，仍是一张图）：
  · 若错误消失 ⇒ 原因是 (A) 并发/共享 workspace；
  · 若错误照旧 ⇒ 原因是 (B) 地址/生命周期。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = P.read_text()
if "DBO-GRAPH-SERIAL" in s:
    print("已打过 DBO-GRAPH-SERIAL，跳过")
    raise SystemExit(0)

old = (
    "        e_fork = torch.npu.Event()\n"
    "        e_join = torch.npu.Event()\n"
    "        e_fork.record(root_stream)   # ★ 必须 record 在捕获根流上\n"
    "        outputs = [None] * len(ubatch_metadata)\n"
)
new = (
    "        import os as _os_s\n"
    "        if _os_s.environ.get(\"V41_DBO_GRAPH_SERIAL\", \"0\") == \"1\":\n"
    "            # [DBO-GRAPH-SERIAL] 判别臂：两半批同流串行（无并发、无事件）\n"
    "            outputs = [None] * len(ubatch_metadata)\n"
    "            for _i, _md in enumerate(ubatch_metadata):\n"
    "                outputs[_i] = _submit(_md, model, root_stream)\n"
    "            return _cat_ubatch_outputs(outputs)\n"
    "        e_fork = torch.npu.Event()\n"
    "        e_join = torch.npu.Event()\n"
    "        e_fork.record(root_stream)   # ★ 必须 record 在捕获根流上\n"
    "        outputs = [None] * len(ubatch_metadata)\n"
)
assert old in s, "fork 锚点未找到"
s = s.replace(old, new, 1)

bak = P.with_suffix(".py.bak_dboserial_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
