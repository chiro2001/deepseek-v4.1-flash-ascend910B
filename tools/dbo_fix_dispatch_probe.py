#!/usr/bin/env python3
"""[DBO-PROBE] 在 ubatch 派发点打栈，回答"谁在真实步调 NPUUBatchWrapper"。

上一轮实测：图模式 DBO 捕获成功（capturing=True -> graph），
但真实 conc=4 步打印 `capturing=False -> thread` 然后挂死。
需要知道这次调用来自哪里（主模型 replay？dspark drafter？piecewise 回退？）。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = P.read_text()
if "DBO-PROBE" in s:
    print("已打过 DBO-PROBE，跳过")
    raise SystemExit(0)

old = (
    "        # If there's no ubatching, just run the runnable directly\n"
    "        if ubatch_slices is None:\n"
    "            return self.runnable(*args, **kwargs)\n"
)
new = (
    "        # If there's no ubatching, just run the runnable directly\n"
    "        if ubatch_slices is None:\n"
    "            return self.runnable(*args, **kwargs)\n"
    "\n"
    "        # [DBO-PROBE 2026-10-06] 谁在真实步调到这里？\n"
    "        import os as _os_p\n"
    "        if _os_p.environ.get(\"V41_DBO_PROBE\", \"0\") == \"1\":\n"
    "            import traceback as _tb_p\n"
    "            _fr = [\"%s:%d\" % (f.filename.rsplit(\"/\", 1)[-1], f.lineno)\n"
    "                   for f in _tb_p.extract_stack()[-6:-1]]\n"
    "            print(\"[DBO-PROBE] capturing=%s mode=%s ntok=%s nub=%d | %s\" % (\n"
    "                getattr(forward_context, \"capturing\", None),\n"
    "                getattr(forward_context, \"cudagraph_runtime_mode\", None),\n"
    "                (input_ids.shape[0] if input_ids is not None else None),\n"
    "                len(ubatch_slices), \" <-\".join(_fr)), flush=True)\n"
)
assert old in s, "派发锚点未找到"
s = s.replace(old, new, 1)

# 插到取 input_ids 之后（上面 new 里已用到 input_ids，需要挪到那之后）
# 简单做法：把探针块移到 input_ids 赋值之后
probe_start = s.index("        # [DBO-PROBE 2026-10-06] 谁在真实步调到这里？")
probe_end = s.index("        compute_stream = torch.npu.current_stream()", probe_start)
probe = s[probe_start:probe_end]
s = s[:probe_start] + s[probe_end:]
anchor2 = "        compute_stream = torch.npu.current_stream()\n"
pos = s.index(anchor2) + len(anchor2)
s = s[:pos] + probe + s[pos:]

bak = P.with_suffix(".py.bak_dboprobe_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
