#!/usr/bin/env python3
"""[DBO-KEEPALIVE2] 修正保活门控：metadata 是在进入 `torch.npu.graph()` **之前**构建的
（`capture_model` 先 `_dummy_run` 再进图），所以 `forward_context.capturing` 那一刻还是 False；
而 `get_forward_context()` 在该路径下直接断言（实测 AssertionError）。

改为**有界无条件保活**：只保留前 5000 条（捕获期必然在最早的几十条里），
内存有界（每条几 KB ⇒ 上限约几十 MB），且捕获期张量永久存活。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
s = P.read_text()

start = s.index("    # [DBO-KEEPALIVE 2026-10-06] 实验：钉住捕获期产出的 metadata。")
end = s.index("    return out\n\n\n_DBO_KEEPALIVE: list = []", start)
new_block = (
    "    # [DBO-KEEPALIVE2 2026-10-06] 有界无条件保活（捕获期张量必须活到进程结束）。\n"
    "    # 为什么不能按 `forward_context.capturing` 门控：metadata 在进入 graph 之前构建，\n"
    "    # 那一刻 capturing 还是 False；且该路径下 get_forward_context() 会断言。\n"
    "    if len(_DBO_KEEPALIVE) < 5000:\n"
    "        _DBO_KEEPALIVE.extend(out)\n"
    "        if not _DBO_KEEPALIVE_LOGGED[0]:\n"
    "            _DBO_KEEPALIVE_LOGGED[0] = True\n"
    "            print(\"[DBO-KEEPALIVE] 开始保活 ubatch metadata（前 5000 条）\", flush=True)\n"
)
s = s[:start] + new_block + s[end:]

bak = P.with_suffix(".py.bak_dbokeep2_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
