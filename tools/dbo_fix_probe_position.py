#!/usr/bin/env python3
"""[DBO-PROBE-FIX2] 从 .bak_dboprobe_* 恢复，再按正确锚点插入探针。

上一版两个错：
  1. 插入点在 `compute_stream` 之后、`input_ids` 之前 ⇒ UnboundLocalError；
  2. 修正脚本用 `attn_metadata = ...` 作为 end 锚点，把中间的 4 个输入提取语句一起删了。
"""
from pathlib import Path
import glob
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/npu_ubatch_wrapper.py")
baks = sorted(glob.glob(str(P) + ".bak_dboprobe_*"))
assert baks, "找不到 dboprobe 备份"
clean = Path(baks[-1]).read_text()
print("从备份恢复:", baks[-1])

anchor = (
    "        inputs_embeds = (\n"
    '            kwargs.get("inputs_embeds") if "inputs_embeds" in kwargs else (args[3] if len(args) > 3 else None)\n'
    "        )\n"
)
assert anchor in clean, "inputs_embeds 锚点未找到"

probe = (
    "\n"
    "        # [DBO-PROBE 2026-10-06] 谁在真实步调到这里？（放在输入提取之后）\n"
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
new = clean.replace(anchor, anchor + probe, 1)

bak = P.with_suffix(".py.bak_dboprobe2_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(new)
P.write_text(new)
print("patched:", P, "backup:", bak)
