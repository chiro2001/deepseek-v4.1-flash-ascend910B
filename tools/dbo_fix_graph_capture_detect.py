#!/usr/bin/env python3
"""[DBO-GRAPH-DETECT] 用 `forward_context.capturing` 检测捕获期（而不是 stream 查询）。

实测：vllm-ascend `compilation/acl_graph.py:183` 在进入 `torch.npu.graph(aclgraph, …)`
之前显式设 `forward_context.capturing = True`；而
`torch.npu.is_current_stream_capturing()` 在我们的 `__call__` 里返回 False
（独立 `torch.npu.graph(g)` 上下文里返回 True）⇒ 上一版检测失效、捕获期仍走 threading 版，
随后捕获阶段 aicore exception(507015)。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = P.read_text()
old = (
    "            try:\n"
    "                _capturing = bool(torch.npu.is_current_stream_capturing())\n"
    "            except Exception:\n"
    "                _capturing = False\n"
)
new = (
    "            # [DBO-GRAPH-DETECT] 首选 vllm-ascend 自己设的标志；\n"
    "            # 实测 `is_current_stream_capturing()` 在这里返回 False，不可靠。\n"
    "            _capturing = bool(getattr(forward_context, \"capturing\", False))\n"
    "            if not _capturing:\n"
    "                try:\n"
    "                    _capturing = bool(torch.npu.is_current_stream_capturing())\n"
    "                except Exception:\n"
    "                    _capturing = False\n"
    "            if os.environ.get(\"V41_DBO_DEBUG\") == \"1\":\n"
    "                print(\"[DBO-GRAPH] capturing=%s -> %s\" % (\n"
    "                    _capturing, \"graph\" if _capturing else \"thread\"), flush=True)\n"
)
if old not in s:
    print("锚点未找到（可能已是新版）；当前是否含 DBO-GRAPH-DETECT: %s" % ("DBO-GRAPH-DETECT" in s))
    raise SystemExit(1)
s = s.replace(old, new, 1)
# 确保 os 已导入（文件顶部有 import threading/torch；os 在各函数里局部 import 过）
if "\nimport os\n" not in s and "\nimport os as _dbo_os\n" not in s:
    s = s.replace("import threading\n", "import os\nimport threading\n", 1)
bak = P.with_suffix(".py.bak_dbographdet_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
