#!/usr/bin/env python3
"""[NPU-STREAM-TLS] 让 NPU 的 set_stream 同步 vLLM 的 current_stream() 线程本地缓存。

实测根因：
  · vLLM 的 `current_stream()`（vllm/utils/torch_utils.py:662）**不调用** torch API，
    而是读线程本地 `_current_stream_tls.value`，只有它自己 patch 过的
    `torch.cuda.set_stream` 会更新这个值；
  · 我们的 NPU ubatch 上下文用 `torch.npu.set_stream(...)` ⇒ **TLS 永远不更新** ⇒
    `current_stream()` 返回陈旧流（实测：set_stream(s2) 后 torch.npu.current_stream()==s2
    但 current_stream() 仍返回旧流）；
  · 后果：ubatching 的 `assert current_stream() == self.comm_stream` 必然失败
    （本轮 [DBO-ALLYIELD] 实验第一次跑就撞上）。

改法：在 NPU 上下文的 update_stream 里同步写一次 TLS（不动 vLLM 源码）。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/npu_ubatch_wrapper.py")
s = P.read_text()
if "NPU-STREAM-TLS" in s:
    print("已打过 NPU-STREAM-TLS，跳过")
    raise SystemExit(0)

old = (
    "class NPUUBatchContext(UBatchContext):\n"
    "    \"\"\"NPU version of UBatchContext that uses torch.npu Stream/Event APIs.\"\"\"\n"
    "\n"
    "    def update_stream(self, stream):\n"
    "        self.current_stream = stream\n"
    "        torch.npu.set_stream(self.current_stream)\n"
)
new = (
    "class NPUUBatchContext(UBatchContext):\n"
    "    \"\"\"NPU version of UBatchContext that uses torch.npu Stream/Event APIs.\"\"\"\n"
    "\n"
    "    def update_stream(self, stream):\n"
    "        self.current_stream = stream\n"
    "        torch.npu.set_stream(self.current_stream)\n"
    "        # [NPU-STREAM-TLS 2026-10-06] vLLM 的 current_stream() 读的是线程本地缓存，\n"
    "        # 只有它 patch 过的 torch.cuda.set_stream 会更新；torch.npu.set_stream 不会。\n"
    "        # 不同步这一份，ubatching 的流断言会看到陈旧流并失败。\n"
    "        try:\n"
    "            from vllm.utils import torch_utils as _tu\n"
    "\n"
    "            _tu._current_stream_tls.value = stream\n"
    "        except Exception:\n"
    "            pass\n"
)
assert old in s, "NPUUBatchContext.update_stream 锚点未找到"
s = s.replace(old, new, 1)
bak = P.with_suffix(".py.bak_streamtls_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
