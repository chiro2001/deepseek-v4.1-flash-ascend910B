#!/usr/bin/env python3
"""[DBO-KEEPALIVE] 验证假设：捕获期切出来的 ubatch metadata 张量被回收 ⇒ replay 读到无效地址。

证据链：
  1. 图内两半批**同流串行**后 MTE 越界一模一样 ⇒ 不是并发/workspace（判别实验已做）；
  2. 上游 `vllm/v1/worker/ubatch_utils.py:125` 的 docstring **自己写着**：
     "Note: This function creates a new tensor to hold the new query_start_locs.
      **This will break cudagraph compatibility.**"
     → `slice_query_start_locs` 用切片相减产生**新张量**，`_make_metadata_with_slice`
     里还有多处 `.clone()`；
  3. 我们的 `ascend_split_attn_metadata` 每个 ubatch 都新建一个
     `AscendCommonAttentionMetadata`，其张量是**临时分配**，捕获期写进图后即被回收。

本实验：把**捕获期**产出的 metadata 全部钉住（引用不释放）。
  · MTE 消失 ⇒ 假设成立，正解 = 预分配 + copy_into（静态缓冲）；
  · MTE 照旧 ⇒ 假设不成立，需要回到张量级地址审计。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
s = P.read_text()
if "DBO-KEEPALIVE" in s:
    print("已打过 DBO-KEEPALIVE，跳过")
    raise SystemExit(0)

old = "    return out\n\n\n@dataclass\nclass GraphCaptureContext:"
new = (
    "    # [DBO-KEEPALIVE 2026-10-06] 实验：钉住捕获期产出的 metadata。\n"
    "    # 上游 slice_query_start_locs 自己写着 \"This will break cudagraph compatibility\"，\n"
    "    # 那里的新张量若在捕获后被回收，replay 就会读到无效地址（实测 MTE invalid GM address）。\n"
    "    try:\n"
    "        from vllm.forward_context import get_forward_context as _gfc\n"
    "\n"
    "        if bool(getattr(_gfc(), \"capturing\", False)):\n"
    "            _DBO_KEEPALIVE.extend(out)\n"
    "            if not _DBO_KEEPALIVE_LOGGED[0]:\n"
    "                _DBO_KEEPALIVE_LOGGED[0] = True\n"
    "                print(\"[DBO-KEEPALIVE] 钉住捕获期 metadata，n=%d\" % len(out), flush=True)\n"
    "    except Exception as _e_ka:\n"
    "        print(\"[DBO-KEEPALIVE] 异常: %r\" % (_e_ka,), flush=True)\n"
    "    return out\n"
    "\n"
    "\n"
    "_DBO_KEEPALIVE: list = []\n"
    "_DBO_KEEPALIVE_LOGGED = [False]\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class GraphCaptureContext:"
)
assert old in s, "ascend_split 返回锚点未找到"
s = s.replace(old, new, 1)

bak = P.with_suffix(".py.bak_dbokeep_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
