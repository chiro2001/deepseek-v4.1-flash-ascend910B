#!/usr/bin/env python3
"""[V41-STEP-PROBE] 与 engram 无关的步长探针。

问题：现有的 `[bneck] hp` 探针（`_BneckState.tick()`）只在 `prepare_engram_inputs()`
里被调用 ⇒ **tiny 跑 ENGRAM=0 时 0 行输出**，而目标把 `[bneck] hp`（引擎侧 ms/step）
定为第 3 道验收门。缺了它，任何优化都只能拿聚合 tok/s 判断（会被接受长度主导）。

做法：在 `DeepseekV41Model.forward`（每步唯一入口）开头插一个墙钟探针：
  · `V41_STEP_PROBE=1` 开启（默认关 ⇒ 零影响）；
  · 用 `positions.shape[0] <= V41_STEP_PROBE_DECODE_TOKENS`（默认 64）区分 decode/prefill；
  · 每 `V41_STEP_PROBE_EVERY`（默认 20）步打印一次窗口均值。
不调用 synchronize（与 bneck 探针口径一致；连续步的墙钟均值即设备受限的步长）。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py")
s = P.read_text()
if "V41-STEP-PROBE" in s:
    print("已打过 V41-STEP-PROBE，跳过")
    raise SystemExit(0)

probe = (
    "\n"
    "# ==== [V41-STEP-PROBE] 与 engram 无关的步长探针（默认关）====\n"
    "import os as _sp_os\n"
    "import time as _sp_time\n"
    "\n"
    "_SP_ON = _sp_os.environ.get(\"V41_STEP_PROBE\", \"0\") == \"1\"\n"
    "_SP_EVERY = int(_sp_os.environ.get(\"V41_STEP_PROBE_EVERY\", \"20\") or 20)\n"
    "_SP_DEC_TOK = int(_sp_os.environ.get(\"V41_STEP_PROBE_DECODE_TOKENS\", \"64\") or 64)\n"
    "_SP = {\"n\": 0, \"dec\": 0, \"last\": None, \"acc\": 0.0, \"nout\": 0}\n"
    "\n"
    "\n"
    "def _sp_probe(num_tokens):\n"
    "    now = _sp_time.perf_counter()\n"
    "    last = _SP[\"last\"]\n"
    "    _SP[\"last\"] = now\n"
    "    if last is None:\n"
    "        return\n"
    "    dt = (now - last) * 1000.0\n"
    "    is_dec = int(num_tokens) <= _SP_DEC_TOK\n"
    "    if not is_dec:\n"
    "        return\n"
    "    _SP[\"n\"] += 1\n"
    "    _SP[\"acc\"] += dt\n"
    "    _SP[\"nout\"] += int(num_tokens)\n"
    "    if _SP_EVERY > 0 and _SP[\"n\"] % _SP_EVERY == 0:\n"
    "        print(\"[step] dec_steps=%d ms/step=%.3f ntok_avg=%.1f\"\n"
    "              % (_SP[\"n\"], _SP[\"acc\"] / _SP[\"n\"], _SP[\"nout\"] / float(_SP[\"n\"])), flush=True)\n"
    "        _SP[\"acc\"] = 0.0\n"
    "        _SP[\"nout\"] = 0\n"
    "        _SP[\"n\"] = 0\n"
    "# ==== [/V41-STEP-PROBE] ====\n"
    "\n"
    "\n"
    "class DeepseekV41Model(DeepseekV4Model):\n"
)

anchor = "class DeepseekV41Model(DeepseekV4Model):\n"
assert anchor in s, "DeepseekV41Model 类锚点未找到"
s = s.replace(anchor, probe, 1)

# 在 forward 开头插入调用
old_fwd = (
    "        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:\n"
    "            raise NotImplementedError(\"V4.1 eager milestone currently requires PP=1\")\n"
)
new_fwd = (
    "        if _SP_ON:\n"
    "            _sp_probe(positions.shape[0])\n"
    "        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:\n"
    "            raise NotImplementedError(\"V4.1 eager milestone currently requires PP=1\")\n"
)
assert old_fwd in s, "forward 开头锚点未找到"
s = s.replace(old_fwd, new_fwd, 1)

bak = P.with_suffix(".py.bak_stepprobe_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
