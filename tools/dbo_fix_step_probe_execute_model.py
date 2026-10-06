#!/usr/bin/env python3
"""[V41-STEP-PROBE-2] 步长探针改挂在 `NPUModelRunner.execute_model`（图捕获之外）。

为什么不在 model.py：第一版把探针插在 `DeepseekV41Model.forward` 里，
而那一段**被 torch.compile/dynamo 追踪** ⇒ `perf_counter()` 直接编译失败
（实测：EngineCore 报错、服务起不来）。已回退。

为什么是 execute_model：它是**每步唯一的普通 Python 入口**（图 replay 在它内部发生），
所以「相邻两次调用的墙钟差」就是引擎侧 ms/step，且不会被 dynamo 追踪。

开关：`V41_STEP_PROBE=1`（默认关）；`V41_STEP_PROBE_EVERY`（默认 20）；
`V41_STEP_PROBE_DECODE_TOKENS`（默认 64，用于区分 decode/prefill）。
"""
from pathlib import Path
import shutil
import time
import ast

P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
s = P.read_text()
if "V41-STEP-PROBE" in s:
    print("已打过 V41-STEP-PROBE，跳过")
    raise SystemExit(0)

probe = (
    "\n"
    "# ==== [V41-STEP-PROBE] 引擎侧 ms/step（与 engram 无关；默认关）====\n"
    "import os as _sp_os\n"
    "\n"
    "_SP_ON = _sp_os.environ.get(\"V41_STEP_PROBE\", \"0\") == \"1\"\n"
    "_SP_EVERY = int(_sp_os.environ.get(\"V41_STEP_PROBE_EVERY\", \"20\") or 20)\n"
    "_SP_DEC_TOK = int(_sp_os.environ.get(\"V41_STEP_PROBE_DECODE_TOKENS\", \"64\") or 64)\n"
    "_SP: dict = {\"n\": 0, \"ntok\": 0, \"acc\": 0.0, \"last\": None}\n"
    "\n"
    "\n"
    "def _sp_tick(n_tokens: int) -> None:\n"
    "    _now = time.perf_counter()\n"
    "    _last = _SP[\"last\"]\n"
    "    _SP[\"last\"] = _now\n"
    "    if _last is None or int(n_tokens) > _SP_DEC_TOK:\n"
    "        return\n"
    "    _SP[\"n\"] += 1\n"
    "    _SP[\"ntok\"] += int(n_tokens)\n"
    "    _SP[\"acc\"] += (_now - _last) * 1000.0\n"
    "    if _SP_EVERY > 0 and _SP[\"n\"] % _SP_EVERY == 0:\n"
    "        print(\"[step] dec_steps=%d ms/step=%.3f tok/step=%.2f\"\n"
    "              % (_SP[\"n\"], _SP[\"acc\"] / _SP[\"n\"], _SP[\"ntok\"] / float(_SP[\"n\"])),\n"
    "              flush=True)\n"
    "        _SP[\"n\"] = 0\n"
    "        _SP[\"ntok\"] = 0\n"
    "        _SP[\"acc\"] = 0.0\n"
    "# ==== [/V41-STEP-PROBE] ====\n"
    "\n"
    "\n"
    "class NPUModelRunner(GPUModelRunner):\n"
)

anchor = "class NPUModelRunner(GPUModelRunner):\n"
assert anchor in s, "NPUModelRunner 锚点未找到"
# 注意：文件里 class NPUModelRunner 出现两次的可能是注释，取第一次是定义
s = s.replace(anchor, probe, 1)

old_fwd = (
    "    ) -> ModelRunnerOutput | IntermediateTensors | None:\n"
    "        if vllm_version_is(\"0.27.1\"):\n"
    "            if self.vllm_config.model_config.enable_return_routed_experts and self.routed_experts_initialized:\n"
    "                self.routed_experts_capturer.clear_buffer()\n"
)
new_fwd = (
    "    ) -> ModelRunnerOutput | IntermediateTensors | None:\n"
    "        if _SP_ON:\n"
    "            try:\n"
    "                _sp_tick(sum(scheduler_output.num_scheduled_tokens.values()))\n"
    "            except Exception:\n"
    "                pass\n"
    "        if vllm_version_is(\"0.27.1\"):\n"
    "            if self.vllm_config.model_config.enable_return_routed_experts and self.routed_experts_initialized:\n"
    "                self.routed_experts_capturer.clear_buffer()\n"
)
assert old_fwd in s, "execute_model 开头锚点未找到"
s = s.replace(old_fwd, new_fwd, 1)

bak = P.with_suffix(".py.bak_stepprobe2_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
ast.parse(s)
P.write_text(s)
print("patched:", P, "backup:", bak)
