#!/usr/bin/env python3
"""[DBO-DP1 2026-10-06] 让 DP=1 的真实 execute_model 也走 ubatch（与 capture/dummy 一致）。

实测：上游 vLLM 只在 `data_parallel_size > 1` 时用 coordinate_batch_across_dp 决定
should_ubatch；本仓 `_determine_batch_execution_and_padding` 里
`should_ubatch, num_tokens_across_dp = False, None` 是硬编码 ⇒ DP=1 的真机路径
**永远不 ubatch**。而我们的 `_dummy_run`（capture/profile）用 check_ubatch_thresholds
算了 should_ubatch ⇒ 两边 slice 数不一致：
  · capture: 32tok/4req 注册 2 个 ATTENTION frontier（两个 ubatch builder）
  · 真机第一次跑同描述符只提交 1 个 ⇒ DeviceMetadataExecutor 抛
    "Device metadata frontiers changed for an existing full-graph batch descriptor"
（本轮实测日志：submitted=1 / expected=2）
修法：DP=1 且 use_ubatching 时按与 dummy 完全相同的阈值公式补 should_ubatch。
"""
from pathlib import Path
import shutil, time, ast
P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
s = P.read_text()
if "DBO-DP1" in s:
    print("已打过 DBO-DP1，跳过"); raise SystemExit(0)
old = ("        # Extra coordination when running data-parallel since we need to coordinate\n"
       "        # across ranks\n"
       "        should_ubatch, num_tokens_across_dp = False, None\n")
assert old in s, "锚点未找到"
new = old + (
    "        # [DBO-DP1 2026-10-06] DP=1 时上游不会计算 should_ubatch（只在 DP>1 的\n"
    "        # coordinate_batch_across_dp 里算）⇒ 真机永远不进 ubatch 路径。\n"
    "        # 用与 _dummy_run/capture 完全相同的公式补上，保证两边 slice 一致。\n"
    "        if self.parallel_config.use_ubatching and not should_ubatch:\n"
    "            try:\n"
    "                from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds as _chk_ub_dp1\n"
    "                should_ubatch = bool(\n"
    "                    _chk_ub_dp1(self.parallel_config, num_tokens_padded, uniform_decode=uniform_decode)\n"
    "                )\n"
    "            except Exception:\n"
    "                should_ubatch = False\n")
bak = P.with_suffix(".py.bak_dbodp1_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak); s = s.replace(old, new, 1); ast.parse(s); P.write_text(s)
print("patched:", P, "backup:", bak)
