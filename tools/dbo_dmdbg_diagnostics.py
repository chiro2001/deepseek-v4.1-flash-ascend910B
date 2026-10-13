#!/usr/bin/env python3
"""[DBO-DMDBG] 给 DeviceMetadataExecutor.submit 加诊断：打印每个 batch_descriptor
首次登记的 frontiers，以及不一致时的 submitted/expected 对比。"""
from pathlib import Path
import shutil, time, ast
P = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/device_metadata.py")
s = P.read_text()
if "DBO-DMDBG" in s:
    print("已打过，跳过"); raise SystemExit(0)
o1 = "        if expected_frontiers is not None and expected_frontiers != submitted_frontiers:\n            raise RuntimeError(\"Device metadata frontiers changed for an existing full-graph batch descriptor\")\n"
n1 = ("        if expected_frontiers is not None and expected_frontiers != submitted_frontiers:\n"
      "            print(\"[DBO-DMDBG] mismatch descriptor=%s\\n  submitted=%s\\n  expected=%s\"\n"
      "                  % (batch_descriptor, submitted_frontiers, expected_frontiers), flush=True)\n"
      "            raise RuntimeError(\"Device metadata frontiers changed for an existing full-graph batch descriptor\")\n")
assert o1 in s, "锚点1未找到"
s = s.replace(o1, n1, 1)
o2 = "        if batch_descriptor is not None and expected_frontiers is None:\n            self._external_frontiers[batch_descriptor] = submitted_frontiers\n"
n2 = ("        if batch_descriptor is not None and expected_frontiers is None:\n"
      "            print(\"[DBO-DMDBG] register descriptor=%s frontiers=%s\"\n"
      "                  % (batch_descriptor, submitted_frontiers), flush=True)\n"
      "            self._external_frontiers[batch_descriptor] = submitted_frontiers\n")
assert o2 in s, "锚点2未找到"
s = s.replace(o2, n2, 1)
bak = P.with_suffix(".py.bak_dmdbg_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak); ast.parse(s); P.write_text(s)
print("patched:", P, "backup:", bak)
