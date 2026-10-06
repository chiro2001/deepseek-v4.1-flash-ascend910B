#!/usr/bin/env python3
"""[DBO-EAGER-DEVMETA 2026-10-06] v41 device-metadata 的"eager 构建 + 跳过 wait"一致性修复。

背景（本轮实测）：
  · dsa_v41 的 DeepseekV41MetadataBuilder.enable_device_metadata() 在 ubatching 下把
    `_device_metadata_enabled` 置 False ⇒ `_publish_task` 走 **就地 eager 执行** 分支，
    任务不进 DeviceMetadataExecutor。
  · 但消费侧（dsa_v41._native_attention:3011 ATTENTION / :2365 COMPRESSOR /
    models/deepseek_v41/indexer.py:251 INDEXER）仍无条件 `wait_for_device_metadata(...)`，
    该 frontier 从未 submit ⇒ `KeyError: (DeviceMetadataStage.ATTENTION, id)`（本轮实测崩溃点）。
修复：记录"eager 构建过"的 buffer id，三处 wait 通过 helper 跳过。
（V41_DBO_NODEVOPS 那个 env 开关是错误杠杆：它会把 SMLA/QLI metadata 整块跳过，
  反而让 indexer 报 "QLI metadata was not built"，这里一并去掉，避免误用。）
"""
from pathlib import Path
import shutil, time, ast

P = Path("/home/l00886679/dcpw/vllm_ascend/attention/dsa_v41.py")
IDX = Path("/home/l00886679/dcpw/vllm_ascend/models/deepseek_v41/indexer.py")
ts = time.strftime("%m%d_%H%M%S")
for f in (P, IDX):
    shutil.copy2(f, f.with_suffix(f.suffix + ".bak_eagerwait_" + ts))

s = P.read_text()
changed = []

# 1) 模块级 helper
helper = (
    "# [DBO-EAGER-DEVMETA 2026-10-06] `_publish_task` 在 `_device_metadata_enabled=False`\n"
    "# 时**就地执行** metadata 内核，不再产生 DeviceMetadataTask。消费侧如果还\n"
    "# `wait_for_device_metadata(...)` 就会 KeyError（frontier 从未 submit）。\n"
    "# 这里记录 eager 构建过的 buffer id，供三处 wait 跳过。\n"
    "_V41_EAGER_META_BUFFER_IDS: set[int] = set()\n"
    "\n"
    "\n"
    "def _v41_wait_device_metadata(stage, group_id) -> None:\n"
    "    if group_id in _V41_EAGER_META_BUFFER_IDS:\n"
    "        return\n"
    "    wait_for_device_metadata(stage, group_id)\n"
    "\n"
)
marker = "class DeepseekV41MetadataBuilder"
assert marker in s, "找不到 DeepseekV41MetadataBuilder 锚点"
if "_V41_EAGER_META_BUFFER_IDS" not in s:
    s = s.replace(marker, helper + "\n" + marker, 1)
    changed.append("helper")
else:
    changed.append("helper已存在")

# 2) _publish_task eager 分支登记 id
old = "        else:\n            run()\n        return buffer\n"
new = "        else:\n            run()\n            _V41_EAGER_META_BUFFER_IDS.add(id(buffer))\n        return buffer\n"
assert old in s, "_publish_task eager 分支锚点未找到"
s = s.replace(old, new, 1)
changed.append("publish_eager_reg")

# 3) 两处 wait 换成 helper
o1 = "            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)"
n1 = "            _v41_wait_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)"
assert o1 in s, "COMPRESSOR wait 锚点未找到"
s = s.replace(o1, n1, 1)
o2 = "        wait_for_device_metadata(\n            DeviceMetadataStage.ATTENTION,\n            id(op_metadata),\n        )"
n2 = "        _v41_wait_device_metadata(\n            DeviceMetadataStage.ATTENTION,\n            id(op_metadata),\n        )"
assert o2 in s, "ATTENTION wait 锚点未找到"
s = s.replace(o2, n2, 1)
changed.append("waits→helper")

# 4) 去掉有害的 V41_DBO_NODEVOPS（会把 SMLA/QLI metadata 整块跳过）
old_g = (
    "        # [DBO-NODEVOPS] ubatching 时强制走 Python metadata 路径：\n"
    "        # AICPU metadata kernel（VllmQuantLightningIndexerMetadata / SparseFlashMlaMetadata 等）\n"
    "        # 在 ubatch 下入参不匹配（实测 22007 / frontier KeyError）。\n"
    "        try:\n"
    "            import os as _o2\n"
    "            if _o2.environ.get(\"V41_DBO_NODEVOPS\", \"0\") == \"1\" and                     bool(getattr(self.vllm_config.parallel_config, \"use_ubatching\", False)):\n"
    "                self._supports_device_ops = False\n"
    "        except Exception:\n"
    "            pass\n"
)
if old_g in s:
    s = s.replace(old_g,
        "        # [DBO-NODEVOPS-RETIRED 2026-10-06] 原 V41_DBO_NODEVOPS 开关已废弃：\n"
        "        # 它把 `_supports_device_ops` 置 False，会**整块跳过** SMLA/QLI metadata 的构建\n"
        "        # ⇒ indexer 报 \"V4.1 QLI metadata was not built\"。ubatching 下正确的做法是：\n"
        "        # 保留 `_supports_device_ops=True`，只让 `_device_metadata_enabled=False`\n"
        "        # （eager 就地构建）+ 消费侧跳过 wait（见 `_v41_wait_device_metadata`）。\n",
        1)
    changed.append("nodeps-retired")
else:
    changed.append("nodeps-已不存在或格式不同")

ast.parse(s)
P.write_text(s)

# 5) indexer.py 的 INDEXER wait
t = IDX.read_text()
o3 = "        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))"
n3 = "        _v41_wait_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))"
if o3 in t:
    t = t.replace(o3, n3, 1)
    changed.append("indexer-wait")
else:
    raise SystemExit("indexer wait 锚点未找到")
# import 补充
if "_v41_wait_device_metadata" not in t.split(o3)[0].split("\n")[0]:
    import re
    m = re.search(r"from vllm_ascend\.attention\.dsa_v41 import \(\n(.*?)\n\)", t, re.S)
    assert m, "dsa_v41 import 块未找到"
    t = t[:m.end()-2] + "    _v41_wait_device_metadata,\n" + t[m.end()-2:]
ast.parse(t)
IDX.write_text(t)

print("patched:", changed)
