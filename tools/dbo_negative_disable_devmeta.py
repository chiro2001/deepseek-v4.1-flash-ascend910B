#!/usr/bin/env python3
"""ubatching 时禁用 device-metadata 任务队列（走 eager 路径），绕开 AICPU metadata 内核的不匹配。"""
import ast, sys
from pathlib import Path
p = Path("/home/l00886679/dcpw/vllm_ascend/attention/dsa_v41.py")
s = p.read_text()
if "DBO-NODEVMD" in s:
    print("已打过"); sys.exit(0)
old = '''    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True'''
new = '''    def enable_device_metadata(self) -> None:
        # [DBO-NODEVMD] ubatching 时把 metadata 计算留在 eager 路径：
        # device-metadata 任务队列（AICPU kernel QuantLightningIndexerV2Metadata 等）
        # 在 ubatch 下的入参尚未适配 ⇒ 实测 aicpu exception(22007)。
        try:
            if getattr(self.vllm_config.parallel_config, "use_ubatching", False):
                self._device_metadata_enabled = False
                self._device_metadata_tasks = ()
                return
        except Exception:
            pass
        self._device_metadata_enabled = True'''
if old not in s:
    print("锚点未命中"); sys.exit(2)
s = s.replace(old, new, 1)
ast.parse(s); p.write_text(s)
print("已在 ubatching 下禁用 device-metadata")
