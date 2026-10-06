#!/usr/bin/env python3
"""[DBO-DRAFT-DEVMD 2026-10-06] 把 dsa_v1(AscendDSAMetadataBuilder) 的 device-metadata 开关
变成**可配**：默认恢复原始行为（enabled=True），只有显式 V41_DBO_NODEVMD_SCOPE=1 才在
ubatching 时关掉。动机：这次失败的 AICPU kernel 是 V1 名(VllmQuantLightningIndexerMetadata)，
而 v41 主模型的 V2 kernel 已被 V41_DBO_NODEVOPS=1 关掉 ⇒ 怀疑上一轮把 dspark draft 的
device-metadata 也一并关掉(它并不在 ubatch 线程里跑)，反而把原本正常的路径改成 eager。
另外给 qli 调用点加诊断打印。
"""
from pathlib import Path
import shutil, time

P = Path("/home/l00886679/cedpd-repo/patches/files/draft/dsa_v1.py")
s = P.read_text()
orig = s

# 1) 守卫变成可配
old = "        self._device_metadata_enabled = not _ubatching\n"
new = (
    "        import os as _os2\n"
    "        _nodevmd_scope = _os2.environ.get(\"V41_DBO_NODEVMD_SCOPE\", \"0\") == \"1\"\n"
    "        _en = not (_ubatching and _nodevmd_scope)\n"
    "        print(\"[DBO-NODEVMD] dsa_v1 enable_device_metadata: use_ubatching=%s scope=%s -> enabled=%s\"\n"
    "              % (_ubatching, _nodevmd_scope, _en), flush=True)\n"
    "        self._device_metadata_enabled = _en\n"
)
if old in s:
    s = s.replace(old, new, 1)
elif "[DBO-NODEVMD]" in s:
    print("guard 已是可配版本，跳过")
else:
    raise SystemExit("guard 锚点未找到")

# 2) qli 调用点诊断打印
anchor = "            qli_metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(\n"
inject = (
    "            import os as _os3, traceback as _tb3\n"
    "            if _os3.environ.get(\"V41_DBO_DEBUG\") == \"1\":\n"
    "                print(\"[QLI1-CALL] device_enabled=%s reqs=%s qsl=%s seqlens=%s maxq=%s maxkv=%s %s\" % (\n"
    "                    self._device_metadata_enabled, len(seq_lens), tuple(query_start_loc.shape),\n"
    "                    tuple(seq_lens.shape), max_seqlen_q, max_seqlen_kv,\n"
    "                    \" | \".join(\"%s:%s\" % (f.name, f.lineno) for f in _tb3.extract_stack()[-5:-1])),\n"
    "                    flush=True)\n"
)
if anchor in s:
    s = s.replace(anchor, inject + anchor, 1)
elif "[QLI1-CALL]" in s:
    print("qli 打印已存在，跳过")
else:
    raise SystemExit("qli 锚点未找到")

bak = P.with_suffix(".py.bak_draftdevmd_%s" % time.strftime("%m%d_%H%M%S"))
shutil.copy2(P, bak)
P.write_text(s)
import ast; ast.parse(s)
print("patched:", P, "backup:", bak)
