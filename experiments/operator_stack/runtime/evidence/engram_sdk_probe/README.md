# Engram SDK 探针证据

结论与范围见 [探针说明](../../../parallel_tiny/engram_sdk_probe/README.md)。

eager NPUGraph `npugraph_direct_v5` 32/32 代通过；每代 host consumed.wait 变体第 2 代被实际 SDK 拒绝。正式选项对齐的 static/nonstatic AOT/GraphEx v6 均在 producer 首次调用 `TypeError: expected Tensor()`，尚未执行 GraphEx 真实回调或取得实际 alias/stream/marker 证明。所有安全 receipt 均为 false。

| 文件 | 用途 |
| --- | --- |
| [summary.json](summary.json) | 分阶段结论、源码 SHA 与尚未完成的能力门。 |
| [collection_receipt.json](collection_receipt.json) | 90 个原始文件的本地相对路径、远端路径、大小与 SHA256。 |
| [SHA256SUMS](SHA256SUMS) | 本目录文件逐项校验。 |
| [direct NPUGraph result](remote_mount/evidence/npugraph_direct_v5/result.json) | 32 代逐代值、epoch、稳定地址、双 fence 与延后 producer 的证据。 |
| [hostwait result](remote_mount/evidence/npugraph_hostwait/result.json) | 第 2 代 single-record/multiple-wait 拒绝与终态策略。 |
| [static v6 result](remote_mount/evidence/graphex_static_v6/result.json) | 正式选项对齐后的失败 traceback 与真实回调 metadata。 |
| [static v6 generated kernel](remote_mount/evidence/graphex_static_v6/producer_generated_kernel_0.py) | 实际 GraphEx 生成代码中的 functionalization clone。 |
| [nonstatic v6 result](remote_mount/evidence/graphex_nonstatic_v6/result.json) | 关闭 static kernel 的诊断对照，仍在真实回调前失败。 |
| [release receipt](remote_host/evidence/sdk_scope_release.json) | 终态 exit、physical4 健康/无进程及 probe 无残留。 |

`remote_host/source*` 保存所有冻结源码；`remote_host/evidence` 保存原始 launch、log、exit 和设备快照；`remote_mount/evidence/<stage>` 保存程序产生的 environment、result、progress、FX 和生成 kernel；`remote_mount/run*.sh` 保存容器 launcher。原始结果保持原样，包括说明文档已标注的 `source_v3` 顺序字段记录错误。

本目录处于 gitignore 的 runtime 路径下，提交时需要对本任务目录使用 `git add -f`。
