# TP8 双 SDK 门证据桥 v9 CPU

本组件只读加载两个根线程收集的 v8 原 job：`static_keep` 和 `nonstatic_keep`。两个实际 job 均已 exit0，八个 rank 各完成 32 代、严格 copy/alias、native 非默认 stream 与原基线恢复，八个自有 child 均正常 reaped 0。组件不会执行 SDK、RPC、NPU 或新 job，不安装消费者，不改写旧收据。

入口为 `evidence_loader.load_tp8_sdk_pair(static_job_dir, nonstatic_job_dir, source_root=None)`。返回 `Tp8SdkPairEvidence`：frozen dataclass、递归不可变 mapping，并提供 `to_dict()` / `as_dict()`。static/nonstatic 各保存原 job dir/id、原大 driver 的 remote path/bytes/SHA、compact、collection、controller、source manifest、cleanup，以及八个原 rank/ABI/kernel 引用和当前 native scope、十三个 owner seals。

`source_contract.json` 固定已核验的 v8 freeze/SHA256SUMS、十四个源件、真实 SDK/native/ABI/bootstrap 源与版本、每种 variant 的实际 compiler options，以及两个原 collection receipt / fetch manifest hash。nonstatic 的 SDK `as_dict()` 省略默认 false 的 static 编译键，按实际原记录核验，不伪补字段。

loader 核对收集 manifest 下每份被使用原文件的完整字节 SHA；大型 driver 原字节保留在远端，compact 必须绑定根线程已确认的原 hash，不能声称本地验证了未传输原字节。八个 rank 原 result、raw RPC、controller nonce/路径/源和 loaded ABI 绑定必须一致；ABI owner、live code hash、normal-init 身份、native owner/current/default/baseline handles、正常恢复、13 seals、实际完整 copy trace、原 marker sources、返回 tuple/cache alias 与 19 Tensor/kwargs 边界逐项验证。32 个原 epoch 的实际 fence 必须递增；两个 producer/reader 原 export 的 BF16 数值、位置、epoch、count/rows/valid 再由 stdlib 重算并验证。

整 job 必须 driver passed、run.exit0、八个实际退出码全 0、无 hard kill/pending/uncaptured。双 job 必须同 frozen source、SDK、outer keep 和物理 8–15，对应 static/nonstatic，使用独立 job/driver/worker 身份。缺 nonstatic、缺 rank、复用 PID、错源/策略、临时 copy、缺 owner/scope、旧/重复 epoch、假 fence、-9 或不完整终态均拒绝。

`protocol_passed=true` 只表示这些受根线程 hash 约束的原证据满足协议。返回对象自己的 `actual_sdk_gate_passed`、device/consumer/prepack/formal 收据恒为 false；真实 SDK 成功只引用原 job，而新消费者覆盖、copy-before-pack、模型数学、质量和性能均需独立实际证据。旧 v1 `require_sdk_receipts` 的 single-chip/`initialized_before=false`/exporter schema 不直接兼容标准 TP8 normal init，不能改旧收据让它通过。未来独立 full43 接线使用本桥保留的更强原 provenance。

19 项 CPU 协议门通过，含真实双 job 正例及必要拒绝反例；未导入 Torch/torch_npu/vllm。测试只在新目录内创建和清理普通文件反例。实际字段规范化结果见 `evidence/actual_pair_protocol.json`，CPU 日志、旧包严格完整性与最终清单见 `evidence/`、`cpu_validation.json`、`freeze.json`、`SHA256SUMS`。旧 v1–v8 和其他 full43/prepack 模块保持不变。
