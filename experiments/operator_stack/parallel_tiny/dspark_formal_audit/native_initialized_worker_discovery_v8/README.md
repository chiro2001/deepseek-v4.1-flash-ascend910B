# SDK init-only 小门：准确 owned copy 集合与正常退出等待 v8

v8 供主线程运行独立 fresh `static_keep` / 32 代小门，仅修订 copy destination 的 owner 集合和正常成功 worker 的退出等待。本 revision 的 NPU 权限与 NPU 作业数均为 0；v8 实际 SDK、32 代、alias、device、full43、prepack 与正式收据均为 false，full43 继续暂停。

v7 `static_default` 的八个 rank 都完成 32 代语义、原生非默认 stream 与正常基线恢复，但 epoch 先写临时 destination、再 copyback，原 alias 门拒绝，整体失败。v7 `static_keep` 的八个原 result 则都通过原 alias 门：每 rank cold-capture 仅八条直接 payload copy、32 代、正常 native stream 恢复。该 v7 作业仍整体失败：rank2/3 正常退出 0，其余六个成功 child 在旧 5 秒正常退出等待后被 verified pidfd 收口为 -9。不能把 rank gate 通过改写成完整 job 通过，原始证据保持。

v8 在 probe allocation 后，从实际原 producer/reader 十二个 Tensor 和 native cache 创建准确十三个 owner seals。每个 seal 记录 owner 名、实际 Tensor object id 与完整 ptr/shape/stride/dtype/device/format；capture 后再从相同 live Tensor 取 seals，要求严格相同。native cache 的返回 alias 和每代原 owner seals 也必须保持。不会从观察到的 copy 反推或新增 owner。

每条 capture copy 都必须使用正确 native stream，实际 destination 与 logical destination 必须精确一致，并匹配封存的原 owner/layout。逐条归类为八个 payload/marker、四个 count 或 native cache；未知、临时、错误 owner、布局变化和 stream 变化都拒绝。payload/marker 的原输入 source 检查、八个必需 payload destination 与原返回 tuple alias 检查保留。新增 count/cache 分类解决旧检查器把所有 copy 交给仅八个 payload ptr 集合的问题，不跳过非 payload copy。

成功清理默认等待从 5 秒改为 45 秒，所有 worker 共享一个 deadline。等待到期仍只对经过 PID/PPID/boot/start/cmdline/pidfd 核验的自有 child 收口；failure 分支不等待成功 grace，原身份检查与 kill/join 逻辑不变。清理收据如实记录本次实际使用的 grace。真实 Linux CPU child 延迟 6.2368 秒后正常 reaped 0，无 hard kill，覆盖旧 5 秒边界；失败忽略 TERM 的普通 CPU child 仍立即以 verified pidfd 回收。

本次只跑直接受影响的 57 项门：contracts 24、owner 19、namespace/bootstrap 4、真实 Linux cleanup 10（skip0）。owner 门包含八个 v7 default 原 trace 的拒绝回归，以及八个 v7 static_keep 原 trace 的接受回归。v7 没有记录 cache object seals；回归中的缺项明确以独立 synthetic CPU setup 补全，绝不视为 v8 NPU 收据。count/cache 准确 owner 正例和未知/temp/layout/owner/source/stream 负例均覆盖。

另在正式容器的隔离 CPU 环境运行两个独立 PyTorch 2.10.0+cpu 双 AOT case，复用原 v5 ABI/transport/program 和 SDK-like backend。外层 keep False 的首调用观察为 21 copies、2 个临时 epoch、6 个 marker copyback；keep True 为 34 copies、4 个临时 epoch、6 个 marker copyback。两臂 CPU 数学、原 owner seals、返回 alias 与 19 Tensor kernel assert 保持。这些是首次 invoke 的 CPU 观测，不能外推实际 SDK 的 cold-capture copy 数或 alias 结论；实机 v7 `static_keep` 的直接 copy 证据优先。未导入 torch_npu/vllm/acl，设备可见性全空、backend autoload 禁用。

数学 program、CopyMode、fw/backend、native capture lifecycle、强 ABI、boxed Tensor/kwargs adapter、SDK/kernel assert 和 tuple transport 保持。旧 v1–v7 冻包逐文件 SHA 不变。除正常 cleanup grace 外，ABI/数学/native 相关八个源文件逐字节一致；driver/controller/SDK worker 只有新包 namespace。完整来源、原 result SHA、日志与 receipts 见 `evidence/`、`cpu_validation.json`、`freeze.json`、`SHA256SUMS`。

主线程重新核验资源与 host 门后，以全新目录运行：

```bash
bash /work/NEW_PRIVATE_SNAPSHOT/stack/run_stack.sh \
  parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v8/init_only_driver.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/NEW_JOB/gate --variant static_keep --generations 32
```

`NEW_JOB/gate` 须尚不存在。candidate 源集仍为 14，每 variant 使用 fresh worker；launcher、既有 AA 依赖与 NPU 窗口由主线程管理。主线程先做 `static_keep`，再以另一个 fresh job 做 `nonstatic_keep`，源码与 outer keep 都相同。两个完整 job 都满足全八 rank 严格 gate、32 代、alias 与最终退出码全 0 后，再由主线程分配独立 full43 接线。

旧 v1 `adapters.py` 的真实双门函数为 `require_sdk_receipts(static, nonstatic, exporter_sha256)`。未来独立 TP8 桥可复用其双门原则，并验证以下原始字段，不能直接把 v8 receipt 当成旧 single-chip schema：

| 双门证据 | v8 的实际来源与核验 |
|---|---|
| static/nonstatic、同 outer keep | 两个 driver 的 `variant`；八个 rank 的 `stage=graphex`、`static_kernel` 与 `outer_keep_inference_input_mutations`，都必须一致对应 |
| 同冻结源与真实 SDK | controller manifest 的十四个 `candidate_sources`；rank 的 `source_gate`、`sources.npugraph_ex`、`sources.npu_fx_compiler`、torch/torch_npu 版本及 `native_capture_scope.source`；两个 job 对应源 SHA 相同 |
| 至少 32 个真实 epoch 与 fence | 每 rank `generations` 连续 1–32，payload/positions/count/cache/returned/stable 布尔全真，`device_fence_monotonic_ns` 是原同步后的实际时间；八个 rank 原证据全部齐全 |
| 当前 owned copy 与 alias | `buffers_before/after_capture`、十三个 `owned_copy_destination_seals_before/after_capture`、`capture_calls`、原 marker sources、返回 tuple/cache layout 与 `copy_alias_structure`；逐条核验正确 native stream，`alias_error=null`、alias 与 stage/pass 都真 |
| fresh TP8 worker 与 native 生命周期 | controller 的 nonce/job/source 绑定、每 rank PID/boot/normal_init_return_ns、driver birth PID/PPID/boot/start/pidfd；两 job 的 worker 身份独立，物理 8–15 对应一致；原 init-return 基线与新 native stream 分开、退出严格恢复 |
| 完整 job 成功 | driver `passed=true`、`worker_exit_codes_complete=true`、八个实际 reaped 退出码全 0，cleanup 无 pending/uncaptured |

旧函数还绑定 v1 gate/exporter SHA、`initialized_before=false`、`device_count_before=1`、旧 epoch/fence 字段与单 `physical_chip`。v8 的 worker 由标准 TP8 正常 init 后执行，真实基线为 initialized，使用 tuple transport 和当前 gate 源；这些事实不能改成旧字段或伪改旧收据。未来桥必须引用两个完整新 TP8 job 的原证据并保留更强的当前 owner/seals 检查。本小门始终不等同于 17 hooks、43 个 device consumers 或 full43/formal 通过。
