# base1 首启显存门故障复核（只读）

日期：2026-10-13。范围只限本地冻结的 base1/A2v4 计划、故障日志、vLLM/vllm-ascend 本地源码和已归档的启动前/失败后快照。未 SSH、未访问在线服务/NPU、未发请求、未重启或重跑、未修改代码。现场回执补充了近窗 NPU PID/HBM 表，但采样无精确 wall-clock timestamp、PID 的 cgroup 和 rank→physical 实测映射缺失；因此本报告区分直接原因、强线索与仍未证实成因。

## 结论

- **已证实的直接原因：** 一个 worker 在 `NPUWorker.init_device()` 的启动显存准入检查处抛出 `ValueError`。日志值为 `10.97/61.27 GiB` 可用/总显存，`gpu_memory_utilization=0.7` 对应请求 `42.89 GiB`；比门槛少 **31.92 GiB**。EngineCore 随后启动失败，`run.exit=1`，服务/API 未 ready。
- **已证实的阶段边界：** 检查发生在该 worker 选设备、`torch.npu.set_device()`、inductor 导入、`gc/empty_cache` 和可能的 local-comm resource setup 之后；在该 worker 的 HCCL distributed init、workspace/ModelRunner 建立、`load_model()` 之前。故不能把这条异常说成已加载模型权重、KV 或图捕获所致。其他 worker 是并发启动的，不能只凭这条异常断言整个进程组完全没有更早的设备分配。
- **最强的近窗线索：** 现场回执 `base1_failure/README.md` 记录了初始化/失败窗口的 NPU 进程表：physical 12 有 `python3` PID `3428147`，报告进程显存 45,387 MB、die HBM 49,230/65,536 MB；该 PID 不在同轮目标 D 容器的 `docker top` 名单中，NPU 工具的 container PID 为 `NA`。如果失败 worker 实际映射到 physical 12，这一进程足以解释显存门附近的大额占用；但采样无精确 wall-clock timestamp，失败 worker 的 rank/物理映射未留存，PID 的 `/proc` status/cgroup 事后不存在，**所以只能判为强相关线索，owner 仍未知**。
- **仍未证实的根因：** 不能把上述 PID归给某个租户、轮换任务或 `fyf_1009`，也不能排除进程实际映射和计划不同、其他启动 worker/进程先分配、或设备初始化阶段本进程分配。异常文本本身也不能证明占用来源。
- **映射配置检查未发现“计划里重复选同一 die”：** base1 的 8 个物理 ID 和两个 visible 列表都是 `8,9,10,11,12,13,14,15`，各自 8 个值互异；`STACK_*` 是应用合同/校验，NPU runtime 实际设备选择使用 `ASCEND_RT_VISIBLE_DEVICES`。本地 Ascend 平台映射按 rank 从该列表取物理 ID，再反查成 visible ordinal；若运行时完整遵守这些值，rank4/local_rank4 对应 visible `npu:4`、物理 ID 12。**这只是代码/计划推导，现场没有每 rank 实际绑定回执，不能当作已观测映射。** `--device-index 0` 只设置 `V41_ENGRAM_DEVICE_INDEX=0`，不选择或平移 NPU rank。

## 配方对照：base1 与 A2v4 成功启动计划

比较 base1 `plan.json` 与 A2v4 停服前实际 `service/plan.json`：`model_kwargs` 逐项相同。8 项环境差异为：

| 项 | base1 首启 | A2v4 已启动实例 | 判读 |
|---|---|---|---|
| `STACK_PHYSICAL_CHIPS` / `STACK_EXPECTED_CHIPS` | `8–15` | `0–7` | 选择物理子集不同 |
| `ASCEND_RT_VISIBLE_DEVICES` / `ASCEND_VISIBLE_DEVICES` | `8–15` | `0–7` | 两个变量各自都为无重复的 8 项列表；与所选子集一致 |
| `DELIVERY_PHYSICAL_WORKER(_MANIFEST_SHA256)` | 无派生 worker | 有严格 pin 的 `0–7` 派生 worker | A2v4 通过仅改一处物理守卫适配 0–7；base1 使用其冻结的 8–15 base worker。不是额外的 NPU 选择参数 |
| `STACK_DSPARK_FORMAL_OUTPUT` / `VLLM_CACHE_ROOT` | base1 专属 jobroot | A2v4 专属 jobroot | 仅输出/cache 路径 |

二者 `V41_ENGRAM_DEVICE_INDEX` 都是 `0`，模型、TP=8、MP、DSpark5、`gpu_memory_utilization=0.7`、固定 4 GiB KV、max len 196608、batch 8192 和其他模型参数一致。A2v4 的 `service_selection.json` 记录 `engine_initialized=true`、`ASGI_startup_completed=true` 和八个 rank 的 `0–7` 派生 worker 说明；但其 `actual_quality_passed=false`、`actual_performance_observed=false`，这里仅把它称为**成功启动/服务 ready**，不是质量或性能通过。

base1 `service_selection.json` 仍带 CPU 计划 runtime fingerprint，`engine_initialized=false`、`ASGI_startup_completed=false`，没有 `rank_descriptions` 或实际 runtime fingerprint。CPU plan/runtime fingerprint 不能当作失败进程已验证的物理映射。

对应启动器 `/work/delivery_matrix_generic_20261013_v7/delivery_matrix/run_service.sh` 先设置 `TMPDIR`、后启动 `delivery_matrix.service`；base1 launcher 显式传入 `--device-index 0 --physical-chips 8,9,10,11,12,13,14,15`。本地 `service.py` 会把选择列表写入四个 `STACK/ASCEND_*` 合同值，而 `device-index` 仅落到 `V41_ENGRAM_DEVICE_INDEX`。因此，**“物理列表 + `--device-index 0` 把卡 0 又映射到每个 rank”不符合这份 launcher/plan 的参数语义**；但仍需现场绑定回执排除环境被容器或其他层重解释。

物理编号背景：仓库 `AGENTS.md` 把 A3 物理 0–15 定义为 16 个 die；`field_gate/a3_operator_revalidation_readonly_20261013_100507.txt` 的 `npu-smi` 表显示 8–15 是 NPU 4–7 的两 die 各一项。计划选择 8 个不同 die，即 4 张卡的两个 die，并非把同一个物理 ID 重复写八次。

## 故障日志与调用时机

`base1_failure/run.log`（原始 46,825 字节）记录服务 PID `1624804`、EngineCore PID `1625665`；错误栈 worker PID `1625867`。日志没有在该失败 PID 的启动段打印 `rank/local_rank`。现场 README 按 spawn 顺序（rank3 后、rank5 前）推测它可能是 rank4；因此本报告对 PID→rank 仍标为推测。失败 PID 的异常帧为容器内 `/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py:449`，抛出：

```text
Free memory on device (10.97/61.27 GiB) on startup is less than desired GPU memory utilization (0.7, 42.89 GiB)
```

对应的本地 `vllm_ascend/worker/worker.py` `_init_device()` 顺序为：

1. 得到 local rank；若没有显式 `assigned_physical_gpu_ids`，检查 rank 小于 `torch.npu.device_count()`。
2. `current_platform.logical_device_id_to_visible_device_id(local_rank)` 计算 visible ordinal，构造 `npu:<ordinal>` 并 `torch.npu.set_device()`。
3. 导入 `torch_npu._inductor`（若可用 Triton），执行 `gc.collect()` 与 `torch.npu.empty_cache()`；有硬件能力时调用 `setup_ascend_local_comm_res()`。
4. 建 `MemorySnapshot(device=device)`，并比较 `free_memory < total_memory * gpu_memory_utilization`；base1 在此处失败。
5. 通过显存门后才调用 `_init_worker_distributed_environment()` / HCCL 初始化，再设置随机种子和 Triton 属性。
6. `init_device()` 再建立 workspace/ModelRunner；vLLM `WorkerProc` 只有在自己的 `init_device()` 返回后才调用该 worker 的 `load_model()`。

本地源码定位：`upstream-v41/vllm-ascend-upstream/vllm_ascend/worker/worker.py:379–481`、`vllm/platforms/interface.py:288–333`、`vllm_ascend/platform.py:82–90`；vLLM MP worker 顺序见 `vllm/v1/executor/multiproc_executor.py:640–680`。运行日志显示可确认的健康 worker 为 rank 0–3、5–6，rank7 未见完成行；PID `1625867` 在显存异常处退出，但其 rank 只由 spawn 顺序推测。EngineCore 于 10:28:14 报 startup failed。日志无权重加载完成、KV 分配或图捕获证据。多进程并行启动，健康 worker 已过各自的门并进入 HCCL 初始化，只能证明它们越过自己的门；不能据此说明其实际物理映射或它们是否与失败 rank 共用了一块 die。

**源码 provenance 边界：** traceback 直接证明这次容器执行了该 `vllm_ascend/worker/worker.py` 路径和上述行号；没有采集到失败容器该文件的内容哈希。上面调用顺序是对仓库源码的本地审阅，与异常点/行号吻合，但不宣称已逐字节证明为容器叶文件。不要把仓库内 `dcp_runtime_20261013_v1` 的复制文件当成 base1 实际挂载/执行源。

## 成因分级

| 解释 | 当前证据与结论 |
|---|---|
| 启动前就有固定外部进程长期占用 8–15 | **未获支持。** 10:20–10:24 启动前 `npu-smi` process table 为空；10:05 只读表中 8–15 也无 NPU 进程。但这只是取样点，不覆盖后续启动窗口，也不证明无进程外分配。 |
| 失败附近 physical 12 有大额占用 | **直接观察到一条强线索，但归属未知。** `base1_failure/README.md` 报告 physical 12 上 PID `3428147` / `python3` 占 45,387 MB，die HBM 49,230/65,536 MB；同轮目标 D `docker top` 未列此 PID，NPU container PID 为 `NA`。NPU 快照无精确时间戳，`/proc` status/cgroup 已缺失；不能把它归给外部租户或具体任务。若计划映射成立，physical12 是 rank4 的目标 die，这会强烈支持该进程造成显存门失败。 |
| 失败 worker 是否实际映射到 physical 12 | **计划上应当如此，现场未实测。** 列表推导为 rank/local_rank4→visible ordinal4→physical12；错误栈 PID 的 rank 本身也仅按 spawn 顺序推测。没有 NPU PID→rank 交叉表或 worker binding receipt。 |
| launcher 把 8 个 rank 重复绑定到同一物理 chip | **计划层没有重复。** 四个列表均为相同的 8 个不同 ID；平台代码的预期 rank→visible→physical 转换也是一一对应。physical12 上有大额未知进程，不能仅由此反推 rank 重复。**实际映射未被运行时观测，故还不能完全排除容器/ACL 映射差异。** |
| base1 在 `--device-index 0` 选到了物理 0 或把 Engram index 当 rank | **与代码语义不符。** 本地 launcher 将其写成 `V41_ENGRAM_DEVICE_INDEX`，NPUWorker 绑定由 local rank 与 `ASCEND_RT_VISIBLE_DEVICES` 决定；成功 A2v4 同样用了 index 0。 |
| rank 0–3/5–6 或 EngineCore parent 在失败 gate 前先分配 | **目标 D 内 worker 不是当前最受支持解释，但仍不能只看日志排尽。** NPU 进程表 physical8、9、10 上列有目标容器 `VLLMWorker`（各117 MB）；physical12 的大额 PID 不在同轮目标 D `docker top` 名单。这使“目标 D 已知 worker 在物理12先分配”较不受支持；但快照精确时间、容器PID与rank交叉、PID cgroup 均缺，尚不能定位3428147来源。 |
| base1 的模型权重或 KV cache 分配过大 | **与失败进程阶段不符。** 异常 worker 在自己的显存门处就退出；代码顺序显示该门早于本进程分布式初始化和模型加载。仍可有初始化上下文/local-comm等显存，但日志没给出其具体分配量。 |

## 最小后续取证（仅在另有授权的下一次启动/现场采样中）

1. 先关联已采集的近窗 physical12 PID `3428147`：若现场仍保有同一时间的宿主进程表、PID namespace 映射或运行时审计，核对其宿主 PID、容器 PID、cmdline、cgroup 和 owner。当前 `/proc` 已不存在时，就把归属保持为 unknown，不用容器之外的 PID 号猜租户。
2. 若未来另有授权启动/采样，在 worker 做显存 snapshot 的同一时间窗保存 physical 8–15 的 HBM/free 表和完整 NPU process table；对每个 NPU PID 同时留宿主 PID、容器 PID、`/proc/<pid>/cgroup`、cmdline/starttime。重点核实失败 worker PID `1625867` 的 local rank、它是否关联 physical12，以及同一 die 是否出现两个 rank/容器 PID。
3. 核对实际 `ASCEND_RT_VISIBLE_DEVICES`、local-rank→visible ordinal→physical ID、`torch.npu.device_count()` 与实际分配的 `assigned_physical_gpu_ids`。计划推导 rank4→physical12 不替代实际映射记录。
4. 同步保存失败前后的每 die HBM，才能区分短时占用 race 与启动进程自身/其他 worker 造成的增量。当前失败后基线表只能说明占用已消失。

这份审计不建议通过下调 `gpu_memory_utilization` 来绕过准入门；那只会改变门槛，不会解释物理设备映射或约 31.92 GiB 的显存缺口。本次任务不做任何重试或服务修改。

## 证据索引与校验值

| 本地证据 | SHA-256 / 边界 |
|---|---|
| `base1_failure/run.log` | `192558e37f8be4e3d6e40528a5cafc5bd5d8143c2081e0ddcac8b7b34c1c491c`（与启动回执所记远端 46,825 字节 run.log 一致） |
| `base1_failure/run.exit` | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`（内容 `1`） |
| `base1_failure/plan.json` | `242295357597407e36eb865bb3ecf551f4add662ed7e8a6dbcfc2257944c495f` |
| `base1_failure/service_selection.json` | `e25982c728d09a78abe023e983fc0d77bf7dc9558ae11eb824ab6fe12f17278d`（CPU 计划 fingerprint；无 rank map） |
| A2v4 参照 | `experiments/delivery_matrix/evidence/a2sim_tp8_v4/prestop_snapshot_20261013/a2_runtime/service/{plan.json,service_selection.json,runner.sh}`；A2v4 仅作 0–7 成功启动对照，不视作 base1 物理 8–15 的运行映射证据 |
| 现场采样回执 | `base1_failure/README.md` 提供近窗 physical12 大额 PID 线索；但无精确时间戳、cgroup 和 rank 映射。`field_gate/baseline1_cpu_plan_receipt.md`、`field_gate/a3_operator_revalidation_readonly_20261013_100507.txt` 是启动前/失败后证据，不能单独归属该进程 |

本地 `vllm-ascend-upstream` checkout 当前 commit 为 `c173a64a44dec4ba97aaba6277b1dfc1562eda19`；工作区中 `worker.py`/`platform.py` 有已暂存的其他修改。报告只读审阅 `_init_device` 映射及显存门路径；其文件字节不作为失败容器文件哈希。
