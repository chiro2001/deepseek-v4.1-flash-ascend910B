# CED v4 新 P/D jobs-path 预检与 recipe 重生成收据（2026-10-13）

状态：现场只读门已采集；基于 root 指定的可写 `jobs/` roots 重生成独立 CPU-only
recipe 与 plan。Root 已实读审核并批准下列新 recipe SHA。**最终启动门于 08:55:46
发现外部容器 `fyf_1009` 的实际 NPU workload 先占用 physical chip 5，后续轮询又在
physical chips 0、1 和 9 发现同名短命进程，且 physical 9 属于 D 目标集，故尚未创建 P5/D5
job、尚未初始化服务、尚未启动 proxy、尚未发送 generation。必须等待该外部进程自然
释放并重新通过 16-chip 门后才可继续。** 原
`p5_launch.sh` / `d5_launch.sh` 和其 CPU-only receipt 均保留，未覆盖。

## 现场来源与资源门

目标仅通过 `ssh a3-21` 读取。容器状态、完整包/源码检查结果、资源/端口/NPU 快照、
路径 mount 与僵尸 RSS/PSS 证明保存在 [`field_gate_20261013/`](field_gate_20261013/)，
逐件 SHA 在其中 `artifact_hashes.sha256`。

- P `dsv41-ced-p0-7-20261013-v2`（CID `16d3045321f8…b6c631bd`）和 D
  `dsv41-real-stack-tp8-20261010-a321`（CID `b1aa15d88003…3b429eb0`）均为
  `running`, restart `0`，没有 Docker Healthcheck。
- 冻结 CED v4 包 manifest 在 P、D 各 `29/29` 通过；包 archive SHA
  `3916d72f0fb798dac9ffe3dab243abd4999e2de42e6f0cc83a7b3785ef940aab`，
  29-file manifest SHA `d57ed5caf1688e29af9090a9403ef989a704c0c8cf49b9e66ed20642d606c89b`。
- 原 frozen source candidate contract SHA 为
  `5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`；P、D
  各 `220/220` 源文件校验通过。Candidate fingerprint 仍为
  `8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`。
- 两端完整 `/usr/local/Ascend/cann-9.1.0/opp/vendors` 树都匹配同一 canonical
  SHA `0f1a8d8206d3fd36ab3012dc6c547f5fafe974d5cdab3c53ace27b4c37c8f2fe`：722
  entries、103 directories、618 regular files、1 symlink、222,548,156 B。
  `config.ini` 两端均 30 B，SHA
  `7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`，内容行
  `load_priority=batch_invariant`。version.info、libcust_opapi、opsproto、opmaster
  四个 BI pins 两端一致并通过。
- P 映射 physical chips `0–7`，D 映射 `8–15`；现场没有任何 0–15 运行中的 NPU
  进程，`npu-smi` 16 芯片 Health 全 `OK`，HBM 为约 `2.87–3.13 GB` idle baseline。
  未进入/修改 `xp_mk` 或其他外部容器。
- host `MemAvailable=2,072,850,208 kB`（约 `1.93 TiB`）；P/D 当前容器 memory
  分别约 `28.68/49.23 GiB`。没有 drop cache。P/D 容器进程表分别有 434/1,117 个
  累积 zombie，所有 zombie RSS 总和与最大值均为 0 kB；PIDs limit 为 unlimited。
  这些不是本轮 worker 数。A2v4 runner/worker 和旧 D4 worker 的代表性 zombie 没有
  address space，PSS 不可测/不适用。无人 signal/reap 或重启容器。
- 宿主无 `18960`、`18990–18992`、`19090–19097` listener。P、D 容器对上述全部端口
  的 `connect_ex` 均为 `111`。未来 P rank 1 通常会拥有 producer group 的 `19091`；
  服务 ready 后需按 listener owner 归因，不能把该 P listener 误报为 D consumer 冲突。
- 原 recipe path `/home/l00886679/tmp/dsv41-ced-runtime-20261013/p5` 在 P 是只读 bind
  mount，`test -w` 失败，因此没有执行旧 recipe。新的预期路径均不存在：
  - P `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/p5`：父目录 `jobs/` 是
    host bind mount，`rw`，`test -w` 通过；新 `p5` root 尚不存在。
  - D `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/d5`：`jobs/` 位于 D 的
    read-write overlay（不是 P 的 host bind，也不是共享路径），`test -w` 通过；新
    `d5` root 尚不存在。D 的产物应通过 D 容器域收集，不能假设宿主/P bind 可直接读到。
- 新 P/D `DELIVERY_TMPDIR` prefix 均 58 B；加 `/` + 36-char UUID + NUL 为
  `96/107` B。实际启动时仍需保留实测 IPC path 复核。

## 新 recipe 和 CPU-only service plans

输入仍为已审候选，不变更模型/source/OPP/core overlay/D-only clamp、`best`
multistream、KV 配置、Engram 或 graph role。两个脚本 `bash -n` 均通过，recipe-plan
都记录 `service_launch_requested=false`。

| Role | 新 recipe SHA-256 | Recipe-plan SHA-256 | CPU-only runtime fp | 新 job/output/cache/TMPDIR |
|---|---|---|---|---|
| P prefill | `23e5916c2048ccc8a64948e88569b7e346b1f531c15b81644abc0ac1cfac443a` | `f6c998e5155fa3aead03ff82449d2de5c9785d3665d71152208f6f0fd7f4ffdd` | `e7674b826f7c38a789ac1e5e0ee176eff2c1447b8f1b670a41710ed0c937f700` | `/jobs/p5`; output `/jobs/p5/service`; native/cache `/jobs/p5/service/{native,cache}`; TMPDIR `/jobs/p5/tmp` |
| D decode | `00331485daab4ba7fca41ae3bbbcecea93954515a78469889b4e12c947198020` | `f74248ba60888698703210cf4deee02353acdd40843f7a948c8f7f0c035c806b` | `d192f5fbd6d068d6079de6b8a2ae8ca552c723eb669a1d2e695f9ed1aa2cc3c9` | `/jobs/d5`; output `/jobs/d5/service`; native/cache `/jobs/d5/service/{native,cache}`; TMPDIR `/jobs/d5/tmp` |

完整绝对路径请以表中目录前缀
`/home/l00886679/tmp/dsv41-ced-runtime-20261013` 拼接。CPU plan SHA：P
`77fef4a039317703c272e33eebd741c1f607c37294bc0863a77b130a419f0f93`；D
`9f9c173032074d452cfa6aed572adb2aef1303ca22e0d5e5e5a4c2541b7ee7a6`。

两份 CPU plans 是由 hash-verified package 的 `make_ced_plan` 纯函数生成，未调用
`ced_service.main()`（没有创建 job/output directory），也未加载 Torch/NPU。CPU plans
中的 `native_opp_receipt=null`，所列 runtime fingerprint **不是未来实际 runtime
fingerprint**；`--execute` 时仍须重检 OPP，实际 plan/fingerprint 将包含 OPP receipt。

- P physical chips `0–7`，TP8、prefill/producer `kv_port=19090`、draft graph 关闭、
  Engram enabled/device index `0`，无 compressor env。
- D physical chips `8–15`，TP8、decode/consumer `kv_port=19091`、draft graph 开启、
  Engram enabled/device index `0`；仅 D 有 compressor path
  `/work/delivery_matrix_ced_20261013_v4/delivery_matrix/ced_compressor_20261013_v1`
  和 manifest SHA `accf27fa871e62a78cb5e7aed3318c789f69feae14a5299db347795519e4e9f0`。
- P/D 均保持 `dsv41-ced-pd-v4`、`max_model_len=1048576`、KV bytes
  `15728022528`、multistream `best`、BI、Engram；不增加 DCP/shared-Engram/DRAM。

## 最新阻断：外部容器 NPU workload 持续活动（physical 5 → 0 → 1 → 9）

在最后只读启动门 `2026-10-13 08:55:46+08:00`，完整 `npu-smi info` 显示 logical
NPU 2 / physical chip 5 上有进程 `rain_fusion_att`，host PID `3215293`，process
memory `304 MB`，AICore `100%`；NPU-SMI 的 container-PID 列为 `NA`。只读检查
`/proc/3215293/cgroup` 指向 Docker ID
`f7f117abc17e338d8051fc253d412dd60ae56f170f8fd3b1dd7c8451958b5419`，只通过
`docker ps -a --no-trunc` 元数据映射到 `fyf_1009`（Up 12 minutes）；没有进入该容器。
该 PID/cgroup 归属记录及完整 NPU 表在 `field_gate_20261013/`。

约 60 秒后的第一轮复核（08:59:23）中，原 PID 已不在 `/proc`；但完整 NPU 表报告同名
`rain_fusion_att` 新 PID `3218039` 正在 logical NPU 0 / physical chip 0 上运行，304 MB、
AICore `100%`，container-PID 仍为 `NA`。08:59:40 只读 `/proc` 显示该 PID elapsed
`1:55`，cgroup ID 仍相同并映射到 `fyf_1009`。因此不能仅凭原 PID 结束判定外部 owner
已释放；目标物理 0–7 仍有实际进程占用。round1 的完整采样与 PID/cgroup 复核均已归档。

第二轮（09:01:37）又在 logical NPU 0 / physical chip 1 观察到 `rain_fusion_att`
PID `3220109`，同为 304 MB / AICore `100%` / container-PID `NA`；此 PID 在 22 秒后的
09:01:59 `/proc` 采样时已消失，但 `fyf_1009` cgroup 对应容器仍 `Up 17 minutes`。
两轮前后可见相同容器中的短命 NPU 进程由 physical 5 变为 0、再变为 1。该 owner 的
实际 workload 仍在活动，不能把单个 PID 结束或单一芯片瞬时空闲作为资源释放门。
round2 全量表与 `/proc`/容器元数据复核均已归档。未获 root 对“全 16 空闲后启动”或
“仅 D”路径的决定前，保持 P/D 两侧都不初始化。

Root 随后明确决定不先初始化单 D。第三轮（09:04:49）又在 logical NPU 4 / physical
chip 9（属于 D 的 `8–15` 芯片集）观察到 `rain_fusion_att` PID `3222742`，304 MB、
AICore `100%`、container-PID `NA`；紧接着 host `/proc` 读到同 PID state `R`、elapsed
`1:04`，cgroup 仍指向 `fyf_1009`（Up 19 minutes）。因此外部 workload 已证实轮到 P
区与 D 区；当前不能启动 P 或 D，也不能采用仅 D 规避路径。完整 16 芯片第三轮采样已归档。

新 goal 回合的权威只读复核时间为 `2026-10-13 09:13:30+08:00`。当时 P/D 容器仍
`running`, restart `0`；完整 NPU 表仍见 logical NPU 2/chip 1（physical chip 5）上的
`rain_fusion_att` host PID `3229365`，304 MB、AICore `100%`、container-PID `NA`；
host `/proc` 显示 PPID `3229363`、state `R`、elapsed `03:29`，cgroup 同为 Docker ID
`f7f117…b5419` → `fyf_1009`（Up 28 minutes）。Docker top（只列 PID/PPID/state/elapsed/
comm，不含 Args）显示其父链 `bash → MainThread → bash → timeout → rain_fusion_att`
仍 live。宿主 listener 与 P/D 的 `18960`、`18990–18992`、`19090–19097` connect 检查
全部空闲/拒绝，`jobs/p5`、`jobs/d5`、`jobs/proxy5/wire` 仍未创建。原始完整小件
SHA-256 `485df324ecc803252170eb21b0fe443523af97160661e4654868d6f7ccbd7b0c`，路径
`field_gate_20261013/fyf_authoritative_recheck_20261013_091330.txt`。本轮没有复核/改动
外部容器，也没有初始化任何项目角色。

此前 `field_gate_20261013/a3_current_resource_port_npu.txt` 的无进程快照已过时，不能
作为本次启动授权依据。保持 P 与 D 都不启动、不停止/触碰 `fyf_1009`；按 root 要求
约每 60 秒只读复核当前 NPU PID/同 owner cgroup 与完整 NPU 表。若该 workload 自然结束且所有 16 芯片
再次无外部运行进程，其他门仍需确认有效后，才可按已批准 recipe init。

新脚本、recipe plans 和 service CPU plans 为 `p5_jobs_*` / `d5_jobs_*` 文件；原
`p5_launch.sh` / `d5_launch.sh` 与旧 CPU-only plans 仍在原位，未覆盖。

## wire wrapper 准备件

独立 wrapper 副本及 `--help` 输出见 [`wire_wrapper_prepared/`](wire_wrapper_prepared/)：
wrapper 6,870 B，SHA `82873d5b2c454d50580b80c39a46970f69e18c7b15118fa8f7dcec78d098c0d1`；
原 proxy SHA `813c5a0e1a8b7e7988ee9f02fd3a83d8fa38ede9a88713a9c65e5cf0cf99eb33` 在工作树、
hash-verified 包和 P 容器源均匹配。Root 允许将同一 wrapper 复制到 P 的可写
`/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/proxy5/ced_wire_audit.py`，并仅运行
`--help`；本地与 P 内 help 输出分别归档。该 `jobs/proxy5` 仅作为源码准备目录存在，
P 内 port 18992 仍 `ECONNREFUSED`，`jobs/proxy5/wire` audit-output 目录不存在。
没有启动 proxy、建立 listener、发送 HTTP 或 generation 请求。

本收据及 blocker snapshots 的逐件 SHA 收录于 `new_artifacts_sha256.sha256`；所有新
artifacts 与原有证据均独立保存。
