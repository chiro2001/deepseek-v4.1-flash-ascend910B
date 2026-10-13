# A3 P0–7 TP8 容器配置与实际部署（2026-10-13）

采集时间：2026-10-13 03:27 +08。基于现有容器 `dsv41-real-stack-tp8-20261010-a321`
（ID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`）的只读
`docker inspect`；P0–7 指 physical chips 0–7，即宿主 NPU 0–3。采集时 NPU 0–3
均无运行进程；DCP8 v3 正在使用 NPU 4–7。注意：BI OPP 目录虽在**当前容器文件系统**
中可见，但父代理随后核实基础镜像 ID `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea`
并不含该 OPP；该目录由当前容器 writable layer 提供（约 214 MiB / 617 files）。
因此新的 P0–7 容器不得直接假设 OPP 随镜像存在；同一 `vendors/` 下的 `config.ini`
也不在基础镜像中。已从当前容器复制完整 `vendors/` 目录到显式宿主任务目录，并分别核验
父树、BI 子树和配置文件哈希。新 P 将以只读 bind mounts 使用，具体证据见
[`opp_config_ini_addendum_20261013.md`](ced_preflight/opp_config_ini_addendum_20261013.md)。

OPP 只读来源审计（offload_research，未操作当前容器/服务）：base image 以
`--network none --read-only`、无 NPU devices 的短命 inspect 确认 `BI_OPP_MISSING`；
适用 A3 安装包是当前 `/work/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run`
（105,731,094 B，SHA-256
`9fc692978e9420336e3fea03a92c2a85df1b50a65a7df50173e3bf8bedaea70e`），不是 910B 包。
当前 writable-layer 安装树关键文件：

| 文件 | SHA-256 |
|---|---|
| `version.info` | `cc3e7001ed18e748e400d78bbcd3a8c51c6d292bd7f252a3ed9c4268dc1a5371` |
| `op_api/lib/libcust_opapi.so` | `ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069` |
| `op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so` | `f8308be503301621117cd421e411240a077dc4fcf04972321e403e7efc818010` |
| `op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so` | `68221652dffb7fa13351616ecdf4e3445425697452fed29ae147f0cc335b02c1` |

额外发现，真正选择 BI OPP 的 `/usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini`
同样仅存在于现 D 容器 writable layer：30 B，内容为 `load_priority=batch_invariant\n`，
SHA-256=`7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`。只挂载
`vendors/batch_invariant` 子树不足以复现 BI 激活状态。P 将只读挂载经核验的 host
`vendors/` 父树；其完整 manifest 同时绑定子树和选择配置，避免重复嵌套 mount。

完整 image/OPP/SDK 只读证据：[`image-and-opp-readonly.md`](../ced_preflight/image-and-opp-readonly.md)，
SHA-256 `337902580f50a5ace2a8f4ad49f583bbeb0e93c0a936a95fc548fc989990dc38`。该审计还确认
相关 vLLM/vllm-ascend 源与 base image 一致；无需额外复制这些源码，关键部署缺口是 OPP。

## 拟复用配置

- Image：`sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea`（tag `local/dsv41-operator-stack-real:20261010-production`）
- Workdir：`/workspace`
- Network：`host`；IPC：`shareable`
- Privileged：`true`；`shm-size`：16 GiB；restart policy：`no`
- `ulimit memlock=-1`；`security-opt=label=disable`
- 设备：`/dev/davinci0`–`/dev/davinci7`，权限 `rwm`；另映射 `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`
- 设备环境差异（相对当前 P8 容器）：
  - `STACK_PHYSICAL_CHIPS=0,1,2,3,4,5,6,7`
  - `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
  - 运行服务时对应设置 `ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`、`LOCAL_WORLD_SIZE=8`
- 正式服务 OPP：`ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant`
  - P 只读挂载完整 `vendors/` 父树；其中 `config.ini` SHA-256=`7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`。
  - 来源完整性：`vendors/` manifest SHA-256=`0f1a8d8206d3fd36ab3012dc6c547f5fafe974d5cdab3c53ace27b4c37c8f2fe`；BI 子树 manifest SHA-256=`3a8b878c9e8f2cd370e881dc248fd05e9bbbf07e7317a0848069b624179feb4a`。

双容器 CED ready 还须做来源绑定：P/D 的 plan/selection 与每个 rank 都应报告
64 位 `ced_overlay_manifest_sha256`；proxy 必须 fail-closed 验证两端 SHA 完全一致后才接受
pair ready。这个 ABI/来源门与容器 image/device profile 分开记录，但应在真正部署包里一起验收。

## 原容器 bind mounts（拟保持原 source、destination 与访问模式）

| Host source | Container destination | Mode |
|---|---|---|
| `/etc/ascend_install.info` | `/etc/ascend_install.info` | ro |
| `/home/l00886679/models` | `/home/l00886679/models` | ro |
| `/home/l00886679/models/out/v41-w4a8-stage1` | `/home/l00886679/models/out/v41-w4a8-stage1` | ro |
| `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-mtpq/engram_int8` | same | rw |
| `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-mtpq/optional` | same | ro |
| `/home/l00886679/models/out/v41-w4a8-engram-dr-vision/engram_int8` | same | rw |
| `/usr/local/dcmi` | `/usr/local/dcmi` | ro |
| `/usr/local/Ascend/driver` | `/usr/local/Ascend/driver` | ro |
| `/usr/local/bin/npu-smi` | `/usr/local/bin/npu-smi` | ro |
| `/home/l00886679/models/out/engram-int8` | same | rw |
| `/home/l00886679/models/out/v41-w4a8-engram-dr-vision` | same | ro |
| `/home/l00886679/projects/dsv41/models/out/v41-w4a8-dspark` | same | ro |
| `/home/l00886679/projects/dsv41/models/out/v41-w4a8-dspark/optional` | same | ro |
| `/usr/local/Ascend/firmware` | `/usr/local/Ascend/firmware` | ro |
| `/home/l00886679/projects/dsv41-real-stack-tp8-20261010` | `/work` | rw |
| `/etc/hccn.conf` | `/etc/hccn.conf` | ro |
| `/home/l00886679/models/out/v41-w4a8-dspark` | same | ro |
| `/home/l00886679/models/out/v41-w4a8-dspark-mtpq` | same | ro |
| `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-mtpq` | same | ro |
| `/home/l00886679/projects/dsv41/models/out/engram-int8` | same | rw |

## 实际部署状态

- 新 P 容器 `dsv41-ced-p0-7-20261013-v2` 已创建并运行，ID
  `16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`，固定 image
  `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea`；仅映射物理
  0–7，`RestartCount=0`。P 全 `vendors/` 父树只读挂载，容器内完整 manifest 与源一致。
- P 的 CED v2 服务 `dsv41-ced-pd-v2` 已在 18990 返回 ready identity：8 rank、
  `engine_initialized=true`、`ASGI_startup_completed=true`、批不变开关为 1。服务/OPP/overlay
  身份、未完成的 D 启动尝试和未执行请求的边界见
  [`ced_v2_startup_attempt_20261013.md`](ced_preflight/ced_v2_startup_attempt_20261013.md)。
- D 复用既有 TP8 项目容器，不创建 D 容器。冻结服务 v2 的 decode 角色完成权重、KV、
  kernel compile 和 graph capture 后，在安装 HTTP guard 时退出（`run.exit=1`），18991 无监听；
  这是服务包中间件安装时序错误，不代表模型/OPP 失败。尚未启动 proxy 或发送 canary，
  因而**CED P/D pair 尚未 ready**。任何 guard 修复都必须作为新候选重新绑定 fingerprint，
  不继承当前 v2 的质量或性能结果。

**不要 `docker commit`**（默认可能暂停正在服务的容器），也不要用未经验证的 910B OPP 包代替。
