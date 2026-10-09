# CED P 侧 DRAM offloading 与 mock decode

状态：2026-10-09，goal active，初版代码与离线检查已完成，硬件验收进行中。

## 分支与资源

- 分支：`feat/ced-p-dram-offload`，基于 CED 已提交基线 `77ef6e1`。
- 隔离工作树：`/home/chiro/projects/dsv41/ced-dram-offload`。
- 原 `ced-pd-release` 工作树的用户未提交改动保留原样，未复制进本分支。
- 用户授权实验卡：a3-21 Phy-ID `6 7 8 9 10 11 12 13`，仅用于真实 P，TP8/EP8。
- 【实测】2026-10-09 11:10 CST，上述卡无 NPU 计算进程；设备锁表未见这些卡的锁。
- 【实测】6–7 Health OK；8–13 Health Alarm，8 的查询返回 `80C98001`，
  `AIC / RAS State / module error can not be fixed`。尚未重置设备。
- 【未确认】8–13 的执行可用性以及跨 6–13 的 TP8/HCCL 通信可用性，须先验证。
- 当前没有 16 张卡；用户后续授权 mock 可使用一张 device 初始化通信和消费 KV，
  不加载 D 模型、不启动真实 vLLM。额外卡在启动前重新核对占用与锁。

## 实现范围

真实 P 维持 CED 层 0–19 + layer-20 全局 KV 源投影。D 的生产协议保持
MooncakeHybridConnector 的 128-token replay 交接契约。仅 P 接入本机 DRAM
二级缓存：HBM 命中直接复用，DRAM 命中搬回 HBM 并补算尾部，两级 miss 冷算。

组合目标为 P 上的 `MultiConnector[MooncakeHybridConnector(kv_producer),
CED-adapted OffloadingConnector(kv_both)]`，DRAM 后端复用 Ascend native
NPUOffloadingSpec 和已验证的 per-group chunk/registered host pool 修复。
该 CED 适配尚待实现，不能把两个既有组件各自通过当成组合已通过。

必须落实的语义：

1. 按 runtime group/spec 识别 P 有效组；保存和查找都排除未计算的上半层
   SWA G7–G11，保留原 group 下标。G1 circular state 不参与 prefix offload，
   由对齐后的尾部计算恢复。保存 G0 与 G2–G6 可供各有效命中边界恢复的内容。
2. 复用每组 chunk 和 APC alignment；对齐由 runtime spec 计算。不能仅保留
   全 prompt 的最后 SWA 窗口后宣称支持任意较短前缀命中。
3. 修复 P 的异步 DRAM 命中恢复，包括 N−1 截断只执行一次、加载完成后的
   computed boundary、full hit 的非零尾部，以及失败恢复。现有
   `core_scheduler_prefill_hit.patch` 只覆盖 `not load_kv_async` 的本地 full hit。
4. PD 消费和 DRAM 保存的完成通知共同决定页何时可以复用；DRAM 完整写入
   后才发布命中。验证 MultiConnector 多 child 完成汇总、SWA 页复用前 flush、
   abort/preemption 和有界 pending write 背压。
5. 打印实际有效组、对齐单位、注册结果、宿主 tensor nbytes/实占、各 rank
   的存取字节/耗时；配置开关必须可验证，不能静默退化。

继续保留 `num_blocks <= 29076` 的寻址上界，禁止扩大 HBM 池来绕过驱逐问题。
P 使用 `SPEC=0` 是 CED 的执行层约束；这里无真实 decode 投机实验。

## mock decode

mock 必须消费真实 P 的 KV 数据，不能只回复 HTTP 200 或只发送释放 ACK。
消费完成后，向对应 P worker 回传真实完成通知，覆盖延迟消费和失败不提前
释放的情形。记录 per-rank / per-group 字节、摘要与阶段耗时。

首先验证 Mooncake ascend transport 的 host buffer 接收。
【实测】无驱动挂载的 CPU-only v3 容器导入 `mooncake.engine` 时报
`libascend_hal.so` 缺失；这只证明依赖未满足，尚不能判断 host 接收是否可用。
补齐只读驱动库和管理设备接口后，仍需初始化 device context；用户已授权 mock
使用一张 device。随后使用与既有起服脚本一致的 privileged/device/firmware 挂载，
在锁保护下 Phy-ID 6 完成基础计算和 Mooncake 自进程 HBM→host 逐元素校验：
1024 B，SHA256 `8808405eec6fbe306fe3369f88daed79dd5613ddbb5e801f632b01d6218c5f08`。
【实测】这证明该 transport 可接收 host buffer；跨进程/八 rank 的实际 KV 消费仍待验收。

初版 `tools/ced_mock_decode.py` 使用单 device context、64 MiB 有界 host 接收缓冲，
消费八个 P rank 的真实 KV；全 rank 成功后才回传 DONE_RECVING。P 的可选
mock 几何元数据描述准确 payload 视图，避免把 padding/未计算 SWA 当成缓存正确性证据。

mock 用于 P 缓存恢复、数据一致性和交接生命周期验收。真实 D 的生成文本
正确性、128-token replay 数值正确性和端到端 TTFT 仍需后续 16-chip 实验。

## 执行顺序与完成判据

1. 离线验证配置、有效组、对齐边界和异步完成语义；运行修改入口所需自检。
2. 持有设备锁后验证 6–13 的设备计算及 TP8 通信。只操作本任务独立命名的
   容器，禁止重置设备、停止他人服务或处理 14–15 的他人预留锁。
3. 真实权重 P + mock，先短上下文小池验证冷、HBM hit、DRAM hit；强制
   清理/挤出 HBM 后须观测真实 H2D 字节、external hit 和尾部计算。
4. 覆盖 full/partial/append、1/127/128/129/1023/1024/1025 等边界，以及
   延迟消费、abort、重入；一致性至少三轮，任一轮不同即失败。
5. 至少四条独立 1Mi 前缀交错访问，证明目标从 HBM 驱逐、DRAM 保留、回访
   真正加载。分开记录 P 排队、DRAM load、P 实算、mock 消费和释放耗时。
6. 归档实际参数、日志、原始计数器和结论；标记【实测】/【推断】/【未确认】。
   硬件未能运行的项目保持未完成，不能仅凭离线通过宣称 goal complete。

性能判据不能依赖 PD 响应的 `cached_tokens`（代理恒为 0）、npu-smi 的
预分配池占用或单独的 `preemptions=0`。以服务端 cache hit、实际 DMA bytes/time、
P 实算 token 数和阶段耗时共同证明恢复路径。

跨机大文件传输使用 COS；实现、配置与实验产物使用各自独立目录。

## 初版离线检查

`tests/test_ced_dram.py` 八项通过，覆盖有效组、动态对齐、物理页去重背压、组合
配置、mock 读范围、保存屏障以及 P async resume。async resume 测试提取实际
镜像 scheduler 经 admission/P-hit/DRAM 三份补丁后的方法执行；save barrier
测试使用 native callback stub，尚不能替代实机 native worker 完成验证。
三份调度补丁已在镜像原始源码上依次应用并通过 Python 编译。
起服脚本语法、docker run 续行链及 DRAM dry-run 通过。

基线 model.py 的两份 MD5 清单已过期，本分支按既有载荷真实字节修正，未修改
模型载荷。包内完整 selfcheck 的 MANIFEST 尚需在提交代码后从 HEAD 更新。

后续增加实际 MultiConnector 方法的双 child 完成顺序验证，共九项通过。
【实测】`evidence/ced_dram_20261009/probe8.log`：锁保护下 Phy-ID 6–13 的
向量运算和 128×128 FP16 矩阵乘法 8/8 通过。
【实测】`evidence/ced_dram_20261009/hccl8.log`：TP8 HCCL all-reduce 三轮，
八 rank 全部得到期望值 36。仍须记录 Alarm 状态并验证完整模型运行。

镜像 native KV split helper 名称为 `_canonicalize_split_attention_cache`；
本地较新 upstream fork 已改名，初版 mock 导入烟测抓到了这个差异，现按实际
目标镜像 API 接入，不把参考代码的版本当作镜像事实。

## 实机推进记录

- 初版代码提交 `0e975b0`，MANIFEST 提交 `197a46a`；包完整 selfcheck 通过。
- 真实镜像导入 `CEDOffloadingConnector`、`CEDNPUOffloadingSpec` 和 mock geometry
  通过；`supports_hma(CEDOffloadingConnector)=True`。
- mock 使用额外 Phy-ID 4（启动前已核对空闲、无锁），容器
  `ced-dram-mock-20261009-1159`，HTTP `127.0.0.1:19191`。64 MiB host buffer
  注册成功；未加载模型或启动真实 vLLM。
- P 第一臂 `ced-dram-p-20261009-1159`，MAX_LEN=32768 / HBM 138477568 B，
  在 KV 最低容量校验阶段失败：最低约 0.78 GiB。没有执行测试请求，已自动清理。
- P 第二臂 `ced-dram-p-20261009-1210`，MAX_LEN=16384 / HBM 512 MiB，同一
  校验报告最低约 0.72 GiB，已退出并清理。该模型的校验有显著固定开销，不能
  按 max_len 线性缩小预算；这两臂不算 DRAM 存取失败。
- P 第三臂 `ced-dram-p-20261009-1219`，MAX_LEN=32768 / HBM 1 GiB / DRAM
  记账 2 GiB，真实权重正在启动。须以容器实际状态和新日志核验进展，禁止仅
  凭旧锁文件或旧 PID 重启。每个 P 实例都由前台 supervisor 保持设备锁至退出。
- 测试 runner `tools/ced_dram_bench.py` 已写入，逐请求保存原始 metrics、P 交接
  参数、八 rank mock 数据指纹及阶段耗时；冷/热/四前缀交错/三轮回访、部分/
  追加和短边界仍待实际运行。P response 和 mock consume 时长不标成真实 TTFT。

按仓库要求读取 cannbot `model-infer-kvcache` 的 §2.1–2.6 与实际 paged tensor
shape 说明：逻辑 token/block、物理 block ID 与物理 stride 必须分开；cache 的
实际 NZ/padded 形状不能从 input_layout 字面推断。本实现沿用目标镜像 native
canonical views，不修改 attention layout/FA 算子配置；mock 读取有效 payload，
未写满的末页和 circular state 消费但不进入 reusable-prefix 指纹。

### 2026-10-09 后续接入修复

- 第三臂已退出，报 `tokens_per_block=32 not divisible by tokens_per_hash=128`。
  这是 non-prefix-cacheable G1 被 offloading config 的通用 hash 校验误纳入；
  已让该校验同 save/lookup 一样排除 CED 无效组，参与组仍严格校验。
- 第四臂 `ced-dram-p-20261009-1225` 在 1 GiB HBM 下成功启动，GPU KV cache
  报告 41,733 tokens；已确认 `/v1/models` 为本实验模型。
- 第一个真实预热请求返回交接参数，并有八 worker 的 DRAM save barrier 完成
  日志，但尚未证明消费或 DRAM 读回成功。
- API 统计汇总随后报 `Connector CEDOffloadingConnector is not registered`。
  构造路径支持外部 module，统计路径仍按 registry 名称解析；已在 module 加入
  注册，实际镜像 `get_connector_class_by_name` 同类身份校验通过。
- mock 在第一次 `/consume` 退出 139。跨线程 device/TE 调用是候选原因，
  尚未确认；改成主线程 HTTPServer，并开启 faulthandler，待重新消费定位。
- 客户端输出改用 user-owned `client_results/`，避免 P supervisor 的 root-owned
  服务目录影响记录。bench 的 `external_hit_tokens` 按 `external_kv_transfer` 读取。
- 本地另补 abort-during-load 的完成屏障保护：request 结束时须等 pending load
  destination 与 store source 一起完成；目前十一项离线测试通过。
- 两端旧实例已确认退出/清理；第五臂 P `ced-dram-p-20261009-1250` 和 mock
  `ced-dram-mock-20261009-1250` 正在启动，真实 KV 消费、挤出读回、三轮一致性、
  边界和 1M 实验仍未完成。不得把 save barrier 日志当成 read-back 证据。
