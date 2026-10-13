# 2026-10-13 三形态交付验收

目标：A2 TP8 P/D 同置（在 A3 上模拟）、A3 8+8 CED-PD、A3 TP8+DCP8
复用 8 ranks，逐一接入 10-13 最优真实 DSpark 配方并验收，随后添加 CED
单侧 KV DRAM 卸载及跨 P/D Engram 文件映射共享。

## 统一口径

- 强制真实权重、Engram、D 侧真实 draft graph；device index 可以关闭。
- BI、自定义 OPP、async 和多流必须记录实际生效值。CED 的历史多流错误
  需要重新验证，不能把 TP8 的结果直接外推。
- 两类负载：固定地火 8 条串行请求；固定 8 个中级编程任务并执行独立测试。
- 冷 prefill 用 PREFIX=0，先暖机后正式测量。至少测 2K、32K、144K 地火；
  1M 和 Vision/GSM 为交付能力回归。编程题使用自身真实输入长度。
- TTFT 与 prompt_tokens/TTFT 是客户端端到端口径；服务端 prefill wall
  与纯设备执行时间分别标注，不能混用。
- decode 同时报 actual rounds、A、ms/step、tok/s、TPOT、输出质量。
  没有真实计数就留空。A3 模拟结果不得标作真实 A2 硬件测量。
- 精度：GSM100、Vision23；CED/DCP 长上下文针；编程任务全部独立测试通过。
  缺失/失败证据保留，不得写“达标”。

## 现场与依赖

- 16 die 已获用户授权；每次启动前仍检查宿主 npu-smi 与项目归属。
- 原 TP8 容器的 v12/v15 失败任务存在存活残留，已由现场代理保留证据后
  仅 stop/start 此项目容器；旧快照和日志 SHA 不变。
- 最优快照 `/work/src_dspark_general_quality_v11`，原工作区基线 cf16be3。
- 当前交付分支缺 DCP overlay；历史 main-merge/shunt 配方与 a3-21
  实际 overlay 需要对账。TP8+DCP8 使用 8 die，不能按 64 ranks 启动。
- 只由 site_runtime 操作服务资源，避免两个代理重复起服。

## 执行顺序

1. 接入 native general API 与实际 step 观测；TP8 A3 模拟实测/精度。
2. 移植 DCP8；相同工作负载与质量门。
3. CED-PD 接入最优配方；相同负载，额外长上下文与传输验收。
4. 更新 README：三形态配置、实际指标、证据与限制。
5. CED 单侧 DRAM offload、host-only 共享 Engram mmap、KV 池唯一存储。
6. CPU 正负控、真实双向卸载、容量/物理内存、精度回归及 README 更新。

本计划不是性能或精度达标收据。goal 只有全部必要项完成后才可完成。
