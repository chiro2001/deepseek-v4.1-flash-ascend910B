# 第二轮：HcPre 和窄 router 已接入 tiny，端到端吞吐提升约 12%

在 a3-21 **chip4** 的同一模型进程中，保留 native/router/hc/both 四套独立 decode 图，交错六组请求，关闭 profiler、logprobs 和路由回传计时。组合方案的 decode 中位从 **29.254 降至 26.122 ms/step**，A=1，吞吐从 **34.183 升至 38.282 token/s**，约 **+12.0% 吞吐 / −10.7% 时延**。六组组合臂均优于同组 native。

优化后的独立服务保留在 `dsv41-tiny-prof-20261009-c4`，当前主机内接口为 **`http://172.17.0.4:18971`**，模型名 `dsv41-tiny-prof-20261009`。重启后 Docker bridge 地址发生变化，访问前以 docker inspect 为准；本地可用 `ssh -L 18971:172.17.0.4:18971 a3-21`。全程只运行于 chip4，排除 chip14–15。

## 改动与适用范围

1. **HcPre 单 token SIMD 路径**：用 Vector 完成投影/RMS，再执行全部 **20 次 Sinkhorn** 和输出组合；绕开原来 Cube↔Vector 握手、全核汇合及两阶段 TPipe 重新初始化。保留 eps、HF32 转换、pre_mix 和四个输出的数学定义。
2. **窄 FP32 router**：仅 `[1,5120]×[8,5120]` 选择 SIMD GEMV。prefill 和其他形状使用 native；N=128 的负对照无收益，未采用。
3. **进程内、可回退接入**：自定义 `TinyPerfWorker` 在正常模型加载后安装 HC hook，避免提前导入绕过平台补丁；使用不透明 custom op 保持动态 batch 分派。编译缓存关闭，捕获覆盖不足则拒绝运行。

代码：[hc_vector.py](scripts/hc_vector.py)、[runtime_patches.py](scripts/runtime_patches.py)、[tiny_perf_worker.py](scripts/tiny_perf_worker.py)、[serve_tiny.sh](scripts/serve_tiny.sh)。未修改模型文件和其他租户服务。

本轮仍是 TP1 tiny dummy 模型。Hc 仅选择单 token、4×5120、24×20480 的验证形状，其他 batch 使用 native。结果不能直接外推到 TP8 W4A8、DSpark 多 token 或真实生产权重吞吐。

## 同进程端到端结果

每次请求 2048 输入 token、48 输出 token，seed=0、prefix cache 关闭、FULL_DECODE_ONLY；先预热两轮，每组正反交错四个图 bank。使用同一模型、同一权重、相同 KV 配置和 CPU 绑定；图切换同时恢复 attention 的事件、workspace 和 task handles。

|模式|decode 中位 ms/step|token/s|配对加速比中位|
|---|---:|---:|---:|
|native|29.254|34.183|1.000|
|router|28.191|35.472|1.042|
|hc|27.602|36.229|1.065|
|both|26.122|38.282|1.119|

组合臂六组配对加速比分别为 1.131、1.100、1.118、1.121、1.120、1.106。所有请求的 48 个输出 token 一致。[性能结果](results/optimization/performance.json) 保存每组数值。

这是新一轮匹配条件下的 native 基线，不能拿第一轮 28.862 ms 的跨会话数字直接相减。profiler 在计时阶段关闭；后续 profiling 单独进行，不进入上述结果。

## 数值与真实路由验证

- HcPre 扩大到 **24 个**零输入、有符号、0.001/1/100 尺度、有/无 pre_mix 用例。四个输出全部通过原有精度门槛，没有放宽容差。[压力验证](results/optimization/hc_stress.json)。
- 单模型验证另将 **40 个 gate、80 个 HC fn/scale/base** 在编译前用固定 seed 赋予非恒定权重；检查真实图中的 gate 标准差约 0.014、HC fn 约 0.0007，避免 dummy 恒定值和编译冻结常量造成空洞验证。
- 四个图 bank 均捕获到 **40 次 router、80 次 HC**。每个请求实际记录 `[2095,40,2]` expert 数据，共 **83800 个 top2 路由对**；ID 在合法范围内，top2 两项不同，出现多个 expert。
- 三组正反交错验证中，router/hc/both 与 native 的**实际路由、48 个输出 token、top-5 logprobs 全部一致**。实际图 router 输出也与重算值一致；HC 辅助输出误差约 10⁻⁷。[模型验证结果](results/optimization/model_audit.json)。完整请求、路由和逐步时延保留在远程 `results/model_combined_audit_nondegenerate/requests.json`。

这验证了独立 tiny 和覆盖的数值用例，不等于真实 checkpoint 的完整质量评测。

## HF32 校准：拒绝错误参考后再接入

首次 SIMD 原型沿用参考脚本的“10 位尾数直接截断”，扩大到有符号输入后仅 **3/24** 用例通过，未采纳。[失败记录](results/optimization/hc_stress_rejected.json) 和旧源码 `scripts/rejected/` 保留。

使用单项输入、精确尾数边界值和 native 的 pre/post 输出反推转换，发现当前 **Ascend910_9382 + CANN 9.1** 的实际行为为 **11 位显式尾数、半值远离零舍入**。实现为 `(bits + 2048) & -4096`；不是 NVIDIA TF32 风格的 10 位截断。校准后 24/24 通过，native 辅助输出差异由约 10⁻⁴ 降到约 10⁻⁷。[位级实验](results/optimization/hf32_bits.json)。

该结论限定于本机当前 native 路径，不能仅由旧 bool API 的注释推断所有 910 型号的位级语义。官方资料亦区分 nearest 舍入及平局规则：[HF32 round mode](https://asc.gitcode.com/api/SIMD-API/c_api/cube_compute/asc_set_hf32_round_mode.html)。

## 匹配 profiling：收益来自哪里

native/both 在同一模型中各采连续 20 个 decode step，全部 trace 可解析、Device_id=4。以下按每步计算，采集耗时只用于结构归因。

|指标|native|both|
|---|---:|---:|
|kernel 数/步|2198|2278|
|原生 HcPre 次数/步|80|0|
|project_hf32 / finish_hc 次数/步|0 / 0|80 / 80|
|SIMD router 次数/步|0|40|
|HC kernel 合计 μs/步|2830.04|1257.18|
|优化 router kernel μs/步|—|143.27|
|全部 kernel 合计 μs/步|21033.54|18074.87|

全部替换 kernel 均为 **AI_VECTOR_CORE**，新 HC 路径中没有 Cube→Vector 标志9握手。虽然拆成两个 Vector kernel 增加了 80 个 task，消除内部握手和控制开销仍带来净收益。不能把 kernel 数量本身当成性能目标。

[验收记录及 CSV SHA256](results/optimization/profile_validation.json)。原始两份完整 trace 位于远程 `results/model_perf_final/prof/`。

## 双缓冲实验与指标边界

对新投影的五个 K tile，只改变自动 multibuffer 编译开关，按 OFF/ON/OFF/ON 交错，在芯片无其他任务时重新测量。中位分别为 **14.348 / 14.449 / 14.405 / 14.446 μs**，没有可测的额外收益。[结果](results/optimization/multibuffer.json)。本次主收益是改变计算分工和去除握手，追加 buffer 不是有效主方向。

原生 HC 已有 UB/L1/L0 双缓冲，第一轮逐核计数证明 wait_id9 约 9.8 μs；新路径使用 Vector，直接去除该跨单元依赖。新 `project_hf32` 的 msprof op Default 成功采集 24 个 Vector 核，频率 1800 MHz，[逐核 CSV](results/optimization/project_default/PipeUtilization.csv)。该工具插桩下的耗时不与未采集基线混比。

再次尝试新 Vector kernel 的 TimelineDetail，工具仍明确报告采集失败。没有把退出码 0、只有 OpBasicInfo 的结果当作指令时间线。UB 容量占用率和原先异常的 AIC MTE2 active bandwidth 仍不用于结论。

## 复现与回退

- 同进程性能与匹配 profiling：`python /work/scripts/bench_router_model.py --arms=native,router,hc,both --pairs=6 --profile --output=/work/results/NEW_ROUND`。
- 非恒定权重验证：另加 `--audit-routes --randomize-validation`；审计耗时不混入正式计时。
- 服务启动：`TINY_PERF_ARM=both bash /work/scripts/serve_tiny.sh`。native/router/hc/both 均可选择，重新启动本次自己的服务进程后生效。
- worker 启动后必须显示 `TINY_PERF_EFFECTIVE` 的 `arm=both, router_calls=40, hc_calls=80, visible_devices=4`；不满足则拒绝运行。
- 重采前先停止本次自己的服务，避免同芯片并行工作污染计时。所有原始数据、失败记录、源码快照和校验保留于 `/home/l00886679/projects/dsv41-tiny-prof-20261009`。

优化服务的 health、模型归属和生成冒烟记录见 [service_smoke_optimized.json](results/optimization/service_smoke_optimized.json)。
