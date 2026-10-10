# 正式TP8叠加审计通过，当前最快26.623ms/step

2026-10-10 · a3-21 physical chip8–15 · 正式40层/W4A8/384专家top6权重

`HCCL_DETERMINISTIC=strict`下的base/core/stack整网图审计全部通过。随后关闭路由/logprob审计、cache探针和profiler，在同一组正式worker中完成12组交错配对。当前最快core为`(26.622860ms/step, A=1, 37.561704tok/s)`，仍未达到≤约19ms目标；不能把tiny的19ms结果当作正式模型验收。

## 精度与图bank生命周期

`formal_stack_graph_strict_audit_v1`正常退出0。候选创建前、core捕获后、stack捕获后的三次base A/A，完整路由、Top5集合和logprob全部一致/delta0；三组base/core/stack配对共6个比较也全部一致。47/48输出交替覆盖C2末步状态。

9个配对请求的72份rank消费者审计中，Indexer functional INT8/FP32-scale、实际INT8 K cache及FP16-scale cache全部逐位一致，max abs0。当前通过条件保留原始路由范围/专家唯一性、Top5集合与`logprob delta<1e-3`，未放宽。

这些数据说明本轮三个图bank在相同worker中切换及重新捕获后，原生对照保持稳定。结论限定当前测试的模型、2K输入、47/48输出与batch1；客户端质量和其他上下文仍待验收。

## 正式关闭审计的配对性能

`formal_stack_graph_strict_perf_v1`正常退出0，2K输入/48输出，max_num_seqs=1，无推测解码，A=1。全部pair计时均在同一实例/8worker中，按正反顺序交错，prefill和前9步不进入decode中位。

|方案|ms/step|A|tok/s|相对base配对加速中位|更快组数|
|---|---:|---:|---:|---:|---:|
|base|30.598535|1|32.681303|1|—|
|core|26.622860|1|37.561704|1.146861|12/12|
|stack（另加Indexer融合）|26.707870|1|37.442147|1.149406|12/12|

core/stack均有约15%的配对提升。整体中位core略低，但stack的相对base配对中位略高；两者差约0.085ms，Indexer增量没有显示稳定、明显的额外收益。按当前整体中位，候选服务选择core，仍需客户端验证。审计态的约30ms数值没有用来算这次收益，跨进程/跨会话差异没有累加。

## 实际启用的优化原理与边界

正式core保留兼容的metadata/多group slot/blockmap准备和Q/KV重叠，减少重复准备与串行等待。正式W4A8的GMM已经原生融合激活，BF16激活候选覆盖0；HC static/HcPost候选因正式数值门未通过而保留原生；TP1的8expert/top2 router/selected-GMM不匹配384/top6，禁用。

stack只额外融合Indexer K源层的RMSNorm、尾64维RoPE、INT8量化与cache写入，保留BF16舍入边界和原生量化顺序。独立/cache精度通过后，本次整网配对也通过，但实际端到端增量小，不能累加其他会话的0.3–0.5%宣称收益。

`strict`在新进程与通信域初始化前设置，消除了本轮原生图的重复差异。当前尚未完成本模型GSM8K/视觉质量验证，且可能有通信性能代价；不会单靠历史100/100或本轮A/A通过就完成交付。

## 下一步如何回收剩余7.6ms

已启动`formal_strict_profile_v1`，在正式base/core上重新采集7组×8rank，预计112份目录。它用于定位strict下的HCCL执行、SDMA/AIV展开、CPU调度和图间隙，再与本轮原生算子/MTE/cache热点对应。采集RPC影响时间线间隙，kernel累计允许重叠，不当成端到端时延或可回收时间。

采集状态更新：v1在W4A8通信域初始化时遇到`EI0020`，NPU adapter的16666端口已绑定；没有采集目录。按CANN官方文档，已在新作业`formal_strict_profile_v2`设置`HCCL_NPU_SOCKET_PORT_RANGE=auto`，由系统选择未占用端口，保持strict配置，不停止任何现有进程。该重试结果仍待完成，不能把v1计为采集成功。

没有UB容量occupancy或双缓冲收益证明；原有小M MatMul的MTE2高、GMM Scalar高仍须分开分析。固定顺序AllGather求和独立精度与图重放已通过，但尚未与正式整网strict对照，也没有端到端收益可计。下一轮依据strict新数据决定通信展开、固定序归约或真实W4A8小M/GMM优化的优先级。

## 客户端与交付准备

已准备受审计/性能证据约束的`serve_formal_tp8.py`，核验正式checkpoint、物理chip、同进程配对、strict环境、有效三元组及所选最低中位方案后才起API。客户端验收仍标为pending，当前未启动已验收的最佳服务。

GSM8K工具支持独立模型名；性能工具支持8个不同prompt的serial采样和模型名严格检查。错误服务负控已通过。a3-21的GSM8K cache、官方encoding与两张官方视觉图片存在；原Python缺少datasets，已创建本任务独立venv，依赖安装正在进行，不修改模型容器运行环境。

新报告及源代码将继续提交、自检和双远端push。Goal保持active；19ms、客户端质量、服务归属与最佳正式部署均尚未完成。

## 紧凑证据

- `evidence/formal_a321/results/formal_stack_graph_strict_audit_v1/`：原生控制、6个配对、72份消费者rank审计、bank覆盖、退出码。
- `evidence/formal_a321/results/formal_stack_graph_strict_perf_v1/`：12组关闭审计配对、正式三元组、启动环境与退出码。
- [strict原生图通过报告v10](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-strict-native-graph-pass-20261010-v10.html)。
