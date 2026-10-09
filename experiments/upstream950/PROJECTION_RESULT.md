# Q/KV、q_b 面板与预取：当前候选均不采纳（A3 tiny）

完成联合Q_a/KV投影、q_b真实NZ预排、连续消费顺序面板、显式下一面板预取四个切口，保留原生decode/prefill。联合投影未通过实际路由一致性门槛，其余正确但没有端到端收益。

## 实现与正确性

联合投影在加载后一次将两份 `[512,5120]` BF16权重组成 `[1024,5120]`，保留输出BF16、原生Norm/RoPE。q_b将 `[32768,512]` 一次预排为 `[256,4,128,128]` `(N tile,K tile,K,N)`，使用M16×N128×K128、FP32累加后BF16输出。理论两份A/B L1面板预算72KiB，L0C8KiB，符合当前DAV_2201资源约束。

显式预取版本在当前dot之前加载下一份A/B，K累加顺序不变。动态loop携带CBUF tensor首先触发BiShengIR的ND/NZ iter_arg类型不匹配，三个交接静态展开后编译成功。编译产物为真实ELF；保存num_stages2、multibuffer及enable_preload配置和各IR/binary的SHA。已实现加载顺序，不只以num_stages参数宣称预取；没有声称得到950的四Buffer具体片上布局。

NZ预排只在生成格式29时临时开启本进程internal-format选项，并通过C API读回原值、finally恢复。后继RoPE需要ND，转换如有必要计入候选实际路径。权重面板在profiling/capture前生成；有版本计数时按storage/version失效，inference权重需显式invalidate。

16个随机用例（T1/T4×8seed）：q_b NZ、panel、prefetch均逐位一致。联合投影出现少量BF16末位变化，最大绝对差QA0.00390625、KV0.000244140625，最大相对RMS误差8.63e-5。单独的BF16数值门槛通过仍不代表模型可采纳。

40层随机模型审计中，joint在pair0改变5条有序路由记录，其中2处专家集合不同；10个Top5 logprob键集合也不同。输出token一致，共同logprob项最大差9.54e-7，QA/KV局部5个元素不同、max_abs1.91e-6。按实际路由一致性要求拒绝，未为此放宽门槛或继续正式性能评估。

NZ/panel三bank与panel/prefetch三bank审计分别通过：全部40个投影调用，QA/KV、q_b、后继Norm/RoPE逐位一致，实际路由与logprobs一致、delta0。正式图无消费者点复制或profiler。

## 同进程性能

固定已有HC/router both、BF16/TP1 tiny、2K prompt/48输出、A=1、FULL_DECODE_ONLY，每轮各12组交错，GPU独占锁。两轮57/56个chip4全程采样均0。

|轮次|路径|`ms/step`|A|tok/s|对本轮原生配对中位|更快组数|
|---|---|---:|---:|---:|---:|---:|
|1|原生|26.454195|1|37.801188|1|—|
|1|NZ|26.759165|1|37.370374|0.989197|2/12|
|1|panel|26.597360|1|37.597717|0.995415|4/12|
|2|原生|26.168845|1|38.213379|1|—|
|2|panel|26.228675|1|38.126211|0.997238|2/12|
|2|prefetch|26.245545|1|38.101705|0.995874|2/12|

NZ/panel/prefetch分别增加配对时延中位0.289065/0.121845（轮1）和panel/prefetch 0.072570/0.108420ms（轮2）。结论依据各组配对，不把两轮整体漂移当收益。

单算子范围包括QA/KV、q_norm、q_b及ND交接：轮1原生30.501861us，NZ30.542521us、panel39.354120us；轮2原生30.386061us，panel40.469160us、prefetch40.169300us。预取相对panel略降，但仍慢于原生，不能只报局部更快而采纳整个候选。

代码：`scripts/projection_panel.py`、`projection_patches.py`。证据在 `evidence/final/projection_precision_v5.json`、`projection_joint_rejection.json`、`projection_model_audit_v2/v3/v4`、`projection_model_perf_v1/v2`、对应micro/card_load和`projection_panel_compiler.json`。当前结论限定BF16/TP1 tiny，不外推到950的不同rank、量化或完整融合结构。
