# 原生 Q_a/KV 合并：独立精度通过，正式 TP8 审计进行中

2026-10-11 · a3-21 · 当前完整正式客户端目标 ≤17ms/step

当前已验收的正式客户端仍为 **(19.289956ms/step,A=1,51.840451tok/s)**，GSM8K100/100、Vision23/23。新候选尚无正式整网或客户端收益，不宣称达到17ms。

## 独立候选

正式层0/20的复制 Q_a `[5120,1280]` 和 KV `[5120,512]` INT8 权重按输出维原样拼接为 `[5120,1792]`，加载为原生NZ/format29；BF16 per-channel scale按相同顺序拼接。仍调用原生 npu_quant_matmul，保持原INT32累加和原生缩放，未复用已否定的自写INT8 GEMV。

v1在直接导入DSA模块时遇到上游循环import，退出1，未进入数值比较。v2改用AST读取源码契约并按已有独立Indexer入口注册原生custom ops，退出0。两层×三量级×四输入形态共24组，Q_a、KV、下游quantized Q、token scale、rank0 Q_b和KV norm六输出均逐位相等，BF16最大ULP0；原1ULP门未放宽。

|范围|原生µs|合并µs|配对加速中位|更快配对|
|---|---:|---:|---:|---:|
|两路投影及共享动态量化|27.631289|18.488428|1.494786|16/16|
|有限多流prolog|43.712199|35.484796|1.231740|16/16|

每图128个独立输入缓冲，计时8次重放，16组交错顺序；所有设备/提交比≥4.795（门限2），profiler关闭。有限prolog不含RoPE、SWA cache、Q头归一化或compressor，不能替代完整模型。每次约8.23µs仅为后续选择依据，不能直接乘40后当作已获得的端到端收益。

## 正式集成与守卫

`tp8qkv`继承当前mask方案，加载期为40层真实参数打包并检查NZ回读、dtype/shape/原权重version。仅M1 decode选择新算子，prefill回退原生。精准替换原多流函数中的两次投影，原事件、norm、RoPE、cache、compressor尾部保持原序。

每rank捕获40个唯一层的实际INT8输入、token scale和投影输出；审计要求Q_a与KV两个消费者都逐位等于原生。在此之外仍独立检查完整路由/token/Top5/logprob、Indexer/cache和mask消费者，不能用局部通过替代。

不可变快照 `/work/src_qkv_model_v2` 的 `formal_qkv_merge_audit_v1` 已在复查空闲/授权Alarm后启动，使用chip8–15、strict、async scheduling，三组base/mask/qkv bank。只有原生A/A、三组完整比较和8rank消费者全过，`continue_qkv_perf.py`守卫才会启动12组关闭审计/profiler的同实例配对。

## 并行与子代理传递修正

用户追加授权所有空闲芯片、同时正式TP8和多个tiny，并指定tiny优化子代理gpt-6.1-sol/max。旧三个代理实际deepseek-flash/max，至少两个确认任务与followup被传成Fernet密文。旧代理已停止；新tiny_qkv、tiny_engram、tiny_ids全部实际模型/effort核验为gpt-6.1-sol/max，明文任务文件与三条准确ACK均已核验。

芯片分工：chip2 QKV tiny、chip3真实Engram投影重放、chip4索引hoist tiny、chip8–15正式TP8。每次启动重新检查实时占用和健康；0/1/5其他租户不操作。tiny各自记录模型/层数/TP/量化差异，只作为筛选诊断，不汇总成正式TP8收益。当前tiny仍在独立环境准备，尚无新tiny性能结论。
