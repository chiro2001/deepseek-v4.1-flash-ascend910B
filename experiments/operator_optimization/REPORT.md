# 激活融合与 Q/KV 重叠：独立 tiny 算子优化

2026-10-09 · `feat/tiny-operator-opt-20261009` · 基于 `origin/main@998c47c`

【实测】a3-21 chip4 同进程六组对照中，routed/shared激活融合与Q/KV多流组合，
将HC/router基线从 **26.456降到24.446 ms/步，A=1**，吞吐从 **37.799升到40.907 token/s**。
本轮吞吐提升 **8.2%**、时延下降约 **7.6%**，六组全部更快。
这是新一轮匹配结果，不与之前跨进程的12%直接相加。

## 改动与范围

- `clamped_swiglu.py`将routed的两个视图clamp及SwiGLU合并，保留输入原地修改；
  shared路径保留实际limit/alpha/beta和必要的BF16中间cast。
- `activation_patches.py`、`operator_worker.py`在正常加载后安装custom op和hook，
  保留独立图bank及attention事件、workspace、task handles。
- `overlap_patches.py`切换框架已有的Q/KV前处理：串行Cube投影，让Q norm与KV matmul、
  Q_b与KV norm/RoPE/cache store重叠，保留阶段事件与最终join。

新激活仅选择NPU BF16、物理ND且连续的routed `[2,512]`、shared `[1,512]`；其他形状保留原路径。
旧HC/router源码快照和结果在 `baseline/`、`evidence/baseline/`。

本轮是TP1 tiny dummy BF16、40层、8专家/top2、intermediate=256，Engram/DSpark关闭。
未验证生产TP8/W4A8、真实checkpoint或长上下文CED-PD；特别不能把本次多流设置
直接移到仓库明确要求关闭多流的CED-PD部署。

## 性能：关闭profiler与审计，正反交错六组

2048输入、48输出token；seed轮换0/1/2；prefix cache关闭、KV=4 GiB、MAX_SEQS=1，
FULL_DECODE_ONLY `[1]`，CPU绑定相同。每臂预热两轮。
计时关闭profiler、logprobs及路由回传；profiling在计时之后单独采集。

第一轮四种激活模式的同进程对照：

|模式|ms/步|A|token/s|配对加速比中位|
|---|---:|---:|---:|---:|
|HC/router基线|25.635|1|39.009|1.000|
|仅routed融合|24.476|1|40.856|1.047|
|仅shared融合|25.242|1|39.617|1.016|
|两种融合|24.058|1|41.566|1.065|

三种候选均为六组全部更快。shared在独立算子图中没有明确收益，但完整模型有稳定改善。
结果在 `evidence/model_performance/result.json`。

最终另在初始化多流投影wrapper的同一进程中比较三种模式：

|模式|ms/步|A|token/s|配对加速比中位|
|---|---:|---:|---:|---:|
|HC/router基线，激活原生、Q/KV单流|26.456|1|37.799|1.000|
|激活融合、Q/KV单流|25.021|1|39.966|1.058|
|激活融合、Q/KV多流|24.446|1|40.907|1.081|

多流也在六组中全部快于同组融合单流，两个中位时延比值约1.0235。
四模式的24.058与这里24.446属于不同进程，不能比较来判断多流优劣。
全部性能请求的48个输出token一致。结果在 `evidence/combined_performance/`。

## 数值、真实路由与图覆盖

【实测】139个独立精度用例全部通过，覆盖有符号、尺度0.001/1/100、limit=1/7/7.9、
alpha/beta组合、非连续视图、clamp附近值、空batch和NaN/Inf。
shared逐位一致，routed最大1 BF16 ULP，特殊值分类及输入clamp比较通过。
测试前固定全元素BF16混合容差及最大1 ULP要求，未按结果放宽阈值；另记录CPU FP64误差。
详见 `evidence/activation/`。

【实测】模型审计在编译前随机化40个gate、80个HC参数组，以及80个routed和80个shared MLP权重。
四种激活模式、fused/overlap两种模式各完成三组验证：实际图激活输入标准差非零，
目标形状输出与native逐位一致。每套图覆盖router=40、HC=80、两种激活各40次；
候选选择数与模式对应，不足则拒绝运行。

每请求实际路由为 `[2095,40,2]`，83800个top2对，ID合法且不退化为全零。
实际路由、48个输出token、top-5 logprobs全部一致，最大logprob差为0。
摘要在 `evidence/model_audit/`、`evidence/overlap_audit/`，完整请求留在远程。

## 匹配profiling与微架构

### task链确实被替换

每份窗口连续20步，CSV/trace/统计文件、Device_id=4、调用数和SHA256均验收。

|激活对照，每步|基线|融合|
|---|---:|---:|
|全部task数|2278|1798|
|Slice / ViewCopy / SwiGlu|80 / 80 / 40|0 / 0 / 0|
|新clamped_swiglu调用数|0|80|
|新routed / shared累计 μs|—|85.606 / 84.803|
|全部kernel时间合计 ms|18.097|16.353|

新kernel均为AI_VECTOR_CORE，routed/shared的Block Num为2/1，每步各40次。
两条激活链约1.7 ms的采集态成本降为约0.170 ms；性能收益仍以无profiler墙钟为准。

### 多流存在真实task区间重叠

最终baseline/overlap为45560/35960条、各20步，HC/router覆盖正确。
aux stream每步40次KV MatMul、40次RMSNorm、40次RoPE、40次Scatter，共160个task。
主辅流执行区间交集平均 **780.644 μs/步**，见
`evidence/combined_performance/overlap_intervals.json`。
这是task区间重叠，不能解释为指令级MTE/Vector重叠率。
最终组合采集态kernel合计16.695 ms/步；重叠、争用与固定开销须结合墙钟评估。

### 新激活逐核Default与MemoryDetail

真实routed `[2,512]` 两组补采成功，各有OpBasicInfo和2个Vector核的非空流水/内存表，
Device Id=4，频率1800 MHz。

Default task为2.900 μs；block0执行2.273 μs，Vector 0.291、MTE2 0.307、MTE3 0.397、
Scalar 0.509、wait_ib 1.103 μs。各计数可重叠，不能相加。
MemoryDetail task为2.780 μs；block0 GM→UB为1 KiB，UB→GM为1.5 KiB，
对应读取两个半行、写回clamp输入和输出。工具逐核路径带宽利用率约0.215%/0.381%，
不是整芯片HBM速率。CSV和日志在 `evidence/activation_microarch/`。

【推断】单tile、很小的传输量和启动/指令等待成本，使追加UB双缓冲优先级较低。
没有有效TimelineDetail或容量Occupancy，不能量化逐指令critical path和UB容量占用。
独立图事件时间与上板task计数器时间口径不同，不混比。

## 失败及未采用方案

- 【实测】flat v1通过精度，但routed独立图224.365 μs，远慢于native 25.013 μs，已拒绝。
  v2调整行/tile划分，BLOCK 1024→256、目标grid 1→2；独立图26.771→14.112 μs，约1.897×。
  `rejected/`保留源码与原因。【推断】慢路径与非连续写回的编译降低有关，尚无指令trace证明。
- 【实测】8组wo_a比较TransposeBatchMatMul与普通BMM：hot为20.394→19.963 μs；
  8份权重旋转、工作集268.4 MB时为41.007→40.821 μs，仅约0.455%改善。
  按40次估计仅约7.4 μs/步，尚未作模型收益验证，未采用；工作集超过L2也不等于实测100% miss。
  当前8组不能进入现有单组 `V41_O_PROJ_2D` 开关，下一轮需研究分组GEMV/tiling。
- CANN初始化后的PYTHONPATH必须追加保留，否则会丢失ACL Python模块。
  统一入口 `run_in_container.sh` 已保留该路径。

## 复现与回退

`sync_to_a3.py`只传小于1 MB的源文件包。核对chip4归属、停止本次自己的服务后：

```bash
bash /work/operator_opt/run_in_container.sh verify_activation.py --output=/work/operator_opt/results/precision_NEW.json
bash /work/operator_opt/run_in_container.sh bench_operator_model.py --pairs=3 --audit --output=/work/operator_opt/results/audit_NEW
bash /work/operator_opt/run_in_container.sh bench_operator_model.py --pairs=3 --audit --test-overlap --output=/work/operator_opt/results/overlap_audit_NEW
bash /work/operator_opt/run_in_container.sh bench_operator_model.py --pairs=6 --test-overlap --arms=baseline,fused,overlap --profile --output=/work/operator_opt/results/combined_NEW
bash /work/operator_opt/run_in_container.sh summarize_model_profile.py --root=/work/operator_opt/results/combined_NEW --arms=baseline,overlap
```

profiling目录的安全权限可能使宿主普通用户忽略其内容，分析应在容器内执行。
服务入口 `serve_operator_tiny.sh` 默认 `OPT_ACT_ARM=overlap`，可选
baseline/routed/shared/fused/overlap；重启本次服务后生效。
baseline保留HC/router并关闭新激活和多流；完全原生回退仍可用既有
`/work/scripts/serve_tiny.sh` 的native模式。
服务验收与源码完整性记录在 `evidence/service/`、`evidence/source_manifest.json`。
【实测】最终服务health=200、模型归属正确，32输入token成功生成16输出token；
启动日志确认overlap=40层、HC=80、router=40，两种新激活各选择40次。
当前主机内地址为 `http://172.17.0.4:18971`，重启容器后需重新核验bridge IP。

远程根目录为 `a3-21:/home/l00886679/projects/dsv41-tiny-prof-20261009/operator_opt`。
原始请求/trace分别在 `results/model_audit_v2`、`model_perf_v1`、`overlap_audit_v1`、
`combined_perf_v1`、`activation_microarch`。仓库保存小型结果及逐核CSV，
两份 `profile_validation.json`包含完整CSV哈希、设备、类型计数和流信息。

## 20 ms goal后续：HC finish/norm筛选与语义核对（2026-10-10）

新增hc_prenorm.py把finish和全5120维RMS放入同一Vector kernel，保留20次Sinkhorn和HC的BF16中间输出。
固定的FP32路由1e-4门槛下，候选未通过，暂未接入模型；BF16路径的独立图改善约1%，8192布局更慢。
v1编译选项错误已单独保留，v2/v3逐步校正原生边界，没有放宽容差。

源码及上板检查证明当前vendor的RmsNormCast将最终BF16输出扩展到FP32，检查max差0。
因此旧机会清单中“不能从BF16输出float替代独立未舍入FP32”的警告不描述当前实现；融合仍必须保持真实舍入和归约顺序。
相关证据在evidence/goal20/rms_norm_cast_semantics.json，适用范围限当前vendor/BF16。

chip6完整模型尚无有效计时：static尝试在运行时流同步等待，nostatic尝试报DSA随机任务分配80字节失败；
可选GOAL20_SERIAL_DUMMY=1保持NPU RNG/seed不变，只在初始化后同步，12451个参数完成加载，但图warmup后仍同步等待。
未重置设备、未操作其他租户进程。此前22.674 ms/step、A=1、44.103 token/s仍为阶段最佳，goal保持active。
