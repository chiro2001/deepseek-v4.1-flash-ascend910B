# 正式原生 DSpark 时间线诊断

只在真实 native ON A/A 已通过后，采集完整40层/TP8/384top6/W4A8/Engram/视觉/MTP3的CPU/NPU时间线。它不放行候选、不授予消费者/质量/20ms通过，也不产生正式性能报价。

新快照必须逐个匹配A/A的九项源码SHA和checkpoint配置；同样K5、target FULL[6]、draft eager、HCCL strict。原生A/A只有2K/47审计请求，本诊断使用2K/256、两warmup后单个真实请求，route/logprob审计关闭，profiler明确开启。不能将采集态数字与关闭profiler的正式客户端相减。

标准LLM worker链初始化后才导入profiling worker，保留原Factory/权重和counter。按照CPU scheduler的实际computed/prompt边界，在第一个真实decode开始profile：5个实际decode warmup，10个active窗口，8rank分别采集PipeUtilization、CPU/NPU与shapes。capture/profile启动及prefill不计窗口；零token清理不计步骤。worker记录实际core ID和15步schedule，设备CSV完整HcPre/attention/算子步数仍需另核验，不能用schedule收据替代设备证据。

计数器RPC和profilerstop都不生成报价；raw counters及大trace留远端并记SHA。profile输出只用于区分target、MTP投影/attention/MoE/logits、主机提交和等待实际区间，不能直接当可回收端到端预算。

```bash
python3 experiments/operator_stack/parallel_tiny/dspark_native_profile/test_phase.py
```

profile launcher另需`--native-profile-aa-job formal_dspark_native_eager_aa_v3 --draft-graph 0`，主线程每次fresh资源/健康门、只允许已授权80C98001，不停止其他租户或迁芯片重试。
