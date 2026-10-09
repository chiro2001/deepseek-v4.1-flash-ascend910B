# 20 ms/step 持续优化状态

目标：语义与实际路由验证通过，无profiler/审计decode ≤20 ms/step，A=1，≥50 token/s。
本goal仍为active。独立分支 `feat/tiny-operator-opt-20261009`，只使用chip4，排除14–15。

## 已完成

- 核验容器label、模型归属、进程和芯片；仅停止自己的API，容器保持运行。
- 当前激活融合+Q/KV多流重测：24.689 ms/step，A=1，40.503 token/s，六组都优于同组融合单流。
- 连续20步profiling验收并更新实际shape热点：1798 tasks/step。
- HC静态HF32预处理和K/output tile共9配置；默认布局独立图约1.056×。
- wo_a六种Vector布局/tile均更慢；修复Cube广播、显式BF16输出后完成五种Cube配置，尚无收益。
- GMM1、shared1出现Vector候选；GMM2/shared2/q_b/wo_b已有负结果，继续记录支持的配置。
- 路由初始化/汇合覆盖全部64 expert对；HcPost八种block/FMA配置已验证。
- 第一轮模型覆盖守卫拒绝HcPost未捕获；修正把batch分派移入opaque op后，第二轮模型审计通过。
- `goal20_model_audit_v2`：七种模式，每种两次warmup及一次对照，非恒定gate/HC/MLP权重；
  路由、48输出token和top-5 logprobs一致。该会话的combo包含慢GMM2，对应当前代码的all_candidates；
  正式计时的combo已移除GMM2。审计计时不作为正式性能。

## 进行中

- `goal20_model_perf_v1`已验收：基线24.270 ms/step，A=1，41.203 token/s；combo23.541 ms/step，A=1，42.479 token/s，六组全快。
  单独hcstatic/hcpost/gmm1未六组全快，route和combo六组全快；下一轮需要更紧邻配对/消融区分小收益。
- `goal20_combo_audit_v1`三组非恒定权重审计全部通过；实际路由和输出token一致，最大logprob差9.537e-7。
- 20步匹配profiling结构验收：1798→1758 task/step，kernel累计16.853→16.284 ms/step；不当作端到端可回收时间。
- `goal20_static_perf_v1`：同进程基线23.461 ms/step，A=1，42.624 token/s；combo22.674 ms/step，A=1，44.103 token/s，六组全快。
  日志确认LOCAL_WORLD_SIZE=1与static shape kernel路径；不跨进程推算开关收益。
- `goal20_static_audit_v1`三组非恒定权重审计全部通过，最大logprob差9.537e-7。
- SWA单kernel Cube十二配置均遇BiSheng代码生成崩溃，未接入。
- 注意力独立复现必须先bootstrap vendor再初始化设备；系统ABI/HCA模板失败均无效。
- C128/C2的Default及MemoryDetail已成功采集，24Cube/48Vector、Device4、1800MHz，逐核CSV及summary已取回。
  只有1个Cube/2个Vector有实质计算，CSA源码mBaseSize=groupSize；C2 Cube wait_id6最大10.807us，对应V0→BMM1。
- 源同步已结束；GMM2 Cube五个支持的多K-loop配置均更慢（较优13.386us，native11.846us），继续保留原生。
- 当前NPU任务 `goal20_headsplit_c128_v1`：静态sink重复模式下，将head分组映射成独立batch，保留完整KV与indices，单独筛选2/4/8/16组；metadata准备成本尚未计入。
- 阶段报告已发布COS并登记links-server，HTML/Markdown及SHA256均上传，内容明确goal仍active。
  https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a3-21-tiny-goal20-operator-milestone-20261010-v1.html
- 候选源码在同目录，原始日志与trace在远程 `/work/operator_opt/results/goal20_*`。
- 已有优化服务暂时停止，以避免污染本轮独立测量；任务结束或需要长期交付时恢复经过验证的服务。

## 待继续

- 验收六组结果、组合三组非恒定审计、匹配profiling和新kernel逐核Default/MemoryDetail。
- GMM/wo_a的Cube与Vector进一步tiling、shared1实际布局筛选和融合epilogue。
- SparseFlashMla/Indexer逐核等待与控制路径诊断；metadata与邻接norm/RoPE/cache store融合。
- static kernel配置筛选与有效性检查，保留同口径证据，禁止跨进程计算百分比。
- 更新工作/原理报告并上传COS及links-server，必要自检和提交；goal达到目标后才标complete。
