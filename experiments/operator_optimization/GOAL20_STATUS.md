# 20 ms/step 持续优化状态

目标：语义与实际路由验证通过，无profiler/审计decode ≤20 ms/step，A=1，≥50 token/s。
本goal仍为active。独立分支 `feat/tiny-operator-opt-20261009`，初期chip4，后因别租户占用迁移chip6，始终排除14–15。

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
- head分组C128初筛约8%改善、C2明显慢；但发现chip4别租户进程占用，两个晚期计时均quarantine，不能据此采纳。精度结果仍为逐位一致。
- 阶段报告已发布COS并登记links-server，HTML/Markdown及SHA256均上传，内容明确goal仍active。
  https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a3-21-tiny-goal20-operator-milestone-20261010-v1.html
- 已提交推送7220c70及manifest49a576c；仓库自检全部通过。
- chip4后有别租户`m00933363_catlass_runtime_bz_a3_1`进程，约29.8GB/100% AICore；未操作其进程。
  自己的goal20服务因空闲显存小于GPU_UTIL要求退出，当前chip4容器仅sleep，无API。
- 空闲chip2/3初始化TSD/OPP失败；chip6计算冒烟通过。
- 新独立容器：`dsv41-tiny-goal20-20261010-c6`，label task=dsv41-tiny-goal20-20261010, chip=6。
  镜像及model/work挂载与c4一致，物理6→逻辑0，容器privileged，其他租户保持运行。
- c6的`goal20_c6_rebaseline_v1`无结果：两次native栈采样证实在_fx_func_run流同步等待，输出目录空；SIGTERM无效后仅停止自己的PID184。
- c6 `goal20_c6_nostatic_v1`退出1：dummy param.uniform_的DSA随机任务分配80字节失败，207001/EL0019，没有模型性能。
- 增加可选GOAL20_SERIAL_DUMMY=1：调用原有NPU RNG/seed，仅每个参数初始化后同步，避免加载阶段任务积压；12451参数完成并进入图捕获。
- c6 `goal20_c6_serial_dummy_v1`两次栈证实capture warmup后stream同步等待；仅自己的PID2165已停止，无有效计时。goal20_c6_blocking_diagnostic_v1在框架配置校验中因ACL graph不兼容LAUNCH_BLOCKING=1被拒绝，没有执行模型。当前任务goal20_c6_timeout_diagnostic_v2仅设置操作超时30000ms，LAUNCH_BLOCKING=0，只用于定位，不能当正式性能。
- run_in_container.sh/serve_goal20_tiny.sh新增GOAL20_CHIP=4或6；源同步session2259可能尚在进行，启动c6重测使用显式环境包装，避免旧脚本强制4。

下一项CPU实现建议：HC finish与全维RMS/RmsNormCast融合。先独立kernel筛选：顺序加载四行形成完整5120维BF16 raw y，
再FP32全维RMS归约，保留BF16中间边界、20次Sinkhorn和独立FP32路由输出。
原模型rms_norm_cast是方法，使用post_attention_layernorm.weight/variance_epsilon，不是独立模块。
模型接入需要合理的prenorm opaque op/forward接入及旧HC refs审计，避免在Dynamo外冻结batch分派。
- 已实现hc_prenorm.py/probe_hc_prenorm.py。v1仅编译选项错误，v2/v3路由精度拒绝；BF16 norm改善约1%，8192布局更慢。
- RmsNormCast源码和上板证明当前vendor的FP32输出=最终BF16.float()（max diff0）；前机会清单“独立未舍入FP32”警告不适用于此实现。
- v3修正final BF16→FP32边界和均值顺序后仅2个元素仍不符1e-4门槛，没有放宽。v4进一步64-lane布局仍未通过路由门槛（2元素），未接入模型；第一次v4源同步先后不完整，需保存源fingerprint再确认，不据此采纳。
- debug工具py-spy仅安装在runtime/debug_tools，未改全局依赖；共享SSH multiplex拥挤，增加可选--ssh-control-path到sync/fetch。
- vLLM现有qk norm/rope/cache融合开关源码限定CUDA/ROCm/XPU，会在Ascend禁用，不作为有效候选。
- 候选源码在同目录，原始日志与trace在远程 `/work/operator_opt/results/goal20_*`。
- 已有优化服务暂时停止，以避免污染本轮独立测量；任务结束或需要长期交付时恢复经过验证的服务。

## 待继续

- 验收六组结果、组合三组非恒定审计、匹配profiling和新kernel逐核Default/MemoryDetail。
- GMM/wo_a的Cube与Vector进一步tiling、shared1实际布局筛选和融合epilogue。
- SparseFlashMla/Indexer逐核等待与控制路径诊断；metadata与邻接norm/RoPE/cache store融合。
- static kernel配置筛选与有效性检查，保留同口径证据，禁止跨进程计算百分比。
- 更新工作/原理报告并上传COS及links-server，必要自检和提交；goal达到目标后才标complete。

## GMM1/activation后续（2026-10-10）

- chip4重新空闲，已恢复在c4实验；c6明确失败算子QuantLightningIndexerV2Metadata，AICPU函数注册/copy so name失败及timeout，507017；无有效模型计时，不重置。
- 新gmm1_activation.py按选中expert同时计算gate/up，保留GMM1 BF16边界与原clamp副作用，融合routed激活。BN16/BK512较优。
- 40份旋转权重：原生链56.254us，已有Vector+activation27.440us，融合18.499us，独立筛选通过。
- composed对native的额外1ULP检查发现已有两kernel基线也在极小值处相差5ULP（abs1.137e-12），融合与已有基线逐位一致；修正融合参照为当前基线，保持1ULP门槛及原矩阵/激活/路由检查。
- goal20_gmmact_model_audit_v3（非static）和static_audit_v4各三组通过：路由、48token一致，logprob最大9.537e-7/0，实际融合40层，输出对已有两kernel0ULP。
- static三模式六组正式会话：baseline23.086ms/A1/43.316tok/s，combo22.220/1/45.004，gmmact21.779/1/45.915；native基线六组全快，对combo五组快一组持平。
- 紧邻两模式十组goal20_gmmact_paired_v2：combo22.276ms/A1/44.892tok/s，gmmact21.991/1/45.474；十组全快，配对约1.31%。不能跨进程相减。
- 匹配20步profiling：1758→1718 task/step，kernel累计14.881→14.572ms/step；vector_gemv40及routed激活40合为gmm1_activation_kernel40，结构/hash验收通过。
- Default/MemoryDetail成功32Vector核、Device4/1800MHz；task18.720/18.680us，最慢核Vec11.080、MTE2 5.719us；逐核GM→UB330.125KiB，UB→GM0.375KiB。
- shared真实N-major投影+激活七种Vector配置均更慢（native约8.26us，较优约10.535us），拒绝，保留结果。
- 已恢复c4 API，默认GOAL20_ARM=gmmact，static=true；health200模型归属正确，32输入/16输出token验收通过。
- 目标尚未20ms。下一项优先：融合kernel的multibuffer/分工、metadata准备/图回放开销，以及剩余norm/RoPE/cache邻接融合。重测前停止仅自己的API并核验芯片。
