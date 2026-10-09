# 持续目标：950优化向A3迁移（独立tiny）

独立分支：feat/tiny-upstream950-a2a3-20261009，base 998c47c。
工作目录：operator-upstream950/experiments/upstream950。
远程tiny：a3-21 /home/l00886679/projects/dsv41-tiny-upstream950-20261009，chip5，容器dsv41-tiny-upstream950-20261009-c5。
旧调研素材位于../../.. /tiny_upstream950_20261009，upstream为本地只读引用，不打包。

|候选|状态|结论|
|---|---|---|
|稀疏专家GMM完整链|完成，拒绝默认采纳|审计逐比特通过；12组A/B仅1组更快，配对加速比0.993465|
|SMLA索引/页表预取|完成，拒绝当前tiny默认采纳|30case及模型审计逐比特通过；12组prefetch/control配对无稳定收益|
|Indexer INT8后处理融合|完成，保留|150case及随机模型逐位通过；两轮12组配对收益0.27%/0.51%|
|inverse RoPE/八组wo_a|完成，拒绝默认采纳|两模型候选12组均变慢；静态八组GMM单算子也无收益|
|Q/KV及q_b权重面板|完成，拒绝当前候选|joint改变2处专家集合；NZ/panel/prefetch审计逐位通过、12组配对均无收益|
|SMLA UB→L1能力验证|完成，当前标准SDK路径不适配|control编译成功；950直写原语在2201目标不可用，公开API经GM软件中转|

2026-10-10 用户收窄范围：只面向可直接访问的A3验证，不准备A2包、不安排A2实机测试。该约束优先于goal初始文字及旧调研计划。

任务采纳门槛：同模型/同进程图bank、单变量、无profiler计时、随机非恒定权重正确性与实际路由；三元组ms/step、A、tok/s。无收益项保留拒绝证据，逐项完成处理。

第一项审计 sparse_gmm_audit_v3 已完成：8次请求（4次预热+2组配对），每次核验40层路由及两次GMM；展开token/row_idx/counts/GMM输出逐比特一致，路由和logprobs一致。审计时延26.506→26.670 ms不能作为正式性能结论；已另起关闭审计的12组A/B。

SMLA候选准备：从容器安装态csrc作独立副本，Up950SparseFlashMla与未改算法的Up950BaseSparseFlashMla两份私有op，保留同工具链控制。只改CSA V0寻址，把当前AIV半tile索引/能容纳的PA表搬入已有v0ValidSizeBuff前384个int32，Vec2从slot384起，不增加UB。源码来源SHA和修改头SHA在远程build/smla/source_manifest.json。当前正在解决首次私有op编译问题，未进行NPU执行。

SMLA构建进度：Torch私有binding已成功编译并加载，binding_manifest.json已有路径和schema。CANN两个private op的host库已编译，现kernel编译失败：Up950[Base]SparseFlashMlaTilingData未定义，正在检查共享tiling结构与重命名边界。此前修复复制过滤误删第三方nested output/build目录、双op header guard和infer helper符号隔离。当前没有GPU任务；不要把编译失败当成能力不支持。build_smla_v5.log记录最早unknown type根因，后续fatbin缺.o是连带错误。

Tiling声明修复采用安装态CANN的gen_tiling_data_stub.py Process._get_tiling_source（含标准padding），只保留POD声明，private kernel cpp显式include。generate_smla_pod.py会先填非零哨兵，验证Host SaveToBuffer的所有scalar字段、嵌套字段、sizeof；通过后才运行kernel。尚在CPU编译校验include paths，smla_pod_layout_v2.log。

CPU tiling布局审计已通过（smla_pod_layout_v3）：prefetch/control均224字节，所有scalar/nested field与SDK Host SaveToBuffer一致。build_smla.sh现自动生成并校验POD，再编译kernel。已启动build_smla_v6.log，待完成后先verify_smla.py --quick，再完整边界验证。

已确认build/binary的源码缓存仍是旧cpp，缺PODinclude/header；build_smla.sh增加在增量编译前同步两个private op的op_kernel到staging，已起build_smla_v7。模型接入smla_patches.py已准备：baseline/control/prefetch三个bank，prefetch仅38个CSA层；审计保留inverse RoPE的输出副作用，随机attention参数和FP32预转换缓存同步。需先完成30个独立边界用例再跑模型。

2026-10-10跨日继续：staging旧cpp问题解决后v7错误变为GET_TILING_DATA_WITH_STRUCT未定义。generate_smla_pod.py改为使用安装态SDK tbe.tikcpp.get_op_tiling.gen_dynamic_shape_v2生成标准copy helper/macros（CCE编译条件中），不自写GM拷贝协议；CPU224B layout验证保持。build_smla_v8进行中。

最新构建：v8两个private kernel均已编译和打包，installer的shared library校验失败，undefined up950_prefetch_sparse_mla_checker::CheckerRunner::Process。根因checkers/checker_sources.cmake全局target去重property沿用原SPARSE_MLA_COMMON_CHECKER_ADDED，导致只编译第一份命名空间。已把两个private op的CMake函数/数组/property分别前缀隔离，prepare_smla.py记录可复现修改；正在build_smla_v9.log。未使用installer --force。

下一步：检查v9尾部是否Private CANN op installed to /work/build/smla/opp；确认vendor路径up950存在；run_remote.py --script verify_smla.py --tag smla_quick_v1 -- --quick。quick通过后完整30case，随后bench_lane_model --candidate smla --arms baseline,control,prefetch --audit --pairs2，再无审计6组perf。smla_patches审核captured output时补inverse RoPE；attention权重随机化和FP32缓存同步已在lane_patches中。其余4候选仍未做。

用户范围更新已落实。SMLA v9已编译并通过shared-library检查，安装目录实际为 /work/build/smla/opp/vendors/up950_transformer（安装器自动追加_transformer）；已修正loader和bench环境路径，开始quick正确性测试。

最新执行状态：2026-10-10范围改为仅A3（用户明确取消A2包/验证）。v9构建成功并安装up950_transformer vendor。smla_quick_v1在control阶段同步报VEC参数无效（core36），尚未执行prefetch；smla_native_probe_v1同输入原生执行通过，说明测试输入和原生路径可用。verify_smla.py增加--arm和分阶段同步日志，需先修好private control再作prefetch性能判断，禁止据此声称寻址优化失败。

关键根因已定位：private kernel JSON opParaSize=8，而原生=232，差额恰好224字节tiling。此前手动SDK宏生成时ascendc_tiling_no_register未设，REGISTER_TILINGDATA_SIZE为空，ELF没有tiling-size section。generate_smla_pod.py现设该flag=True调用SDK生成器，再恢复；开始build_smla_v10.log。下次先确认两个private JSON的opParaSize=232再quick（原生已通过，芯片health=OK，无需重置）。

v10增量构建只重建host库，两个旧kernel的.done依然命中，安装后opParaSize仍8；quick_v2仍control VEC异常。已增加私有.done失效处理与232字节产物检查，loader必须有kernel_parameter_validation.json才允许加载。build_smla_v11正在强制编译两个kernel，检查通过前不再设备试跑。

v11强制kernel编译揭示SDK now自动注入tiling class，和fallback POD重复定义；旧binary仍被上游build.sh打包（其日志管道吞错误），232门禁正确阻止执行。fallback头改为仅在实际SDK宏未定义时提供；清除两个私有旧binary目录避免误打包，v12编译中。

v12成功：kernel参数门禁通过，两个JSON opParaSize均232，kernel_parameter_validation.json已落盘。私有control/prefetch编译产物真实更新，开始smla_quick_v3设备正确性验证。

SMLA独立算子30case全部通过：ratio1/2、长度127/128/129/2048/2049、TopK512/1024、负索引holes、随机非恒定Q/KV、乱序物理页，control/prefetch输出及LSE逐比特等于原生。参数232B和Host布局224B证据已入evidence/smla。正在smla_model_audit_v1三bank整网随机权重审计。

模型审计v1/v2失败发生在baseline自比：两个原生functional replay输出各32768个NaN，q有限，非candidate特有。不能以equal_nan放宽通过。新增stored_nan、seq lengths、sink finite、index范围诊断，正在smla_model_audit_v3；需区分capture保存的metadata陈旧与模型执行本身NaN。三个bank覆盖40/40/38CSA已确认，30case独立验证通过。

模型审计NaN根因明确：stored输出有限（0 NaN），ratio2 cmp_len1047而保存TopK引用末态范围[0,2094]，共享TopK buffer已被decoder层覆写，复算引用错误。smla_patches.py在audit图消费者点clone q/TopK/raw输出，保存原始输出避免后续inverse RoPE副作用；正式性能图关闭clones。正在smla_model_audit_v4。

SMLA模型审计v4完成：三个bank各40次SMLA，prefetch覆盖38CSA；12次请求（6预热+2组×3臂），216个attention权重及HC/router/MLP随机化，消费者快照functional/graph raw输出逐比特一致，路由及logprobs一致，max_abs=0。审计clones不用于正式计时。下一步无审计三臂12组A/B；之后归档决定采纳，再进入Indexer INT8后处理融合。

SMLA正式三臂12组A/B完成并归档SMLA_RESULT.md：prefetch/native配对中位0.9892355，仅1/12组快；prefetch/control配对也无稳定收益。当前方案拒绝默认采纳，保留独立.cpp生成/回退。下一项Indexer K后处理。

下一项设计固定到INDEXER_DESIGN.md。已确认RMSNorm直接npu_rms_norm(BF16 gamma)；cache slots为[T,2]坐标，cache/scales可有layer-outermost物理stride，必须保留。当前无GPU实验进程（仅owned container sleep），源工作树已提交。不要重新跑已完成的GMM/SMLA性能；从Indexer数值边界/真实cache布局继续。

Indexer第一版Triton后处理已写scripts/indexer_post.py：每行128维、FP32 RMS归约、BF16舍入、interleave RoPE尾64、INT8 nearest-even、FP16 scale、实际cache stride及[T,2]坐标直接写。独立probe原生zero quant scale=0；norm.gamma BF16，RoPE BF16走ComputeCastFp32并CAST_RINT已查源码。verify_indexer_post.py对照各stage及整个strided parent cache；首次probe被模型import cycle阻止，已直接用相同native scatter API避免循环，正在indexer_post_precision_v2.log。该版本尚未证明精度，不可接入模型或报收益。

Indexer精度扩大102case（16seed×3幅度×2行数，加zero/constant/quant-half ties）。v1 Norm/RoPE/scale均逐位一致，但4case INT8有±1差异；未放宽接受。逆scale计算由127/maximum改为1/已舍入FP32 scale，保持原生scale分母边界，stress_v2进行中。基本6case逐阶段/全cache已通过，实际RoPE要求4D cos/sin，native测试已修正view。

Indexer quant诊断：Norm/RoPE/scale在102case全逐位一致，INT8差异集中在±63.5附近，单纯等价浮点重排不能保持原生。已找到安装态SDK CPP，dynamic_quant_single_row.h/multi_row.h均Div(127,max)再Mul(x,coefficient)，RINT到int32/int16再cast到int8。v1近似Div有4bad、reciprocal(scale)6bad、(x/max)*127有9bad、div_rn(x,scale)有7bad；尚未接受。当前按真实源码改coefficient=tl.div_rn(127,max)、乘x，stress_v5进行中。

Indexer stress_v5的scalar div_rn(127,max)触发Triton编译器TypeRange null-types断言，进程终止，无设备实验结果。已将127扩为128-lane vector，沿用之前可编译vector div_rn路径；stress_v6运行中。SDK源码路径为.../ops_nn/ascendc/dynamic_quant/{dynamic_quant_single_row.h,dynamic_quant_multi_row.h}，真实Div(127,max)->Mul->RINT；这不是改精度容差。

quant DivRN常量127 scalar和tl.full128两种形式(v5/v6)均触发TypeRange null-types编译器断言，未知是否常量折叠bug；vector(value)/scale RN之前可编译。v7把固定127向量放进一次性NPU常量tensor（按device缓存）并tl.load，避免编译器常量scalar化，语义仍DivRN(127,max)->Mul。stress_v7运行中，未接入模型。精度门槛维持cache bytes逐位。

Indexer stress_v7已完成：102case的Norm/RoPE/INT8/scale以及完整strided parent cache全部逐位一致，证据precision_102.json。RN系数Div(127,max)使用持久NPU向量避免编译器scalar折叠断言。正在补48项布局用例（T1/4×8seed×3布局；FP32 trig、负坐标、strided input），并添加四K源层独立图bank与消费者点cache行快照审计。正式性能计时仍等chip4空闲，不混入audit复制。

layout_v1失败限于T4 strided-input的8case；FP32 trig与混合负坐标均通过。诊断发现debug使用empty_like保留输入转置stride，但debug stores固定row-major；同时原生RoPE只接收contiguous GM，测试将非连续trig直接传入会改变原生语义。修正debug输出为连续；非连续输入验证以逻辑contiguous值送原生作为reference，候选仍读取实际输入stride。生产wk输出本身连续，不变更融合精度门槛。

layout_v2的8个失败case已用第三参考定位：Norm完全相同；native RoPE、连续candidate及直接torch数学参考均相同，仅candidate非连续trig错误。不是native计算公式。候选原rope_col=col-64在前64 masked lanes形成负地址；改为col&63让所有lane地址在bounds，排除Ascend strided masked load lowering问题，启动layout_v3。仍保留真实cache strides、全部原精度检查。

Indexer layout_v3全48项通过：用col&63消除mask前缀负地址后，非连续X/cos/sin也与contiguous native逻辑参考逐位一致；FP32 trig、int64 strided coords、混合负坐标和整个parent cache全部通过。已启动indexer_model_audit_v1，四K源层、原生/fused两个bank、随机attention/MLP与路由；审计请求交替47/48输出token以覆盖C2完成/未完成步。下一项epilogue只作本地实现准备，还未NPU执行或采纳。

indexer_model_audit_v1的图capture/bank覆盖与模型生成已成功，第一次离图原生复算时报Inference tensors cannot be saved for backward：RPC audit在默认autograd上下文调用带Parameter的k_norm。lane.audit添加torch.inference_mode，属于验证器上下文修复，无NPU精度失败。准备audit_v2；UB→L1源码核实已发现dav_c220的TSCM拷贝走KFC软件模拟，不可把公开DataCopy API当成直写硬件能力，后续仍会最小编译核验。

Indexer model_audit_v2完整通过：两bank各4个K源后处理调用，8次请求覆盖C2有效4行/仅C1有效1行，消费者点缓存逐位一致，functional INT8与FP32 scale一致，40层实际路由和logprobs逐位一致（delta0）。真实cache K的page stride为131072/147712，scale为65536/73856；cos为FP32，均已覆盖。审计时延不作性能结论。chip4目前100%且HBM63GB，已安排indexer_perf_idle_queue等chip4及本容器GPU实验连续空闲30秒再做最终102case、单算子及12组整网A/B；没有操作其他线的进程。

Indexer初次正式12组A/B完成（idle_queue_v2，开跑前chip4连续闲30秒）：基线(26.447915ms,A=1,37.810164tok/s)，fused(26.372360ms,A=1,37.918487tok/s)，配对加速中位1.002683，10/12组更快。单算子T1 17.862205→14.206690us，配对1.257485；T4 23.784215→14.257905us，配对1.667945。最终102case全阶段/完整cache逐位通过。收益较小，做第二轮12组确认并补全程chip4负载采样；不先宣称默认启用。

Epilogue model_audit_v1完成：40层native/packed两bank，wo_a/wo_b及实际路由/logprobs逐位一致。packed用torch.bmm在审计态明显变慢，仍需无审计性能证据。新增只改inverse RoPE的inplace候选（4head/program、只读写尾64维、融合负sin），保留原生八组TransposeBatchMatMul，避免T1重排全部nope数据；准备32case及三bank模型审计。当前尚未完成第四项性能，不能据审计时延采纳/拒绝全部输出投影方向。

Indexer第二轮确认（12组，锁保护，全程41次chip4采样均0）：基线(26.121245ms,A=1,38.283014tok/s)，fused(26.010020ms,A=1,38.446722tok/s)，配对加速中位1.005068，11/12组更快。保留融合实现，初始化127常量向量移到install/capture前，核心计算未变。

Epilogue正式三臂12组结束（56次chip4采样均0）：native(26.155935ms,A=1,38.232240tok/s)，packed(30.667795ms,A=1,32.607496tok/s)，inplace(27.222755ms,A=1,36.733975tok/s)。配对中位0.854201/0.964095，两种方案均0/12快，拒绝采纳。另试静态八组GMM：32随机case逐位通过，单算子25.188080→25.909870us，配对0.972292；因同一段其余计算保持原生，微观无收益，未接入整网/未默认启用。

UB→L1能力门禁完成：dav_c220 SDK明确UB→GM workspace→KFC通知Cube GM→L1，公开TSCM是软件模拟。纯GM/UB control在dav-c220-vec编译成功；950的CopyUbufToCbuf在2201的vec/cube目标均undeclared identifier，未执行不受支持原语。结论限定当前标准SDK路径不提供可移除GM中转的950直写，保留已验证的原生SMLA，不声称非公开混合核路径绝对不可能。

Projection precision_v1 16case：panel q_b全部逐位一致；joint QA/KV个别末位差异；NZ实际format2（allow_internal_format默认false），不能把它当NZ对照。修正限定预排时启用internal-format，并用BF16逐元素误差+RMS相对<=1e-3记录joint浮点重排，而后续Norm/RoPE仍需对本次输入逐位正确且路由/logprobs通过。v2首先暴露torch.npu.config只写无getter，尚未运行数值检查；正在查正确读取当前选项方式，避免改baseline的进程配置。

最终收尾：六个方向全部按门禁处理完成，详见README与六份RESULT。q_b NZ/panel/prefetch16case逐位、40层图bank审计及路由/logprobs逐位；两轮三臂各12组正式配对均无收益，57/56次chip4采样均0。joint的BF16重排改变2处专家集合和10个Top5 logprob键集合，拒绝采纳。显式预取的CBUF loop类型问题通过静态展开三个交接解决，精度通过；实际性能仍慢于原生。

服务已恢复在本lane私有容器，默认Indexer fused，k_source_calls=4/fused_post_calls=4；本容器私有/tmp/mysvc.sh精确ID断言返回MINE，32输入/16输出HTTP冒烟成功。A3主机内桥接地址 http://172.17.0.6:18973，运行时仅npu:0/物理chip5。宿主/tmp身份脚本及其他容器未改动。GPU锁防止服务与新实验误并行；原生回退UP950_ARM=baseline或UP950_CANDIDATE=none。

归档校验通过：67个紧凑JSON，所有已归档摘要/SHA manifest一致、全部三元组1000*A/ms与tok/s自洽、全部正式配对样本12组、所有采样chip4=0。修正重复导出的manifest自包含旧hash问题（数据hash不变）；完整大requests仍留远程。源码Python编译、起服/身份脚本bash语法、git diff --check通过。包selfcheck初次系统Python缺pytest；改用已有Miniforge Python/pytest环境后全部通过，无需安装依赖或修改全局环境。
