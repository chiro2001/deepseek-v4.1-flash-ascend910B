# 正式 W4A8 小 M：六槽设备调度，复用原 MSD 数值体

2026-10-11。阶段仅为只读源码审查与 CPU 契约设计，**NPU启动0次**。芯片4已转 Engram，本 lane 不使用任何芯片、不修改正式文件、不提交。已有 prefix/cumsum、TP1八专家 selected-GMM、自写INT8投影GEMV负路径保持关闭。

建议准备新的 `sparse_rows_msd` 算子入口：从已有 `[1,6]` 全局专家 IDs 在设备上生成最多六条本地专家描述符，直接选择完整48专家 NZ 权重/scale/assist 的地址；只改 MSD 的 group 遍历、row 起点与 expert 指针索引，沿用原生 INT4 Cube、分支反量化、FP16中间值和 activation/quant 代码。先做 GMM1，GMM2继续使用原生counts；通过独立门后再做单独的 GMM2版本。当前没有 NPU内核、精度通过或收益声明。

## 原生稀疏描述符不能直接套当前 W4 路径

安装态源码和原参数头/坐标样本已记录在 [source_evidence.json](source_evidence.json)、[adapter_boundary_evidence.json](adapter_boundary_evidence.json)，所有引用都有 SHA256。总计只读17份源文件，SSH两个包428,225和47,406字节；权重留远端。

| 实际位置 | 已确认契约 |
|---|---|
| `.../quantization/methods/w4a8/w4a8.py:206`、`:447`、`:532` | `apply`进入实际通信路径；`apply_gmm1_act_quant`和`apply_gmm2`是实际MLP消费者 |
| `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/*base_tiling.cpp:179`、`*fusion_tiling.cpp:71` | fused W4 MSD 明确只接受 `group_list_type=0/1`，其他值拒绝 |
| CANN9.1 `grouped_matmul/grouped_matmul_utils.h:182`、`:312` | 通用 GMM定义type2；描述符为 `[expert_index,row_count]`，第二列是counts，不是prefix边界 |
| CANN9.1 `grouped_matmul_antiquant_a8w4_msd_pre.h:137` | 当前A8W4 MSD pre只读取前`groupNum`个元素并求和；没有type2列解码 |
| 同目录 `...msd.h:262`、`...msd_new.h:241` | A8W4控制循环直接`GetValue(groupIdx)`读取dense counts，没有type2分支 |

通用GMM的type2存在，不证明正式A8W4模板可用。CPU示例 `[[47,1],[0,0]]` 表示一行；现有MSD pre若把前两个平铺元素当counts，会计算48行，超出容量6。当前禁止通过简单修改 `group_list_type=2` 启用。

另一个独立边界更关键：Torch fused adapter调用 `aclnnGroupedMatmulSwigluQuantWeightNzV2`；这个API在单张量场景从 `group_list.shape[0]` 取得expertNum，并强制设置 NZ storage `[expertNum,N/64,K/16,16,8]`。六槽描述符若交给旧API，会把真实E48错误描述成E6。新入口须独立接受静态`weight_experts=48`和动态`active_groups≤6`，保持全量权重的storage描述符。不能先放宽0/1检查再复用旧adapter。

## 真实参数、地址与数值顺序

正式checkpoint：`/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`，40层/hidden5120/FFN2304/384专家/top6，per-channel `group_size=0`，TP8/EP8，每rank完整48专家，FFN不按TP除8。

抽样为层0/20 × 全局专家0/47/383 × w1/w3/w2，共18组真实参数。专家0/47对应rank0两端，专家383对应rank7最后一个专家。只读取safetensors头和少量坐标字节，没有初始化框架/NPU。

| 参数阶段 | 实际形状/dtype |
|---|---|
| 原文件 w1/w3 | I8 `[1152,5120]`；一个字节沿输出N打包两个signed INT4 |
| 原文件 w2 | I8 `[2560,2304]` |
| 原 weight_scale/offset | F32 `[N,1]`；所取offset样本均0，不将抽样扩展成全参数结论 |
| 原 down scale_bias | F32 `[5120,16]`；EP使scheme.tp_size=1，按原loader对16项求和 |
| 处理后的 w13 | INT32逻辑 `[48,5120,576]`；N=4608 |
| 处理后的 w2 | INT32逻辑 `[48,2304,640]`；N=5120 |
| 实际packed NZ storage | INT32 `[48,N/64,K/16,16,8]` |
| w13 scale/assist | INT64 `[48,4608]` / FP32 `[48,4608]` |
| w2 scale/assist | INT64 `[48,1,5120]` / FP32 `[48,5120]` |

`w4a8.py`先沿K/N转置raw packed bytes、将weight_scale的F32 bit pattern零扩展成INT64，再在加载期转NZ，最后做INT32 view。一个INT32承载八个INT4，低位nibble是较小的输出N坐标。CANN `gmm_dsq_base.h:526` 的 `UnpackInt32ToInt4` 实际只调整ACL dtype/view/storage位宽：逻辑末轴×8、NZ末轴8→64，weight_scale INT64→UINT64；不是每step把权重展开成INT8。

令`e`为本地专家、`k`/`n`为逻辑矩阵坐标，packed NZ的32bit word地址为：

```text
word = ((((e*(N/64) + n/64)*(K/16) + k/16)*16 + k%16)*8) + (n%64)/8
nibble_shift = 4*(n%8)
signed_w4 = sign_extend4((word_value >> nibble_shift) & 15)
```

专家矩阵byte基址为 `e*K*N/2`，scale基址为 `e*N*8`，assist基址为 `e*N*4`。native `GlobalTensor<int4b_t>`的索引单位是nibble，不能把它当byte指针直接加`e*K*N`。CPU契约校验式子与独立的NZ tile循环打包，但**尚未做真实NPU物理NZ回读**。

原生 GMM1 MSD 数值体必须完整保留：

1. pre：`high=floor(x_int8/16)`、`low=(x_int8&15)-8`，每行按`[high,low]`交错存储为两行INT4。
2. mid：原 `MatmulType<ND,int4; NZ,int4; ND,half>`、原INT32累加、原`SetQuantVector(encoded_scale)`、原K切分/`GetTensorC`行为；两分支分别得到FP16中间值。
3. post：FP16→FP32，再`high*16 + low`，加原assist，乘原token_scale，再执行原clamp/SwiGLU、ReduceMax/127、reciprocal和原FP32→half→INT8 RINT cast。

post使用后半N和前半N的原指针/API次序；不重写SwiGLU公式或修改clamp上下界。GMM2还存在API-dequant与vector-dequant模板，未来必须以实际kernel/tiling选择对应原body，不能根据名字任选一个。

CPU随机case已找到可复查的舍入反例：原分支顺序0.013519287109375，合并成一次整数dot后统一缩放0.01338958740234375，相差**3 BF16 ULP**。INT32恒等式成立也不足以满足原1ULP门；这里只证明替换顺序不安全，不是NPU误差测量。

## 具体新入口和设备调度

初版接口维持静态容量与全量权重：

```text
row_plan(topk_ids INT32[1,6], ep_first static=rank*48) -> descriptor UINT8[128]
gmm1_sparse_native_msd(
  x INT8[6,5120], token_scale FP32[6],
  w13 INT32[48,5120,576] NZ, weight_scale INT64[48,4608],
  assist FP32[48,4608], descriptor[128], native_counts INT64[48]
) -> (quant INT8[6,2304], act_scale FP32[6])
```

static选择门要求 decode单token、top6、384全局/48本地专家、连续EP范围、EPLB/LoRA关闭、SILU/per-channel、原dtype/storage和quantGroupNum。其他几何走原生；不能用CPU读取IDs/active count决定路径。实际 `MoEMlpComputeInput`已有`topk_ids`和`expanded_row_idx`字段，无需新增D2H。

设备planner加载六个原topk IDs，按静态本地范围筛选并用固定8元素sorting network排序（两个补位sentinel）。M1 top6 IDs唯一，每个保留的expert有一行；输出每槽 `(local_expert,row_begin,row_count)`，inactive槽为`expert=-1,count=0`。128B header保存`total_rows`和`active_groups`，剩余六槽每槽16B。原counts48、expanded_row_idx、masked topk_weights不改变。CPU以独立counts48展开oracle验证所有八rank的排序/行起点。

首个独立probe使用一条小planner kernel，然后同stream调用新MSD入口；GMM2复用该descriptor。**计时必须包含planner成本**，不能只报新的mid。若初版有净收益，再将planner融合到原init-routing生产阶段以消除额外launch；不能预先计入这一未实现收益。

GMM1的明确源码差分范围：

| 原函数 | 新调度差分，数学体保持 |
|---|---|
| `pipeline::InitWorkSpaceSplitConfig` | 从device header读取R；静态capacity6/原mLimit不变；不让所有core重新读48counts |
| `pipeline::UpdateWorkSpaceSplitConfig` | 最多六个非空group，row_begin/count来自descriptor；R≤6只需原单workspace区间 |
| `mid::Process` | `groupIdx=desc.expert`、`m=2*desc.row_count`、`xBaseOffset=2*row_begin*K`、`yBaseOffset=2*row_begin*N`；复用原`MMCompute(groupIdx,...)` |
| `post::UpdateVecConfig`/`UpdateAuxiliaryMatrix` | 当前和下一expert来自compact槽，assist仍按原expertId读取；其余cast、scale、Swiglu、Quant原样 |
| adapter/tiling | 独立算子名与完整E48 tensor descriptor；原MM tiling、NZ format、K/N、scale/bias指针与字段明确绑定 |

原single-weight分支的`weightGM`指向完整参数，expert地址由**设备**`desc.expert`计算。直接复用原 `MMCompute`中 `e*N*K + tailN*K` 的NZ nibble起点；所有scale/assist也按同一e索引。没有 `.item()`、`.tolist()`、CPU expert列表、动态Python TensorList或完整专家矩阵`index_select`。

新adapter对ATen输入只读，为每个bank创建独立ACL描述符，避免原API的`const_cast`改共享descriptor；模型原Tensor dtype/format/storage/version保持。每层持有独立128B workspace，FULL graph replay每步更新内容；mutable workspace须正确标注，不把首轮expert ID固定在capture。所有core使用统一R0控制和原同步纪律，不能在部分core分支绕过`SyncAll`。

## 内存预算与ND分支

每rank每层：w13为540MiB，w2为270MiB，共810MiB；INT64 scales约3.56MiB、FP32 assists约1.78MiB。40层再保留一份完整ND权重会额外**31.640625GiB/rank**，不能在已有约38GiB权重/运行占用上默认增加。

首候选保持原NZ，不需要复制权重；40层descriptor总共5120字节。容量6对应GMM1/GMM2原MSD FP16中间阵分别最多108KiB/120KiB，沿用原workspace规划，不把它们当新的一份全量模型。

六个专家整矩阵materialization一次就达101.25MiB/层；初版选址只构造地址，不按动态ID搬运这批矩阵。ND仅作为独立单层诊断变量：安装期一次转换一层810MiB并逐位回读，使用支持ND的独立合法adapter及对应原INT4 primitive，完全计入额外HBM。现有WeightNzV2强制NZ、base tiling也设NZ，不能给它传ND再期待自动正确。整网ND替换还涉及native回退和双bank预算，暂不作为首候选。

## 最小独立NPU验证计划（本阶段未执行）

1. 在新授权芯片和私有快照准备真实层0/20，遍历8个EP专家范围。由正式loader生成原48专家INT32 NZ、编码scale与原assist，锁定版本/format/storage，并比较NZ↔ND物理回读；权重和trace继续留远端。
2. 使用实际已dispatch的`x_int8`、token_scale、原topk_ids/counts/expanded_row_idx。覆盖R0/1/2/3及六个本地expert边界，另外加入R6、local0/47、sentinel/padding和clamp/量化边界输入。先做同实例原生A/A，capture前后原生A/A；所有输入与consumer引用在图外导出，capture内不D2H。
3. 同输入比较原MSD和candidate：valid R行的两份FP16中间值、INT8激活和FP32 act_scale目标逐位相同；再把candidate的两输出交给**原生GMM2**，保留原 **≤1 BF16 ULP** consumer门。unwritten capacity tail本来不属于有效GMM输出，单独验证原expanded索引/本地mask/最终unpermute，尤其R0；不能将空行比较当完整consumer通过。
4. 检查真实捕获和重放：每份实际48expert所有者、六ID→真实expert指针、各layer/rank、R与descriptor每步变化。完整模型随后仍要求strict原生规约、路由/token/Top5/logprob、cache消费者和正式客户端门；独立probe不能代替它们。
5. 仅精度/coverage通过后关闭全部观察器，同实例交错配对，比较`planner + GMM1 + 原GMM2 + unpermute`的原1token consumer包络；额外profile准确count与Scalar/等待，记录新增launch/SyncAll/UB/L0布局。约2.1ms与1.33ms家族累计是采集热点，不是保证可回收收益。

GMM2候选独立做第二门：先确认当次native A8W4模板，再仅替换其dense group控制/原pointer offset，保留pre、INT32/FP16分支、bias/per-token/output BF16顺序。没有证明模板兼容前，GMM2继续原生counts。

## 本次可执行CPU产物

- [cpu_contract.py](cpu_contract.py) 与 [cpu_contract_result.json](cpu_contract_result.json)：17份源码SHA、18真实参数集合/648坐标、12,288个signed NZ字节序/地址点、4,096个MSD乘积恒等式、原舍入反例与精确内存预算。
- [sparse_row_schedule.hpp](sparse_row_schedule.hpp) 定义128B ABI、全量参数地址和六topk/EP CPU参考planner；[verify_sparse_row_schedule.cpp](verify_sparse_row_schedule.cpp)验证7000分布和8000 top6/EP案例及formal边界地址，结果与编译器/source SHA见 [cpp_contract_result.json](cpp_contract_result.json)。
- `read_cpu_evidence.py`只用标准库读取正式容器源和参数头，没有框架/NPU import。引用快照与原source SHA一并保存。

本次CPU检查通过，只证明布局/地址/调度契约。物理NZ、native adapter实际调用、NPU ≤1ULP、capture消费者以及性能都仍需主线程另行授权资源后的独立门。
