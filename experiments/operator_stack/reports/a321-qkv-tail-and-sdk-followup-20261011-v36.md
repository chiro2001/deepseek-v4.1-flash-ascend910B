# 正式 QKV 提前 KV tail 审计与并行 SDK 验证

2026-10-11 · a3-21 · 正式40层/TP8/W4A8/Engram int8/完整视觉权重 · HCCL strict

当前已验收正式客户端仍为 **(19.289956ms/step,A=1,51.840451tok/s)**，GSM8K100/100、Vision23/23；**最新目标为正式TP8包含DSpark≤20ms/step**，尚未完成。这套报价关闭 DSpark 的MTP草稿生成与验证，包含主模型、Engram、metadata、通信与已启用Q/KV多流的实际decode开销。DSpark开启后必须重新报(ms/step,A,tok/s)并重新验收，不能沿用当前时延。

## 新 QKV 调度

上一版完整数值审计通过但端到端0/12更快，未采用。源码对照发现独立probe在合并投影后提前提交KV norm，而正式补丁仍保留空的Part2 matmul事件，KV norm/RoPE/cache直到Q norm/quant结束才开始。

新增early-kv-tail-v2只改变独立KV tail的提交位置：加载期仍拼接原INT8 Q_a/KV权重与scale，M1仍用原量化MatMul；完整原Q norm/quant、KV norm/RoPE/SWA cache和最终Q_b/join/Q RoPE函数体逐字保留，分别冻结SHA。非M1原生回退。移除空stage事件，保留主流对aux最终join；不能将源码独立性当作数值或收益证明。

不可变`/work/src_qkv_tail_model_v1`中的formal_qkv_tail_audit_v1退出0：三个原生A/A控制、两个候选各三次路由/token/Top5/logprob比较全部一致，最大差0。8rank每rank40真实唯一QKV消费者逐位通过，48份逐请求rank摘要的实际Indexer/cache、metadata、mask门通过；大requests留远端并记录SHA。

完整精度与补丁/rewrite SHA守卫通过后才启动formal_qkv_tail_perf_v1的12组无审计/profiler同实例交错配对。性能作业退出0：base(20.402537ms,A1,49.013512tok/s)、mask(19.208722ms,A1,52.059685tok/s)、QKV(19.307216ms,A1,51.794106tok/s)。QKV对mask配对中位节省-0.098764ms，0/12更快，继续拒绝部署；不能跨会话与旧版本相减作为恢复收益。审计态时延不作为收益。服务入口额外校验audit/perf/实际启动三方调度variant一致，避免测early而部署native。未启动新服务。

## 旧 QKV 回退的真实时间线

16份mask/qkv×8rank PipeUtilization已离线导出。真实worker schedule收据和160个完整decode步通过，每步80HcPre，共6400层窗口；没有用收据代替设备步完整性。

实际NZ权重qa[40,320,16,32]/kv[16,320,16,32]变为merge[56,320,16,32]，合并投影40次/step。没有新增设备Slice/ViewCopy/Contiguous，Slice均7、Cast均117/step；1792输出切片的设备copy为0。当前证据不支持“切片复制抵消收益”的猜测。

两bank Q norm/KV norm重叠均0。mask的KV Cube与Q norm平均重叠2.216µs/层，qkv为0；qkv KV norm开始距合并投影结束7.811µs、距Q动态quant结束0.631µs，支持原part3门仍延后KV tail。原mask/qkv主图采集态跨度18.573/18.801ms，图前0.8619/0.8595ms基本相同，record均604/step、wait约484.9/step。上述为采集态区间、累计或重叠统计，不是可回收端到端预算；新调度仍以关闭观察器的配对结果为准。

## Engram 与稀疏 MSD 的边界

完整MatMul N25600/K6144 Engram提前计算原型仍未接入正式ARM。主线程补强同地址下输入shape/stride/dtype/device/format守卫，sealed audit输出与参数，禁止捕获后重新分配；CPU/mock31个反例通过，不能启用NPU。

独占健康空闲physical4的SDK探针已证明：pristine wait-before-record与32代record-before-wait可行；一次capture图内record/wait再32代普通NPUGraph回放可行，producer延后时ordinary consumer_done.query为False、当前代输出/marker/epoch一致。但每代host consumed.wait在第2代触发single-record/multiple-wait错误；原型已禁止此模式，要求显式submission-stream普通完成fence证据。正式对齐的AOT/GraphEx static与nonstatic均在producer首次调用、实际custom-op callback之前报TypeError expected Tensor。生成kernel含clone，仅为静态代码证据，尚无实际alias拒绝/提交流证明；safe_formal_receipt全false。不能套用普通图结果作为完整安全receipt。

新GMM2稀疏MSD保持完整48专家NZ storage和原17个数学函数体，仅改AIC最多六槽遍历。CPU native/candidate原tiling2032B逐位相同；编译后.text对照证明descriptor guard/slot循环/rowBegin读取未被常量消除。系统workspace16MiB与128B planner前缀已锁定。入口adapter ABI、真实NPU原参数/≤1ULP、图消费者和完整模型仍待完成，NPU权限当前0，不记性能。

用户先要求专属DSpark tiny，最终将正式目标改为包含DSpark≤20ms/step，取代无DSpark17ms及短暂的24ms。三个已核验gpt-6.1-sol/max代理均明确ACK最新20ms：tiny_engram负责physical4专属DSpark tiny，SDK已释放4；tiny_qkv只CPU准备正式full40 DSpark入口与真实轮次计数；tiny_ids收口MSD CPU产物，NPU权限0。DSpark报价以实际target/verifier decode rounds计步，逐请求A=decoded_tokens/rounds，ms=1000*decode_s/rounds，tok/s=decoded_tokens/decode_s，包含草稿/验证/调度/cache，不能用全局MeanAcceptance或前端poll次数替代。当前尚无正式DSpark开启的20ms结果。芯片2/3/7和完整tiny的6历史故障保留，不reset、不删设备内部文件、不操作其他租户。SSH仅传<1MB紧凑证据。

源码、精度/资源与CPU/SDK紧凑证据后续提交，报告发布与从HEAD生成MANIFEST、自检、双远端push随后完成。当前没有包含DSpark的20ms达成或最优服务完成声明。
