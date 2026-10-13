# geometry2k256 v2：修复原生 chunked prefill 的诊断 bookkeeping

v1 的真实 rank0 OFF warmup 已生成 256 token，实际为 `2044 sampled=[]`、`4 sampled=[87930]` 两段 prefill，加 128 个 decode、130 次成功 native helper/propose；v1 在单 prefill 断言被拒绝，取得 0 geometry snapshot、0 数值比较。本目录使用独立复制件，保留 v1、timing85 和 v40 全部冻结字节，不把 v1 当通过。

只修改 harness：允许合法且连续的 native prefill 分段，总和必须为 2048，未完成 prompt 的段必须 sampled=[]，只有最后一段提供唯一的一个采样 seed。全部 prefill 段的原生 draft 调用都计入，OFF helper 数为总 round 数，ON helper 数为真实 prefill 段数；每个调用必须成功、NONE/非 capture 并绑定真实 round。initial、可采样 seed、partial-prefill 和 terminal 分开报告，最终 fence覆盖全部提交。

继续独立推导 decode 数和输出 token；拒绝缺 seed、多 seed、错 token、非法进度/顺序或丢失 native helper。chunked prefill 保持原样，不改 native forward、数学、参数、草稿调用或调度。

原始数值 audit 继续覆盖每个真实 round 的 43 层 route/weights、proposal 和 3 层 attention；新的 geometry/cache 快照只对真实 decode step 执行。每个 decode 的三组实际 sparse input、665 读槽、动态 live table、seq_len/positions、native -1 padding、active null page及 finite/bitwise 门均保留，未放宽。prefill 段不伪装成完整 decode geometry。

配置和前置门保持：2048/256、SWA128/K5、max len4096、batched2048、capture3072、GPU budget0.70/KV2GiB，43 before-capture owner及每请求前后 source/cache seal。2 warmup+9数值门从零开始，性能关闭、dummy quality false、tiny/formal admission=false。

CPU 最小正例直接使用 v1 原字节 raw_001，核对两段 prefill、唯一 seed、128 decode、130 helper/propose；负例包括缺/multiple seed、错误 token、错 prefill 进度、非法顺序、丢 partial-prefill propose/helper。其他配置、SWA/native builder、capture 无读值和正常 HBM 控制继承 v1。CPU 容器禁 backend autoload、visible devices为空，未导入 NPU SDK。

当前只授权 CPU 准备，NPU 权限为 0。建议窗口名 `dspark_lane_20261011_c4_graph_geometry2k256_v2`，命令由新 readiness 提供；主线程另行分配 physical4 后才可启动。
