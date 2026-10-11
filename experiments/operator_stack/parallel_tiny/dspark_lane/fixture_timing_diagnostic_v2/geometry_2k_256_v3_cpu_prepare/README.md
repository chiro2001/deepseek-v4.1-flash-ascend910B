# geometry2k256 v3：绑定真实 worker 的 geometry RPC

v2 的两个非审计 warmup 已完成，首个 `native-off-0-A` 的首次 decode 在 geometry 前置组合断言 exit 1，0 geometry snapshot、0 数值比较。v2 原证据与 178 文件交付不修改。取回该 job 的 native vendor 源码后，trace 与源码共同定位到对象不一致：`UniProcExecutor.collective_rpc` 给 callable 传 `WorkerWrapperBase`，而 `GeometryQualityWorker.load_model(self)` 安装保存的是内层真实 worker。v2 未导出失败入口的运行时对象 id，后两项状态在短路断言中没有执行。

v3 使用独立复制件，仅让 bench 以 string RPC 调用 `GeometryQualityWorker.snapshot_geometry_step`。真实 `WorkerWrapperBase.__getattr__` 返回内层 worker 的绑定方法，adapter 以真实 `self` 调用原探针。identity 仍严格要求 `worker is installed_worker`，不接受 wrapper、同名对象或其他 owner。

三个入口门分别给出 `capture_forbidden`、`worker_identity_mismatch`、`audit_disabled` 原因。capture 门先执行，拒绝路径无任何 D2H、synchronize、诊断文件写入或日志。非 capture 入口只写一次已有 host 标量状态与对象 id；该文件不构成 geometry、有限性或数值证明。后续同步、实际 reader665、cache、finite 与 receipt 的完整 AST 保持 v2 原样。

CPU 控制执行取回并按 SHA 冻结的真实 `UniProcExecutor.collective_rpc`、`run_method`、`WorkerWrapperBase.__init__/__getattr__`、`get_forward_context` 方法体 AST，及 v3 adapter/入口门、原 audit capturing helper。仅用 CPU worker payload 隔离设备部分，RPC 传参规则不替换。覆盖旧 callable-wrapper 失败、新 string-worker 通过、错 owner/audit/capture 拒绝、capture 先拒绝且无 I/O、一次 host receipt 与原 reader body AST 不变。其余 34 个 CPU 控制沿用合法 `[2044,4]` chunked prefill 和原 SWA/native builder/正常 HBM 门。

配置保持 2048 输入/256 输出、SWA128/K5、max len4096、batched2048、capture3072、GPU budget0.70/KV2GiB，以及 43 个 before-capture owner buffer、每请求 source/cache seal、每个真实 round 的 43 route/weights、proposal、3 attention 和 9 次独立比较。保留所有 partial/initial/seed/terminal native draft 调用与终态 fence。capture 不读设备值；性能关闭、dummy quality false、tiny/formal admission=false。

当前只授权 CPU 准备，NPU 权限为 0。新 readiness 提供 `dspark_lane_20261011_c4_graph_geometry2k256_v3` 的未来窗口命令；主线程另行分配 fresh physical4 后才可启动，不借用 v1/v2 geometry 或数值结论。
