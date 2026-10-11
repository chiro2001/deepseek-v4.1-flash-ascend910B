# 固定 owner fixture 的 128/32 机制计时诊断

本目录独立于 v7 和 graph_fixture_v1 冻结交付。`frozen_shared/` 中 6 个共享源逐字节复制；固定缓存初始化器仍为 SHA256 `7dfa11336f7be4c8c55d0f09eb302673ef31ac51d582686d31cffc432af6fe78`。它在正常 dummy 初始化之后、所有 compile/profile/capture 之前按原生公式注册 target40+draft3 的非持久 owner buffer，原始 forward、算子和参数保持。

计时 worker 不安装 route/logprob audit、first-value trace 或 profiler。`DSPARK_FIXTURE_VALIDATE_CONSUMERS=0` 禁止逐层缓存 D2H；诊断读值、哈希、参数 metadata 和文件写入 helper 在活动窗口内均会硬拒绝。主机计数器保留 `_propose`、实际 native helper、真实 target/draft graph replay 和请求末尾 NPU fence。原生 scheduler/bookkeeping 自身需要的 CPU 数据路径保持，守卫不声称禁止这些必要读取。

每请求开始前和终态 fence 完成后，分别检查全部 43 个 owner/source/cache 的有限性、绑定、view 和字节值封印。这些大检查均在计时窗外。decode 窗口从初始 target bookkeeping 返回开始，覆盖 initial draft、全部 16 个 target decode 和 terminal 已提交 draft，直到 `read_receipts` 的设备完成 fence。分母按真实 16 decode/31 输出 token 计算，完整请求仍是 128 输入和 32 输出。

预声明 2 个 warmup、8 个 OFF/ON 配对，即计数请求每臂 8 个；配对次序交错 OFF/ON 和 ON/OFF，同一 pair 的 prompt/seed 完全相同。新运行独立核对 17 个实际 round、K5/6-token verify 几何、initial/terminal 各 1 次 draft、冷 helper 调用/返回 1/1，以及 OFF 17 个 helper、ON 1 个 initial eager helper和 16 个 draft replay。每请求和按总时间/总步数/总 token 聚合的 `(ms/step, A, tok/s)` 都须自洽。

输出必须与冻结的 128/32 token 参考一致，并单独报告周期 2 复读。所有 dummy clean=0、tiny/formal admission=false；时延数字只描述这个 tiny 机制诊断，不能按复读 A 宣称收益，也不构成正式 TP8 全 main+MTP+verify+scheduler/cache ≤20 ms/step。

CPU 门验证诊断调用的窗口拒绝、同一冻结 fixture 在关闭逐层读值时原样转发 native 调用、helper 参数/返回保持、冷 capture 计数、丢失 initial/terminal、错误 verify 几何、缺失 replay/fence、foreign 地址和三元组计算。CPU 门不构成 NPU 证明。

先执行 `python3 check_fixture_timing_cpu.py` 和 `python3 launch_fixture_timing.py --freeze-only`。冻结 SHA 和完整启动命令 ACK 发给主线程后，用 `--launch` 启动一个全新的 physical4 job；launcher 复用已冻结的 fresh 健康/空 owner/容器映射门，在 SDK 和模型初始化之前分别再检查。只授权 physical4，禁止 reset、dropcache 或停止未知进程；终态需再次取回 physical4 健康及释放证据。

2K 输入/256 输出是尚未执行的另一套 geometry/cache 门；此目录的 128/32 结果不为其提供准入。
