# 2048 输入 / 256 输出独立 geometry/cache 门：CPU 已准备，NPU 未授权

本目录是下一窗口的独立代码，排除在 timing v2 的 85 文件交付之外。旧 v40、timing v2 源码和结果保持原字节。主线程已独立核验本代理为 `gpt-6.1-sol/max`；本目录不创建或启动其他代理。

读取真实模型配置确认 SWA window=128、K=5、target40+MTP3、hidden5120、8 专家/top2。读取并封印真实 native cache registry，确认 DraftSWASpec 使用 SlidingWindowManager，block 0 是 null block。新门不会扫描或要求已经淘汰的全前缀缓存有限，也不会沿用 128/32 的 16 步或两页结论。

预声明配置是 `max_model_len=4096`、`max_num_batched_tokens=2048`、capture 代表 `seq_len=3072`、GPU memory utilization=0.70、KV cache=2 GiB。代表值加 K 为 3077，落在模型长度内；正常内存预算保持。启动器在 initial、SDK 前、模型前都重新检查 physical4 健康、无进程、容器映射及当前 HBM；可用 HBM 至少要达到总量的 0.70，随后仍执行原生正常初始化预算检查。旧 release 证据不作为新门的资源授权或预算证明。

固定 initializer 仍是 SHA256 `7dfa11336f7be4c8c55d0f09eb302673ef31ac51d582686d31cffc432af6fe78`，在所有 compile/profile/capture 前注册全部 43 个派生 owner buffers。每请求前后检查 source/cache/finite/view/value seal，不改原始权重，不在 capture 后回填缓存。

计划仍为 2 warmup、3 组 OFF A/A、3 组 OFF/ON、3 组 ON A/A，共 20 请求和 9 项数值门。每请求 2048 输入/256 输出，actual round 数从新 receipts 推导；同时硬核对每个 round 的 native `_propose`、完整 initial/terminal、cold helper 调用/成功返回、真实 target/draft replay 与最终设备 fence。比较 token、Top5/logprob、43 层有序 route/weights、proposal 和三层 attention，保持原生串行 hidden/pre_mix 链。

新增 probe 只在捕获期保留实际 SparseAttnSharedkv 的原始输入引用，原样转发参数和返回值；捕获中不读设备值。每个真实 step 完成后，依据三组实际 query positions、seq_len、kernel block size、live/native block table 和实际 sparse indices，检查当前可见的 SWA128+K5 共 133 逻辑位置。五个 query 保留全部 665 个重复读槽，另 615 个 padding 项严格依据原生 `-1` ABI 排除，不使用 finite 筛选；active null block、旧块表宽度、stale capture positions/seq_len 一律拒绝。

旧 audit 的 finish 读取 `0..max_position` 所有页，会跨越 SlidingWindowManager 已淘汰的区域。这里仅替换诊断 finish/compare：按实际可见逻辑位置比较全 step 的三组 native cache 值，保持全部原始 forward、参数、算子和调度。性能阶段完全移除，dummy clean=0、tiny/formal admission=false。

CPU 门使用真实冻结 native `build_dspark_swa_indices` 在 CPU PyTorch 上交叉检查 2053/2309 边界，验证 43/128/255 三种动态 round 数，捕获引用原样转发、旧两页表、必读槽被过滤、active null page、stale positions/seq_len 等负控。另有当前 HBM 3289/7465 正控和 24000/缺失预算负控。CPU 容器设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、visible devices 为空，断言未导入 torch_npu/acl，未初始化 NPU SDK。

尚待新 NPU 窗口证明的内容是三组实际 capture reader 绑定、每组块表刷新及 index 地址稳定、初始逻辑列 15/16 与尾部 17/18 访问、每真实 step 的全部 665 读槽有限且 A/A/OFF/ON 逐字节一致。CPU 通过不能代替这些证明。

CPU 命令：`python3 run_cpu_controls.py`，随后 `python3 launch_geometry_gate.py --job dspark_lane_20261011_c4_graph_geometry2k256_v1 --prepare`。未来建议启动命令为 `python3 launch_geometry_gate.py --job dspark_lane_20261011_c4_graph_geometry2k256_v1 --launch`；仅在主线程明确分配新 physical4 窗口后执行，当前 NPU 权限为 0。
