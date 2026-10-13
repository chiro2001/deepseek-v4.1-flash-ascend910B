# DP2TP4 v4 初始化 OOM 与恢复候选（2026-10-13）

这是失败诊断和候选准备收据，不是新精度/性能 PASS。
唯一现场操作者只读核查目标容器与 kernel；未停止外部任务、清 pagecache、
重启容器或占 chip0–7。旧原件未覆盖。

## 失败证据

目标 CID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`。
v4 job `d24h` 的 run.exit=1，engine/ASGI ready 均 false，没有 HTTP/canary/arm/metrics。
DP1 父进程 08:23:20 UTC 已开始 shutdown；随后三个 worker 的 TERM/KILL 和 DP0
ready pipe 的 BrokenPipe 都是退出链后果。EP5 日志提前截止，但不能精确指认被杀 host PID 的 TP rank。

`sudo -n journalctl -k --utc` 的时间戳原件确认 08:23:20.008750 UTC：
host PID1509209（VLLM::Worker_DP，task_memcg 为上述 CID）被 OOM killer 杀死。
anon-rss=29,022,164 kB，file-rss=1,528,436 kB。
约束为 `CONSTRAINT_MEMORY_POLICY,nodemask=6,global_oom`。
触发 cpuset 是外部容器 `5985c930…`，caller 是 `numa_6_t_11`、CPU502、PID1516900；
被杀 task_memcg 才是本项目。不能用触发者的 nodemask 反推本项目 worker 自身策略。
目标 cgroup unlimited/failcnt=0；持久 oom_kill=1 与历史 max_usage 无时间戳，不单独用于归因。

新原件目录 `/home/chiro/tmp/dsv41-dp-matrix-20261013/v4-oom-research`，
filtered kernel SHA `440a46df…0b60720`，完整458行事件约76KB、SHA `97fa22c8…e7a9b22c`。
后续 archive 保留原件及完整哈希清单。宿主与本机时钟有数分钟差，按原件 UTC 记录。

## CPU-only 内存策略门

宿主/SSH shell/容器 init 允许 nodes0–7，target cpuset 不限制单节点；numactl 默认 policy=default。
全机 MemAvailable 曾约1.48TiB，node6 free 同期约6.3GiB；free不是available。
现场 vLLM numa_bind 默认 false，Ascend enable_cpu_binding 实际 false。
Engram host shard 是普通 CPU tensor（pin_memory=False）；小 staging buffer 的 native
torch_npu allocator 行为未作设备验证。

CPU-only `numactl --interleave=0,1,2,3 python3 -S` 成功：get_mempolicy mode=3、mask=0xf；
至多1MiB匿名映射触页后，numa_maps 显示256页分布于0–3各64页。
无 torch/site import、无 NPU 初始化/模型加载；映射释放后进程自然退出。
只证明系统允许该 CPU 策略，不证明设备初始化、模型加载和推理后仍满足策略。

## 新候选与边界

v5 包在 `/home/chiro/tmp/dsv41-dp-matrix-20261013/package-dp2tp4-v5-numa`，
remote `/work/dp_matrix_20261013_dp2tp4_v5`、新 job `d24k`、model `dsv41-dp-dp2tp4-v5`。
DP2/TP4/DCP1/EP8、MAXLEN196608/util0.85/自动KV、Engram/BI/async/DSpark5保持不变。
只新增启动资源策略：loader threads128→16、host NUMA interleave0,1,2,3。
不改变权重精度；host落点可能影响性能，不当成严格原配置 A/B。

包117,558B SHA `9d62a056e3fc04a7c2b100723ce9dae894100286539227ce6ac0de7e3e123060`；
launch4,682B SHA `79c8477749719187639f97eaa8c557de023bca887d5409a5edb095d808abb557`。
策略进入计划/指纹，numactl失败不回退；service/worker加载前及describe验证实际get_mempolicy，
不以env声明当实测。最新 CPU 回归148 passed /349 subtests passed。

最新 chip8–15 被外部容器 `dsv4-flash-tp8ep8-npu4-7-20261013` 的八个 TP worker
占用，每 chip约55GiB；cgroup 与本项目不同。仅核 PID/NSpid/CID，不发信号。
未运行 NPU 继承探针、v5初始化或新质量/性能；本项目仍没有ready API。
须释放/预留8–15后重核现场门。DP4TP2、DP4TP2DCP2仍是未启动的独立候选。

## 本回合最终现场门

v5与cl4已分别stage到上述v5路径及 `/work/dp_matrix_client_20261013_cl4`，未启动服务。
v5包内SHA全过、launch bash-n通过，三项native guard与现场SHA匹配。
cl4三包与本地冻结SHA一致、Code8 manifest匹配。
新包的 `dp_matrix.host_memory.verify_interleave` 在远端 `python3 -S`、numactl0–3下
直接调用成功：mode3/nodes[0,1,2,3]/policy_verified=true，未加载torch/torch_npu/vLLM。
这是CPU门，不声称设备context之后仍生效。

最终唯一复查（宿主2026-10-13 09:03:04 UTC）外部CID
`31b1f424966473427bdf9895c00d967dd597b65424427e88b005fcf58d223642`
仍占8–15，整chip HBM用量58,023–58,287MB/65,536MB（其中worker约55,193–55,199MB）。
本项目目标容器running；v5未启动、无ready API、没有请求/设备探针，不再后台轮询。
全目录原件归档 `dp2tp4_v4_oom_v5_preflight.tgz`，SHA
`945b545653ef3fb4a68294b23fe6b7b6cfefdb3e9be3c492aebe5f194b69b257`；
原目录保留，逐文件evidence.sha256已核验。
