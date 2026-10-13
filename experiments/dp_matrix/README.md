# 八 chip 原生 DP 测试（2026-10-13）

用户最新授权范围为 a3-21 physical0–7 的 DP2TP4、DP4TP2、DP4TP2DCP2；
此前8–15窗口及下面的v3/v4/v5均保留为历史，不能继承到新的0–7 SID。
旧 CED-PD/DRAM 大目标仍暂停。现有 TP8 结果仅作历史参照，不继承精度或性能验收。

三种几何的 `DP × TP = 8`；DCP2 在 TP2 组内部划分 KV，不额外占用芯片。

服务/prepare/freeze/canary/benchmark/Code8 replay/score均提供 `--physical-chips`；
只准两组完整、顺序固定的0–7或8–15，拒重复/混杂/自动选卡。为兼容旧入口默认仍为8–15，
本轮命令必须明确 `--physical-chips 0,1,2,3,4,5,6,7`。
SDK/model/draft/counter/fence原件不变；新derivedworker仅把旧硬编码物理assert替换为
严格runtime guard，并保存源SHA/manifest。实际PID/NPU/cgroup仍独立取证。
当前0–7首轮chip4有外部 `xh-mk` 进程，未启动NPU探针或新服务；容器/source门正在核对。
沿用基线已有的 EP 开关，预期 expert 通信组跨八个 rank，不能用独立 TP 副本冒充
原生 DP。实际通信组、PID/cgroup、路由及每个副本的质量必须在新服务中核验。

当前阶段：DP2TP4 v3 曾真实初始化，现已受控停止；v4 权重加载/worker-ready 阶段
初始化失败，`run.exit=1`，engine/ASGI ready 均为 false，目前没有可用推理 API。
目标容器仍 running/restart=0。v2/util0.90 首次初始化因可用显存门失败。
v3/util0.85 使用自动 profiling，SID `a77dcfc1c52346749bd7c2fce76638da`、
runtime fingerprint `c160fc637cc3305cdf92a2a614a5b3595fbf619e8074393238109ef33aa1ad38`。
8 worker 的 PID/NSpid/cgroup 与 physical8–15 已独立对账，TP 两组和 EP8 正确。
启动日志每副本逻辑 KV token 等价容量为 849714/849648，num_blocks=12824/12823，
每 rank backing storage=6936860672/6936319744 bytes；这不是长请求验证。
原件在 `/home/chiro/tmp/dsv41-site-runtime-v3-audit`。

DP0 唯一 canary 的 HTTP200/DONE/length/usage2048+256/clean 均通过；四 TP 各135
verifier rounds、135 graph calls、1 eager call。旧 reducer 把初始 prefill eager 一并
拒绝而报 FAIL；原件在 `/home/chiro/tmp/dsv41-dp-matrix-20261013/canary-v3/dp0-failure`，
不改写该失败。DP1 未执行。v4 候选新增从真实 counter.requests 导出的分阶段草稿
收据，严格要求每个 decode round 全图、无 eager fallback，单独核 initial prefill。
v4 已实际部署但初始化失败，未调用新增 describe/capacity RPC，未发 HTTP/canary；
其余两个拓扑尚无实机结果。v4 原件在
`/home/chiro/tmp/dsv41-dp-matrix-20261013/v4-preflight`。
原归档说明将 EP4 的 terminated 误作首个失败；按完整日志，DP1 父进程 08:23:20 UTC
先 shutdown，08:23:25/29 才 TERM/KILL 存活 worker。EP5/PID2169866 的日志在
08:22:08 截止、无加载完成或 Python traceback。后续 `sudo -n journalctl -k` 只读日志
确认 08:23:20 UTC，本 CID 的 host PID1509209 被 node6 `global_oom` 杀死，约束为
`CONSTRAINT_MEMORY_POLICY,nodemask=6`；缺少完整 PID 对应原件，不能精确绑定某 TP。
触发 cpuset 属于外部容器 `5985c930…`，被杀 task_memcg 属于本项目；不可把触发者
的 nodemask 当成本项目 worker 自身的 mempolicy。目标 cgroup unlimited/failcnt=0；
全机 available约1.48 TiB，一次复查 node6 free约6.3 GiB；free并非available。
本次归因来自有时间戳的 kernel 事件，不是无时间戳的 cgroup `oom_kill=1`。
原文件及 SHA 保留，不覆盖历史说明；内存策略/NUMA 恢复方案尚待现场门。

v5 新候选已冻结：loader threads16、host NUMA interleave0,1,2,3，原数学条件不变。
CPU-only 1MiB探针验证策略和页分布；服务及worker加载前校验实际策略，失败不回退。
最新 physical8–15 被外部 `dsv4-flash-tp8ep8-npu4-7-20261013` 占用，各约55GiB；
因此未做 NPU 继承探针/v5初始化/新canary。先协调释放并预留窗口，不停止外部任务。
见[独立诊断与恢复收据](evidence/V4_OOM_AND_RECOVERY.md)。

`V41_ENGRAM_DEVICE_INDEX=0` 的实际语义是**关闭 device-index 快路径**，不是选择
NPU ordinal 0。Engram 本身启用且配置层为 [1,14]；v3 有成功模型/完整 Engram 权重
加载链证据，没有直接对象/设备 consumer RPC。v4 新增对象/group/device 元数据，
不读 tensor 值，也不以 metadata 冒充 device consumer proof。

v1 包保留为 CPU 准备证据，禁止执行旧 v1 launch。
`topology.py` 只提供 CPU 预期布局与身份失败关闭门，不证明设备可用或正确性通过。
最终性能和容量结论以实际冻结收据为准；不修改旧客户端及旧原始结果的 SHA。

v2 服务快照：`dp_matrix.tgz` 63,469 bytes，SHA-256
`cab8ee5ca56684180c87ec215fd2ac4448bd038137ce58e065a0fe10a34118af`。
部署候选为 `dp2-tp4`，MAXLEN196608/BAT8192/每副本SEQ1/PREFIX0，
Engram on/device-index快路径off、DSpark5/真实 draft graph/BI/async/双流不变；KV 参数 unset，
GPU utilization=0.90，使用实际自动 profiling，而不是旧固定4GiB池。
新路径 `/work/dp_matrix_20261013_v2`，job `d24b`，model
`dsv41-dp-dp2tp4-v2`；路径及计划不是在线证明。

通信组成员校验依据目标 D 实际 `parallel_state.py`（SHA
`bbad6969765f3c5eb48ec1b42c458464ba58841c26244a6d42f293ec199bcbe2`）
的 `ExternalDP × DP × PP × PCP × TP` 布局，检查 TP/DCP/EP 的实际全局
rank 列表；physical IDs 8–15 仍需独立 PID/NPU/cgroup 证据，不能由 rank 声明推出。

新独立 `benchmark.py` 定向使用 native DP header/tag，分别保存每请求 HTTP/SSE、
arm baseline、actual TP receipts 以及 HTTP/server-fenced 三元组与 worker-prefill。
输入只读复用历史八条地火字节与 SHA，不继承历史测量或质量结果。
`replicas` 模式是每 DP 一请求波次，**不证明最大饱和吞吐**；高并发多请求
counter/graph 与饱和扫描仍需另行实现。编程性能和严格质量执行也尚未完成。
`measurement.py` 只做 CPU 归约；最终 completion-minus-seed tokens 不跨 TP/DCP
相加，actual verifier rounds 不替换成估计步数。

冻结时 v2 服务相关 CPU 门为 85 项通过。最新工作区 pytest 为
148 passed / 349 subtests passed（含加载线程限制和实际 NUMA 策略门）；
这是 CPU 合同证据，不是实机精度或性能 PASS。

独立质量 proxy 流式 tee 保存 request gzip 和 response 原始字节，校验 rank 回显；
Code8 已找到 SHA 匹配的冻结 client/scorer，`code8_replay.py` 只消费原 split prompts，
`code8_score.py` 使用原版 workloads 的 bwrap 评分器，尚无新模型评分结果。
DCP2 的专用 manifest/启动前 bootstrap/Engram contract 已接入本地候选；没有启动或
继承 DCP8 精度。旧 CPU/部署包均保留，当前工作区改动不自动进入已冻结在线包。

CPU 门（零 HTTP/NPU）：

```bash
mkdir -p "$HOME/tmp/dsv41-dp-matrix-20261013/cpu-tests"
TMPDIR="$HOME/tmp/dsv41-dp-matrix-20261013/cpu-tests" \
  python3 -m unittest experiments.dp_matrix.test_topology
```
