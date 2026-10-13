# DP2TP4 v3 原始证据归档

下列 archive 是原任务目录的副本，原目录和原件未删除、未改写。tar 内保留原始
`evidence.sha256`、HTTP/body、日志、PID/cgroup/NPU 证据与阶段文档；原清单里的绝对
路径仍指向旧目录，解包后可按 basename 重新核对内容 SHA。

| 原路径 | 仓库 archive | bytes | archive SHA-256 |
|---|---|---:|---|
| `/home/chiro/tmp/dsv41-site-runtime-v3-audit` | `dp2tp4_v3_ready.tgz` | 103027 | `ae4aebb120885a8044bf8d80da47a8dbac058e8b55e97dc451dc2d2aa0b0b481` |
| `/home/chiro/tmp/dsv41-dp-matrix-20261013/canary-v3/dp0-failure` | `dp2tp4_v3_canary_fail.tgz` | 106338 | `89ec3ad7cd61c957281b1eafadbfffa3100412e363c18786d1fc02d3ed1cce8a` |
| `/home/chiro/tmp/dsv41-ced-pd-v4-bench/client_source_ced_20261013_v2/delivery_matrix/{benchmark.py,workloads.py}` | `frozen_code_sources.tgz` | 34073 | `0cf739cf49423a3c2c696b73081bed1d8cc5a8cc37323f162ee445dbf0c9c922` |

v4 初始化失败原件也已复制归档：
`/home/chiro/tmp/dsv41-dp-matrix-20261013/v4-preflight` →
`dp2tp4_v4_init_fail.tgz`，SHA-256
`aca9b704cfd0306f90ec8219115b74373f38b60c67785a013dda6980dcadd430`。
保留原说明及清单；其中“EP4 是首个失败”的历史判断已被完整时序纠正，不能当成根因。
后续 kernel/NUMA 独立收据见 `V4_OOM_AND_RECOVERY.md`。

后续只读 OOM/NUMA、外部占卡映射与 v5/cl4 staging 原件：
`/home/chiro/tmp/dsv41-dp-matrix-20261013/v4-oom-research` →
`dp2tp4_v4_oom_v5_preflight.tgz`，SHA-256
`945b545653ef3fb4a68294b23fe6b7b6cfefdb3e9be3c492aebe5f194b69b257`。
该目录的 `evidence.sha256` 已逐文件校验；原路径和原件仍保留。

就绪 SID=`a77dcfc1c52346749bd7c2fce76638da`，runtime fingerprint=
`c160fc637cc3305cdf92a2a614a5b3595fbf619e8074393238109ef33aa1ad38`。
该实例是 DP2TP4/DCP1/EP8、util0.85、自动 KV、SEQ1、Engram 开启且 device-index
快路径关闭。v3 初始化和容量 profiling 成功不替代精度/性能或长上下文验收。

唯一 DP0 canary 的旧 reduction FAIL 保留。HTTP/SSE 完整、clean，四 TP 各135
decode rounds、135 graph calls、1 eager call；旧接口没有输出 per-phase 原件。
v4 新增从实际 counter 的 initial/decode proposal 列表派生的收据，独立验收，不覆盖
该 archive。DP1 没有 v3 canary 请求，未用旧 TP8 或 DCP8 结果替代。

冻结 Code8 源 archive 是可复现所需的字节原件：benchmark.py SHA=
`33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816`，
workloads.py SHA=`62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35`。
新 DP 回放器只调用其普通 stream 函数和隔离评分函数，不运行旧 TP8 private API 流程。
