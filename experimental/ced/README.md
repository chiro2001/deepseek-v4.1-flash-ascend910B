# CED-PD 连接器实验副本

`mooncake_hybrid_connector.py` 取自 A3 固定镜像
`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`，原文件 SHA-256：
`94593f29efe54986288205bcab8b93174e25d0a6e24e739c2f80a9a0bc9e4214`。
对应容器内路径：
`/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`。
实验服务只在设置 `V41_CED_ROLE` 且 `PATCH_MODE=mount` 时挂载本副本；普通服务不使用它。

真实权重的 A3 双 TP8 实验请使用 `bash scripts/serve_a3_ced_pd.sh prefill`
和 `bash scripts/serve_a3_ced_pd.sh decode`。包装器会设置并校验
`V41_CED_ROLE`，同时关掉尚不支持的 DSpark、prefix cache 和 draft graph。
`scripts/serve_a3_pd.sh` 是全 40 层 PD 基线入口；直接用它启动 CED 的 D 侧
若漏设 `V41_CED_ROLE=decode`，服务仍可返回 HTTP 200，但不会安装 replay
调度与缺失 SWA 处理，回答可能严重错误。启动后须核对 D 容器环境为
`V41_CED_ROLE=decode`，且挂载了 `ced_scheduler_replay.patch` 与实验版
`dsa_v41.py`。

数值排障可在两个角色启动时设置 `CED_SNAPSHOT_POS=<绝对token位置>`，
导出 D 的逐层 SWA、层 20 全局 KV/Indexer 行；另设
`CED_H20_SNAPSHOT_POS=<同一位置>` 导出 P/D 进入层 20 前的多流隐状态与
`pre_mix`。两组快照分别写入本次 `RUN_ID` 的 `snapshots/` 和
`h20_snapshots/`，只用于单请求隔离实验。完整模型基线可通过
`V41_CED_ROLE=''`、`KV_ARGS_EXTRA=''` 的 `scripts/serve_a3.sh` 启动并
显式传入 `V41_CED_SNAPSHOT_*`、`V41_CED_H20_SNAPSHOT_*`；它们的目标位置
应与 P/D 相同。`tools/compare_ced_cache_snapshots.py` 和
`tools/compare_ced_h20_snapshots.py` 可生成逐层数值差异。

当前变更只处理 **P 侧缓存组有效性**：P 跳过层 20–39 后，将 G7–G11 的
`remote_block_ids` 置空，并在交接元数据写入 `ced_replay_tokens=128`、缺失组、
有效前缀长度。这防止把未写入的上半层 SWA 当成有效缓存交给普通 D。

下一版 D 原型增加 `core_scheduler_replay.patch`：以 P 的 `N−1` 命中为起点，
先调度最多 128 个旧 token，再单独调度未缓存的末 token。D worker 对未传的
G7–G11 本地页清零；`dsa_v41.py` 在 replay 块内只重建 SWA，不回写已收到的
全局 KV/FP32 环。`dsa_v41.py` 的基底来自同一 A3 镜像，原文件 SHA-256：
`56d3ce83098babf585289d3e078d1efff0442547f61fb0955a11f0b42bfc6ba3`。
调度补丁基于该镜像应用 admission gate 后的 live `scheduler.py`，基底 SHA-256：
`533eed493cb307e6d4423ff550910278f6434d71f00581737ce420d60298e8bc`。
该 D 原型尚未在 A3 验证；注意力对 replay 起点的 SWA 截断仍需单独实现或证明
现有物理页窗口足够，详情见 `D_REPLAY_NOTES.md`。
