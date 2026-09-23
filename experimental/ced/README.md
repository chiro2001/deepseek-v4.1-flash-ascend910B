# CED-PD 连接器实验副本

`mooncake_hybrid_connector.py` 取自 A3 固定镜像
`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`，原文件 SHA-256：
`94593f29efe54986288205bcab8b93174e25d0a6e24e739c2f80a9a0bc9e4214`。
对应容器内路径：
`/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`。
实验服务只在设置 `V41_CED_ROLE` 且 `PATCH_MODE=mount` 时挂载本副本；普通服务不使用它。

当前变更只处理 **P 侧缓存组有效性**：P 跳过层 20–39 后，将 G7–G11 的
`remote_block_ids` 置空，并在交接元数据写入 `ced_replay_tokens=128`、缺失组、
有效前缀长度。收到该标记的 D 端在 SWA replay 实现之前直接拒绝请求。
这防止把未写入的上半层 SWA 当成有效缓存交给普通 D。尚未实现 D replay。
