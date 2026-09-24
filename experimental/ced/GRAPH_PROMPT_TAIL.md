# CED D：只让末尾 prompt token 走 eager 的图模式诊断

## 证据与假说

A3-21 真实权重 TP8+TP8、BF16 KV、`MAX_LEN=1048576`，同一 P 和同一
22-token 请求下，D 全 eager 的 `max_tokens=1/2` 分别输出 `Z`/`ZQ`；
D `FULL_DECODE_ONLY` 分别输出 `正确答案`/`正确答案只有一个`。两臂均
HTTP 200，且都重放位置 `0..20`。原始证据见主实验分支的
`evidence/ced_graph_ab_20260924/`。

D 从 P 加载前 `N−1` 个 token 的 KV 后，先重放这段，然后单独计算尚未
缓存的第 `N` 个 prompt token。单卡 trace 显示末 token 的 attention 元数据
仍是 `num_prefills=1, max_query_len=1`，但当前 runner 在建立元数据前按
单 token 形状选择 FULL 图；捕获时使用的是 decode dummy batch。
这提示单 token prefill 误入 decode 图，**尚不是根因定论**。跨流 event
依赖是否也有问题，需结合 profiler 排查。

## 受控改动

`core_model_runner_prompt_tail.patch` 只改动固定镜像的
`vllm_ascend/worker/model_runner_v1.py`。仅在
`V41_CED_ROLE=decode`、`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` 且批内有尚未
算完的 prompt token、最大调度长度为 1 时，给图调度器传 `force_eager=True`。
生成阶段的 `num_computed_tokens >= num_prompt_tokens`，继续使用 FULL 图。
混合批有任一请求处于这种状态时，整个批次走 eager；这是正确性优先的
诊断行为，后续需量化并发开销。

启动器在容器起服务前校验 runner 基底 SHA-256
`67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e`，
再现场应用小补丁；不覆盖整个 runner 文件。诊断臂 D 额外设置：

```bash
V41_CED_GRAPH_PROMPT_TAIL_EAGER=1 GRAPH=1 EAGER=0 \
  bash scripts/serve_a3_ced_pd.sh decode
```

P 保持原实例；端口、芯片、模型目录和其他参数须与此前 graph 失败臂一致。
首先对同一 SHA 的 `max_tokens=1/2` 短针取证；若恢复，再跑完整短针，
并检查日志同时出现 `[CED-GRAPH] one-token prompt tail forced eager` 与
`Replaying aclgraph`。只有这两个运行路径及答案同时成立，才说明补丁
保留了生成阶段的图执行。此包尚未真机验证，不能作为交付配置。
