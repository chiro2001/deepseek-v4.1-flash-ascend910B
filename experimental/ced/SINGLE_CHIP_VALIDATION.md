# A3-22 单卡对拍通道

先前的框架在 `a2/agents/L1_dummy/` 和 `a2/logs/010-offload-tiny-dummy.md`、
`013-single-chip-fix-verify.md`。它的 `make_tiny_config.py` 只缩 MoE 专家数、
中间宽度、LoRA rank 与 RoPE 表，并关闭 Engram/DSpark；仍保留 40 层、
来源层 2/8/14/20、C2 环和 10 个 SWA 组。A3-21 现存的 6.1 MB
`model-tiny` 已以 SHA-256 为
`04086dd074dd0e350b4bfc1a4e9736af12df491735767b84eb313ba5486f96a1`
的压缩包复制到 A3-22 的
`~/projects/dsv41-ced-singlechip/model-tiny`。它以 `--load-format dummy`、
TP1 和固定 `--seed 0` 起服；dummy 权重不证明真权重的语义质量。

A3-22 当前空闲的是 Phy-ID **6、7**。用户预留 **chip0 和 chip1**，本实验
的 [启动脚本](../../scripts/serve_a3_ced_single.sh)仅允许使用 6 或 7。
P 使用 6、D 使用 7，各自独立容器与端口 18960/18961；代理拟用 18962。
全 40 层基线在停掉 D 后复用 chip7，端口 18963；保持同一 tiny 配置与 seed。

```bash
MODEL="$HOME/projects/dsv41-ced-singlechip/model-tiny" \
  bash scripts/serve_a3_ced_single.sh prefill
MODEL="$HOME/projects/dsv41-ced-singlechip/model-tiny" \
  bash scripts/serve_a3_ced_single.sh decode
# D 停止并确认 chip7 空闲后：
MODEL="$HOME/projects/dsv41-ced-singlechip/model-tiny" \
  bash scripts/serve_a3_ced_single.sh baseline
```

此包装器使用 BF16 KV、12 组、`SPEC=0`、`ENGRAM=0`、`MAX_LEN=8192`、
`KV_CACHE_MEMORY_BYTES=1 GiB`，保留本分支的 CED 模型/连接器/Core replay
补丁。`serve_v2.sh` 的 `QUANTIZATION=none` 只作用于 tiny dummy；默认仍为
`ascend`，并显式跳过 dummy 情况下无意义的 lazy safetensors 参数。

先对拍长度 1/127/128/129/256/4096 的同一 token 序列：同进程多次运行的
噪声地板、普通全 40 层输出、CED P→D 输出，检查生成 token/logprob、
层 20 来源主 KV/Indexer K、G7–G11 重放后的有效页和 D 的 replay 起点。
短 prompt ≤128 若随机权重相同，应能建立数值基线；长 prompt 的 bounded
replay 本来允许近似，须报告差值分布并回到 8+8 真权重验收。尚未声称此
单卡精度已通过。

## 第一轮 1+1 实测（`b6ca283`）

- A3-22 Phy-ID 6 的 P 和 Phy-ID 7 的 D，dummy tiny、BF16 KV、`SEED=0`、
  `MAX_LEN=8192`、1 GiB HBM KV；代理端口 18962。0/1 预留卡没有使用。
- 用固定 8-token 模板循环构造长度 2、127、128、129、130、256、512、4096，
  各发两遍 `/v1/completions` 且取 top-20 logprobs，**16/16 HTTP 200**、
  每个长度两次选中 token 和其 logprob 一致。原始响应摘要在
  [`pd_precision.json`](../../evidence/ced_tiny_pd_b6ca283/pd_precision.json)。
  这是 PD 自身的重复性门，尚不是与全 40 层基线的精度比较。
- 长度 1 返回 HTTP 502，D 的 `scheduler.py` 报
  `CED replay prefix boundary mismatch: prompt=1 loaded=1 declared=1`；
  P 不截去单 token，当前 D 校验只覆盖 `N>1`，且异常使 D 引擎退出。
  [`pd_n1_error.json`](../../evidence/ced_tiny_pd_b6ca283/pd_n1_error.json)
  和 [D 原始日志](../../evidence/ced_tiny_pd_b6ca283/d_serve.log.gz)
  保留失败证据，日志解压 SHA-256 为
  `7ea66d979d96e3ae0c03664309d82ee48ea51c55a6e49a6d939d99d76e2c3a3e`。
  D 和代理已停止，P 仍健康；接下来先在 chip7 跑全 40 层基线。

## 全 40 层基线 API 对照（`ccf5762`）

在同一台 A3-22 上停止 D/代理后，用 chip7、同一个 tiny 模型、dummy 权重、
`SEED=0`、BF16 KV 和同一批显式 token IDs 启动无 CED 角色的全 40 层基线。
两臂的长度 2、127、128、129、130、256、512、4096 各重复两遍：
**16/16** 选中 token 相同、选中 token 的 logprob 相同、top-20 候选集合
均为 **20/20** 相同；共同候选的最大 logprob 绝对差
`9.5367431640625e-7`。原始 [基线响应](../../evidence/ced_tiny_pd_b6ca283/baseline_precision.json)
和[逐行对照](../../evidence/ced_tiny_pd_b6ca283/pd_vs_baseline.json)已归档。
这证明 API 可见 top-20 数值接近，仍不足以证明所有 SWA 页逐值相等。

下一诊断已在工作树加入按指定位置导出 40 层 SWA 行以及层 20 全局 KV、
Indexer K/scale 的开关 `CED_SNAPSHOT_POS`，并提供 NumPy 对照脚本。
它**尚未上卡**；应先在位置 126 对齐短 prompt 的 D replay 与全 40 层基线，
再决定是否需要修订 replay 起点的注意力掩码。
