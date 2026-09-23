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

```bash
MODEL="$HOME/projects/dsv41-ced-singlechip/model-tiny" \
  bash scripts/serve_a3_ced_single.sh prefill
MODEL="$HOME/projects/dsv41-ced-singlechip/model-tiny" \
  bash scripts/serve_a3_ced_single.sh decode
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
单卡实例已启动或精度已通过。
