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

## 缓存行快照对照（`2df46ec`，2026-09-24）

已把 `V41_CED_SNAPSHOT_POS` 诊断挂到 A3-22 的临时隔离包
`pkg-snapshot-2df46ec`，运行前后均核对容器内 `dsa_v41.py` 的 SHA-256 为
`6f93b6eabc97dfeaaf126a7836a7eb74992275443c6dea652d5e3321e2a2bc7e`。
使用 Phy-ID 6 的既有 P 与 Phy-ID 7 顺序运行 D、全 40 层 baseline；未使用
用户预留的 chip0/1。两个位置都用同一固定 token pattern，BF16 KV、dummy
权重、`SEED=0`、`SPEC=0`。D 与 baseline 各发一条 API 请求，输出 token 和
top-20 logprobs；比较位置 126、127、254 的 40 层 SWA 行，并比较层 20 的全局
KV、Indexer K/scale。原始 NPZ、请求响应和逐层 NumPy 指标在
[`evidence/ced_tiny_cache_snapshot_20260924/`](../../evidence/ced_tiny_cache_snapshot_20260924/)。

| prompt 长度 / 位置 | D replay 范围 | API 对照 | SWA 对照 |
|---|---|---|---|
| 128 / 126 | `0..126`（127 tokens） | 选中 token、logprob 相同；top-20 20/20，最大共同 logprob 差 0 | 首个 bitwise 差异 L7：`2.98e-8` / rel L2 `2.19e-6`；L19 `1.91e-6` / `1.72e-4`，L20 `1.91e-6` / `1.60e-4`，最差 L27 rel L2 `6.26e-4`，L39 `5.01e-5` |
| 256 / 127 | `127..254`（128 tokens） | 选中 token、logprob 相同；top-20 common 19 / union 21，最大共同 logprob 差 `9.54e-7` | L0 精确；首个差异 L1：max abs `1.526e-5` / rel L2 `1.435e-3`；L19 `3.967e-3`，L20 `4.069e-3`，L21 `4.321e-3`，L39 `6.574e-3` |
| 256 / 254 | `127..254`（128 tokens） | 选中 token、logprob 相同；top-20 common 19 / union 21，最大共同 logprob 差 `9.54e-7` | 首个 bitwise 差异 L2：`2.98e-8` / rel L2 `2.44e-6`；首个 max abs ≥`1e-6` 在 L4：`7.63e-6` / `5.81e-4`；L19 `1.91e-6` / `3.15e-4`，L20 `7.63e-6` / `8.04e-4`，L21 `3.81e-6` / `5.21e-4`，L39 `7.63e-6` / `9.46e-4` |

长度256/pos127 的 SWA 差异从首个 replay 行 L1 开始并逐层累积，L39 rel L2
为 `6.574e-3`；同一请求 pos254 末行 L1 精确、L39 rel L2 为 `9.458e-4`。
这与 replay 起点窗口读取 pre-start SWA 状态的假设一致，但 NPZ 只记录最终
缓存行，不能直接证明实际读取的物理页。pos254 是 replay 最后一行，不能单独
判断起点边界。

长度 256 的层 20 `long_kv` 差为 max abs `7.63e-6`、rel L2 `6.66e-4`、
cosine `0.999999779`；`index_k` max abs `1`、rel L2 `4.27e-3`、cosine
`0.999990893`；`index_scale` 精确相同。两条 API 都 HTTP 200，D 日志没有
越界异常。结果只说明 dummy 权重下的缓存误差幅度；不能据此解释或排除
真实权重的乱码。

在长度256/pos127，层20 `long_kv` max abs `2.98e-8`、rel L2 `2.82e-6`、
cosine `0.999999999996`；`index_k` 和 `index_scale` 精确相同。该点的 SWA
相对误差增大而全局源几乎不变，说明偏差集中在局部 SWA 重建路径；绝对差仍
只有 `1.526e-5` 量级，且仅为 tiny dummy 结果。

## 单 token replay 边界修复（独立分支 `bfe00c1`）

原长度1失败来自 D 校验把 `ced_prefix_tokens` 固定要求为
`num_prompt_tokens - 1`。现在仅在独立 worktree
`fix/ced-single-token-boundary` 中增加 N=1 分支：当 P 已加载唯一 token 时，
D 校验 `prompt=loaded=declared=1`，令 cursor 回到0并重算这个 token；
`max_query_len=1` 不进入 >1-token replay fast path，因此层20正常写全局源。
patch 从 A3 P 容器中的完整 scheduler 原件生成，原件 SHA-256 为
`533eed493cb307e6d4423ff550910278f6434d71f00581737ce420d60298e8bc`；
A3-22 实际运行基于源码提交 `bfe00c1`。后续提交 `a915533` 只补 MANIFEST，
运行逻辑相同；两者尚未并入本分支。

A3-22 使用既有 Phy-ID 6 P，在 Phy-ID 7 顺序运行 D 和全 40 层 baseline；
仍为 dummy 权重、BF16 KV、seed 0、chip0/1 未使用。结果如下：

| prompt 长度 | API 及 PD / baseline 对照 |
|---|---|
| 1 | 两边 HTTP 200，选中 token `Apart`、logprob 相同，top-20 20/20 且最大共同差 0；两边 `completion_tokens=1`，D 只记录一次位置0重算，无多生成或退出 |
| 2 | 两边 HTTP 200，选中 token/logprob 相同，top-20 20/20，最大共同差 `9.54e-7` |
| 127 | 两边 HTTP 200，选中 token/logprob 相同，top-20 20/20，最大共同差 `9.54e-7` |
| 128 | 两边 HTTP 200，选中 token/logprob 相同，top-20 20/20，最大共同差 0 |
| 129 | 两边 HTTP 200，选中 token/logprob 相同，top-20 20/20，最大共同差 0 |

D 日志还确认原有 replay 区间分别为 `0..0`、`0..125`、`0..126`、`0..127`，
无 prefix mismatch、Traceback 或错误输出。原始请求、边界对照、容器信息和
服务日志保存在
[`evidence/ced_tiny_n1fix_20260924/`](../../evidence/ced_tiny_n1fix_20260924/)。
该修复通过 tiny dummy 的边界和 API 精度对照；真实权重仍需验收。

## 生产 `_C_ascend` SMLA 合成算子 A/B（2026-09-24）

在 A3-22 的隔离 chip7 容器沿生产 `vllm_ascend.utils.enable_custom_op()`
注册路径调用 `_C_ascend`，比较全局原始 KV 视图 A 与 replay scratch-local
视图 B。两臂的 `cmp_ratio=2` metadata 均通过，shape `[1024]`、dtype
`int32`；随后 A attention 返回 finite 后才运行 B，两个输出均无 NaN、无
CANN 错误。A/B 输出对 CPU 闭式参考的全张量最大绝对误差分别为
`0.001519607843137205`、`0.0015040106951871857`（覆盖全部 128 个 query
rows；证据 JSON 保存的是全局最大值，不是逐行误差向量）。

CSA 上界控制行 dim1 输出为：row125 A/B 均为 0；row126 A=`0.392578125`、
B=`0.39453125`；row127 两臂均为 `0.392578125`，与相同 CSA 和原始窗口
共同归一化的闭式值一致。此前 `cann_ops_transformer` adapter 对 ratio2 的
metadata 调用被 ratio4 checker 拒绝；本测试走的是生产 `_C_ascend` namespace，
不能把 adapter 的拒绝当成生产路径结论。

这是 **synthetic operator mechanism** 验证，不使用真实模型权重，不证明
CED/PD 的真实权重精度，也不是乱码修复；scratch-local 方案未接入生产服务。
测试容器已停止，chip6 P 保留，预留 chip0/1 未触碰。原始脚本、stdout、JSON、
容器与 NPU 占用记录及 `SHA256SUMS` 见
[`evidence/ced_smla_ori_view_ab_20260924/`](../../evidence/ced_smla_ori_view_ab_20260924/)。
该证据目录从私有 COS 包 `share/xfer/ced_smla_ori_view_ab_20260924.tar.gz`
取回；包 SHA-256 为
`f52829be0b8828b12e08ffd26a0239c2f9e44cf782ba8b9740a19d722ba3d4eb`，与远端
打包前记录一致。包内 `SHA256SUMS` 有 31 项，本机解压后 31/31 校验通过；
清理后的 `npu-smi` 记录确认 chip7 无残留 PID，chip6 P 仍在。
双端校验与压缩包回执见
[`COS_TRANSFER.md`](../../evidence/ced_smla_ori_view_ab_20260924/COS_TRANSFER.md)。
