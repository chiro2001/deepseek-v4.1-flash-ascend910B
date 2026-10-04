# F2 判决：W4A8 下的 `rms_norm_dynamic_quant` 融合 —— **不做**（2026-10-05）

> 背景：R4 的 `reports/scalar-bound-op-audit.md` 提出 F2：把 `RmsNorm + DynamicQuant`
> 两跳合成一个 `npu_rms_norm_dynamic_quant`，估 **0.25–0.40 ms/步**。
> 但 R5 的所有文档（`LEVERS-R5` / `FINAL-R5` / `REMAINING-TARGETS`）**都没有提过 F2**
> ⇒ 本轮先把它当作"未交付的候选"重新评估。结论是**不做**，理由见下（三条独立证据）。
>
> 环境：微型实例（`dsv41-tinyspark`，TP2，die2/3），**未占用交付实例**。
> 标注：【实测】。

## 1. 现状：交付口径下 `RmsNorm` 与 `DynamicQuant` 确实成对出现

decode 步（M=6）时间线（`DELIVERY-PROFILE-R6` §5.5 的同一份 profile）：

```
17.994  RmsNorm       4.3 µs  "6,5120;5120"     ← input_layernorm
17.998  DynamicQuant  3.6 µs  "6,5120"          ← 给 wq_a / wkv 的激活量化
18.003  QuantBatchMatmulV3 12.5 µs
18.016  RmsNorm       5.0 µs  "6,1280;1280"     ← q_norm
18.021  DynamicQuant  2.6 µs  "6,1280"          ← 给 wq_b 的激活量化
18.028  QuantBatchMatmulV3 11.6 µs
```

即每层 2 对 × 40 层 = **80 对 / 步**（160 个算子）。

## 2. 证据一：融合算子**不是逐位等价**【实测】

夹具（`f2_equiv.py`）：同一 `x`（bf16）、同一 `w`、同一 `eps=1e-6`，比较

* A：`npu_rms_norm(x, w, eps)` → `npu_dynamic_quant(·)`
* B：`npu_rms_norm_dynamic_quant(x, w, epsilon=eps)`

| 形状 | int8 **逐位相同** | 差异元素 | scale max\|Δ\| |
|---|---:|---:|---:|
| (6, 5120) | **否** | **1245 / 30720（4.1%）** | 4.9e-5 |
| (6, 1280) | **否** | **358 / 7680（4.7%）** | 5.0e-5 |
| (256, 5120) | **否** | **63715 / 1310720（4.9%）** | 1.2e-4 |

⇒ 两条路径的**归约顺序/舍入不同**，量级是"int8 的 1 LSB，约 4–5% 的元素"。
这不是"更粗的量化"，而是**同一 int8 管线里的另一种取整**，但它**会改变输出轨迹**
（见 `BATCH-DEPENDENT-OUTPUT-20261005.md`：连批大小不同都会让 7/8 条 prompt 的输出分叉）。

## 3. 证据二：融合在 **prefill 尺度上反而更慢**【实测】

同一夹具的流水化耗时（host 入队 + 设备消化，逐次同步取总时间）：

| 形状 | split（norm+quant） | fused | Δ |
|---|---:|---:|---:|
| (6, 5120) | 53.9 µs | **42.3 µs** | **−11.6** |
| (6, 1280) | 50.2 µs | **39.9 µs** | **−10.3** |
| **(8064, 5120)** | **252.9 µs** | **297.4 µs** | **+44.5** |

* decode（M=6）**省 ~11 µs/对**；
* **prefill（M=8064）反而慢 44 µs/次** ⇒ 每个 prefill chunk（40 层 × 2 对）多花
  ≈ 3.5 ms（该 chunk 约 817 ms，**−0.4%**），把 decode 的收益抵掉一部分。

## 4. 证据三：收益量级只有 1–1.6%

按交付实例实测的"每算子 ~4 µs（尾部分析）"折算：80 对 ⇒ 少 80 次下发
≈ **0.32 ms/步（1.3%）**；与 R4 的 0.25–0.40 ms 估计一致。

## 5. 判决与理由

**不做 F2。** 三条独立理由：

1. **收益小且有反向**：decode 端 ~1.3%，prefill 端 ≈ −0.4%；
2. **数值改变**：4–5% 的激活元素差 1 LSB ⇒ 必须重跑完整精度验证
   （144K 四针 + 并发 2 + regress2），而且**无法先验判断是变好还是变坏**；
3. **与交付口径的优先级不符**：当前四个维度全部达标，用"可能翻车的数值改动"
   换 1.3% 的 decode，风险收益比不划算 —— 这与之前否掉 `wo_a` 量化（1.30–1.34% 误差）
   和 `HcPre+RMSNorm 融合`（净亏 672 µs）是同一条判据。

> **更一般的结论**（写给后续所有 fusion 候选）：
> 在这个工作负载上，**任何改变算子形状/归约顺序的融合都会改变 argmax 轨迹**
> （因为解码是贪心、且已证实对批大小都敏感）。所以"融合省 1–2%"这一类候选，
> 必须同时提交**精度证据**，否则默认按"不可采纳"处理。
> 唯一例外是**不改变数值**的改动：减少 kernel **数量**但保持每个 kernel 的
> 输入/形状/累加顺序不变（例如 armF 的 gmm1 降 SyncAll、A1 的自适应 K_L0）。

## 6. 复现

```bash
docker cp ~/tmp/f2_equiv.py dsv41-tinyspark:/tmp/
docker exec -e ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
  -e ASCEND_RT_VISIBLE_DEVICES=0 dsv41-tinyspark bash -lc "cd /workspace && python /tmp/f2_equiv.py"
# 速度对比把 f2_equiv.py 换成 f2_speed.py（同目录）
```
