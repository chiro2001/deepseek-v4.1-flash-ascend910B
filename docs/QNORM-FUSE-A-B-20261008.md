# 线① 落地：`q_norm + wq_b.quantize` 融合，端到端 **−0.62%**（2026-10-08）

> 承接 `HCPRE-NEIGHBORHOOD-LINE1-RELOCATED`。该文识别出 `dsa_v41.py` 里两处 norm+quant 对，
> 本文完成**第一处的融合并实测验收**。全部【实测】。

---

## 0. 一页纸

| 项 | 值 |
|---|---|
| 改动 | `dsa_v41.py::multistream_preprocess`：非 index_source 层用融合算子 |
| 门控 | `V41_QNORM_FUSE`（默认 0） |
| **端到端** | **−0.149 ms/步 = −0.62%**（23.913 vs 24.062，n=280/272） |
| kernel 验证 | `RmsNormDynamicQuant` **3 → 35/步**；`RmsNorm` **140 → 108/步** |
| 融合发生在 | **主流 s146**（关键路径）✓ |
| 预测 vs 实测 | 预测 35×4.24=0.148 ms；**实测 0.149 ms** |

---

## 1. 为什么只有 35 层（而不是 43）

| 配置 | 值 |
|---|---|
| 总层数 | 43 |
| `index_source_layers` | `[2,8,14,20,24,28,32,36]`（**8 层**） |
| `kv_source_layers` | `[2,8,14,20]`（**4 层**，⊂ index_source） |
| **可融合层** | **43 − 8 = 35** |

**依据**：`q_norm` 的输出 `qr`（bf16）**只**被 `_select_sparse_indices` 使用，
且只在 `self.role.is_index_source` 分支里被真正读取：

```python
def _select_sparse_indices(self, attn, hidden_states, qr, ...):
    if not self.role.has_long_context: return None
    if not self.role.is_index_source:
        return shared.topk_indices[: hidden_states.shape[0]]   # ← 不用 qr
    ...
    selected, candidates = attn.indexer.select(hidden_states, qr, ...)  # ← 用 qr
```

⇒ **非 index_source 层可以把 `qr` 置 None**，直接用融合算子出 `(int8, scale)`。

---

## 2. 补丁

```python
# dsa_v41.py::multistream_preprocess（替换原 2 行）
_qf_ok = (V41_QNORM_FUSE and not self.role.is_index_source
          and not wq_b._has_communication and wq_b._is_w8a8_dynamic)
if _qf_ok:
    q_b_quant, q_b_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
        q_a, attn.q_norm.weight, epsilon=v1_impl.eps)
    qr = None
else:
    qr = attn.q_norm(q_a)
    q_b_quant, q_b_scale = wq_b.quantize(qr)
```

**文件落位**：`/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py`
（**非挂载**，`docker cp` 部署，md5 `b91e2237`）

---

## 3. 验收（三层证据）

### 3.1 端到端 A/B（同一份文件，仅 env 开关不同）

| 臂 | n | **hp p50** | p25 | p75 | min |
|---|---:|---:|---:|---:|---:|
| **FUSE=1** | 280 | **23.913** | 23.708 | 24.043 | 23.391 |
| **FUSE=0** | 272 | **24.062** | 23.853 | 24.234 | 23.552 |
| | | **−0.149 ms（−0.62%）** | | | |

**分布完全不重叠**（FUSE=1 的 p75 24.043 < FUSE=0 的 p25 23.853 之上但整体错开）。

### 3.2 kernel 数（profile 实测）

| kernel | FUSE=0 | FUSE=1 | 变化 |
|---|---:|---:|---|
| `RmsNormDynamicQuant` | 3.00/步 | **35.00/步** | **+32** ✓ |
| `RmsNorm` | 140.00/步 | **108.00/步** | **−32** ✓ |
| `RmsNormCast` | 43.00/步 | 43.00/步 | 不变 ✓ |

### 3.3 融合发生在关键路径上

| 臂 | fused 的流分布 |
|---|---|
| FUSE=0 | `s142: 333`（draft 侧流） |
| **FUSE=1** | **`s146: 3840`（主流）** + s142: 360 + s47: 35 |

**⇒ 35 个融合 kernel 全部落在主流 s146 上**（此前只有 draft 的 3 个在侧流）。

### 3.4 正确性

冒烟 `/v1/chat/completions` → `'2'` ✅

---

## 4. 四个融合点的账（含未做的三个）

用实测值（`RmsNorm` 4.329 µs、`DynamicQuant` 3.565 µs / 融合 4.08 µs @5120；
5.161 + 2.542 / 3.20 @1280）：

| # | 融合 | 覆盖层 | 当前 µs | 融合后 | **收益/步** | **占比** | 阻塞 |
|---|---|---:|---:|---:|---:|---:|---|
| **A** | `q_norm` + `wq_b.quantize` | 35 | 7.70 | 3.20 | **0.149 ms** | **0.62%** | ✅ 已完成 |
| **B** | `input_layernorm` + `wq_a.quantize` | 35 | 7.89 | 4.08 | 0.133 ms | 0.52% | 需把 layernorm 挪进 attention（结构改动） |
| **C** | B 的三输出版（补剩余 8 层） | 8 | 7.89 | ~5.0 | 0.023 ms | 0.09% | 需写 kernel |
| **D** | **`HcPre` + `input_layernorm` + `quant`** | 43 | **41.4** | ~35 | **0.275 ms** | **1.08%** | 需改 HcPre kernel |

### 4.1 D 为什么最有价值

`HcPre` 的 y 输出（`[6,5120]`）**只有 `input_layernorm` 一个消费者**：

```python
x, attn_post, attn_comb, attn_pre = self.hc_pre(...)   # x = [6,5120]
x = self.input_layernorm(x)                            # ← 唯一消费者
x = self.self_attn(positions, x, ...)
```

而 `HcPre` 是**自研 kernel**（`csrc/moe/hc_pre/`，4082 行，含 op_host/op_kernel）
⇒ 可在其 vector 段尾部接 norm+quant，省掉 **2 次 kernel 下发 + 2 次中间张量往返**。

**成本结构**（实测）：HcPre 33.5 µs（数学 2.1 + 标量 11.8 + MTE 10.8 + 未归因 8.8）
+ RmsNorm 4.3 + DynamicQuant 3.6 = **41.4 µs**。

---

## 5. 下一步

| # | 动作 | 收益 | 前提 |
|---:|---|---:|---|
| 1 | **A 转默认**（`V41_QNORM_FUSE=1`）+ 长文针验收 | 0.62% | 需过四道门 |
| 2 | **B：把 `input_layernorm` 挪进 attention** | 0.52% | 结构改动，需处理 layer-20 特例 |
| 3 | **D：改 HcPre kernel，尾部接 norm+quant** | 1.08% | 改自研 kernel |

**建议顺序：1 → 2 → 3**（1 已就绪，2 用现成算子铺路，3 收益最大）。

---

## 6. 复现

```bash
# 部署
docker cp dsa_v41_fused.py dsv41-tp8k5:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py
docker exec dsv41-tp8k5 rm -f .../attention/__pycache__/dsa_v41*.pyc
# A/B（同一份文件，只翻 env）
ssh a3-21 'bash /tmp/deploy_fuse.sh'      # FUSE=1
ssh a3-21 'bash /tmp/launch_nofuse.sh'    # FUSE=0
# kernel 计数对比
ssh a3-21 'docker exec dsv41-tp8k5 python3 /tmp/kc.py'
```
