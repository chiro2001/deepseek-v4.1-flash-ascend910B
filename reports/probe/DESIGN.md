# 稀疏状态插针（sparse-state probe）设计

> 2026-09-18|<test-host>|目标：让"输出退化"这类问题可**事后取证**，不必靠重放碰运气
> 状态：**设计定稿，待落地**

---

## 0. 为什么不是"dump 整个 KV cache"

用户最初的直觉是"把一个状态的 kvcache 全 dump 下来"。我评估后**不建议作为第一选择**：

| 对象 | 单 rank 大小 | 对定位非确定性的价值 |
|---|---|---|
| long_kv_cache（4 个 source 层） | 每层 `2 × 2B × tokens × 576`；512K token 时 **≈1.2 GB/层，4 层≈4.7 GB** | **低**。同输入下 KV 是确定的；除非写入路径本身有 bug，否则 dump 出来也只是"正确的中间态" |
| indexer.k_cache（8 个 source 层） | 同上量级 | 低 |
| `qr` + `positions`（index source 层的输入） | `5120 × seq × 2B` ≈ 10 MB @1K tok | **高**。所有下游误差的入口 |
| **`selected`（topk_indices）** | `seq × 512 × int32` ≈ 2 MB @1K tok | **最高**。这是"模型实际看到了哪些历史位置" |
| **`candidates`（块级候选）** | `seq × 2048 × int32` ≈ 8 MB @1K tok | **最高**。报告 H1 的嫌疑对象 |

**关键理由**：按 `nondeterminism-rootcause.md` 的结论，退化沿
"稀疏选择 → 注意力 → 输出"传导。选择结果只有几 MB，可以**逐 step 全量保留**；
KV cache 上百 GB，只能抽样，且抽到"正确的那一份"并不能解释任何事。

⇒ 插针的**第一优先级是 `selected` / `candidates`**，KV cache 作为**可选的低频快照**。

---

## 1. 插针位置

`vllm_ascend/attention/dsa_v41.py:451-481` 的 `_select_sparse_indices()`：

```python
def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
    if not self.role.has_long_context:
        return None
    shared = attn.shared_state
    if shared is None:
        raise RuntimeError("V4.1 shared attention state is not initialized")
    if not self.role.is_index_source:
        return shared.topk_indices[: hidden_states.shape[0]]
    if attn.indexer is None or metadata.indexer is None:
        raise RuntimeError("V4.1 index source is missing indexer metadata")

    context = get_forward_context().no_compile_layers
    source_layer = context[self.index_k_source_prefix]
    selected, candidates = attn.indexer.select(
        hidden_states, qr, positions, cos, sin,
        source_layer.kv_cache[0], metadata.indexer.cache,
        is_candidate_source=self.role.is_candidate_source,
        uses_candidate_filter=self.role.uses_candidate_filter,
        candidate_topk_blocks=self.topology.candidate_topk_blocks,
        candidate_block_size=self.topology.candidate_block_size,
        candidates=shared.candidates[: hidden_states.shape[0]],
    )
    shared.topk_indices[: selected.shape[0]].copy_(selected)
    if self.role.is_candidate_source:
        shared.candidates[: candidates.shape[0]].copy_(candidates)
    return shared.topk_indices[: hidden_states.shape[0]]
```

**插针就放在 `attn.indexer.select(...)` 返回之后**——这里同时拿得到
输入（`qr`/`positions`）、输出（`selected`/`candidates`）、以及角色信息
（`layer_idx` / `is_candidate_source` / `uses_candidate_filter`）。

### 1.1 拓扑常量（本模型实测）

| 项 | 值 |
|---|---|
| `num_hidden_layers` | 40 |
| `kv_source_layer_ids` | `[2, 8, 14, 20]` |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` |
| `candidate_source_layer_id` | **20** |
| `index_topk` | 512 |
| `candidate_topk_blocks` | **2048** |
| `candidate_block_size` | **8**（内核硬性要求，不可改） |

⇒ 本模型共 **8 个 index source 层**，每次 forward 会命中 8 次插针；
其中 **layer 20 是 candidate source**（唯一会产出 `candidates` 的层）。

**注意**：`candidate_topk_blocks × candidate_block_size = 2048 × 8 = 16384`
正好是报告里那个"16385 阈值"——即"候选块刚好覆盖不满一段上下文"的临界点。

---

## 2. 两层设计

### L1：元数据（默认常开，成本≈0）

每次命中记一行 JSONL，**不落任何张量**：

```json
{"req":"<req_id>","step":123,"layer":20,"role":"candidate_source",
 "seq":8192,"n_selected":8192,"sel_min":0,"sel_max":8191,"sel_unique":8192,
 "cand_min":0,"cand_max":2047,"cand_unique":2048,"sel_hash":"a1b2c3d4","cand_hash":"e5f6a7b8",
 "dirty":false,"elapsed_ms":0.83}
```

* `sel_hash` / `cand_hash`：对张量做 `xxhash` 或 `torch.sum` 之类的**廉价指纹**，
  用来回答"同输入下这次和上次选的是不是同一批"——这正是 H1/H2 要判的问题。
* `dirty`：**关键的廉价正确性信号**——`candidates` 里是否出现
  本 forward 序列长度之外的位置（如 `>= seq`，或 `-1` 之外的非法值）。
  若 H2（读到非本 forward 的候选行）成立，这里会直接亮。
* 成本：每 step 8 行、每行约 200 B ⇒ **1.6 KB/step**。
  按 1000 tok/s、平均 8 tok/step 估算，**约 200 KB/min，一天约 280 MB**。

### L2：张量快照（按需，默认关闭）

由**运行时开关文件**控制，可在不重启服务的情况下打开：

```bash
~/probe_capture/ENABLE          # 存在即开启；内容可取 ring:N / once / trigger
```

打开后按环形缓冲保留**最近 N 个 step** 的 `selected`/`candidates`（`torch.save`，
`int32` 存原始 dtype；实测 `seq=8K` 时 `selected`≈16 MB、`candidates`≈64 MB）。

**环形缓冲是关键设计**：退化发生时人往往是事后才知道的，
"保留最近 N 个 step 并随新 step 滚动"意味着**故障发生时现场还在**。
N 可在开关文件里配（默认 4）。

---

## 3. 资源预算（<test-host> 实测）

| 资源 | 现状 | 插针占用 |
|---|---|---|
| `/home`（14 TB） | 已用 2.2 T（17%），可用 **11 T** | L1 约 **280 MB/天**；L2 打开时按 4 个 step × 8 层 × 80 MB ≈ **2.5 GB**（一次性，环滚动覆盖） |
| `/tmp`（tmpfs 1007 G，**吃内存**） | 已用 34 G | **不落 /tmp**，避免重演历史事故 |
| 系统内存 2013 G | available **868 G** | L2 全在设备侧 `torch.save` 直写磁盘；**不在 host 侧缓存整张量** |
| NPU 侧显存 | 单卡 61 GB 已用 | 插针**不额外驻留**显存：只在回调里读一次、立即落盘 |

**红线**：L2 单次快照不得超过 1 GB；超过则自动降级为"只 dump `selected`，跳过 `candidates`"，
并打警告。L1 总量超过 5 GB 时自动轮转（保留最近 7 天）。

---

## 4. 与"能否事后定位"的关系

有了这套，任何一次退化都可以立刻回答下列问题（当前**只能靠重放碰运气**）：

1. 退化的那次请求，**哪一层**的 `selected` 第一次出现异常（越界/重复/与上一步不一致）？
2. `candidates` 是否出现"非本 forward 的位置"（H2 的直接检验）？
3. 同样的 prompt，成功和失败两次的 `sel_hash` 是否不同？**从哪一层开始分叉**？
4. 退化是**单层**现象还是**层层累积**？（对应 H1 vs H2）

---

## 5. 与 dev mode 的关系

用户同时要求打开 `VLLM_SERVER_DEV_MODE=1`。两者**互补**：

| | dev mode | 插针 |
|---|---|---|
| 能做什么 | 清缓存 / 暂停 / 休眠唤醒 | 记录中间态 |
| 回答"哪一层坏了" | ❌ | ✅ |
| 回答"能否不重启恢复" | ✅ | ❌ |

一起开：出错时**先 dump 现场**（插针），**再试恢复手段**（dev mode），
两条线同时有答案。
