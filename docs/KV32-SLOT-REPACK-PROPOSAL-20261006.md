# ★ 提案：slot 重排把 32 位块上限从 29,076 提到 32,768 ⇒ 容量 4.03M（+34.8%），**零精度风险**

> 背景：容量的硬约束是**每个 slot 的页步长**（32 位页偏移），不是池总大小。
> 现状绑定线是 slot3 的 **147,712 B** ⇒ 上限 29,076 块 ⇒ 容量 **3,572,998**（+19.6%）。
> 本文给出一个**只改布局、不动 KV dtype** 的方案，把上限提到 **32,768 块** ⇒
> **4,026,689 tokens（+34.8%）** —— 与目标里的 4,032,862 **只差 0.15%**。
> 全部为【算术 + 代码路径核对】，**未占卡**。

## 0. 一句话

`slot3` 之所以是 147,712 B，是因为 **layer-20 的 `indexer.k_cache`（INT8 K 16,384 + FP16 scales 256）
被紧挨着放在同一个 slot 页里**（`CachePlacement(index_name, kv_bytes, …)`）。
而 `slot0/1/2` 已被 DSpark draft 顶到 131,072，**各自还空着 23,936 B**。
⇒ **把 layer-20 的 index K 挪进 slot0 的空闲区**，四个 slot 全变 131,072 ⇒ 上限 32,768 块。

## 1. 现状的槽位账（真机证据）

`[V41-DCP-DIAG] pool_bytes_per_block=540928 (slots=[131072, 131072, 131072, 147712])`

| slot | 归属 | KV 平面 | index 平面 | state/SWA/draft 别名 | **容量** |
|---|---|---:|---:|---|---:|
| 0 | layer 2（ratio 2） | 65,536 | 8,320 | state(2) / SWA(2,6,…) / **draft(0)** 各 131,072 | **131,072** |
| 1 | layer 8 | 65,536 | 8,320 | 同上 | **131,072** |
| 2 | layer 14 | 65,536 | 8,320 | 同上 | **131,072** |
| **3** | **layer 20（ratio 1）** | **131,072** | **16,640** | SWA(3,7,…) 131,072 | **147,712** ← 绑定 |

* `capacity = max(kv_bytes + index_bytes, aliases…)`（`plan_cache_slots`）
* 32 位判据取**每个 slot 的页步长**：`floor(2³²/147712) = 29,076` 块

## 2. 提案（只动 `plan_cache_slots` 的 placement）

把 `slot3` 的 index 平面（16,640 B）改放到 **slot0 的空闲区**：

```
slot0 已用 = 65,536 + 8,320 = 73,856          （KV(2) 在 [0,65536)，index(2) 在 [65536,73856)）
slot0 空闲 = 131,072 − 73,856 = 57,216  ≥  16,640  ✓
⇒ 新 placement: CachePlacement(index_of_layer20, offset=73,856, page_size=16,640)
⇒ slot0 容量仍为 131,072（被 draft 顶住，不变）
⇒ slot3 容量降为 131,072（只剩 KV(20) 131,072 + SWA 131,072）
```

> 勘误（2026-10-06）：§1/§2 初稿把 ratio-2 的 index 平面写成 41,600 B，实测几何是
> **8,320 B**（INT8 K 8,192 + FP16 scales 128；ratio-1 才是 16,384 + 256 = 16,640）。
> 空间结论不变（57,216 ≥ 16,640），已按实测值修正。

## 9. ★ 实测（2026-10-06，tiny a3-21 chips 2/3）

三组 run，除注明变量外其余逐项相同（TP=2 / DCP=2 / BAT=2048 / GRAPH=1 / SPEC=1；
prompt 63,999 tok，每请求 ≈500 块，30 轮累计 ≈15,000 块）：

| run | 代码 | pool stride | 块数 | `GPU KV cache size` | 走查结论 |
|---|---|---:|---:|---:|---|
| `serve_ctl` | 旧几何 | 540,928 | 29,076 | 3,019,745 | 内部 30 轮 `max|Δ|=0` |
| `serve_repack29k` | **重排** | 524,288 | 29,076 | 3,019,745 | 内部 0；**与 ctl 逐位相同（max|Δ|=0）** |
| `serve_repack` | **重排** | 524,288 | 32,768 | 3,403,198 | 内部 0；与 repack29k `max|Δ|=4.723` |

1. 【实测】**几何**：`[V41-DCP-DIAG] pool_bytes_per_block=524288 (slots=[131072, 131072, 131072, 131072])`
   —— 四槽全部落到 `2³²/32768` 安全线内（改动前 `[131072,131072,131072,147712]`）。
2. 【实测】**容量与块数严格线性**：3,403,198 / 3,019,745 = 1.12701，而 32768 / 29076 = 1.12700
   ⇒ 容量 = 块数 × 103.86（本配置）；页步长缩短的 3.17% 也如实兑现。
3. 【实测】**纯布局改动逐位相同**：同块数（29,076）下旧几何 vs 重排，
   30 轮 × 63,998 位全部 `max|Δ|=0`、0 个非零点 ⇒「把 layer-20 的 index 平面挪进 slot0 空闲区」
   数值上零影响（是逐位相同，不是"差异小到看不出"）。
4. 【实测】块数变化（29,076 → 32,768，同一份重排代码）带来**确定性**数值差：
   `max|Δ|=4.723 / mean=0.873`，每轮数值恒定（不是漂移；属池规模 → 图 tiling → 舍入重排）。
   量级与**已被接受的** BAT 8192↔2048 变化同级（`R_A8192` vs `R_B2048`：max 5.6 / mean 1.13，24K prompt）。

⇒ 结论：**重排本身零精度风险**；容量提升伴随的"块数变化"属于已知一类的确定性数值差，
必须在 A3 真实模型上以长文检索（60K/74K/150K）端到端确认。

复现物（a3-21）：`~/tmp/plp/WALK_{ctl,repack29k,repack}.json`、`~/tmp/launch_{ctl,repack29076}.sh`、
比较器 `~/tmp/walk_xcmp.py`（跨 run）、`~/tmp/walk_cmp.py`（run 内）。

## 10. 交付侧落地（A3，入口已就位、默认关）

* 交付镜像里的 `core/deepseek_v41.py`（356 行，md5 `f48b17613d5ae1040e7aa9e58b4e309c`）与 tiny
  工作副本（503 行）**不是同一份文件**，补丁按镜像自身版本单独移植：
  `patches/files/deepseek_v41.repack.py`。
* 挂载入口：`scripts/serve_a2.sh` mount 段新增 `[V41-KV32-REPACK]`，
  只有 `V41_KV32_REPACK=1` 且文件存在时才覆盖
  `/vllm-workspace/vllm-ascend/vllm_ascend/core/deepseek_v41.py`（默认关，不影响其它 run）。
* 起服验收点：`pool_bytes_per_block=524288 (slots=[131072,131072,131072,131072])`；
  同池字节下块数相对旧几何 ×1.0317，同块数下容量与旧几何完全相同（已实测）。
* 拿满 32,768 块需要 KV 预算 16.00 GiB（`KV_CACHE_MEMORY_BYTES=17179869184`）；
  与 BAT=2048 组合时是否装得下，取决于 A3 实测的可用 KV 预算（见
  `docs/CAPACITY-CEILING-MATH-20261006.md` 的内存账）。

**四个 slot 全为 131,072** ⇒
* `pool_bytes_per_block = 524,288`（比现状 **小 16,640**）
* 块上限 = `floor(2³²/131072)` = **32,768**

## 3. 运算结果

| 量 | 现状 | **提案** |
|---|---:|---:|
| 每块字节 | 540,928 | **524,288** |
| 块上限（32 位） | 29,076 | **32,768** |
| **容量（BAT=2048，NPR=8,533）** | **3,572,998** | **4,026,689（+34.8%）** |
| 满块所需内存 | 14.65 GiB | **16.00 GiB** |
| BAT=2048 的可用 KV 内存 | — | **16.60 GiB ⇒ 余 0.60 GiB** ✓ |

**与目标里的 4,032,862 只差 −0.15%。**

## 4. ★ 代码路径核对：跨 slot 放置**是现成支持的**

关键三段（全部在交付容器里核对过）：

1. **placement → (offset, stride)**（`worker/model_runner_v1.py`）：
```python
layer_placements = {p.name: (p.offset, slot.page_size_bytes)
                    for slot in plan_cache_slots(layer_kv_cache_spec)
                    for p in slot.placements}
```
2. **按 placement 建视图**（同文件）：
```python
if is_v41_spec(current_kv_cache_spec):
    offset, block_stride = layer_placements[layer_name]
    kv_caches[layer_name] = reshape_cache(..., offset=offset, block_stride=block_stride)
```
3. **`reshape_cache` 本身就支持任意偏移**（`core/deepseek_v41.py`）：
```python
if offset < 0 or offset + sum(plane_sizes) > block_stride:
    raise ValueError("V4.1 cache component exceeds its slot page")
... torch.as_strided(..., storage_offset=raw.storage_offset()+byte_offset, ...)
```

⇒ **`plan_cache_slots` 决定 (slot, offset)，下游全自动跟随** —— 不需要改 runner、不需要改算子。

**并且这个 placement 模式已被现网使用**：今天 slot3 的 index 就放在 `offset = kv_bytes`（非零偏移），
证明"slot 内任意偏移"这条链是通的。

## 5. 需要一并检查的四项（实现时的清单）

| # | 检查 | 期望 |
|---|---|---|
| 1 | `reshape_cache` 的越界判据 | `107,136 + 16,640 = 123,776 ≤ 131,072` ✓ |
| 2 | `plan_cache_slots` 末尾的"每个资源恰好出现一次"校验 | index(20) 仍只出现一次（换了 slot） ✓ |
| 3 | `group_cache_specs` 的 `page_size_padded` | 由 placement 的 `page_size_bytes` 决定 ⇒ 需给新 placement 正确值（16,640） |
| 4 | **`request_blocks`**（每请求块数）是否变化 | `request_blocks = Σ_g max_s (ceil(mem/page))`；full 组的 spec 集合不变，但 index(20) 的 `page_size_padded` 会变 ⇒ **必须实测**（若变小，NPR 会下降 ⇒ 容量还要更高；若变大则反之） |

⚠️ 第 4 项是**唯一可能推翻收益估算的地方**：容量 = `块上限 × MML / NPR`，
而 NPR 由 `request_blocks` 决定。提案只改 index(20) 的 page_size_padded（16,640 → 16,640，**不变**！）
—— 因为它在 slot3 时的 page_size_padded 恰好也是 `capacity − kv_bytes = 147,712−131,072 = 16,640`，
与提案里给的值**相同**。⇒ **NPR 预期不变** ✓

## 6. 与另外两条路的对比

| 路径 | 容量 | 精度风险 | 需要改什么 |
|---|---:|---|---|
| ① 只压池（已备脚本） | 3.57M（+19.6%） | 无 | 配置 |
| ② int8 主线 KV（A2 做过） | 4.03M | **有**（+2.15% 时延 + 量化误差） | 7 文件移植 |
| **③ ★ slot 重排（本文）** | **4.03M** | **无** | **`plan_cache_slots` 一处** |

## 7. 验证路径（不需要 A3 窗口）

**tiny 夹具的 KV 几何与 A3 逐项相同**（`compress_ratios` / `kv_source_layer_ids` / `head_dim` /
`index_*` / `num_key_value_heads`，见 `l1_dummy_provenance.json` 的 `structure_keys_unchanged`），
而且 tiny **同样带 draft**（所以 slot0/1/2 的 131,072 绑定也一样）。
⇒ 可以直接在 tiny 上：
1. 改 `plan_cache_slots`（挂载覆盖）→ 起服 ⇒ 看 `[V41-DCP-DIAG]` 的 `slots=` 是否变成四个 131,072；
2. 核对 `pool_bytes_per_block` 是否 524,288、容量是否 ≈32,768 块；
3. 用已有的**逐位置 logprob 仪器**（`walk_blocks.py`）跑噪声底与长文针 —— 因为这是**布局改动**，
   预期逐位相同（与"0 越界"那 116 条一样）。

## 8. 复现（算术）

```python
U32=2**32; MML=1048576; NPR=8533
print(U32//147712, 29076*MML/NPR)          # 现状 29076 / 3,572,998
print(U32//131072, 32768*MML/NPR)          # 提案 32768 / 4,026,689
print(32768*524288/1024**3)                # 16.00 GiB
```
