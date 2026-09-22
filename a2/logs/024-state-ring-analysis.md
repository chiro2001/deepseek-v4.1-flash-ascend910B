# P0-2 分析：compressor state ring 能不能缩（KV8 容量分母的真凶）

> 2026-09-22 01:5x，主代理前台**只读代码**分析（不占卡）。
> 起因：`logs/020` 实测 —— 把 SWA 也量化后容量**只到 ×1.135**，
> 真凶是 **FP32 compressor state ring（131072 B/block）顶住了 3 个 ratio-2 槽的页大小**。

---

## 一、它的定义（**代码事实**）

`vllm_ascend/models/deepseek_v41/compressor.py:38-44`：

```python
self.state_cache = DeepseekV41CacheLayer(
    vllm_config, f"{prefix}.state_cache",
    CircularBufferSpec(
        block_size=STATE_RING_ROWS,      # 32（cache_config.py:22）
        num_kv_heads=1,
        head_size=2 * self.width,        # ★ 2 倍！
        dtype=torch.float32,             # ★ FP32
        head_size_v=0,
    ),
)
```

### 算一下

```
page = block_size × num_kv_heads × head_size × dtype_size
     = 32 × 1 × (2 × 512) × 4
     = 32 × 1024 × 4
     = 131072 B          ← ★ 与 logs/020 的实测**逐字吻合**
```

（反推 `width = 512`，与 `head_size = 1024` 一致。）

---

## 二、★ 关键结构：它**自己很小，但顶着整个页**

| 事实 | 值 | 出处 |
|---|---|---|
| state ring 是**每请求 1 页** | `max_num_blocks_per_req = 1` | `logs/013` §（`AscendCircularBufferManager`） |
| 它**不参与前缀缓存** | `prefix_cacheable = False` | `logs/001` §3 |
| **总占用**（`max_num_seqs = 64`） | 64 × 131072 = **8 MB** | 上两行算 |
| **但它决定 hybrid slot 的页大小** | `page_size = max(kv + index, *aliases)` | `cache_config.py::get_layer_tuples` |

**⇒ 所以"缩 state ring"的意义不在省它自己（才 8 MB），而在**
**让 hybrid slot 的页不再被它顶住。**

---

## 三、三条可能的缩法（按代价排序）

| # | 缩法 | 页从 131072 变 | 代价 | 可行性 |
|---|---|---|---|---|
| **A** | **`dtype: float32 → bfloat16`** | **65536 B**（一半） | ★ **数值敏感**：ring 是**累积**状态（`compressor_from_projected` 每步读改写），精度影响会**随步数累积** | ⚠️ 要精度验证 |
| **B** | `head_size: 2×width → width`（若那 "2" 里有一半可省） | 65536 B | 要看 `compressor_from_projected` 到底用了几份 | ⚠️ 要读懂算子 |
| **C** | `block_size: 32 → 16` | 65536 B | ring 行数减半 ⇒ **改变算法语义**（环形缓冲的长度） | ⛔ 大概率不行 |

### ★ 缩了之后能拿回多少？

`logs/020` 的实测页构成（ratio-2 槽）：`476416 B/block`（SWA 量化后）。
若 state ring → 65536 B，则页的 `max(...)` 里那一项不再主导，
页大小由 **SWA 页 66560 B** 或 **long-KV + indexer 页**决定
⇒ **页可能降到 ~66560–74000 量级**，即 **476416 → ~70000 ⇒ ≈×6.8**。

（**这只是上界估计**：要按 `get_layer_tuples` 的 `capacity = max(kv+index, *aliases)` 精确重算，
而且 4 个槽的构成不同。**别把 6.8 当结论**。）

> ⚠️ 注意：这与 `logs/020` 说的 "×1.91"（缩 state ring 后的推算）**不一致** ——
> 020 的 ×1.91 是"只缩 state ring、不改 SWA"，我这里算的是"**SWA 已缩 + 再缩 state ring**"。
> **两者要合并成一张表才算准**，见 §四。

---

## 四、把两条缩法合起来算（**需要精确重算，不是估计**）

| 配置 | long-KV 页 | SWA 页 | state ring | ratio-2 槽的页（`max`） |
|---|---:|---:|---:|---:|
| 现状（BF16 全） | 73856 | 131072 | 131072 | **131072 + 其它** ⇒ 540928 总 |
| 只缩 SWA（020 实测） | 41088 | 66560 | **131072** | **仍 131072** ⇒ 476416（×1.135） |
| **缩 SWA + 缩 state ring** | 41088 | 66560 | **65536** | **由 SWA 66560 主导** ⇒ 待算 |

**⇒ 这一步要**照着 `get_layer_tuples()` 逐槽重算**（4 个槽的 alias 构成不同），
并**在单卡上用 `logs/017` 的 `probe_predicate.py`（不占卡）验证算式**。

---

## 五、结论与建议

| # | 结论 |
|---|---|
| 1 | **state ring 的 `head_size = 2 × width` 与 `dtype = float32` 是页大小的**唯一**来源**（131072 逐字吻合） |
| 2 | **它自己只占 8 MB**，缩它的意义是**解开 hybrid slot 的页** |
| 3 | **缩法 A（FP32 → BF16）最直接**，但**必须做精度验证**（ring 是累积状态） |
| 4 | **建议顺序**：**先等 `logs/023`（KV8_gather 的性能修复）** —— 因为**性能不达标时，容量再大也是负收益**；性能修好后，再决定要不要为容量去动 state ring 的精度 |
| 5 | **这条超出 KV8 范围**（它是 compressor 的数值问题，不是 KV 量化），所以**要单独作为一个候选**，不要混进 KV8 的叙事 |

---

## 六、取证

```bash
cd ~/projects/dsv41/upstream-v41/vllm-ascend-upstream
# state ring 的定义
git show origin/main:vllm_ascend/models/deepseek_v41/compressor.py | sed -n '36,46p'
# STATE_RING_ROWS
git show origin/main:vllm_ascend/models/deepseek_v41/cache_config.py | grep -n STATE_RING_ROWS
# 页大小的算法
git show origin/main:vllm_ascend/models/deepseek_v41/cache_config.py | sed -n '55,80p'
```
