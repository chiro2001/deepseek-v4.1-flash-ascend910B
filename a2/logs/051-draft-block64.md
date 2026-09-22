# 051 — ②c（draft block 128→64，保 BF16、保投机）：**64 档本来就存在**，改动是「一个参数 + 一条断言」

> 2026-09-22 11:1x–11:3x CST。执行：子代理 **T_draftceiling**（静态 + 不占卡的容器内 CPU 自检），
> **主代理独立复核**了本节 §1 的表结构（读 `graph_prep/src/vllm_ascend` 真源码）。
> 没写 `upstream-v41/`；没用 `/tmp`；没占任何 die。
> 标记：**【实测】**= 有判别力的原始数据；**【算术】**= 由源码几何算出；**【推断】**；**【未确认】**。

---

## 0. 一句话

**②c 不是"给 draft 新造一档 64 行块"，而是"把 draft 的 `block_size` 从 128 改成 64"** ——
算子的块大小档位表里 **64 本来就有**，而且那一档的 SWA 值**就是 64**、`page_size_padded_t2` **正好 65,536**（= 我们要的 draft 页）。
**改动面 2 个文件 / 2 处**，**默认关**（`VLLM_V41_DRAFT_BLOCK` 不设 ⇒ 逐字旧行为），**保留 BF16（无精度风险）+ 保留投机解码**。

**收益（8 卡真权重口径，`050` 的零参数模型的预测）**：档 C **427,643 → 595,404**、**档 D 485,610 → 777,318（×1.8177 vs 档 B）**。
**代价**：卸载池需求 **+5.0%**（draft 组 2→3 unit）。**【预测，端到端臂待跑】**

---

## 1. ★★ 关键事实：`64` 档在真源码里（主代理独立复核）

`graph_prep/src/vllm_ascend/models/layer/attention/layer.py:32-56`：

```python
def get_dsv4_block_sizes(use_a5_bf16_kv: bool = False):
    # cache_config.block_size: [mla, swa, c4 state, c128 state], [page_size_padded_t1, page_size_padded_t2]
    _DSV4_BLOCK_SIZES = {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64:  [[ 64,  64, 4, 16], [ 8320,  65536]],   # ← ★ 有 64 档
        32:  [[ 32,  32, 2,  8], [ 4160,  32768]],
    }
    _DSV4_COMPRESSED_BLOCK_SIZES = {128: [[128,128,8,16],[16896,81920]],
                                    64:  [[ 64, 64,4, 8],[ 8448,40960]], ...}   # ← 64 档的 swa 仍是 64
    _DSV4_BLOCK_SIZES_A5_BF16    = {128: ..., 64: [[64,64,4,8],[8448,65536]], ...}
    if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
        return _DSV4_COMPRESSED_BLOCK_SIZES if not use_a5_bf16_kv else _DSV4_BLOCK_SIZES_A5_BF16
    return _DSV4_BLOCK_SIZES
```

| 复核点 | 结论 |
|---|---|
| 三张表**都有 64 档**，且 `[0][1]`（swa）在每张表里都是 **64** | ✅【实测·源码】 |
| `AscendDeepseekV4SWACache.__init__` 取 `DSV4_BLOCK_SIZES[cfg.block_size][0][1]` = **swa 列** ⇒ 64 档给的就是 64 | ✅【实测·源码】 |
| 64 档的 `page_size_padded_t2` = **65,536**（= 128×512×2/2，正是 draft 要的页大小） | ✅【实测·源码】 |
| ★ 硬件档案：`DSV4_COMPRESSED_CACHE` **只挂在 `AscendDeviceType.A5`**（`device/hardware_profile.py:228`），**A3 的 profile 没有它** ⇒ **A2/A3 走的是 `_DSV4_BLOCK_SIZES`（64 ⇒ t2 = 65,536）** | ✅【实测·源码】（主代理复核） |
| 上游 vLLM 的 `DeepseekV4SWACache.__init__` 本来就**硬编码 64**，后端 `get_supported_kernel_block_sizes() = [MultipleOf(64)]` ⇒ 64 与 128 都是这批算子认的 | ✅【实测·源码】 |

> ⚠️ **一个留给端到端臂的核对点**：`[1]` 那对 `page_size_padded_t1/t2` 是**给 target 的 padded page** 用的；
> draft 组的页大小在本模型里由**它自己的 spec**算（`storage_block_size × 1 × 512 × 2`）⇒ 与 `t2` 无关。
> 若端到端的实测 `Σslot_pages` 不等于 **369,280（档 C）/ 282,880（档 D）**，就说明这条【推断】要改，**回退按 §4**。

---

## 2. ★★ 精确改动清单（2 文件 / 2 处，默认关）

### ① `vllm_ascend/models/deepseek_v41/dspark.py` — 给 draft 单独块大小（**唯一**决定页大小的开关）

```python
class DeepseekV41DSparkSWACache:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = int(os.environ.get("VLLM_V41_DRAFT_BLOCK", "128"))  # 默认 128 = 逐字旧行为
        if self.block_size % 64 != 0:
            raise ValueError(...)
```

理由链（逐跳）：`self.block_size` → `get_kv_cache_spec()` → `AscendSlidingWindowMLASpec(block_size=…)`
→ `DeepseekV41DraftSWASpec(block_size=spec.block_size)` → `real_page_size_bytes = storage_block_size × num_kv_heads × head_size × itemsize`
⇒ **128→131,072 / 64→65,536**。**target 的 40 个 SWA 层走另一个类（`AscendDeepseekV4SWACache`）⇒ 一个字不动。**

### ② `vllm_ascend/core/deepseek_v41.py::plan_cache_slots` — 只放宽 draft 的**块大小相等**检查

```
现在：draft_spec.block_size != swa_spec.block_size  or  draft_spec.head_size != … or … ⇒ raise
改成：swa_spec.block_size % draft_spec.block_size != 0 or draft_spec.head_size != … or …
```
（128 % 64 == 0 通过；**head_size / sliding_window /"页装得下 Σdraft ≤ capacity"三条一条不放松**。）

### 逐条说明「哪些**不动**」（避免看日志的人以为漏了）

| 组件 | 要不要改 | 理由 |
|---|---|---|
| `real_page_size_bytes` | 不动 | 由 `block_size` 自动派生 |
| `reshape_cache` | 不动 | 逐字用该组自己的 `spec.storage_block_size` |
| `metadata` | 不动 | `storage_block_size/logical_block_size` 取自该组 spec（`dsa_v41.py:1252,1261`） |
| `slot_key` | 不动 | 按 `(ratio, storage_block_size)` 分键 ⇒ 64 自带一格 |
| 卸载层（scheduler/worker） | 不动 | `tokens_per_block` 从 spec 现算；**但池配额要复算（+5.0%）** |
| `DraftSWASpec.__post_init__` | 不动 | 继续强制 **BF16**（②c 要的正是这个） |
| `kv8_ori_plane` 的 `pages_per_req = 2` | 不动 | 它在 **int8 平面**上；②c（BF16 draft）**不走它**。★ 只有 **②a+②c 组合**才要改成 `cdiv(window, block)+1` |

★ **"窗口跨块"是不是真风险？（三条查过）** draft 窗口 128 token 在 block=64 下跨 **2 块**（带投机 query 共 133 token ⇒ 最坏 4 块），但：
① 算子是**绝对 token 坐标寻址**（`block = pos // storage_block_size`）；
② 块表是**全长行**（`cdiv(max_len, block)`，与窗口无关）；
③ KV manager 的 `_contiguous_blocks_for_hit = cdiv(window−1, block)` **自动变 2**。
⇒ 三处都**没有"窗口 ≤ 1 块"假设**。★ 同槽混块也早有先例：`state` 组就是 **32 行页**与 128 行页共享同一 block ID。

★ **一个必须记住的算术细节**：②c 在**档 C**上的 slots0–2 = **73,856**（不是 66,560）——
因为档 C 的 long-KV 还是 BF16：`kv+index(73,856) > SWA(66,560) > draft(65,536)`
⇒ **②c 在档 C 上的收益来自"把 draft 拉到 kv+index 之下"**；在**档 D**上才是 SWA binding。两者的预测值因此不同（C 595,404 / D 777,318）。

---

## 3. ★★ 三臂对称的单元自检（**不占卡**，容器内 CPU）

`a2/agents/T_draftceiling/scripts/selfcheck_draft_blk.py`（生成器 `patch/patch_draft_blk.py`：锚点计数 + AST 结构自检 + `py_compile` 2/2）。

| 判据 | `upstream`（纯上游） | `draftaware`（R 的 capacity 补丁，**8 卡实测用的就是这件**） | **`patched`（②c）** |
|---|---|---|---|
| 档 B Σ | **540,928** ✅ | 540,928 ✅ | 540,928 ✅ |
| 档 C Σ | **raise** ✅（= 8 卡实测到的同一条断言） | **540,928** ✅（×1.0000，与 8 卡逐字同） | 540,928 ✅ |
| 档 D Σ | raise ✅ | **476,416** ✅（与 8 卡逐字同） | 476,416 ✅ |
| **draft=64 档 C** | raise ✅ | **raise** ✅（⇒ ②c 补丁**必需**） | ★ **369,280**（slots = [73,856×3, 147,712]） |
| **draft=64 档 D** | raise ✅ | raise ✅ | ★ **282,880**（slots = [**66,560×3**, 83,200]，**SWA binding**，draft 65,536 已不顶） |
| 预测 token（8 卡口径） | — | B 427,643 / C 427,643 / D 485,610 —— **三个都逐字命中 8 卡实测** | ②c：档 C **595,404** / 档 D **777,318** |
| 卸载池 Σ权重 | — | 20 | **21**（draft 组 2→3）⇒ 池需求 **+5.0%** |
| 无 draft 组的 tiny 几何 | — | — | **逐字 no-op**（540,928 / 282,880）✅ |

★ **判别力**（`AGENTS §5b` 第 3 条）：同一批断言在 `upstream` / `draftaware` 上**要么 raise、要么给出不同的数**
⇒ 不是"永远为真的空断言"。

---

## 4. 回退与边界

### 4.0 ★★ 2026-09-22 13:2x：**主代理的一次误判与撤回**（留给后人，勿重蹈）

横向对账时我（**主代理**）看到 `051` 的 slot 算术说"②c 档 D 的 Σ 降到 **282,880**"
（`3×66,560 + 83,200`），而 `050 §1.6` 的零参数模型里 Σ=282,880 是 **tiny 几何**（**无 draft 组**），
于是判断"两者矛盾、②c 的 Σ 应该是 476,416" —— 并据此去改
`agents/T_draftceiling/scripts/selfcheck_draft_blk.py`，还加了一条 `assert`。

**这是错的。** 正确的推理只需要看清"**块大小是逐组的**"：

```
②c（draft_block=64）下：full / state / 10 个 target SWA 组 **仍然是 128 行块**，
                        只有 draft 组是 64 行块 ⇒ draft 页 131,072 → 65,536 B
档 D 的 slots 0–2 = max(kv+index 41,600, state 65,536, swa 66,560, draft 65,536) = 66,560
                                                     ↑ 注意：这个 66,560 是 **BF16 SWA 页**
档 D 的 slot 3   = max(kv+index 83,200, swa 66,560) = 83,200
⇒ Σ = 3×66,560 + 83,200 = 282,880   ★ 与 tiny 的 Σ **数值巧合相同**，但不是同一件事
```

用 `050` 的 canonical 式复核 8 个点，**全部逐字命中**：
```
8 卡 B/C/D = 427,643 / 427,643 / 485,610   ← 8 卡实测
②c 档 C/D  =  595,404 /  777,318           ← 051/050 的预测
```

⇒ **`tokens()` 本来就是对的**（`cdiv(max_len, BLOCK)` 用模块常量 `BLOCK=128` 管 full 与 10 个 SWA，
**只有 draft 项**用传入的 `block` —— 这正是"逐组块大小"）。我的"修正"**已完全撤回**，
`selfcheck_draft_blk.py` 与版本库**净差异为零**（`git diff --stat` 空）。

★★ **教训（比这个 bug 值钱）**：**横向对账时先读代码、再下结论** ——
我把"参数名 `block`"误读成"对所有组生效"，差点**改坏一个正确的脚本**，
而它承载的 ②c 预测（595,404 / 777,318）正是端到端臂要验的判据。
这与 `AGENTS §5b` 那些坑同源：**"看起来矛盾" ≠ "有矛盾"**。

**回退（三档）**：`VLLM_V41_DRAFT_BLOCK=128`（或删掉该 env）⇒ ②c 逐字 no-op；两处补丁都不挂 ⇒ 逐字回现状；
`draftaware` 的 capacity 补丁若要单独回退 ⇒ 档 C/D 会回到 `raise`（**不能只回退 ②c 而留着 capacity 补丁去跑 int8**）。

**已在 `050` 里被杀掉的兄弟支线**（别重开）：②b（draft FP16）= **0 收益**（FP16/BF16 同为 2 B/token，页仍 131,072）；
②d（只缩 sliding_window）= **0**；①（把 state 移出 draft 槽）= **0 或负**。

**诚实边界（全部标【未确认】）**：
1. **②c 端到端零实测** —— 本日志的全部收益都是 **`050` 零参数模型的预测**（该模型 6/6 逐字命中实测，但**不含 draft=64 这一点**）；
2. ★ **在 A2 真权重、图模式、含投机解码下跑过一条臂**才算数 —— 判据 = 容量 **595,404 / 777,318** + 图捕获成功 + **`SpecDecoding` 四项不降** + `sha` 与冷算参考一致；
3. **卸载池 `OFFLOAD_GB=56` 要复算**（+5.0%）；**短 prompt（<1024）的命中损失**（`047` §11-2）在 ②c 下未重测；
4. **池里 `sw_chunks` 1→2** 的副作用会叠在 §3 的 +5.0% 之上 —— 端到端臂要同时看 `BlockRemoved:CPU` 是否为 0。

---

## 5. 与用户已定决策的关系（★ 别写反）

**用户 2026-09-22 决策：保留投机解码** ⇒
* **⑤a（`SPEC_ON=0`）已否决**，只作诊断：它确实跑出了 `GPU KV cache size = 863,318`（与零参数预测逐字相同），
  但那条臂**自检 FATAL（P1/D2/L1 全 0，exe 早期失败）** ⇒ 它的价值只剩"证明 draft 组是 slots0–2 的 binding"这一条归因；
* **②c 是"保投机"路线的序 1**（`INT8-CASE.md` / `A2-DEPLOY-NOW.md` 的表已按此重排）。

## 6. 交付物

| 件 | 位置 |
|---|---|
| 生成器（2 处改动的实现） | `a2/agents/T_draftceiling/patch/patch_draft_blk.py` |
| 三臂单元自检 | `a2/agents/T_draftceiling/scripts/selfcheck_draft_blk.py` |
| 逐槽算术复跑 | `a2/agents/T_draftceiling/slot_arith.py` + `out/slot_arith.{txt,json}` |
| 本日志 | `a2/logs/051-20260922-draft-block64.md` |
