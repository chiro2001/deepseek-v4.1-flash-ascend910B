# 050 — draft 天花板：**DSpark 的 BF16 窗口面顶住 slots 0–2，int8 在 A2 上只剩 ×1.1356**

> 2026-09-22 10:4x–11:xx CST。执行：子代理 **T_draftceiling**。机器：**<workstation>（A3 的只读侧）**，
> **本轮（1)(2) 全程不占卡**；只读 `A3-node1` 的 `R_8card_int8` 原始产物（ssh 只读 `grep`，无写、无 scp）。
> 没写 `upstream-v41/`；没用 `/tmp`（`TMPDIR=~/tmp/20260922/T_draftceiling`）；没碰任何别人的容器/文件。
> 产物：本日志 + `logs/raw/050-draft-ceiling/` + `agents/T_draftceiling/`。
> 标记约定：**【实测】**= 有判别力的原始数据；**【算术】**= 由源码几何逐项算出（可复跑脚本）；
> **【推断】**= 由【实测】/【算术】外推但没直接测；**【未确认】**= 没跑到。

> ★★ **后续更正（2026-09-22 11:3x，用户决策）**：**保留投机解码** —— A2 是单流场景，DSpark（接受长度中位 3.58）的收益不可替代。
> ⇒ 本日志 §3.6 推荐表里的 **⑤a（`SPEC_ON=0`，原序 0）已否决**，只作诊断/归因臂，**不得作为交付配置**；
> 交付路线按 **②c（保 BF16、保投机）→ ②a → ③c** 排序，判据必须含 `SpecDecoding` 四项不降。

---

## 0. 一句话结论

| 问题 | 答案 |
|---|---|
| **draft 是不是"顶住"了？** | **是，但只在 slots 0–2，而且它顶的是"本来就已经是 131,072 的那个顶"**：档 C/D 里 slots0–2 的候选是 `kv+index 73,856/41,600`、`state 65,536`、`SWA 66,560`、**`draft 131,072`** ⇒ **draft 独占 binding**。**【算术 + 反证】** |
| **slot3 呢？** | **与 draft 无关**：binder 是 **ratio-1 的 long_kv+index = 147,712**（★ 任务书里的 73,856 是 ratio-2 的值，它一点都不 binding）。**【算术】** |
| **档 C（int8 SWA + ring16）在 A2 值多少？** | **×1.0000**（8 卡实测 427,643 = 档 B 逐字相同）。**【实测】** |
| **int8 在 A2 上完全没价值吗？** | **不是**：★ 档 D（再加 long-KV int8 + prefill）实测 **485,610 = ×1.1356**。**【实测，本轮新读到的原始数据】** |
| **能修吗？** | **能**，但必须动 **draft 的页**（int8 或更小的 block）或 **分配器**；动 target 侧（state/SWA/long-KV）永远是 0 收益。修好后的天花板 = **×1.4648（档 C）/ ×1.9122（档 D）**，相对现在的档 D 再涨 **×1.6842（+68%）**。**【算术 + 推断】** |
| **★ 用户决策（2026-09-22 11:2x）** | **推测解码必须保留**（A2 单流场景，DSpark 接受长度中位 3.58 不可替代）⇒ **⑤a（关投机）作废**；**②c（draft block 128→64，保 BF16、保投机）升为序 1**。 |

---

## 1. ★★ 逐槽算术（任务书 §1(1)）—— 谁是 binding

### 1.1 几何来源（全部只读，逐条可查）

| 量 | 值 | 来源 |
|---|---|---|
| block_size / head_size | 128 / 512 | `DSV4_BLOCK_SIZES[128][0] = [128,128,8,32]`（`models/layer/attention/layer.py:34-36`）+ `AscendDeepseekV4SWACache.__init__` 取 `[0][1]` |
| ratio-2 槽 `kv+index` | **73,856** = 64×512×2 + (64×128×1 + 64×2×1) | `020` F2_binding 实测；`_cache_plane_sizes`（`core/deepseek_v41.py:177-184` KV8 版） |
| ratio-1 槽 `kv+index` | **147,712** = 128×512×2 + (128×128 + 128×2) | 同上（020） |
| int8 long-KV | 每行 `512×1 + 8×1 = 520 B` ⇒ ratio-2 41,600 / ratio-1 83,200 | `015`/`020`（"520 B/token"） |
| state ring（F32 / FP16） | **131,072 / 65,536** = 32×1024×(4 / 2) | `compressor.py:86-91`（`head_size = 2*width`）+ `020` |
| SWA（BF16 / int8） | **131,072 / 66,560** = 128×512×2 / 128×(512×1 + 4×2) | `020 §2`（`reshape_cache` stride 实测 131072 / 65536、scale 1024） |
| **draft（DSpark，g12）** | **131,072** = 128×512×2（BF16、单平面、`compress_ratio=1`） | `DeepseekV41DraftSWASpec.__post_init__`（`core/deepseek_v41.py:51-56`）；`plan_cache_slots` 的相等检查要求它和 target SWA 同 block/head/window |

### 1.2 四个 slot 的 `capacity = max(哪些项)`（数值）

布局（`plan_cache_slots`：`aliases = [state] + swa[i::4]`，draft 的第 i 层落在 slot i）：

| slot | 候选（档 B → 档 C → 档 D） | capacity B / C / D | **binding** |
|---|---|---|---|
| **slot0**（ratio2，state[0]、swa{0,4,…,36}、draft mtp.0） | `kv+index` 73,856 → 73,856 → **41,600**；`state` 131,072 → **65,536**；`swa×10` 131,072 → **66,560**；**`draft` 131,072（不变）** | **131,072 / 131,072 / 131,072** | B：**state=swa=draft 三并列**；C/D：**draft 独占** |
| **slot1**（同构，state[1]） | 同上（swa{1,5,…,37}、mtp.1） | **131,072 / 131,072 / 131,072** | 同上 |
| **slot2**（同构，state[2]） | 同上（swa{2,6,…,38}、mtp.2） | **131,072 / 131,072 / 131,072** | 同上 |
| **slot3**（ratio1，**无 state / 无 draft**） | `kv+index` **147,712 → 147,712 → 83,200**；`swa×10` 131,072 → **66,560** | **147,712 / 147,712 / 83,200** | B/C：**long_kv+index**；D：**long_kv+index** |
| **Σ slot_pages** | | **540,928 / 540,928 / 476,416** B/block | ×1.0000 / ×1.0000 / **×1.1354** |

★ 三句话：
1. **档 C/D 的 slots0–2 只有 draft 顶住**（73,856 / 65,536 / 66,560 全 < 131,072）；
2. **slot3 与 draft 无关**，是 long_kv+index 顶住（★ 更正任务书：73,856 是 ratio-2 的值，slot3 是 147,712）；
3. **档 B 的 slots0–2 本来就已经是 131,072**（state / SWA / draft 三者同值并列）⇒ 所以"draft 顶住"这件事在 int8 之前**看不出来**，一旦把 state+SWA 压小，draft 就成了唯一的天花板。

### 1.3 ★ 反证（有判别力的对称证据）

* **未打** `draft-aware` 补丁的 int8 臂（`r8-b2-tierC-graph` / `b2-tierC-eager` / `b2-tierD-graph` / `b3-tierC-graph` / `b3-tierC-eager` / `b3-tierD-graph`，共 **6 条**）在 `plan_cache_slots` **直接 raise**：
  `ValueError: Aurora DSpark geometry must match target SWA and fit its existing slot`；
* 同一批里**档 B（BF16）从不 raise**（`r8-a-tierB-graph` 正常起服）。
  ⇒ 这条 raise 就是 **"max(kv+index, state, swa) = 73,856 < draft 131,072"** 的实测证据（对称跑过：BF16 臂不炸、int8 臂炸）。
* int8 在档 C 里**确实生效**（排除"其实是 BF16 基线"）：档 C 图模式臂的捕获期崩栈落在
  `dsa_v41.py:436 in kv8_ori_plane` —— 该函数只在 SWA 平面是 `(payload, scale)` 元组（= int8）时才会被调用。

### 1.4 与已实测对账（4 点全部 <0.06%）

`predicted = anchor_tokens × (Σ_anchor/Σ_arm) × (每请求块数_anchor/每请求块数_arm)`
（依据 `kv_cache_utils.py:937-959`：`max_concurrency = num_blocks / Σ_groups cdiv(max_mem, page)`；
每请求块数 = full 1040 + state 1 + 10×SWA 各 1 + draft k）

| 格 | Σslot_pages | 每请求块数 | predicted | **实测** | 误差 | 来源 |
|---|---:|---:|---:|---:|---:|---|
| tiny 档 C（无 draft） | 369,280 | 1051 | 33,279 | **33,295** | −0.047% | `034`/`047` c1 |
| tiny 档 D（无 draft） | 282,880 | 1051 | 43,444 | **43,469** | −0.058% | `034`/`035` c2 |
| **8 卡 档 C（有 draft）** | 540,928 | 1052 | 427,643 | **427,643** | **0.000%** | `R_8card` `c2-tierC-{graph,eager}` |
| **8 卡 档 D（有 draft）** | 476,416 | 1052 | 485,551 | **485,610** | −0.012% | `R_8card` `c2-tierD-graph` |

★ 档 D 的原始行（本轮从 A3 只读取出，**此前没进过任何日志**）：
```
r8_r8-c2-tierD-graph_20260922_102329/serve.log:882:
  (EngineCore pid=1170) INFO 09-22 02:27:27 [kv_cache_utils.py:2235] GPU KV cache size: 485,610 tokens
```
与档 B 的**命令行逐字相同**（`--max-model-len 133120 --block-size 128 --kv-cache-memory-bytes 4294967296
--gpu-memory-utilization 0.92 --max-num-seqs 32 --speculative-config {...num_speculative_tokens:5...}`），
唯一差别是挂了 int8 影子包 ⇒ **差值只可能来自 Σslot_pages**（这也排除了"多给 HBM"的解释）。

### 1.5 ★ "如果 draft 也缩"（【推断】：`__post_init__` 现在会拒绝非 BF16 draft）

draft INT8（128×512×1 + 128×4×2 = **66,560**，与 target 的 int8 SWA 面**同形**）：

| 档 | 现在 Σ（draft BF16） | 现在 token | draft ≤ 后 Σ | draft ≤ 后 token | 相对现在 |
|---|---:|---:|---:|---:|---:|
| 档 C | 540,928 | 427,643（×1.0000） | **369,280** | **626,419**（×1.4648） | ×1.4648 |
| 档 D | 476,416 | 485,551（×1.1354） | **282,880** | **817,746**（×1.9122） | **×1.6842** |

★ 两条**必须一起记住**的边界：
1. **只要 draft ≤ 73,856（档 C）/ ≤ 66,560（档 D）就够** —— 不需要 int8：**64 行块的 BF16 draft**（64×512×2 = **65,536**）也行 ⇒ 净 **816,970（×1.9104）**，代价只是 draft 每请求 1→2 块（1052→1053）。
2. **FP16 draft 一分钱都省不下来**（FP16 与 BF16 同为 2 B/token，页仍是 131,072）⇒ Σ 与 token **逐字不变**。**【算术】**（这条看起来像"低精度就行"，实际不是。）

### 1.6 ★★★ 零参数精确容量模型（**6 个实测点逐字复现**）—— 修 §1.5 里的两个数

上一版的"每请求 1051/1052 块"是**反解出来的拟合值**（它只在比值里出现，所以比值预测没错，但**绝对 token 数**和"draft 块数 1 或 2"都不可靠）。本轮把它落成**从源码推出来的零参数模型**：

```
num_blocks            = avail // Σslot_pages − 1        ★ −1 = null block（0 号块保留）
每请求块数 BPR(臂)     = cdiv(max_len, block)            ← full 组（= max_memory_usage_pages）
                        + 1                             ← state 组（AscendCircularBufferSpec.max_memory_usage_bytes == page_size_bytes）
                        + 10 × P_swa                    ← 10 个 target SWA 组
                        + P_draft                       ← draft 组（关投机时为 0）
              P_x      = cdiv(min(window − 1 + max_in_flight, max_len), block_x) + 1
                        （`SlidingWindowSpec.max_admission_blocks_per_request`，kv_cache_interface.py:587-612）
max_in_flight         = max_concurrent_batches × max_num_batched_tokens = 2 × 8192 = 16,384
                        （config/vllm.py:540-561；mcb=2 因为 pp=1 + async_scheduling **默认开**，
                          而 vllm.py:1095-1144 的禁用名单**明确放行 dspark**）
GPU KV cache size     = int(num_blocks / BPR × max_len)   （kv_cache_utils.py:2227-2235）
```

| 格 | Σslot_pages | 每请求块数 | **预测** | **实测** |
|---|---:|---:|---:|---:|
| tiny 档 B（ring F32 / SWA BF16） | 540,928 | 715 | **22,719** | 22,719 ✅ |
| tiny 档 C（SWA int8 + ring16） | 369,280 | 715 | **33,295** | 33,295 ✅ |
| tiny 档 D（+ long-KV int8） | 282,880 | 715 | **43,469** | 43,469 ✅ |
| 8 卡 档 B | 540,928 | 2471 | **427,643** | 427,643 ✅ |
| 8 卡 档 C | 540,928 | 2471 | **427,643** | 427,643 ✅ |
| 8 卡 档 D | 476,416 | 2471 | **485,610** | 485,610 ✅ |

（tiny：avail=1 GiB、max_len=8192、bat=8192 ⇒ P = cdiv(8192,128)+1 = 65，BPR = 64+1+10×65 = 715；
 8 卡：avail=4 GiB、max_len=133,120、bat=8192 ⇒ P = cdiv(16,511,128)+1 = **130**，BPR = 1040+1+10×130+130 = **2471**。）
★ `min(...)` 在 tiny 上被 `max_model_len=8192` 夹住（⇒ 65 而不是 130），这也是**两套几何必须分开算**的原因。

**⇒ 两个修正**（本节的数**取代** §1.5 与第一报里的 what-if 数）：
1. **②c（draft block 64）没有原来那么香**：draft 的窗口 128 token 跨 **2 个 64 行块**，而 `max_admission_blocks_per_request` 是按 **block 数** 记的 ⇒ draft 组每请求的页数 **130 → 259**（BPR 2471→2600）
   ⇒ 净 **777,318（×1.8177 vs 档B；相对档 D 是 ×1.6009）**，而不是 816,970（×1.9104）。
2. **⑤a（关投机解码）比原来更值**：draft 组**整体消失** ⇒ 少掉它那 130 页/请求（BPR 2471→2341）
   ⇒ **863,318（×2.0188 vs 档B；相对档 D 是 ×1.7776）**，而不是 817,746。
3. ②a（draft int8，block 仍 128）不受此影响：**817,898（×1.9126）**（BPR 不变）。

---

## 2. 为什么 tiny 六轮全绿也没发现（第 4 格）

| 事实 | 证据 |
|---|---|
| tiny 的 config **没有 draft 组** | `agents/L1_dummy/models/model-tiny/config.json`：`num_nextn_predict_layers = 0`、`dspark_target_layer_ids = []`（`l1_dummy_provenance.json` 显示这是**故意**从真权重的 3 / `[37,38,39]` 改掉的） |
| 真权重有 3 层 draft | 同 config 的 `num_nextn_predict_layers = 3`、`dspark_target_layer_ids = [37,38,39]` |
| 代码里 draft 分支整段跳过 | `plan_cache_slots` 的 `if slot_idx < len(draft):`（`core/deepseek_v41.py:237`）⇒ tiny 上**一次也没执行** |
| 组数 / 张量数 | tiny = 12 组 / Σ权重 18；真权重 = **13 组（多 g12）/ Σ权重 20**；worker 侧真分量：真权重 = `[[0],[1..12]]`（档 B）/ `[[0,2..11],[1,12]]`（int8），tiny = `[[0],[1..11]]` |

⇒ **tiny 的 ×1.4655 / ×1.9133 是"无 draft 几何"的倍率**；在 A2 真权重（DSpark 开）上分别塌成 **×1.0000 / ×1.1356**。

---

## 3. ★★ 四条路线 + 两条自加（任务书 §1(2)）

口径：`Σ` 见 §1.2；"净倍率"含每请求块数。

### ① 把 state ring 移出 draft 所在 slot —— **【死路】收益 0 或负**
* 数：档 C 里 state 65,536 < draft 131,072 ⇒ 移出后 slots0–2 仍 131,072 ⇒ **0**；档 D 里 state 65,536 < SWA 66,560 ⇒ 仍 66,560 ⇒ **0**。（把 draft 修好之后也一样：档 C 的 binder 是 73,856、档 D 是 66,560，都不是 state。）
* "移出"在 4 槽结构里做不到：4 个 slot 被 4 个 long-KV 层绑死，state 只能落到 slot0/1/2；要新增第 5 槽就得给 Σ **加** 65,536（档 C 540,928→606,464 = **×0.892**）。
* 改动面：`plan_cache_slots` 别名分组 + `allocate_cache_config`（5 槽）+ 卸载层张量数 + `P2_COMP_JSON`。风险：动 GPU 布局。**【算术】**

### ② 把 draft 的窗口面做小 —— **【唯一能恢复全额倍率的方向】**

| 支线 | 页大小 | 档 C Σ / token | 档 D Σ / token | 改动面 | 风险 |
|---|---:|---|---|---|---|
| **②a draft INT8** | 66,560 | 369,280 → **626,419（×1.4648）** | 282,880 → **817,746（×1.9122）** | **3 处**：`DeepseekV41DraftSWASpec.__post_init__` 放行 int8+scale；`dspark.py::DeepseekV41DSparkSWACache.get_kv_cache_spec` 给 `dtype=int8, scale_dim=4, scale_dtype=fp16`；`plan_cache_slots` 的 draft 检查（已满足） | ★ 读写路径**自动复用** KV8 SWA（`reshape_cache` 用 `getattr(spec,'scale_dim',0)` 决定是否返回 tuple）⇒ 但 draft 是 **non-causal multi-token decode**，q_len=6 ⇒ 走 `kv8_ori_plane` 的 **prefill 分支（带 `.item()` 宿主同步）** ⇒ **图捕获期必炸** = `048` 的 `S_graphfix` 阻塞。**不能单独上。** |
| **②b draft FP16** | 131,072 | 540,928 → 427,643（**×1.0000**） | 476,416 → 485,551（×1.1354） | — | **0 收益，判死** |
| **②c draft block 128→64（保持 BF16）** | **65,536** | 369,280 → **661,278（×1.5463）** | 282,880 → **777,318（×1.8177）** | 2 处：`dspark.py` 给 draft 单独 block_size；放宽 `plan_cache_slots` 的 `block_size == swa_spec.block_size` 检查 | **无精度风险、无 dtype 变更**；但 ★ **"窗口跨块"是真的**，见 §3.6（draft 组每请求页数 130→259 ⇒ **吃掉了 5.2% 的收益**；另有 DRAM 池 +4.8%）。**【推断，需一条臂验证】** |
| **②d 只缩 `sliding_window`** | 131,072 | 不变 | 不变 | — | **0 收益，判死**（页大小只由 `block_size × head_size × itemsize` 定） |

### ③ 之前先补：★★ ⑤a（关投机解码）的前提已核实 ★ **该路线已否决，本节仅作机制归因**

**问题**：关掉 spec-decode 后，draft 组是"从 `kv_cache_groups` 里消失"还是"仍在、只是不用"？（这决定 ×2.0188 成不成立。）

**答：消失。两条独立证据。**

1. **代码面**：drafter（以及它的 draft 模型、mtp 层）**只在 `speculative_config` 存在时构造**：
   `vllm/v1/worker/gpu_model_runner.py:584` → `if self.speculative_config and get_pp_group().is_last_rank(): self.drafter = ...`；
   而 `DeepseekV41DraftSWASpec` 的**唯一产地**是 draft 模型自己的 cache 层（`models/deepseek_v41/dspark.py:24-33 → DeepseekV41DSparkSWACache.get_kv_cache_spec`），
   它又被 `get_layers_from_vllm_config(AttentionLayerBase)` 收进 runner 的 spec 字典（`gpu_model_runner.py:7800-7830`）。
   ⇒ 没有 `speculative_config` ⇒ 没有 mtp 层 ⇒ **没有第 13 个 group**。
2. **实测面（组清单的直接判据）**：
   ```
   8 卡 + `--speculative-config dspark`（= 本轮全部实测臂）：
     [D2_offload] KV 卸载 group 清单 n=13: [..., (12, 'DeepseekV41DraftSWASpec', 128, 3, 'DeepseekV41DraftSWASpec', True)]
     [D2_offload] 参与卸载的组：full_attention=[0] sliding_window=[2..12]；被排除的组=[1]
   tiny / 任何不带 spec 的臂（本机 raw 里 381 条，A3 上 250 条）：
     [D2_offload] KV 卸载 group 清单 n=12: [..., (11, 'DeepseekV41SWASpec', 128, 4, ...)]   ← 到 11 就结束，没有 12
   ```
   ⇒ **判据 = 起服日志的 `group 清单 n`**：**n=12 且末组是 `DeepseekV41SWASpec`** = draft 组真的没了（不是"在但不用"）。

**⇒ 写进结论（★ 已否决，本节只作机制归因）**：~~关 spec 是"今天就能拿满容量、且图模式可用"的唯一选项（零代码改动）。~~
★ **用户决策（2026-09-22）：保留投机解码** ⇒ ⑤a 的价值只剩"证明 draft 组撑住 slots0–2"，不进交付配置。
**代价（如实）**：丢掉 spec-decode 的吞吐收益 —— 具体多少**本轮没能测**（压测口径是 `--max-tokens 1`，
投机解码只用得上 1 步；现存的唯一读数是档 B 一次近乎空请求的 `Mean acceptance length: 1.50 / 10 tokens drafted`）
⇒ **该代价标【未确认】**，要在真实生成口径下另测。

★ **已跑完（诊断存档）**：`TAG=t-draft-e1-tierD-nospec TIER=D GRAPH=1 EAGER=0 SPEC_ON=0`。
**实测结果**：`GPU KV cache size = 863,318 tokens`（= 零参数模型的**第 7 个预测点、第 1 个纯【外推】命中**，逐字相同）、
`EE1016 = 0`（图捕获成功）、起服日志里 `DraftSWASpec` 出现 **0** 次（组真的消失）。
⇒ ⑤a 的机制归因成立；但**用户已决定保留投机解码 ⇒ 该路线作废**，本格只作诊断。
**三条可判伪的预测**：(a) `GPU KV cache size = **863,318**`（±0.02%）；(b) **图捕获成功**（无 `EE1016/dsa_v41.py:436`）；(c) `group 清单 n=12`。
若 (a) 落在 863.3k 附近而 (c) 是 12 ⇒ 组消失且模型成立；若 n 仍是 13 ⇒ 我的前提被推翻，×2.02 作废。

### ③.6 ★★ ②c 的"窗口跨块"风险（回答主代理 §4）

**问题**：draft 的窗口 `sliding_window=128` 在 block=64 下要跨块 —— 读路径是否假设了"窗口 ≤ 1 块"？

| 检查点 | 结论 | 依据 |
|---|---|---|
| **跨几个块** | ★ 不是 2 个，**最坏 3 个**（纯 decode）；带投机（q_len=6）时窗口+query 共 133 token ⇒ **最坏 4 个** | 纯算术：`[l−128, l−1]` 的 64 行块数 = `floor((l−1)/64) − floor((l−128)/64) + 1`，在 `l ≡ 63 (mod 64)` 时取 3 |
| **算子（`npu_sparse_flash_mla`）** | **无"≤1 块"假设**：它按**绝对 token 坐标**自己算 `block = pos // storage_block_size`、`offset = pos % ...`、`page = ori_block_table[b, block]`，块表是**全长行**（`max_num_blocks_per_req = cdiv(max_len, block)`，与窗口无关） | `kv8_ori_plane` 的 docstring（`dsa_v41.py:313-345`，引自 `sparse_flash_mla_swa_block_vector.h` 的 `GetOriSparseKeyGmOffset`）+ `kv_cache_interface.py:139-149` |
| **KV manager** | **无假设**：`_contiguous_blocks_for_hit = cdiv(window−1, block)`（64 行块 ⇒ **2**，`use_eagle` 再 +1）、`max_admission_blocks_per_request = cdiv(min(window−1+max_in_flight, max_len), block) + 1`（⇒ 130→**259**） | `single_type_kv_cache_manager.py:883-894`、`kv_cache_interface.py:587-612` |
| **`kv8_ori_plane` 的快路径** | ⚠️ **有一条硬编码**：decode 分支写死 `pages_per_req = 2`，注释是"the span is exactly one window and covers at most two pages" —— **block=64 时不成立（最多 3 页）**。**但这条只在 int8 平面上跑** ⇒ **②c（draft 仍 BF16）根本不走它**；**②a（draft int8）+ block 64** 才需要把它改成 `cdiv(window, block) + 1` | `dsa_v41.py:428-436` |
| **容量副作用 1（HBM）** | draft 组每请求页数 **130 → 259** ⇒ BPR 2471 → 2600 ⇒ 收益从 ×1.9104 掉到 **×1.8177**（相对档 D ×1.6009） | §1.6 的精确模型 |
| **容量副作用 2（DRAM 池）** | 卸载层 `sw_chunks = cdiv(sliding_window, span)`，span=块 ⇒ **1 → 2**，再 `+ is_eagle(=1)` ⇒ 该组每段 unit **2 → 3**；按 `042 §3` 的 23.5 unit/1024 token 反解，draft 占 ~2.25 ⇒ 总需求 **+4.8%**（A2 的 `OFFLOAD_GB=56` 要复算） | `K_l1_8card/patched/p2_pool.py:566-579`、`042 §3` |
| **②c 的验证判据（3 条，可判伪）** | ① `GPU KV cache size → 777,318`（若仍读到 863k 附近 ⇒ 说明 draft 页没变/组没变）；② 起服日志里 draft 组的 `tokens_per_block`（或 `max_admission_blocks_per_request`）**= 64 / 259**；③ 一条**正向对照**：同一权重、同一 prompt，block=64 的 draft 与 block=128 的 draft 的 **SpecDecoding metrics 四个数**必须一致（若接受率下降 ⇒ 说明窗口/寻址变了语义） | 本节 |

⇒ **结论**：**②c 不引入"窗口跨块"的新 bug 面**（算子/KV manager 都是按块通用的），但它**有一个必须记住的硬编码地雷**（`pages_per_req=2`，只在 int8 平面上，**是 ②a+②c 组合的前置条件**），以及**两个可量化的容量副作用**。

### ③ capacity 不再取 max / draft 走独立路径
* **③a "给 draft 单独开 slot（页仍 BF16）"：比现状更差 —— 判死。**
  draft 3 层共用 1 个 block ID ⇒ 需要 **3 个页**（同一 group 的层共享块 ID，不能把 3 层塞进一页）。
  不别名后：Σ(档 C) = 369,280 + 3×131,072 = **762,496（×0.709）**；Σ(档 D) = 282,880 + 393,216 = **676,096（×0.800）**。**【算术】**
* **③b 给 draft 一个独立"小池"**（`num_blocks ≈ max_seqs + 1 = 33`）：draft 的 HBM ≈ 3×131,072×33 = **12.6 MB ≈ 0** ⇒ 效果 = ②a 的 ×1.4648 / ×1.9122。**但** vLLM v1 的 `num_blocks` 是**全局一套**（`allocate_cache_config` 只返回一个 `num_blocks`，8 rank 还要取 min）⇒ 要改 per-group 块池（块表偏移 + 图捕获 + 卸载层 unit 记账）。
  **改动面大、风险高**；只有在 ②a/②c 都失败时才值得投。
* **③c（我加的）把 draft 从 `kv_cache_groups` 拿出来，做成模型自持的 per-request scratch**（`[max_seqs, 128, 512]`，3 层共 12.6 MB，**按请求槽位**而不是 block ID 索引）⇒ 与 ③b 同效，但**不动 vLLM 分配器/块池**；代价 = draft 的读写路径要接一套新视图（且必须 graph-stable）。**中等改动**，是 ③b 的可落地替身。
  ★★ **③c 的第一风险 = 图稳定（graph-stable）**，不是容量：draft 的 attention 是**被捕获的 decode 图的一部分**（生产 `FULL_DECODE_ONLY`，capture sizes 到 192），
  而 ③c 要把 draft 的 `ori_kv`/`block_table` 从"调度器分配的块表"换成"模型自持的请求槽位表" ⇒ 这个替换**必须只依赖图内张量（不能有 `.item()`、不能有 host 侧索引、不能每次 capture 重新分配）**，
  否则会复现 `048` 的同款 `EE1016 ... during the capture stage`（那次的根因正是宿主同步）。
  ★ **判别判据**：① 捕获阶段零 `.item()`/零 host 同步（静态检查 + `EE1016` 无命中）；② 图内 replay 与 eager **逐比特一致**（同 prompt、同权重）；
  ③ draft 的 `SpecDecoding metrics` 四个数与现状一致（接受率不降）；④ 捕获前后的 scratch 地址不变（`data_ptr()` 稳定，证明是**复用的稳定缓冲**而不是每次重建）。
  ★ 收益口径：等同 ②a（**×1.9126**）—— 因为 ③c 让 draft 页**完全不占全局块池**，效果与"draft 页足够小"等价。

### ④ 缩 long-KV 页
* **更正**：73,856 是 ratio-2（slots0–2）的 `kv+index`，**它不 binding**；**slot3 的 binder 是 ratio-1 的 147,712**。
* 只加 int8 long-KV、SWA 保持 BF16：slot3 = max(83,200, **131,072**) ⇒ **只有 ×1.0317**（= `020` 的 `long1_swa0` 行）。
  ⇒ **档 D 那 13.5% 里，SWA 量化的唯一贡献就是"把 slot3 压到 83,200 以下"**（增量 ×1.1005）；**slots0–2 的 SWA 量化零收益**。
* 再压 index 平面：已 int8（ratio-1 = 128×(128+2) = 16,640）⇒ 最多再省 1,024 B/块（scale_dim 2→1，<0.5%），不值。
* ★ **降风险建议（不是容量杠杆）**：把 int8 SWA 只用在 **slot3 的 10 层**（layers 3,7,…,39），其余 30 层留 BF16 ⇒ **Σ 与 token 完全不变**（档 D 仍 476,416 / 485,610），但 int8 读路径暴露面 **40→10 层**。代价：SWA 的 10 个组要**按 slot 重分成 4 组**（组内 dtype 必须一致）⇒ 改卸载层组清单 / `P2_COMP_JSON` / 并发记账。**【推断】**

### ⑤ 其他（我加的）
* **⑤a 关掉投机解码（`SPEC_ON=0`）：draft 组整体消失 ⇒ 档 D = ×1.9122（预测 817.7k token）**，
  **且 `048` 的图捕获炸点同时消失**（`query_rows != num_reqs` 只在 spec-decode 下成立）。
  代价 = 丢掉 spec-decode 的吞吐。★ ~~这是**今天就能拿满 ×1.91 + graph 模式**的唯一选项（建议在 c0 补一条 `SPEC_ON=0 TIER=D`，20 min）。~~
  ★★ **已否决（用户决策 2026-09-22）**：A2 单流场景必须保留投机解码；该臂只作诊断，若阻塞 ②c 则取消。
* ⑤b draft 层数 3→2：`plan_cache_slots` 硬要求 mtp.0/1/2 恰好 3 层 ⇒ **不可**。
* ⑤c 让 draft 与 target SWA 共用同一页：两者是**不同权重**的投影（draft 是 mtp 层自己的 `wkv`），字节不能共用 ⇒ **判死**。
* ⑤d draft 不缓存、每步重算窗口：decode 步里没有过去 128 token 的 hidden_states（要么留 128 步历史、要么从 embedding 重放）⇒ **判死**。
* ⑤e draft KV 放 host/DRAM（每步 H2D ≈ 393 KB/请求）：容量上等价 ③c，实现上多一层 DMA 同步 ⇒ 列为 ③c 的备选。**【推断】**

---

## 4. "draft 能不能 int8"：约束是**必要性**还是**保守**？

* 代码原文（`core/deepseek_v41.py:51-56`）：`if self.dtype != torch.bfloat16 or self.num_kv_heads != 1 or self.compress_ratio != 1: raise ValueError("Aurora DSpark requires one uncompressed BF16 KV plane")`。
  三个条件里，**`num_kv_heads == 1` 是结构性不变量**（`032` 用它证明"canonical 页里没有 TP 维"）；
  **`compress_ratio == 1` 是语义约束**（draft 是未压缩 SWA）；
  而 **`dtype == bfloat16` 是"喂给算子的平面必须是 BF16"**——这一条**可以被 KV8 的既有机制绕过**：int8 SWA 的做法本来就是
  **"写侧量化存 int8、读侧只把这一步真正要读的那几页重建回 BF16 scratch"**（`kv8_swa_store` / `kv8_ori_plane`），
  算子看到的仍然是 BF16。⇒ **这一条更像"保守/未实现"，不是物理必要性**；但它现在**真的会拦**（硬 raise）。
  **【推断】**（没有上游注释/历史说明它为什么必须是 BF16；`validate_cache_runtime` 里倒是有一句 *"Aurora's planes are always BF16. Pin the inherited DSV4 draft backend to the same layout"* ⇒ 更像"钉死布局"的工程选择。）
* **量化 draft 会不会伤接受率？**
  * **输出正确性不受影响**（投机解码的 draft 只负责提候选，target 端验证/拒绝不变；只有"接受长度"变）——
    但注意 `037` 的既存不确定性：批形状变化会让 `temperature=0` 的首 token 也抖（与 KV 无关）。
  * **服务自带现成判据**：`SpecDecoding metrics: Mean acceptance length / Accepted / Drafted / Per-position acceptance rate / Avg Draft acceptance rate`（档 B 的 serve.log:1839 就有）⇒ **A/B 直接读这四个数**，不需要新写探针。
  * **可复用的精度方法论**（`034 §4`）：真 `wkv` + 真 kernel + CPU fp64 golden，层 2/8/14 × L=1K/8K/32K ⇒ 地板 `1.6565e-3~1.6583e-3`、**FP16 只高 +0.41%**、L 涨 32× 漂 ≤+0.7%（**无累积**）。draft 侧应照抄这套口径，但**判据要换成"draft top-1 与 target argmax 的一致率"**（这才是接受率的直接前身），而不是 hidden 误差。
* **建议的最小实验**（都是【未确认】，等 ② 落地后做）：
  1. 档 D + draft int8，跑同一批 prompt，读 `Avg Draft acceptance rate` / `Mean acceptance length`；
  2. 对称跑 `SPEC_ON=0`（无 draft）的 sha 作为**输出正确性锚点**；
  3. 反例臂：`SPEC_ON=1` + draft BF16（现状）⇒ 若两者接受率差异 < 噪声，则 draft int8 可用。

---

## 5. 对交付包的影响（主代理要知道的一句）

* 任务书里的"**档 C/D 仅 tiny 几何成立，A2 真权重零收益**"应改成**分档写**：
  * **档 C（SWA int8 + ring16）在 A2 真权重 = ×1.0000**【实测 3 条臂：`c2-tierC-graph`、`c2-tierC-eager`、`a-tierB-graph` 全部 427,643】；
  * **档 D（+long-KV int8 + prefill）= ×1.1356**【实测 485,610 / 427,643】，**不是零**；其中贡献全部来自 slot3（SWA int8 的增量 ×1.1005、long-KV int8 单独只有 ×1.0317、ring16 = 0）。
  * **×1.9133（以及 ×1.4655）在 A2 上只有在"draft 页 ≤ 66,560 / 73,856"或"关掉 DSpark"时才成立**。
* ★★ **推荐路线（按投入产出，含本轮修正后的数）**：

| 序 | 路线 | 8 卡预测 token | vs 档B | vs 档D（现状） | 改动面 / 前置 |
|---|---|---:|---:|---:|---|
| **1** | ★ **②c draft block 64（保 BF16、保投机）** | **777,318** | **×1.8177** | ×1.6009 | 2 处 + DRAM 池 +4.8%；无精度风险；**判据含 SpecDecoding 四项不降** |
| 2 | ②a draft INT8（保投机） | 817,898 | ×1.9126 | ×1.6841 | 3 处；**等 `S_graphfix`**（draft q_len=6 的 `.item()`） |
| 3 | ③c draft 自持 scratch | 817,898 | ×1.9126 | ×1.6841 | 中等；**第一风险 = 图稳定** |
| ~~0~~ | ~~⑤a 关投机解码（`SPEC_ON=0`）~~ | ~~863,318~~ | ~~×2.0188~~ | ~~×1.7776~~ | ★ **已否决（用户决策）**，只作诊断/归因 |
| — | ①/②d/③a/③b | 0 / 0 / 更差 / 大改 | — | — | 见 §3 |

* ★ 一张图看懂：**A2 的 HBM 池 = 3 个 draft 窗口页 + 1 个 long-KV 页**（`540,928 = 3×131,072 + 147,712`）；
  target 的 state/SWA 面**都叠在 draft 页里面** ⇒ **int8 只能压第 4 页**（所以档 C 是 ×1.0000、档 D 只有 ×1.1356）。
  **要动就动 draft（页或归属），动 target 侧永远是 0。**

---

## 6. 诚实边界

* 本轮 (1)(2) 全部是**静态算术 + 只读对账**，**没占卡**；§1.2 的 `capacity` 值是【算术】，但因为 4 个实测点（2 tiny + 2 八卡）都在 0.06% 内闭合，**几何模型本身可视为被验证**。
* ★ **§1.6 的精确模型是零参数的**（6 个实测点**逐字**命中，不是拟合）⇒ 由此得到的 **②a/②c/③c/⑤a 的 token 预测可以当作"可判伪的预测"用**（误差来源只剩 `int()` 取整与 `num_blocks` 的 -1 约定）。
* **未做**：②c（draft 64 行块）的寻址验证、②a（draft int8）的端到端、③b/③c 的实现、draft int8 的接受率 A/B、**关 spec 的吞吐代价**。**全部标【推断】/【未确认】。**
* **⑤a（`SPEC_ON=0`）已跑完并降级为诊断**（用户决策：保投机）：三条判据**全部命中** = `863,318` ✅ / 图捕获成功 ✅ / `DraftSWASpec` 出现 0 次 ✅。**该路线作废，不进交付。**
* 本轮**没有改任何生产代码**，也**没有动 `R_8card_int8` / `S_graphfix` 的文件**（只在 c0 的锁队列里挂了一条自己的臂，`OUT/LOGD` 指向 `agents/T_draftceiling/`）。
* `R_8card_int8` 的 `r8-c2-*` 臂在**自检层被判 invalid**（`R8_trace_lines=0`、`int8_mounts<8` 之一），但：
  * 它们的 `GPU KV cache size` 行是**引擎分配阶段**打印的（在捕获/自检失败之前）；
  * 档 C 的 int8 生效有**独立旁证**（崩栈落在 `dsa_v41.py:436 kv8_ori_plane`）；
  * 档 D 的 485,610 与【算术】预测差 0.012%，而"int8 没生效"会给 427,643（差 12%）。
  ⇒ 本轮把它们当**【实测】**使用，但**建议 R 在 `048` 里补一条"int8 生效"的直读判据**（例如 `R8_SLOT_TRACE` 那次没落盘，`[R8-SLOTS]` 在结果日志里 0 命中）。

---

## 7. ★★ ②c 的交付（**序 1**，保 BF16 / 保投机解码）—— 改动清单 + 单元自检

### 7.1 ③（前置问题）`DSV4_BLOCK_SIZES` 里**有 64 吗** —— **有**

`vllm_ascend/models/layer/attention/layer.py:32-56`（三套表，按硬件档案/压缩缓存能力选）：

```
_DSV4_BLOCK_SIZES            = {128: [[128,128, 8,32], [16640,131072]],
                                64: [[ 64, 64, 4,16], [ 8320, 65536]],   ← ★ 有 64
                                32: [[ 32, 32, 2, 8], [ 4160, 32768]]}
_DSV4_COMPRESSED_BLOCK_SIZES = {128/64/32: …}      _DSV4_BLOCK_SIZES_A5_BF16 = {128/64/32: …}
```

列含义：`[0] = [mla, swa, c4_state, c128_state]`、`[1] = [page_size_padded_t1, t2]`。
`AscendDeepseekV4SWACache.__init__` 取的 `[0][1]` **正是 `swa` 那一格** ⇒ 64 档的 SWA 值就是 64，
且它的 `page_size_padded_t2 = 65,536` **恰好等于我们要的 draft 页大小**（64×512×2）
⇒ **②c 是"改一个参数"，不是"新增一档"**。
★ 上游 vLLM 的 `DeepseekV4SWACache.__init__` 本来就**硬编码 `self.block_size = 64`**
（`vllm/v1/attention/backends/mla/sparse_swa.py:79`），后端 `get_supported_kernel_block_sizes() = [MultipleOf(64)]`
⇒ **64 与 128 都是这批算子认的块大小**（128 反而是 Ascend 侧抬上去的）。

### 7.2 【精确改动清单】2 个文件 / 2 处（生成器：`agents/T_draftceiling/patch/patch_draft_blk.py`）

| # | 文件 | 锚点 | 改动 | 理由 |
|---|---|---|---|---|
| **①** | `vllm_ascend/models/deepseek_v41/dspark.py` | `class DeepseekV41DSparkSWACache(AscendDeepseekV4SWACache):`（:24） | 新增 `__init__`：`super().__init__(…)` 后 `self.block_size = int(os.environ.get("VLLM_V41_DRAFT_BLOCK","128"))`（校验 64 的正倍数） | `self.block_size` → `get_kv_cache_spec()` → `AscendSlidingWindowMLASpec(block_size=…)` → `DeepseekV41DraftSWASpec(block_size=spec.block_size)` → `real_page_size_bytes = (block//cr) × 1 × 512 × 2` ⇒ **128→131,072 / 64→65,536**。**target 的 40 层走另一个类，不受影响** |
| **②** | `vllm_ascend/core/deepseek_v41.py::plan_cache_slots` | `draft_spec.block_size != swa_spec.block_size`（:241） | 改成 `swa_spec.block_size % draft_spec.block_size != 0` | ②c 之后 draft=64 / target=128。**只放宽"块大小相等"这一条**；`head_size` / `sliding_window` / `Σdraft ≤ capacity` 三条**一条不放松** |

**不动的（逐条给理由）**：`AscendSlidingWindowMLASpec.real_page_size_bytes`（自动）、`reshape_cache`（逐字用 `spec.storage_block_size`）、
`DeepseekV41MetadataBuilder.build()`（`storage_block_size/logical_block_size` **取自该组自己的 spec**，`dsa_v41.py:1252,1261`）、
`slot_key`（按 `(ratio, storage_block_size)` 分键，64 自带一格）、
卸载层 `p2_pool.py::compute_weights`（`tokens_per_block` 从 spec 现算 ⇒ `sw_chunks = cdiv(128,64) = 2`、`reachable_tail = 2+eagle = 3`；**无需改代码，但池配额要复算**）、
`DeepseekV41DraftSWASpec.__post_init__`（仍强制 BF16 = ②c 要的）。
★ **同槽混块先例**：state 组就是 **32 行页**与 128 行页共享同一个 block ID（`AscendCircularBufferSpec`, `STATE_RING_ROWS=32`）。
★ **窗口跨块已判**：draft 的 `sliding_window=128` 在 block=64 下跨 2 块（带投机 133 token ⇒ 最坏 4 块），但
①算子按**绝对 token 坐标**寻址（`block=pos//storage_block_size`，`dsa_v41.py:313-345`）；
②块表是**全长行**（`max_num_blocks_per_req = cdiv(max_len, block)`）；
③KV manager 的 `_contiguous_blocks_for_hit = cdiv(window−1, block)` 自动变 2
⇒ **三处都没有"窗口 ≤ 1 块"假设**。唯一硬编码是 `kv8_ori_plane` 的 `pages_per_req = 2`，**它在 int8 平面上** ⇒ **②c（BF16 draft）不走它**。

### 7.3 ★★ 单元自检（**三臂对称跑**，容器内 CPU、不占 die；`scripts/selfcheck_draft_blk.py`）

| 判据 | `upstream`（纯上游） | `draftaware`（= R 的 capacity 补丁，**8 卡实测用的就是这件**） | **`patched`（②c）** |
|---|---|---|---|
| 档 B Σslot_pages | **540,928** ✅ | 540,928 ✅ | 540,928 ✅ |
| 档 C Σ | **raise** ✅（= 8 卡实测到的同一条断言） | **540,928** ✅（×1.0000，与 8 卡逐字同） | 540,928 ✅ |
| 档 D Σ | raise ✅ | **476,416** ✅（与 8 卡逐字同） | 476,416 ✅ |
| `draft64` 档 C | raise ✅ | raise ✅（⇒ ②c 补丁**必需**） | ★ **369,280**（slots=[73,856×3, 147,712]） |
| `draft64` 档 D | raise ✅ | raise ✅ | ★ **282,880**（slots=[**66,560×3**, 83,200] ⇒ **SWA binding**，draft 65,536 已不顶） |
| 预测 `GPU KV cache size` | — | B 427,643 / C 427,643 / D 485,610 **三个都逐字命中实测** | ②c：C **595,404** / **D 777,318** |
| 卸载池 Σ权重 | — | 20 | **21**（draft 组 2→3 ⇒ ★ 池需求 **+5.0%**） |
| 无 draft 组（tiny 几何） | — | — | **逐字 no-op**（540,928 / 282,880）✅ |

★ **判别力**：同一批判据在 `upstream`/`draftaware` 上**要么 raise、要么给出不同数** ⇒ 不是空断言（`AGENTS §5b` 第 3 条）。
★ 一个细节：**②c-C 的 slots0–2 = 73,856**（不是 66,560）—— 因为档 C 的 long-KV 还是 BF16，
`kv+index(73,856) > SWA(66,560) > draft(65,536)` ⇒ **②c 在档 C 上的收益来自"把 draft 拉到 kv+index 之下"**，在档 D 上才是"SWA binding"。

### 7.4 真权重端到端臂（**在 c0 排队**；`scripts/chain_2c.sh`）

```
臂 1  t-dc2-b-D ：draft block 128（flag 关）= 基线，同几何同 workload
臂 2  t-dc2-c-D ：draft block 64（flag 开）= ②c
workload：8 × 4096 → max_tokens 64（★ 不是 max_tokens=1：那样 SpecDecoding 只有 ~10 个 drafted token，没有统计功效）
交付判据（4 组）：
  ① GPU KV cache size：基线 485,610 / ②c 777,318（【算术】预测）
  ② ★ SpecDecoding 四项（Mean acceptance length / Drafted throughput / Avg Draft acceptance rate / Per-position）
     与基线逐项对比，**不许回退**
  ③ 四条功能判据：BlockStored:CPU / CPU→GPU>0 / hits>0 / replay ≪ fill
  ④ 起服日志的 `KV 卸载 group 清单` 里 index 12 那项的块大小
★ 工程说明：②c 的**生产形状**是"dspark.py 读 env"（§7.2 ①）；但 8 卡 runner 的 `inner.sh`
  **只转发白名单 env**，无法把 `VLLM_V41_DRAFT_BLOCK` 递进容器 ⇒ 端到端臂用一个**等价的文件开关变体**
  （`--variant core-only`：块大小覆写放在 spec 的 `__post_init__`，读 `/work/agents/T_draftceiling/draft_block_64.flag`）。
  语义等价的依据：对 Ascend 的 DraftSWASpec 而言 **spec.block_size 是页几何的唯一来源**
  （`real_page_size_bytes` / `reshape_cache` / metadata / 块表 / slot_key 全部由它派生），
  dspark 里那个 `self.block_size` 只用来构造这个 spec。**两臂只有这一个变量的差别。**
```

---

## 8. 复跑方式（不占卡）

```bash
python3 a2/agents/T_draftceiling/slot_arith.py --json a2/agents/T_draftceiling/out/slot_arith.json
#   → 逐槽目标/候选/binding + 4 点实测对账 + draft what-if（本日志 §1.2/1.4/1.5 的原始输出）

# ★ ②c 的单元自检（三臂对称；容器内 CPU，不占 die）
python3 a2/agents/T_draftceiling/patch/patch_draft_blk.py --core <core/deepseek_v41.py> \
        --dspark <models/deepseek_v41/dspark.py> --out-dir <patched> [--variant core-only]
for m in upstream draftaware patched; do
  PYTHONPATH=<overlay-$m> python3 a2/agents/T_draftceiling/scripts/selfcheck_draft_blk.py --mode $m
done
#   → §7.3 的表（upstream/draftaware 会 raise 或给不同数 ⇒ 判据有判别力）
```
原始数据：`logs/raw/050-draft-ceiling/{slot_arith.txt, slot_arith.json, kvsize_and_raises.txt}`。
代码：`agents/T_draftceiling/{slot_arith.py, patch/patch_draft_blk.py, scripts/{selfcheck_draft_blk.py, chain_2c.sh, run_tiny_draft_probe.sh, tdc_selfcheck.py}}`。
★ **tiny 的 mini 判别臂**（在 tiny 上合成 g12，三臂）：`GPU KV cache size` = **20,826（B，draft 在场）** →
**int8 无补丁 arm 复现 8 卡同一条 `Aurora DSpark geometry` 断言（4 次）** → **int8 + R 的 draft-aware 补丁 = 20,826（×1.0000）**
⇒ 与 8 卡档 C 的现象**逐字同构**（详见 `agents/T_draftceiling/out/t-tiny-draft-*.tdc_verdict.txt`）。
★ **这两条 tiny 臂的诚实标注**：它们的 KV size 行**有效**（分配阶段打印），但两臂**都在 runner 的探针自检门上报 FATAL（rc=9）**
（B 臂 `D2=0 scheduler=0`、Ci8fixC 臂同）—— 该门的判据是**卸载/卸载探针**，与本结论（页几何）无关。
另有一条**中间臂**先跑了"int8 SWA + long-KV int8（= 档 D 几何）"得 **23,651**，我在同一批里补跑了**真正的档 C**（`long_kv_int8=False`，自检打印 `swa_int8=True / long_kv_int8=False`）才拿到 20,826
—— ★ **这正是"不许用相邻数字顶替缺的那格"的一次实际应用**。
