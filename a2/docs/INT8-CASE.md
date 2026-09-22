# int8 KV 容量杠杆：一页论证（`logs/044`→`047`）

> ## ⚠️⚠️ **2026-09-22 10:4x 更正（`050` 的逐槽算术）：不是"零收益"，是「档 C ×1.000 / 档 D ×1.1356」**
>
> ### ★★★ 但 int8 在 A2 上**有一条 HBM 容量之外的收益：宿主内存 ×1.3146**（`048` 实测）
> ```
> 宿主实占（8 rank）：
>   档 B（16 张量，Σpage=910,208 B）  = 197.21 GiB（per-worker 24.65 GiB）
>   档 C（20 张量，Σpage=832,128 B）  = ★ 150.01 GiB（per-worker 18.75 GiB）
>   ⇒ ★ 省 47.20 GiB = ×1.3146
> ```
> **机制**：20 张量里 **SWA 平面真的缩了**（`pages=[65536, 8192, 128, …, 1024, …]`，Σpage 910,208→**832,128**），
> 而 `worker_kv_bytes_per_block` 仍是 131,072 ⇒ **L1 的行空间变小** ⇒ 同样 `OFFLOAD_GB=56` 的宿主占用降 **24%**。
> ★★ **所以 int8 在 A2 上的真实价值是"省 DRAM 池"，不是"HBM 装更多 token"** ——
> 这与 tiny 上的表现（HBM ×1.4655）**完全不同**，因为 tiny 没有 draft 组。
> ⇒ **16×128K 的池：197.21 → 150.01 GiB**（同样的 `OFFLOAD_GB=56`）。
>
> **`R_8card_int8` 的两条 8 卡真权重实测**：
> ```
> 档 B（纯 BF16）             ：GPU KV cache size = 427,643 tokens
> 档 C（int8 SWA+ring16）     ：GPU KV cache size = 427,643 tokens  ← ★ 逐字相同 ⇒ ×1.0000
> 档 D（+ KV8 双平面）        ：GPU KV cache size = 485,610 tokens  ← ★ **×1.1356**
> ```
> ⇒ **档 C 零收益；档 D 有 ×1.1356**（tiny 上是 ×1.4655 / ×1.9133）。
>
> **根因（`050` 的逐槽算术，与 4 点实测闭合）**：**draft 组把 slots 0–2 的页顶住了**
> ```python
> # vllm_ascend/core/deepseek_v41.py:51-56
> class DeepseekV41DraftSWASpec(AscendSlidingWindowMLASpec):
>     """DSpark SWA owned by G12, aliasing target slots at distinct block IDs."""
>     def __post_init__(self):
>         if self.dtype != torch.bfloat16 or ...:
>             raise ValueError("Aurora DSpark requires one uncompressed BF16 KV plane")
> ```
> | slot | 候选（档 B → 档 C → 档 D） | capacity B / C / D | binding |
> |---|---|---|---|
> | **slot0–2（×3）** | kv+index(r2) 73,856→73,856→**41,600**；state 131,072→**65,536**；swa×10 131,072→**66,560**；**draft 131,072（不变）** | **131,072 / 131,072 / 131,072** | 档 B：**state=swa=draft 三并列**；档 C/D：**draft 独占** |
> | **slot3（×1）** | kv+index(r1) **147,712→147,712→83,200**；swa×10 131,072→66,560 | **147,712 / 147,712 / 83,200** | 档 B/C/D：**long_kv+index**（与 draft 无关） |
> | **Σ** | | **540,928 / 540,928 / 476,416** B/block | **×1.0000 / ×1.0000 / ×1.1354** |
>
> ⇒ **档 C 零收益的机制**：档 B 的 slots0–2 **本来就已经是 131,072**（state FP32 = SWA BF16 = draft BF16 **三并列**）；
> 档 C 只把 state/SWA 压到 65,536/66,560，**draft 仍 131,072** ⇒ **页逐字不变**。
> ⇒ **档 D 有收益的机制**：它把 **slot3 的 `long_kv+index` 从 147,712 压到 83,200**（那一格与 draft 无关）。
>
> ★★ **为什么 tiny 六轮全绿也没发现**：**tiny 没有 draft 组**
> ```
> tiny  config: num_nextn_predict_layers = 0, dspark_target_layer_ids = []
> 真权重 config: num_nextn_predict_layers = 3  ⇒ 多一个 draft 组（13 组 vs 12 组）
> ```
> ⇒ `plan_cache_slots` 的 draft 分支**整段跳过** ⇒ **这一格从没被跑过**。
>
> ★★ **而且 `T_draftceiling` 顺手纠正了我两处算术**：
> 1. **slot3 的 `kv+index` 是 147,712（不是 73,856）** —— 73,856 是 slot0–2 的 ratio-2 值，且**它根本不 binding**；
> 2. **FP16 draft 一分钱都省不下来** —— FP16 与 BF16 同为 2 B/token，页还是 131,072。
>
> ★ **实测对账（4 点闭合，误差 ≤0.06%）**：
> ```
> tiny 档C 33,279/33,295  tiny 档D 43,444/43,469  8卡 档C 427,643/427,643  8卡 档D 485,551/485,610
> ```
>
> ★ **另一个独立阻塞**：档 C 在 **`FULL_DECODE_ONLY`** 下**捕获期炸**（`dsa_v41.py:436` 的 `.item()` 被 spec-decode 误判 ⇒ `EE1016`）；
> **而档 D 那次捕获成功**（该臂 serve.log 里 `EE1016` 计数 = 0）—— 两者的差别需要 `S_graphfix`（049）说清。
>
> **⇒ 若 draft 也能缩到 ≤73,856（档 C）/ ≤66,560（档 D），可恢复到 ×1.4648 / ×1.9122**【推断】
> （`T_draftceiling` 说**不需要 int8**：64 行块的 BF16 draft（65,536）也行 ⇒ ×1.9104）。
> **⇒ 见 `logs/050` 的四条路线评估。**

> **读者**：决定"要不要开 int8"的人。**一句话**：`×1.4655` / `×1.9133` 的容量**已实测拿到**，
> 代价是**每个命中请求多算 ≤1023 个 token**，且 **store 侧零改动、池需求不变**。
> 结论标记：【实测】/【推断】/【未确认】。

---

## 1. 值多少（★ 2026-09-22 12:5x **重写**：把"tiny 比例"与"A2 实测"分开，并交代**原始目标**）

### 1.0 ★★ 先说原始目标：**×1.84（4421 → 2405 B/token）没有达到** —— 如实交代

任务书里写的是 **4421 → 2405 B/token（×1.84，3.50M → 6.43M token）**。**实测结果是：**

| | 目标 | **A2 真权重实测** | 差距的原因 |
|---|---|---|---|
| HBM 容量 | ×1.84 | **档 C ×1.0000 / 档 D ×1.1356** | ★ **draft 组（BF16、block=128）把 slots 0–2 的页顶死**（`050` 逐槽算术，4 点实测闭合） |
| 宿主 DRAM 池 | —（任务书未定） | ★ **×1.3146**（197.21 → **150.01 GiB**） | int8 把 SWA 平面真的压小了 ⇒ L1 行空间变小 |

⇒ **"×1.84"这个数只有一种走法能碰到：②c（draft block 128→64）**，预测 **×1.8177**（`051`，端到端在验）。
**在那条落地之前，不要对外说"int8 拿到了 ×1.84"。**

### 1.1 两套几何必须分开看（**这就是之前那张表最大的问题**）

| 几何 | 档 B | 档 C | 档 D | 说明 |
|---|---:|---:|---:|---|
| **tiny**（无 draft 组）| 22,719 | **33,295**（×1.4655） | **43,469**（×1.9133） | `050` 的零参数模型 6/6 逐字命中 |
| ★ **A2 真权重**（多一个 draft 组）| 427,643 | **427,643**（**×1.0000**） | **485,610**（**×1.1356**） | 8 卡实测（`048`/`050`） |

★ **两张表的倍率差 3 倍以上**（tiny ×1.9133 vs 真权重 ×1.1356）——
**拿 tiny 的倍率去讲 A2 的收益，是本项目踩过的最贵的一次口径错误**（`050` 专门更正过）。

### 1.2 那 int8 在 A2 上到底值什么

```
HBM 容量     档 C ×1.0000（无收益）、档 D ×1.1356
★ 宿主内存   档 C 197.21 → 150.01 GiB（×1.3146，省 47.20 GiB）   ← 这才是主要收益
代价         int8 的 +1.3~1.8% decode 时延
```

#### ★★ 1.2b 换算成"并发 × 上下文"的边界（`logs/061`，零参数模型算出来的）

**这是"容量倍数"真正能让人感知的形式**（口径：`avail = 56 GiB = 60,129,542,144 B`、`max_len = 131072`、
`max_seqs = 16` —— 与 `logs/042` 的 8 卡实测同口径，模型在那一点**逐字命中**）：

| 档 | 每请求 KV（宿主） | 16 × 128K | **32 × 128K** | 64 × 128K |
|---|---:|---|---|---|
| 档 B | **13.16 GiB** | 210.6 GiB | **421.3 GiB**（= 余量 442 的 **95%**，⚠️ 紧到不可用） | ⛔ 不可能 |
| **★ 档 C** | **8.76 GiB** | 140.2 GiB | ★ **280.4 GiB**（**63%**，可行） | ⛔ 仍不可能 |
| **★ 档 D** | **7.10 GiB** | 113.6 GiB | ★ **227.3 GiB**（**51%**，宽松） | ⛔ 仍不可能 |
| ②c（预测） | 6.29 GiB | 100.7 GiB | 201.4 GiB | ⛔ |
| ②a（预测） | 5.99 GiB | 95.8 GiB | 191.6 GiB | ⛔ |

⇒ ★★ **一句话**：**"32 × 128K 从不可能变可行"是档 C 带来的；档 D 让它变宽松。**
★ 但 **64 × 128K 在任何档下都不可能**（最少也要 383 GiB 宿主，且那还是模型值）——
**别把"×1.91"读成"并发能翻好几倍"**：它翻的是**单请求 KV 的账**，
而"能开多少会话"还要乘上**服务本身与宿主余量**。

> ⚠️ 上表的**宿主口径有 ±10~15% 的不确定度**（`P2_WORKER_HOST_BYTES` 只测过"张量总量"，
> `torch` 分配器的碎片/对齐/`aclrtHostRegister` 的额外占用都没进账）⇒ **按 1.15× 留余量**：
> 档 C 的 32×128K 实际按 **≈322 GiB** 规划，仍在 442 GiB 之内。
> **精确值是 `logs/042` 的实测（197.21 / 150.01 GiB），本表是把它按公式外推**——口径见 `logs/061`。
★ **它改变的是"能开多少会话"**（按宿主算）：
```
16 × 128K：档 B 197 GiB（余量 442 的 45%）→ 档 C 150 GiB（34%）
32 × 128K：档 B ≈394 GiB（紧）           → 档 C ≈300 GiB（可行）
1M 场景  ：1M = 4421 B/token × 1,048,576 = **4.32 GiB/rank**（8 rank 合计 34.5 GiB HBM，见 KV-CACHE-ACCOUNTING.md）
           ⇒ 档 C 的 150 GiB 宿主池 ≈ 覆盖 **11 个 1M 会话**的 KV
```

---

## 2. 为什么需要"APC 对齐"这个补丁（**这是 int8 能用的前提**）

```
① 上游 max_cache_hit_length = num_tokens - 1  只对齐 block_size、不知道压缩比
   ⇒ 4096-token prompt 的命中边界 = 4095（奇数）⇒ compress_ratio=2 的组【跨在边界上】
② replay 时必须回读 compressor state ring 里 token 4094 那一行
   而 ring 组 prefix_cacheable=False、【不参与卸载】⇒ 那一行的原始投影【从未被存过】
③ int8 几何把它【解读成 NaN】（int8 平面字节 × FP16 视角）⇒ 翻 token（D/F ❌ 14/16、15/16）
   ★ 纯 BF16 几何也在读别家字节（31/32 行/步）但 F32 视下【永不产生 NaN】⇒ 静默、不翻 token
```

**修法（`VLLM_V41_APC_ALIGN=3`）**：命中长度向下对齐到**段栅格**
（= 参与卸载的 full-attention 组的 `tokens_per_chunk`，V4.1 = **1024**，运行期现算）。
`4096 → 3072` ⇒ **落点正好是 store 侧已经保留的段尾 chunk** ⇒ ★ **store 零改动**。

---

## 3. 代价（**就这一条**）

```
每个命中请求的可复用前缀从「4095 token」变成「3072 token」
⇒ 每请求多算 ≤1023 token = 按 128K 生产口径 +0.78% prefill
⇒ store 侧逐字不变：BlockStored:CPU = 714、GPU→CPU = 196,689,920 B（与基线逐字相同）
⇒ 021 的 4.89× 倍率不变 ⇒ A2 的 OFFLOAD_GB=56 与池需求【不需要重算】
⇒ 时延：D 几何三条臂 fill p50 542.2 / 543.1 / 546.0 ms（±0.7%）
```

**为什么这个代价比"对齐到 ratio"小**：
| 对齐单位 | 命中边界 | store 侧 | 池需求 | 是否采纳 |
|---|---:|---|---|---|
| **段栅格（1024）** | **3072** | ★ **零改动** | **不变** | ★ **采纳** |
| ratio（2） | 4094 | 要把 SWA 尾部 1→2（**+24%**） | **144 MiB 池装不下 ⇒ 溢出归零** | ⛔ 否决 |
| block_size（128） | 4096 | 零改动 | 不变 | ⛔ hit 为 0（不可用） |

---

## 4. 证据（**十一条判据全过**，`047`）

| # | 判据 | 结果 |
|---|---|---|
| ① | **D 几何** J2 ❌14/16 → ✅ | ★ **✅ 0/16**，`fill = replay = 24b570535f58…`（与冷算参考逐字相同），**真命中**（`CPU→GPU=184.4 MB`、`hits=49,152`）|
| ② | 探针读数 | ✅ `pre_len` 4095→**3072**、`used` 1→**1024**、`rows_changed` 1/32→**32/32**、`nan_rows` 32→**0** |
| ③ | **F 几何** J2 ❌15/16 → ✅ | ★ **✅ 0/16**，容量 **43,469 = ×1.9133** |
| ④ | C0 守门员 | ✅ 0/16，容量 22,719 不变 |
| ⑤ | D + 池 1 MiB 冷算 | ✅ `q-a8` 冷算 sha == 热臂 replay sha |
| ⑥ | 容量不退化 | ✅ **33,295 / 43,469 / 22,719** 逐字不变 |
| ⑦ | 四条判据不回归 | ✅ store 侧逐字相同（见 §3）|
| ⑧ | **`021` 变长前缀** | ✅ `hot-replay == cold-replay`，**且 4.89× 倍率不变** |
| ⑨ | 复跑同 sha | ✅ `q-a9` 与 `q-a3` 的 fill/replay/hits 逐字相同 |
| ⑩ | prefill 时延 | ✅ ±0.7% |
| ⑪ | `ratio=1` no-op | ✅ 6 组假 config + `n=1..4096` 全扫 + `mode 0` 连 hook 都不装 |
| ★ | **反例臂** | ★ `APC_ALIGN=0` **逐字复现 ❌14/16 + `6a47dd65f1ff`** ⇒ **判据有判别力** |

**两条可复跑的判决脚本**（不占卡、只读原始 `client.json`）：
```bash
python3 a2/agents/Q_apcrecord/scripts/offline_selfcheck.py   # 75 PASS / 0 FAIL
python3 a2/agents/Q_apcrecord/scripts/judge_047.py           # 33 PASS / 0 FAIL（含反例臂判据）
python3 a2/scripts/selftest_apc_align.py                     # 38 PASS / 0 FAIL（发布包两版各 19）
```

---

## 5. ⚠️ 上线前必须知道的三件事

| # | 事项 | 状态 |
|---|---|---|
| **1** | ⛔⛔ **8 卡真权重 + 图模式（生产是 `FULL_DECODE_ONLY`）→ 捕获期直接炸** | ★ **这是当前唯一的阻塞**（见 §5.1） |
| **2** | ★ **短 prompt（<1024 token）在 mode3 下不再命中池** | 【未确认】：请求照常重算、**无正确性影响**；若将来要服务短 prompt 需另设更小的对齐单位 |
| **3** | ★★ **`J2 ✅` 本身不能单独当判据** | 已固化成结构性判据：**任何命中臂必须 `CPU→GPU > 0` 且 `hits > 0`**（否则是"池溢出→整段重算→sha 当然等于冷算参考"的假阳性） |

### 4.1 ⚠️⚠️ **2026-09-22 14:2x：「热 replay == 冷算参考」这条判据在 8 卡上出现问题（正在定性）**

> ★ **这是从 `R_8card_int8` 的 8 卡真权重实测里发现的**，**在定性出来之前，本文件与交付件里
> "逐字相同 ⇒ 取回保真"的说法一律降级为【未确认】**。

```
热臂 r1  replay1 sha = bc2e797ab069f09c…   ← 档 C
热臂 r2  replay1 sha = bc2e797ab069f09c…   ← ★ 与 r1 逐字相同（热路径【完全可复现】）
冷参考 r1 replay1 sha = cfac77743d575952…   ← ★ 与热臂【不同】
fill（三臂 + 档 B）    = d524172f9f5ae368…   ← 全部逐字相同
逐 prompt：hot r1 vs hot r2 = 16/16 相同；★ hot r1 vs cold r1 =【只有 2/16 不同】
冷臂自证：CPU→GPU=0 / hits=0 / 无 BlockStored:CPU / replay TTFT 9154.9 ms = 热臂的 5.8×
```

**两种可能的解释（判决实验在做）**：

| 解释 | 含义 | 判决判据 |
|---|---|---|
| **(a) 系统性差异 = 真缺陷** | int8 命中取回与"全量重算"不等价 | ★ **档 B（BF16 无损池）的 hot vs cold 是否逐字相同** —— 若**相同** ⇒ 差异只来自 int8 量化 ⇒ 不是缺陷；若**不同** ⇒ 真路径不一致 ⇒ (a) |
| **(b) 量化地板（合法）** | 热路径用"池里那行的 **int8 反量化值**"更新 state，冷路径用"刚算出的 **BF16 投影**" ⇒ **取值来源不同** | `015`/`034` 实测 int8 的 `rel_L2 = 5.43e-3 / cos = 0.9999857` 就是**量化地板**；2/16 的边缘 argmax 翻转**在预期内** |

★★ **一个重要的判据修正（无论 (a)/(b)）**：
**"冷参考可复现"并不能证明"差异是缺陷"** —— 因为**量化本身是确定性的**，
一个**完全合法**的量化翻转也**必然**系统性可复现。
⇒ 正确的判据组合是：
```
(i)  热臂【自身可复现】（已有：hot r1 == hot r2，16/16）
(ii) 热 vs 冷 的差异【落在量化地板量级内】，且只在极少数 prompt 上翻转 token
(iii) ★ 上限：不允许出现"整段崩坏"（那才是真缺陷）
✗ 作废："热 replay == 冷算参考【逐字相同】"（在 int8 有损池上这条不成立是正常的）
```

★ **与目标三判据的关系（重要，别混）**：目标要求的是
**`BlockStored>0` / `CPU→GPU>0` / `hits>0` + replay TTFT ≪ fill**（已过，**12.5×**）；
**"逐字相同"是我们自己加的加强判据** ⇒ 它可以降级，**但对外口径必须跟着降**：
不能再写"取回后输出与全量重算逐字一致"。

**另**：档 C 的 `dsa_v41.py` 必须带 **scratch role 分键**（`035` §3.3 的静默覆盖 bug），
否则测出来的是"档 C + 一个已知静默 bug"的混合体。

### 5.1 ⛔⛔ 图模式阻塞（`048`，8 卡真权重实测）

```
档 C（KV8_SWA=1 RING_FP16=1 APC_ALIGN=3）在 8 卡 + FULL_DECODE_ONLY 下，图捕获阶段炸：
  Worker_TP0..7 同时：
    capture failed: Not_Supported(EE1016): Synchronizing a stream failed.
      Reason: Stream (stream_id=31) during the capture stage is not supported.
  Python 栈（8 rank 逐字一致）：
    model_runner_v1.py:5594 capture_model → dsa_v41.py:898 forward
      → :711 _attention → :797 _native_attention → ★ dsa_v41.py:436 in kv8_ori_plane

第 436 行（宿主同步）：
  pages_per_req = int(((lens - 1) // block_size - window_start // block_size + 1).max().item())
★ 旁边代码自己写着 `# Prefill: ... Eager only, hence the host syncs`
  ⇒ 该分支被假定"只在 eager 的 prefill 里跑"，但图捕获时被走到了。

根因（分支判据）：
  if query_rows == num_reqs:   # ← decode 分支（device-side、capture-safe）
  else:                        # ← prefill 分支（.item() ⇒ 捕获期炸）
  捕获时是【spec-decode 的 decode 批】（num_spec_tokens=5 ⇒ 每请求 6 行 query）
  ⇒ query_rows = 6 × num_reqs ≠ num_reqs ⇒ 【误走 prefill 分支】
★ 判据：speculative-config 里 num_speculative_tokens=5；
  且档 B（纯 BF16）同一条链、同样图模式、同样 spec 配置【捕获成功】
  （BF16 的 SWA 平面不走 kv8_ori_plane）。
```

★ **这不是今天的补丁引入的**：`[APC_ALIGN]` 在调度器侧、**不在被捕获的 forward 里**。
它是 **int8 KV8 代码自身的既存缺陷**（`if query_rows == num_reqs` 在 spec-decode 下不成立）。

★ **也解释了为什么单卡 tiny 六轮全绿**：tiny 上**从来没同时具备**
`int8 + spec-decode + 图模式` 这三个条件 —— **这是单卡验证的盲区**，
说明"必须在 8 卡真权重 + 真实 spec 配置上图模式跑一次"这一步不可替代。

★ **另两个只在真权重上暴露的真问题**（`R_8card_int8` 已定位并修好）：
1. **槽位页被 draft 顶爆**：int8 让 `Σstate`/`Σswa` 同时缩小 ⇒ 页缩到 draft 的 BF16 窗口面以下
   ⇒ 上游 `raise Aurora DSpark geometry must match target SWA and fit its existing slot`。
   **tiny 只有 12 组（无 draft 组）⇒ 这一格从没被跑过**；用 `patch_slots_draft.py` 把容量改成
   `max(kv+index, aliases, draft)` 后已通过。
2. **8 卡链自己挂了一份生产 `model.py`**，而 KV8 接线也在 `model.py` 里 ⇒ `Duplicate mount point`；
   用 `merge_model.py`（difflib 现算 3 个 hunk + 自证）合成后已通过。

**⇒ 修法**（`S_graphfix` 在做）：把 decode 分支的判据改成 spec-decode aware，并消除该分支内的 `.item()`。
**在它修好之前，档 C / 档 D 只能以 `--enforce-eager` 运行**（性能代价另算）。

---

## 6. 怎么开

```bash
# 档 C
KV8_SWA=1 KV8_RING_FP16=1  bash a2/scripts/serve_a2_offload.sh
# 档 D
KV8_SWA=1 KV8_RING_FP16=1 KV8_FULL=1 KV8_PREFILL=1  bash a2/scripts/serve_a2_offload.sh
# ★ APC_ALIGN 自动置 3（若开了 int8 却没给，脚本会打印告警）
# ★ 开 int8 后 P2_COMP_JSON 必须换成 20 张量那套：[[0,2,3,4,5,6,7,8,9,10,11],[1]]
# ★ 必挂 0001（scheduler 卸载补丁），否则 assert isinstance(kv_cache_spec, FullAttentionSpec) 必炸
```

**回退**：`VLLM_V41_APC_ALIGN=0` 或 `KV8_*=0` ⇒ 逐字回到档 B。
> ★ **逐槽算术已可复跑**（`python3 a2/agents/T_draftceiling/slot_arith.py`，不占卡、不 import torch）：
> **四个锚点全部闭合**【实测】——
> ```
> tiny 档C 33,279/33,295 (−0.047%)   tiny 档D 43,444/43,469 (−0.058%)
> 8卡  档C 427,643/427,643 (0.000%)  8卡  档D 485,551/485,610 (−0.012%)
> ```
> ★ **A2 池的结构事实**：`540,928 = 3 × 131,072(draft 窗口面) + 147,712(long_kv+index)`
> ⇒ **容量天花板是被"3 个投机解码的窗口页"钉死的，不是被量化精度钉死的。**
>
> ★★ **决策（2026-09-22，用户）：保留投机解码。** A2 是单流/小并发场景，DSpark（接受长度中位 3.58）对单流吞吐不可替代
> ⇒ **⑤a（关投机）已否决，只作诊断/归因臂**；所有交付路线必须在 `--speculative-config dspark` 开启下成立。
>
> ★★★ **2026-09-22 13:2x 追加：②c 的机制已在 slot 层实测（单 die，`054`）** ——
> `capacity = max(kv+index, aliases_max, draft)` 这一行，四个臂逐字给出：
> ```
> 档 B（aliases_max=131072）:  draft 131072 → capacity 131072
>                             draft  65536 → capacity 131072   ← ★ 纹丝不动（aliases 顶住）
> 档 D（aliases_max= 66560）:  draft 131072 → capacity 131072   [draft-aware]（legacy 66560）
>                             draft  65536 → capacity  66560   ← ★ 减半
> ```
> ⇒ **②c 在档 B 上只拿到副作用（draft 页数 130→259 ⇒ 容量 ×0.9242），在档 D 上才把 draft 拉下 binding ⇒ ×1.5570。**
> ★ 这也把 `050` 那句"draft 是 slots 0–2 的 binding 项"从**算术推断**升级为 **slot 层直接实测**。
>
> ★ **保投机的三条能解开天花板的路线**（`050`，按投入产出排序）：
> | 路 | 收益（8 卡） | 改动面 | 风险 |
> |---|---|---|---|
> | **① ②c draft block 128→64（保 BF16、保投机）★ 改动清单已定稿** | **×1.8177**（**777,318**） | ★ **2 文件 / 2 处，默认关**（见 `051`） | 风险已查清：窗口跨块（`051` §2 三条判据）；DRAM 池 **+5.0%** |
> | **② ②a draft 也 int8（保投机）** | ×1.9122 | 3 处 | ⚠️ 依赖 `S_graphfix`（draft 走 prefill 分支的 `.item()`） |
> | **③ ③c draft 做 per-request scratch** | ×1.9122 | 中等 | ⚠️ graph-stable |
> | ~~④ ⑤a 关投机解码~~ | ~~×2.0188（863,318）~~ | 零代码 | ★ **已否决**（用户决策）—— 数据只用于证明"draft 组是 slots0–2 的 binding" |
> ★ **②b（draft FP16）零收益**（FP16/BF16 同为 2 B/token，页还是 131,072）—— 已判死。
>
> ### ★★ `050` 的"零参数精确容量模型"（**6 个实测点逐字命中，不是拟合**）
> ```
> num_blocks        = avail // Σslot_pages − 1          （−1 = null block）
> 每请求块数 BPR    = cdiv(max_len, block)              ← full 组
>                     + 1                              ← state 组
>                     + 10 × P_swa                     ← 10 个 target SWA 组
>                     + P_draft                        ← draft 组（关投机时为 0）
>   P_x = cdiv(min(window − 1 + max_in_flight, max_len), block_x) + 1
>   max_in_flight = max_concurrent_batches(2) × max_num_batched_tokens(8192)
> tokens = int(num_blocks / BPR × max_len)
> ```
> | 格 | Σ | BPR | 预测 | 实测 |
> |---|---:|---:|---:|---:|
> | tiny B/C/D | 540,928 / 369,280 / 282,880 | 715 | 22,719 / 33,295 / 43,469 | **逐字 ✅** |
> | 8 卡 B / C | 540,928 | 2,471 | 427,643 | **逐字 ✅** |
> | 8 卡 D | 476,416 | 2,471 | 485,610 | **逐字 ✅** |
> | ~~⑤a（诊断臂·已否决）~~ | ~~282,880~~ | ~~2,341~~ | ~~863,318~~ | 只作归因：证 draft 组是 binding |
> | **②c（预告）** | **282,880** | **2,600** | **777,318** | ⏳ 待验 |
>
> ★ **为什么"关投机"能解锁这么多（机制说明，不是路线推荐）**：draft 不只是"自己占一页"，**它同时是 slots 0–2 的 binding 项**
> ```
> 档 D 现状：slot0-2 = max(kv+idx 41,600, state 65,536, swa 66,560, draft 131,072) = 131,072
> ⑤a 关spec：slot0-2 = max(41,600, 65,536, 66,560)                               =  66,560  ← 跟着缩
> ```
>
> ★ **②c 的风险（已查清，不是"未知"）**：
> 1. **算子是绝对 token 坐标寻址**（`block = pos // storage_block_size`、`page = table[b, block]`），
>    **块表是全长行**（`cdiv(max_len, block)`，与窗口无关）⇒ 窗口跨块不影响寻址；
> 2. **KV manager 无假设**（`max_admission_blocks_per_request` 按块数记 ⇒ draft 每请求页数 130→259）；
> 3. ⚠️ **唯一硬编码**：`kv8_ori_plane` 的 decode 分支写死 `pages_per_req = 2`（注释"at most two pages"）
>    ⇒ block=64 时最多 3 页。**但那条只在 int8 平面上跑 ⇒ ②c（draft 保 BF16）不走它**；
>    **只有「②a + ②c 组合」才需要把它改成 `cdiv(window, block) + 1`**。
> 4. 两个**可量化副作用**：HBM 收益从 ×1.91 掉到 **×1.8177**；DRAM 池 `sw_chunks` 1→2 ⇒ 该组每段 unit 2→3
>    ⇒ 按 `042 §3` 反解总需求 **+4.8%**（`OFFLOAD_GB=56` 要复算）。
