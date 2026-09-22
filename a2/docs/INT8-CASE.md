# int8 KV 容量杠杆：一页论证（`logs/044`→`047`）

> ## ⚠️⚠️ **2026-09-22 10:4x 更正（`050` 的逐槽算术）：不是"零收益"，是「档 C ×1.000 / 档 D ×1.1356」**
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

## 1. 值多少

| 档 | 配置 | 容量 | tiny `GPU KV cache size` | **16×128K 的 DRAM 池** | **HBM KV cache** |
|---|---|---:|---:|---:|---:|
| **档 B** | L5 + L1 | ×1.000 | 22,719 | **197.21 GiB**【实测·8 卡】 | 3.50M token |
| **档 C** | + SWA-int8 + ring16 + APC 对齐 | **×1.4655** | **33,295** | **≈135 GiB** | **≈5.13M token** |
| **档 D** | + KV8 双平面 + prefill 融合 | **×1.9133** | **43,469** | **≈103 GiB** | **≈6.70M token** |

★ **它改变的是"能开多少会话"**：
```
档 B：16 × 128K 已用掉 197 GiB（余量 442 的 45%）
档 C：同样 16 × 128K 只要 135 GiB ⇒ 可上 32 × 128K（269→184 GiB）
档 D：32 × 128K 只要 141 GiB ⇒ 64 × 128K（281 GiB）从"不可能"变"可行"
1M 场景：单会话 82 GiB（档 B）→ 56 GiB（档 C）→ 43 GiB（档 D）
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
