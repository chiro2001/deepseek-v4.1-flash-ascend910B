# 104 — 1M + A2 同规模几何**跑通第一关**；并找到 **33.1 ms/step 的零代价修复**（chunk 连续视图）

> 2026-09-23 01:2x–01:5x CST（A3-node1 Phy-ID 8–15 + 单卡槽 c0/c1/c2）。
> 执行：**主代理**（裁决/复核）+ 子代理 `ARM_1M`（8 卡 1M 臂）/ `SWA_COMPACT`（c2）/ `BF16_AB`（c1）/ `FUSE_MULTIROW`（纯 CPU）。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

① **1M 长上下文 + A2 同规模卸载：起服与定容这一关过了** —— `GPU KV cache size = 1,943,421 tokens`
   （判据 ≥1,048,576，**1.85×**），且**换算率与 A2 差 0.006%**（242,928 vs 242,941 token/GiB）
   ⇒ "A2 同规模"从**参数像**升级为**【实测】**。
② **找到一条 33.1 ms/step、零 HBM/零宿主代价的修复**：把 SWA int8 的 `kv8_gather_pages` 从
   **非连续页视图**改成**连续 chunk 视图**（`torch.equal = True`，逐比特相同）。

---

## 1. 【实测】1M 臂（`r8-r8-1m-a2scale2`）

| 项 | 值 |
|---|---|
| 参数 | `MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=2048`、`KV_MEM_BYTES=8589934592`（8 GiB）、`OFFLOAD_BYTES=91268055040`（= A2 `OFFLOAD_GB=85` ⇒ 87,040 unit = **3.00 个 1M 会话**）、`TIER=C ENGRAM=1 DRAFT_GRAPH=1 ENGRAM_DEVICE_INDEX=0` |
| **`GPU KV cache size`** | **1,943,421 tokens**（判据 ≥1,048,576 ⇒ **通过**） |
| 错误码 | `EE1016=0 / 507057=0 / EH0012=0 / 207001=0` |
| device-index | 容器内 `=0`，日志 `[DEVICE-INDEX]` 行数 **0** |
| 池几何 | 宿主实占 `旧=67.454 GiB → 新=56.898 GiB = 1.19×`；`tensors/pages` 与 4 GiB 臂**同形**（`KV_MEM` 不影响池） |
| 起服耗时 | t0=01:34:27 → KV 定容 01:38:58（**4m31s**） |

### ★★ 换算率：与 A2 差 0.006%

- **A3（本次）**：1,943,421 / 8.0 GiB = **242,928 token/GiB**
- **A2 生产**：3,498,354 / 14.40 GiB = **242,941 token/GiB**
- 上一次 4 GiB 失败信息给的是 968,064 / 4.0 = 242,016（差 0.4%，同一个数）

⇒ **`MAX_SEQS=4 / BAT_TOKENS=2048` 的几何就是 A2 同规模口径**（对比 `MAX_SEQS=32` 的 106,911 token/GiB，差 2.3×）。

★ **首次起服失败根因（已修）**：`KV_MEM_BYTES` 仍是 `run_arm_r8.sh:54` 的 **4 GiB 默认值** ⇒
`To serve at least one request with the model's max seq len (1048576), (4.32 GiB KV cache is needed,
which is larger than the available KV cache memory (4.0 GiB)`。**不是引擎缺陷**；改成 8 GiB 后通过。

### ★ 一个**待测的容量分界**（阶段 D 的关键）

`OFFLOAD_GB=85` 的池能容 **3 个 1M 会话**，但 **8 GiB 的 HBM 只有 1.94M token** ⇒ **装不下 3 个并发 1M**。
而 **A2 是 14.40 GiB ⇒ 3.50M token ⇒ 装得下**。⇒ 复刻 A2 的"3 会话常驻"需
`KV_MEM_BYTES=15032385536`（14 GiB ≈ 3.40M token，与 A2 差 3%）。
**两种口径都要测**：8 GiB 测"HBM 不够时池是否兜得住"，14 GiB 测"A2 真口径"。

---

## 2. ★★ 【实测】chunk 连续视图：33.1 ms/step，零代价

### 2.1 机制（`SWA_COMPACT` 在 c2，主代理已复核其原始 JSON）

生产的 SWA int8 面**不是独立张量**，是**混合槽页上的 `as_strided` 视图**（`core/deepseek_v41.py:374-381`）：
`stride(0)` = 整个槽页字节数，而该面自身载荷只有 65,536 B
⇒ `torch.index_select` 在**非连续**视图上退化成"24 段各 64 KB、跨大间隔的散拷贝"。

| stride | 4,096 页 harness `gather_i8` |
|---:|---:|
| 476,416（档 D 的 Σ） | 445.5 µs |
| **131,072（档 C slot0–2 真值）** | **440.6** |
| **147,712（档 C slot3 真值）** | **448.7** |
| 66,560（载荷+scale 的"紧凑页"） | **443.7** ← ★ 紧凑页也无效 |
| 65,536（= 载荷本身） | **5.2** |
| gap 扫描（512 / 65,536） | 422 / 421 ← ★ 与跨步距离无关 |

⇒ **触发条件 = "源视图是否连续"**，不是跨步距离、不是页大小。
这一条同时**否掉了 `SWA_COMPACT` 自己的原方案**（紧凑独立池 ⇒ 容量 ×1.4922 的代价），
因为 chunk 视图把它**支配**了：**同收益、零代价**。

### 2.2 修复与判据（真算子路径）

改用 `kv8_chunk_view`（整块 storage 的**连续**视图 + `ids = phys*per_page + [0..payload)`）：

| 臂（真 `npu_sparse_flash_mla` + 真 rebuild；7,938 页 / 表宽 8,192） | µs/层 |
|---|---:|
| `swa_bf16` | **26.29** |
| `swa_int8_current`（页视图） | **1117.06** |
| **`swa_int8_fast`（chunk 视图）** | **288.90** |
| ⇒ Δ | **828.16 µs/层 × 40 = 33.1 ms/step** |

★ **`chunk_vs_page_bit_equal = True`**（真算子路径、int8 与 scale 两面、两种步长）
⇒ **精度按构造不变**（读的是同一批线性字节）。
★ chunk 视图在 **320 / 1,024 / 4,096 / 7,938 页全部 ≈13.0 µs**（与页数无关）。
★ 补丁 `agents/SWA_COMPACT/patches/chunk_page_gather.patch` **只改 `kv8_gather_pages` 的函数体**（签名不变）
⇒ **与 graphsafe 正交**（graphsafe 改的是 `kv8_ori_plane` 的调用者 + `_kv8_graph_rows_bound` + `_kv8_cmp_plane`）。

### 2.3 ★ 它顺带**独立印证了融合件的坑**（第二个例证）

`SWA_COMPACT` 的 p6 里 `accuracy_lossless.D` 与 `accuracy_real_quant` 两格都 ERROR，报
`TypeError: fast_ori_plane() got an unexpected keyword argument 'rows_bound'`，位置 `dsa_v41.py:1040`。

⇒ 这是 **`kv8_fuse_triton.py` 的旧 monkey-patch**（`fused_ori_plane` 只吃 8 个位置实参、**不收 `rows_bound`**）
⇒ 与 `FUSE_MULTIROW` 的负对照**互相印证**（两条独立路径都撞上"丢 `rows_bound` ⇒ 捕获期 `EE1016`"）。
⇒ **故该 ERROR 不是 chunk-page 补丁的缺陷**；但它的精度数字还缺（已让 `SWA_COMPACT` 在**不挂 fuse** 的路径下补 `torch.equal` + `rel_L2/cos`）。

---

## 3. 【实测】单旋钮拆分：**`KV8_SWA` 吃全部代价，`RING_FP16` 等于不存在**

`BF16_AB` 在 c1 的四臂（同 harness、同 dsa `94aeebb7`、`GRAPH_SAFE=1`）；
**主代理用自己的脚本重算原始 `kernel_details.csv`** 复核：

| 臂 | KV8_SWA | RING_FP16 | 设备每步 | `IndexSelect`/步 | 占设备时长 | `GPU KV cache size` |
|---|:--:|:--:|---:|---:|---:|---:|
| C | 0 | 0 | **20.67 ms** | — | — | 1,710,896 |
| **S** | **1** | **0** | **162.84** | **160** | **78.2%** | 1,710,896 |
| **R** | **0** | **1** | **20.70**（= 噪声） | **0** | **0%** | 1,710,896 |
| B | 1 | 1 | 163.26 | 160 | 77.9% | 1,710,896 |

⇒ **`KV8_SWA=1` = 全部速度代价（单卡口径 +142.2 ms）；`RING_FP16=1` = 免费。**
⇒ **R 臂的 `IndexSelect` 行数 = 0**（不是少，是零）⇒ 该算子是 `KV8_SWA` 专有。
⇒ **四臂 KV cache size 逐字相同** ⇒ 与 8 卡侧一致：**档 C 的 SWA int8 在 HBM 上 ×1.000**。

### 3.1 ★ 破了"200× 分歧"：不是伪影，是**双峰**（主代理独立复核原始 CSV）

同一份 CSV 上四条推断给出 8.4 / 129 / 683 / 1672 µs，看似互相矛盾，实为**同名 kernel 的两条调用路径**：

- 分位数：`min 2.7 / p50 43.5 / p90 3197.6 / max 3360 µs`
- 分箱：`<10µs: 3120 / 10–100: 480 / 100–1000: 1200 / ≥1000: 1600`
- **每 decode 步：`≥1000µs` 40 次（= 每层 1 次）⇒ 128.2 ms/步；`<10µs` 80 次 ⇒ 0.30 ms/步**

⇒ `PROF_int8` 的 micro 量的是**便宜那批**，`BF16_AB` 的 avg 是**两批混合** —— **两边都是真的**。
★ 这也解释了"微基准 5.5 ms vs 真机 49 ms"的 9× 缺口：**微基准只复现了便宜的那条路径**。

---

## 4. 三条路线到 ≤24 ms 的算术（【推断】，待 8 卡验证）

| 路线 | 可回收 ms/step | HBM 容量 | 宿主 DRAM | 风险 |
|---|---:|---|---|---|
| **A. chunk 视图**（叶子 helper，一处改动） | **33.1**【实测】 | **0** | **0** | 低（产物逐比特相同） |
| B. 关 `KV8_SWA`（保 `RING_FP16`） | **43.6**【单卡外推】 | **0** | **吐回 ≈20–24 pp**（【推断·强】） | 低（精度还略升） |
| C. 紧凑独立池（**已否决**） | 33.2 | **×1.4922（427,643 → 286,603，−33%）** | 同向变差 | **被 A 支配** |

再叠加 `D2H_advance` 的同步点上限 `H_after ≈ 9 ms`：

- **A + H_after** = 75.911 − 33.1 − 9 ≈ **33.8 ms**（零代价）
- **A + 融合（≈4.4，可加性待验）+ H_after** ≈ **29.4 ms**
- **B + H_after** ≈ **23.3 ms** ✅ **但要用 20–24 pp 宿主 DRAM 换**

★ **口径警告**：目标 24 ms 的出处是 `draft_ab.py --seq-len 1032`（≈1K、**进程内**配对 A/B），
而服务端 quote 基线是 8K **30.45** / 32K 31.52 / 128K 30.92 ⇒ **两者不是同一把尺子**（已向用户求证口径）。

---

## 5. 下一步（已排期）

| # | 事项 | 谁 | 判据 |
|---:|---|---|---|
| 1 | **1M 臂阶段 B/C/D/E**（128K 冒烟 → 真 1M 单发 → 3 会话+池覆盖 → 文本正确） | `ARM_1M` | `failed=0`、`BlockRemoved:CPU`、textprobe |
| 2 | 8 卡 **S1 / R1** 单旋钮阶梯（`OV_SWA`/`OV_RING` 覆盖，`COMP_JSON` 与 tierB 逐字相同） | `ARM_1M` | 哪一翻值 43.6 ms + 宿主归属 |
| 3 | **三份候选合成一份 `dsa_final_merged.py`**（chunk 视图 + 多行融合 + 加固 tail），含逐处 provenance | `FUSE_MULTIROW` | `py_compile` + 分派矩阵 + **融合是否也读非连续视图** |
| 4 | chunk-page 补丁的精度数字（不挂 fuse）+ 定位"每层 1 次贵调用"的行号 | `SWA_COMPACT` | `torch.equal` + 代码行号 |
| 5 | `IndexSelect` 双峰的**调用点归属**（`kv8_gather_pages` 还是 `kv8_gather_rows`） | `BF16_AB` | 形状 / stride dump |
