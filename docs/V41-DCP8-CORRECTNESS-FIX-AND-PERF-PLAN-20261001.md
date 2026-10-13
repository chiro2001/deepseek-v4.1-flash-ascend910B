# DCP8 正确性修复 + 性能分解（2026-10-01）

## 0. 结论

* **① 正确性：已修复并验证**。根因两层，均在 host 侧，**不需要改算子二进制**。
* **② 性能：未达标**。DCP1 = 26.82 ms/step，DCP8 = 34.57 ms/step ⇒ **1.289×**，目标 <1.2×（≤32.18 ms）。

## 1. 根因（两个，都实测闭环）

### 1.1 `seqused_cmp_kv` 坐标系错（主因）

内核（`arch22/sparse_flash_mla_csa_kernel.h:406-430`）的稀疏因果界：

```
cmpMaskS2Size = actualCmpS2Size*cmpRatio + residual
thresHold     = (cmpMaskRight + s1EndIdx + 1) / cmpRatio
              = actualCmpS2Size + s1EndIdx + 1 − actS1Size     （ratio=1, residual=0）
```

它**隐含假设 `actualCmpS2Size ≈ actS1Size`**（压缩序列与原始序列同长）。
DCP8 下我们传的是**本 rank 分片**的压缩行数（T=907 时 =128），而 `actS1Size=907` 是全局：

```
s2IdLimit = t + 128 − 907 = t − 778
GetKeyGmOffset: realS2Idx >= s2IdLimit ⇒ 丢弃该键
```

后果（单卡 + torch 参考实现实测）：
* `t < 778` 的 query 行 **一个 cmp 键都拿不到**；row0 的单键最优解 = ori pos0，**误差 1.2e-7**；
* 其余行的键按**索引值**被截断 ⇒ 向量阶段少搬若干行，而矩阵阶段仍按
  `actualSingleProcessSInnerSize` 读同一块 `kvMergeGm_` ⇒ **读到上次调用残留的内存**。

**单卡判据（决定性）**：同一份生产 dump、同一张卡、同一进程

| 调用序列 | lse_sum |
|---|---|
| holes #1（进程首调） | 286387.19 |
| holes #2/#3 | 286253.09 |
| 中间插入一次 `zero` 变体后再跑 holes | **287870.44** → 稳态 287654.91 |

⇒ 结果取决于**此前跑过什么** ⇒ 确实读了未写过的内存。

**修复**：`seqused_cmp_kv` 传**全局压缩长度**（`floor(seq_lens/cmp_ratio)`），
metadata op 仍用本地值（喂全局值会让 `SparseFlashMlaMetadata` AICPU 崩，
实测 run `dcpcap_1001_010130`）。

**修复后单卡判据**：同一输入 4/4 位一致；5 行全部与 torch 参考吻合到 ≤2e-6
（修复前：3.472 / 1.301 / 1.696 / 0.0338 / 9.5e-7）。

### 1.2 块表页号与页大小错配（第二缺陷，已修）

* 算子寻址：`blockTableIdx = logicalIdx / paOriBlockSize`（Bs 轴 = 128），
  块表里的 **0 不是"跳过"而是"第 0 块"**。
* 修复前 SLOTTRACE：`ori_bt=[3,0,0,…]`，位置 0–127 → 块 3，**位置 ≥128 全部 → 块 0**（别的请求的页）。
* 修复：worker 侧 `BlockTable` 对"复制态非 mamba/circular"组打开 hybrid block 展开
  （`logical = phys*dcp + j`）。修复后 `ori_bt=[3,4,5,6,7,8,9,10,0,0]`，每 128-token 一页。

## 2. 验收（修复后）

| 项 | 修复前 | 修复后 |
|---|---|---|
| 同一 T=904 针连发 6 次 | 1/6（第 1 次 Q7，之后 #） | **6/6 Q7** |
| 长针 2000 A7/K3 | 1/2 / 0/2 | **2/2 / 2/2** |
| 长针 8000 A7/K3 | 0/2 / 0/2 | **2/2 / 2/2** |
| 长针 16000 A7/K3 | 0/2 / 0/2 | **2/2 / 2/2** |
| 短问答 | 5/6 | 5/6（无回退） |
| 长度扫描 140→2000 | 880 起全错 | **10/10 全对**（边界消失） |
| 容量 | 6,082,458 token | 6,082,458（4.90×，不变） |

## 2.5 ★ 本轮性能优化的实测结果（同会话配对，256 token × 5 轮）

| 臂 | 5 轮 ms/step | 中位数 |
|---|---|---:|
| **DCP1**（同一 overlay，DCP=1） | 27.06 / 27.00 / 26.76 / 26.81 / 26.89 | **26.89** |
| **DCP8**（已修 + 已优化） | 33.85 / 33.62 / 33.82 / 33.68 / 33.80 | **33.82** |

⇒ **比值 1.258×**（优化前同口径 34.27/26.52 = **1.29×**）。
目标 ≤1.2× ⇒ 还需再砍 **1.55 ms/step**（DCP8 需到 32.27）。

### 已实施并实测有效的优化

| # | 改动 | 收益 |
|---|---|---|
| 1 | **`_v41_pack_for_reduce`**：merge 打包不再用 `F.pad`，改为按 `(T,H,D)` 缓存的**常驻零初始化缓冲**（只写 `[...,:512]` 与 `[...,512:513]`） | `PadV3` 0.234 + `PadV3AiCore_MemSet` **0.603** 全部归零，代价是 2 次 `ViewCopy` ≈ 0.31 ⇒ **净 −0.52 ms/step** |

### 已实测**证伪**的两条（避免后人重踩）

| 假设 | 实测 | 判定 |
|---|---|---|
| 用 `cumsum+scatter` 替掉 `argsort` 做稳定压缩（数学等价，400 组随机用例 0 不一致） | `ScatterElements` **+19.672 ms/20fwd（123 µs/次！）**、`Cumsum` +8.165、其余 +1.56；而 `Sort` 只省 14.455 ⇒ **净 +0.75 ms/step** | ⛔ 这台 NPU 上 `ScatterElements`/`Cumsum` 是慢路径；`argsort` 反而更快。**已回退**（`_v41_stable_compact` 保留 argsort 并写明原因） |
| `reduce_scatter` 替 `all_reduce`（每 rank 只收自己 8 个 head，接收量 1/8、ring 搬运减半） | 同会话同 build **33.17 vs 33.18** ⇒ **无差异**；eager 微基准 T=1 也是 `RS 87.6 > AR 80.3 µs` | ⛔ 集合通信是**纯延迟 bound**（16 B 与 1 MB 同价），减字节无效。已在 `V41_DCP_RS_MERGE` 后保留实现，**默认关** |

### ★ 本轮踩到并修掉的一个**我自己引入的回归**（重要教训）

为省一次 `FloorDiv`，我把 `_v41_global_cmp_lens(seq, 1)` 写成**直接 `return seq_lens`**。
后果：`seqused_ori_kv` 与 `seqused_cmp_kv` **指向同一块显存**（别名）⇒
短问答稳定回归（`17 × 23 等于多少？只回答数字。` 连续 6 次乱码，此前是 `391`）。
改为物化新张量后立刻恢复（连续 3 次 `391`，完整短问答 5/6）。
⇒ **教训：DCP 路径里"省一个 kernel"的改动必须先过短问答回归。**

## 2.6 ★ 第二轮性能优化（2026-10-01 04:20–05:10）

### 做了什么

| # | 改动 | 结果 |
|---|---|---|
| 1 | **`argsort` 键 int64 → int32**（`_v41_stable_compact`）。动机：配置里 `index_source_layer_ids` 有 **8 个**条 ⇒ 每步必然 8 次 remap、8 次 argsort；profiler 里 `Sort` 8 次/step、90 µs/次 = **0.72 ms/step**（DCP1 为 0），而键值只有 `0..2·width` | 保留 |
| 2 | **折叠 `dcp·_keep`**：`_onum = _oi*(_keep*dcp)`，下游 `scaled - _onum` ⇒ 每层省 1 次 `Muls` | 保留 |
| 3 | `denclamp=1`（默认**关**）：分母用 `clamp_min(1e-30)` 替掉 `where(wsum>0, wsum, ones_like)`，省 `Greater`+`SelectV2`+`Fill` | 保留（默认关，未 A/B 出结论） |
| 4 | `PACKDIRECT`：`output*w` 直写常驻缓冲（省 `Cast`+`Mul`+`ViewCopy`） | ★ **已回退 —— 它破坏了正确性** |

### 4 的实测教训（硬门槛优先）

`torch.mul(bf16_output, fp32_weights, out=<非连续 fp32 视图>)` + 不再预先算 `scaled`
⇒ **T=904 针立刻回归**（输出 `'特'` / `'7'`，正确值 `Q7`，连续 2 次均错）；
同一 overlay 的上一版是 6/6。回退后立刻恢复 6/6。
该优化只值 ~0.2 ms/step ⇒ 直接放弃、不再深挖，代码里留了 `V41-PACKDIRECT-ABORT` 记录。

### 本轮口径的最终实测（同会话配对，256 token）

| 臂 | ms/step（3–4 轮） | 中位数 |
|---|---|---:|
| DCP1 | 27.06 / 27.00 / 26.76 / 26.81 / 26.89 | **26.89** |
| **DCP8 最终（opt5）** | 33.50 / 33.56 / 33.63 | **33.56** |

⇒ **比值 1.248×**（起点 1.29×）。**距 1.2× 还差 1.17 ms/step**（需 ≤32.27）。

### 这一轮的净结论

* 微优化（省几次下发）已经在**跨会话噪声 σ≈0.3 ms** 以内，继续在"逐算子减一个"上投入收益递减。
* 真正剩下的量级只在两处：
  1. **8 次 remap × (argsort + 坐标换算)** ≈ 1.0 ms/step —— 想再降必须把它做成**一个融合 kernel**；
  2. **DCP 专属集合通信 ≈ 3.0 ms/step**（merge allReduce 47 次 × 49 µs + q gather 39 次 × 18.5 µs），
     已实测**纯延迟 bound**（减字节无效、reduce_scatter 无效）⇒ 只能减**次数**或做**跨层重叠**。
* 正确性侧无回退：T=904 6/6 `Q7`、长针 2000/8000/16000 **6/6**、短问答 **5/6**。

> 注：短问答第 3 条（《七律·到韶山》）在 `max_tokens=64` 下**会被截断**，
> 同一实例不同轮次 4/6 与 5/6 交替出现；把该条放宽到 200 token 即给出正确续句。
> 该项**不是**正确性信号，判读时应看汇总里的其它条目。

## 2.7 ★ 第三轮性能优化（2026-10-01 05:20–06:20）

### 做了什么（都在 `_v41_stable_compact` / `_v41_global_cmp_lens` / merge 分母）

| # | 改动 | 依据 | 结果 |
|---|---|---|---|
| 1 | `_v41_global_cmp_lens` 加 **每步缓存**（键 = 步序号 + storage 指针 + shape + ratio）。步序号由 `_refresh_perf_flags()`（`build()` 每步调用）递增 | 该函数被**每层**调用但值在同一步内对所有层相同；`FloorDiv` 线上 104 次/step（DCP1 对照 53） | 保留。**带 `cmplens_verify=1` 一致性自检**：实测 `same=True cached=[304] fresh=[304]`、`cached=[453] fresh=[453]` |
| 2 | 分母用 `clamp_min` 替 `where(wsum>0, wsum, ones_like)` | INV 探针实测 `wsum_min ∈ [1.004, 1.041] ≥ 1` 恒成立 | 保留（默认开，`denclamp=0` 可回退） |
| 3 | 稳定压缩的排序键 **int64/int32 → uint8 掩码** | 单卡微基准：int64 123.9 µs / int32 123.3 µs / **uint8 77.9 µs** / `cumsum+scatter` 220.7 µs；三版输出逐位一致 | 保留 |
| 4 | remap 里的 `indices.to(torch.int64)` → `int32` | 值域 `idx*ratio ≤ 2e6`、结果 ≤ 2.5e5，远在 int32 内 | 保留 |

微基准脚本 `~/tmp/sort_bench.py`（a3-21），可复跑。

### 结果（同会话，256 token × 5 轮）

| 臂 | 5 轮 ms/step | 中位数 |
|---|---|---:|
| DCP8（第二/三轮优化后，opt7） | 33.51 / 33.45 / 33.41 / **33.07** / 33.25 | **33.41** |
| DCP1（同 overlay，**紧邻测量**：先跑完 DCP8 五轮再切 DCP1 五轮） | 26.83 / 26.70 / 26.60 / **26.47** / 26.69 | **26.69** |

⇒ **比值 1.252×**（起点 1.29×）。以最佳单轮计 33.07/26.47 = 1.249。

> 说明：DCP1 在同一时间窗内实测 **26.47–26.83**（比更早一次 26.89 略快），
> 所以按同会话配对口径，比值应记为 **1.25×**，而不是拿旧的 26.89 去算出的 1.242。

设备侧总时间（20 forwards，profiler `op_statistic`）：
**429.30 ms → 415.54 ms（−0.688 ms/step）**。

### 正确性（同一构建 opt7 实测）

| 项 | 结果 |
|---|---|
| T=904 针连发 ×6 | **6/6 `Q7`** |
| 长针 2000/8000/16000 × A7/K3 | **6/6** |
| 短问答 | **5/6** |
| 容量 | 6,082,458 token（4.90×） |

## 2.8 剩下的差距在哪（实测定位）

DCP8 相对 DCP1 的额外开销（同 profiler 口径，ms/step）：

| 类别 | 估计 | 已证实的性质 |
|---|---:|---|
| 集合通信（38 q-gather + 38 merge-allReduce） | **~3.6** | **纯延迟 bound**：16 B 与 1 MB 同价；减字节无效、`reduce_scatter` 无效 |
| `Sort`（8 次/step） | 0.69 → 换 uint8 后 ~0.45 | 已优化 |
| `ViewCopy`（打包，2/层） | 0.36 | 是"去掉 `F.pad`"的代价，净收益仍为正 |
| `Cast`/`Mul`/`Sub`/`RealDiv` 等后处理 | ~1.2 | 每层 ~9 个元素级小算子，**只能靠融合 kernel 降低** |
| `FloorDiv`/`FloorMod`（remap 坐标 + builder） | ~0.5 | 部分已由每步缓存削掉 |

**结论【实测】：靠"省的算子个数"已经打到收益递减**（设备时间降 0.69 ms/step，wall clock 只降 ~0.2 ms/step ——
wall clock 由集合通信延迟主导）。要把 1.242 压到 1.2 还需要 ~1.1 ms/step，
只能来自两处**结构性**改动：

1. **把 merge 的后处理融合成一个 kernel**（现在每层 9 个元素级小算子 × 38 层 = 342 次下发/step）；
2. 或**减少集合通信的相位**（当前 76 次/step，每次 20–50 µs 的固定延迟）。

这两项都需要写自定义算子/重构通信，属多日工作量，不在本轮范围内。

## 2.9 ★ 第四轮：精确通信账 + 死代码清除 + 两次回退（2026-10-01 06:30–08:10）

### 2.9.1 首次拿到"精确"的 DCP 专属通信账

用**同一份 profiler 结构**对 DCP1 与 DCP8 分别解析 `communication.json`，按 `(算子, process group)` 分组：

| 通信 | DCP8 | DCP1 | 归属 |
|---|---|---|---|
| allReduce `group=097` | **76.0 /step（= 2×38）**, 0.976 ms/step | **无** | **DCP merge** |
| allGather `group=374` | **36.1 /step**, 0.670 ms/step | **无** | **DCP q-gather** |
| allReduce `group=503` | 82.0 /step | **82.0 /step（逐字相同）** | TP/EP，与 DCP 无关 |

⇒ **DCP 专属集合通信 = 1.646 ms/step，且每项都可精确归因**。
`76 = 2×38` 说明 merge 每层做了**两次** all_reduce（164 KB 的包 + 256 B 的分母）。

### 2.9.2 合并链路：干净的 DCP1 vs DCP8 算子差分（同 profiler 口径）

| op | DCP1 ms/step | DCP8 ms/step | Δ | 次数 DCP1→DCP8 |
|---|---:|---:|---:|---|
| 设备总 | **16.514** | **20.224** | **+3.711** | — |
| Sort | 0 | 0.677 | +0.677 | 0 → 160 |
| Cast | 0.304 | 0.802 | +0.498 | 4895 → 9675 |
| ViewCopy | 0.010 | 0.329 | +0.319 | 25 → 1545 |
| FloorDiv | 0.071 | 0.382 | +0.310 | 1060 → 2080 |
| Mul | 0.020 | 0.315 | +0.295 | 258 → 2798 |
| **ConcatD** | **0** | **0.241** | **+0.241** | **0 → 760** |
| SparseFlashMla | 1.212 | 1.428 | +0.216 | 1560 → 1560 |
| Sub | 0.012 | 0.189 | +0.177 | 139 → 2639 |
| SparseFlashMlaMetadata | 0.450 | 0.621 | +0.171 | 60 → 60 |
| RealDiv / Slice / ClipByValueV2 / NanToNum / Exp | 0 | ~0.43 | +0.43 | 0 → 38 各 |

### 2.9.3 本轮改动（一保留，三回退）

| # | 改动 | 结果 |
|---|---|---|
| 1 | **删除死代码 `torch.cat([scaled, weights])`**：它紧接的 padding 已被 `_v41_pack_for_reduce`（常驻缓冲）取代，**结果从未被使用**，但 `ConcatD` 内核照样每层下发 | ✅ **保留**，−0.20 ms/step（0.241 的设备时间） |
| 2 | `contigw` 作默认（省掉 38 次/step 的分母 all_reduce） | ❌ **回退**：设备时间 −0.55 ms/step，但 **wall clock 无变化**（32.95 vs 32.85，噪声内），**且短问答回归**（见下） |
| 3 | `torch.sub(pack, ori, alpha=kd)` 一次替掉 `Cast+Mul+Sub` | ❌ **回退**：离线微基准**逐位一致**且快 3.1×，真实路径同样触发短问答回归 |
| 4 | 修 `_rs_applied` / `else` 分支里 `dcp * _onum` 的 **dcp² 潜在 bug**（`_onum` 自 ONUMFOLD 起已含 dcp） | ✅ 保留（非默认分支，不影响当前数值） |

### 2.9.4 ★ 短问答是比 T=904 更灵敏的回归判据（重要方法论）

`contigw` 的整轮验证经过：

| 判据 | 结果 |
|---|---|
| T=904 针 ×4 | **4/4 `Q7` 通过** ← 会误判为"没坏" |
| 长针 2000/8000/16000 | **6/6 通过** ← 也会误判 |
| **短问答 `17 × 23 等于多少？只回答数字。`** | **6/6 输出乱码**（`# 标题：老婆背叛后…`），而上一版稳定 `391` |

⇒ **今后任何 DCP 路径的改动，必须把这条短问答纳入最小回归集**；
只跑 T=904 / 长针会放过它。回退 `contigw` 后立刻恢复 `391`（4/4）。

### 2.9.5 Triton 在本平台**不能**用于小算子融合（决定性负结果）

原型：把 merge 的前/后处理各写成一个 Triton-Ascend 逐元素 kernel（`~/tmp/triton_proto.py`）。

| 方案 | `[1,64,512]` 逐元素 |
|---|---:|
| 8 个原生小算子合计 | ~16 µs |
| **一个 Triton kernel** | **58.5 µs**（前）/ **57.6 µs**（后） |

⇒ **Triton 比它要替代的原生算子串慢 3.6×**。与邻居目录 `dsv41-refresh/op_peak` 的结论一致：
该平台上小规模 kernel 的 Triton 启动/编解码开销极大，突破必须走 **AscendC 自定义算子**（其估算 2–4 周）。

### 2.9.6 本轮最终数据（同窗口，256 token）

| 臂 | 5 轮 ms/step | 中位数 |
|---|---|---:|
| DCP1（`dcp1prof2`，同窗口紧邻测量） | 26.87 / 26.97 / 26.98 | **26.97** |
| **DCP8 最终（`dcpr2`）** | 32.89 / 32.93 / **32.61** / 33.09 / 32.95 | **32.95** |

⇒ **比值 1.222×**（本轮起点 1.229×，本轮之前 1.29×）。

正确性（同一构建 `dcpr2` 实测）：

| 项 | 结果 |
|---|---|
| T=904 针 ×4 | **4/4 `Q7`** |
| 长针 2000/8000/16000 × A7/K3 | **6/6** |
| 短问答 | **5/6**（含 `391` 恢复） |
| 容量 | 6,082,458 token（4.90×） |

### 2.9.7 还差 0.59 ms/step（DCP8 需 ≤32.36）

剩余可动的只有两处，且都**已被实测封死**：

1. **merge 后处理 ~1.9 ms/step 的小算子**（Cast/Mul/Sub/RealDiv/Slice/Clip/NanToNum/ViewCopy）——
   Triton 融合慢 3.6×；所有 Python 层面的"合并算子"尝试（PACKDIRECT / SUBALPHA / CONTIGW）
   **三次全部破坏正确性**。⇒ 只能写 **AscendC** 自定义算子。
2. **集合通信 1.646 ms/step**（76→74 次/step）——**纯延迟 bound**，
   已证：减字节无效、`reduce_scatter` 无效、去掉一次 all_reduce 在 wall clock 上也无变化
   （说明它本来就被掩盖）。⇒ 只能减**相位**或做**跨层重叠**，属结构性改动。

## 2.10 ★ 第五轮：contigw 根因假设被证伪 + 最终数据（2026-10-01 08:30–09:30）

### 2.10.1 对 `contigw` 失败根因的一次可证伪尝试（结果：证伪）

**假设**：`scaled` 与 `weights` 被放进**同一个** all_reduce 缓冲；本文件历史实测已记录
"HCCL 在 buffer 含 Inf/NaN 时行为异常"（当年加 `sanitize` 开关的原因）。
分母列在这个"脏"缓冲里 ⇒ 读出的 `Σ_r w_r` 是垃圾 ⇒ 短输入先崩、长上下文侥幸通过。

**实验**：`contigw` 与 `sanitize` **同时打开**（单一变量组合），重启后跑短问答。

**结果（实测）**：短问答 `17 × 23 等于多少？只回答数字。` **仍然 6/6 乱码**
（`# 标题：老婆背叛后…`、`# 标题：AI时代…`）。

⇒ **假设证伪**：根因不是 Inf/NaN 污染联合 all_reduce。
`contigw` 永久回退为默认关，真实根因标记【未确认】。
代价仅 38 次/step 集合通信（wall 上只值 0.1–0.4 ms，且被噪声淹没），不再追。

### 2.10.2 最终数据（同窗口配对，256 token × 5 轮）

| 臂 | 5 轮 ms/step | 中位数 |
|---|---|---:|
| **DCP1**（`dcp1pair2`，紧邻测量） | 26.68 / 26.61 / 26.89 / 26.67 / 27.15 | **26.68** |
| **DCP8 最终**（`dcpfin2`） | 32.91 / 32.57 / 32.66 / 32.82 / 32.79 | **32.79** |

⇒ **配对比值 1.229×**。若用本周所有同口径 DCP1 测量的中位数（26.82，样本 15 轮，
范围 26.47–27.15）作分母，则为 **1.223×**。

**起点对比**：最初 34.27 / 26.52 = **1.29×** ⇒ 本轮累计改善 **−1.48 ms/step（−4.3%）**。

### 2.10.3 最终构建的完整验收（实测）

| 项 | 结果 |
|---|---|
| T=904 针 | **4/4 `Q7`** |
| 短问答 `17 × 23` | **4/4 `391`** |
| 完整短问答 6 条 | **5/6**（第 4 条无标准答案，第 3 条为长答） |
| 长针 2000/8000/16000 × A7/K3 | **6/6** |
| 容量 | 6,082,458 token（4.90×） |
| 设备侧总时间（profiler） | **19.474 ms/step** vs DCP1 **16.514** ⇒ 设备比值 **1.179** |

### 2.10.4 为什么"设备比值 1.179 < 1.2"却"wall 比值 1.229 > 1.2"

这是本目标最后一个关键事实【实测】：

* DCP8：wall 32.79 − 设备 19.47 = **13.32 ms/step 的非设备开销**
* DCP1：wall 26.68 − 设备 16.51 = **10.17 ms/step 的非设备开销**
* 差 **+3.15 ms/step** —— 全部来自 DCP：

| 项 | ms/step | 已证性质 |
|---|---:|---|
| merge allReduce（76 次/step） | 0.98 | 纯延迟 bound；去掉一次（contigw）在 wall 上只值 0.1–0.4 |
| q-gather（36 次/step） | 0.67 | 同上 |
| 余量（112 次集合通信的启动/等待，未被 `Elapse Time` 计入） | ~1.5 | 112 次 × ~27 µs ≈ 3.0 ms |

⇒ **wall 的差距几乎全部是"DCP 专属集合通信的调用次数"**，而不是算子耗时。
要跨过 1.2 必须**减少相位**（例如 SFA 式 `all_to_all` 替掉 `q-gather + all_reduce`），
或把这些通信做进**跨层流水**——两者都是结构性改动。

### 2.10.5 本轮所有尝试的最终台账

| # | 改动 | 结论 |
|---|---|---|
| 1 | 删除死代码 `torch.cat`（`ConcatD` 0→760 次/20fwd） | ✅ 保留，**−0.20 ms/step** |
| 2 | `_v41_global_cmp_lens` 每步缓存（含 `cmplens_verify` 自检） | ✅ 保留 |
| 3 | 稳定压缩排序键 `int64→int32→uint8` | ✅ 保留（微基准 123.9→77.9 µs） |
| 4 | remap `indices.to(int64)→int32` | ✅ 保留 |
| 5 | `denclamp`（分母 `clamp_min` 替 `where+ones_like`） | ✅ 保留 |
| 6 | 修 `dcp * _onum` 的 dcp² 潜在 bug | ✅ 保留 |
| 7 | `contigw` 默认（去 38 次集合通信） | ❌ 回退（短问答回归，根因未确认） |
| 8 | `contigw` + `sanitize`（验证 Inf/NaN 假设） | ❌ **假设证伪**，一并回退 |
| 9 | `torch.sub(pack, ori, alpha=kd)` 融合 Cast+Mul+Sub | ❌ 回退（短问答回归；离线微基准逐位一致也无用） |
| 10 | `PACKDIRECT`（`torch.mul(out=<strided view>)`） | ❌ 回退（短问答回归） |
| 11 | Triton 融合 merge 前/后处理 | ❌ 负结果（58 µs vs 原生 16 µs，慢 3.6×） |
| 12 | `reduce_scatter` 替 `all_reduce` | ❌ 无差异（纯延迟 bound） |
| 13 | `cumsum+scatter` 替 `argsort` | ❌ 反向优化（+0.75 ms/step） |

**★ 一条贯穿全程的方法论结论**：本平台上**任何"把多个逐元素算子融合/改写"的写法都不可信**
（#7/#9/#10 三次都是"离线微基准逐位一致、真实路径破坏正确性"）。
而**"删掉真正多余的算子"是安全且有效的**（#1）。

## 2.11 距离目标的最终差距与建议

目标 ≤1.2× ⇒ 以 DCP1=26.68 计需 ≤**32.02 ms/step**，当前 **32.79**，**还差 0.77 ms/step**。

剩余 0.77 的可达路径（按可行性排序）：

| # | 路径 | 预估 | 前置条件 |
|---|---|---|---|
| 1 | **reduce q-gather 相位**：用 SFA 式 `all_to_all`（`dcp_a2a.py` 已有参考实现）替掉 `q all_gather(36/step) + merge all_reduce(76/step)` | 1.5–2.5 ms | 需重做 DCP 合并链路，数天 |
| 2 | **AscendC 自定义合并算子**：把 merge 的 ~19 个逐元素算子/层压成 1–2 个 | 1.0–1.5 ms | 需写 AscendC，2–3 天（1+1 环境可快速迭代） |
| 3 | 跨层流水（把第 L 层的合并通信与第 L+1 层的投影重叠） | 未知 | 需重构调度 |

第 1 项收益最大且能同时压掉大半 wall 差距；第 2 项只能压设备侧（设备比值已 1.179，
单靠它**不足以**让 wall 达标）。

## 2.12 ★ 第六轮：图节点成本模型 + 类型提升优化（2026-10-01 09:00–10:00）

### 2.12.1 先纠正一个此前的前提错误

我一直以为 `EAGER=1` 关掉了图捕获。**实测否定**：起服日志里
`enforce_eager=False`、`cudagraph_mode: FULL_DECODE_ONLY`、`cudagraph_capture_sizes=[1,2,...]`
——**decode 一直是图捕获的**。`EAGER=1` 在这个脚本里并没有传 `--enforce-eager`。
另外 `@eager_break_during_capture` 在本机是 **no-op**（`VLLM_USE_BREAKABLE_CUDAGRAPH=0`），
所以 DSA attention **也在图里**，不存在"host 侧 Python 开销"这条解释。

### 2.12.2 关键路径分析：设备**空闲占比 40–60%**

用 `kernel_details.csv` 的 `Start Time/Duration`，按 SMLA 计数把时间线切成 forward（78 SMLA/forward）：

| | forward 跨度 | 内核占用（并集） | 空闲 | 空闲占比 |
|---|---:|---:|---:|---:|
| DCP1 稳态 | 29.7 ms | 17.7 ms | **12.0 ms** | 40% |
| DCP8 稳态 | 36.4 ms | 22.3 ms | **14.1 ms** | 39% |

⇒ 设备有近 40% 的时间在**空闲**（依赖等待 + 内核间空隙）。

### 2.12.3 ★ 图节点的定量成本：**~3 µs / 节点**

本轮删掉了 76 个/step 的 `Cast` 节点（见下），实测：

| | 节点数/step | 设备 ms/step | wall ms/step |
|---|---:|---:|---:|
| 改前 | 3206 | 19.474 | 32.79 |
| 改后 | **3130**（−76） | **19.359**（−0.115） | **32.56**（−0.23） |

⇒ **每个图节点约值 3 µs 墙钟**（= 节点自身耗时 + 图回放的派生开销）。
这条把"接下来还需要多少"变成了可计算的问题：**要达到 1.2× 还需再砍约 200 个节点/step。**

### 2.12.4 本轮落地：靠**类型提升**省 Cast（逐位等价，已验证）

`bf16_tensor * fp32_tensor` 在 PyTorch 里会**自动提升到 fp32**，与
`bf16_tensor.to(fp32) * fp32_tensor` **逐位一致**（单卡验证 `torch.equal=True, max|d|=0`），
但**少一个 Cast 节点**。注意：**0 维** fp32 张量会被当标量、**不提升**，必须用 **1 维**张量。

| 位置 | 改法 | 微基准 |
|---|---|---|
| `scaled = output.to(fp32) * weights` | `scaled = output * weights` | 27.31 → 10.70 µs/层 |
| `_onum = _oi.to(fp32) * (_keep*dcp)` | `_onum = _oi * _v41_scalar_t(_keep*dcp, _oi)`（缓存 1 元素 fp32 张量） | 30.72 → 11.35 µs/层 |

实测：节点 −76/step，`Cast` 0.818 → 0.527 ms/step，设备 −0.115，wall −0.23 ms/step。
**两个回归判据都一次通过**（短问答 `391` ×6、T=904 `Q7` ×6）。

### 2.12.5 本轮最终数据（同窗口配对）

| 臂 | 5 轮 ms/step | 中位数 |
|---|---|---:|
| **DCP1**（紧邻测量） | 26.55 / 26.63 / 26.52 / 26.65 / 26.77 | **26.63** |
| **DCP8（含本轮优化）** | 32.55 / 32.62 / 32.56 / 32.45 / 32.58 | **32.56** |

⇒ **配对比值 1.223×**（起点 1.29×，累计 −1.71 ms/step）。
设备比值 **19.359 / 16.514 = 1.172×**（已 < 1.2）。

完整验收（同一构建）：T=904 **6/6 `Q7`**；短问答 `17×23` **6/6 `391`**、
完整短问答 **5/6**（第 3 条为 `max_tokens=64` 截断假象，给 200 token 即给出正确续句，已在文档说明）；
长针 2000/8000/16000 **6/6**；容量 **6,082,458 token（4.90×）**。

### 2.12.6 还剩 **0.60 ms/step**（需 ≤31.96）—— 逐项已封死

| 候选 | 节点/step | 估算 | 状态 |
|---|---:|---:|---|
| `Sort`（8 个 index source 各一次） | 8 | 0.65 ms 设备 | **已到地板**：微基准 8 种写法全在 45–47 µs（`cumsum+scatter` 220 µs 更慢、`nonzero` 92 µs 更慢）；架构上 8 个 index source 必须有 8 次 remap |
| `Cast` 残余 | ~163 | ~0.5 ms | 剩下的是 `output→fp32` 的**必要**转换（已合并系数）、`_oi→fp32`、以及最终 `.to(bf16)`；已无冗余 |
| `Sub` | 125 | ~0.4 ms | 三处都是语义必需（`lse−ori_lse`、`Σw−const`、`pack−onum`） |
| `ViewCopy`（打包 2 次/层） | 76 | ~0.3 ms | 缓冲 dtype 必须 fp32 且 512B 对齐；`out=` 直写 strided 视图 = PACKDIRECT，**已实测破坏正确性** |
| `Clip`（clamp60 + clamp_min） | 76 | ~0.2 ms | `nan_to_num`/clamp 是 NaN 防护，删除即引入正确性风险 |
| 第 2 次 merge allReduce + clone | 76 | ~0.16 ms | = `contigw`，**已两次实测破坏正确性**（含 `sanitize` 联合验证，假设被证伪） |
| 融合成 AscendC 算子 | — | 1.0–1.5 ms | Triton 已证不可用（58 µs vs 原生 16 µs）；AscendC 需 2–3 天 |

**结论【实测+推断】**：以 ~3 µs/节点 的口径，要再省 0.6 ms 必须**一次性去掉约 200 个节点**——
这已经超出"逐个删冗余算子"能覆盖的范围（剩余算子基本都有语义必要性），
只能靠**AscendC 融合**（把 merge 的 ~15 个元素级算子/层压成 1–2 个）或
**结构性减少集合通信相位**。两者都是多日工程量。

## 2.13 ★★ 自制 AscendC 融合算子：把 merge 的 15 个小算子压成 2 个 kernel

> **目标修正（用户 2026-10-01 明确）**：**优化绝对性能，不是优化 DCP8/DCP1 的比值。**
> 这条解除"对称优化无用"的限制 ⇒ 邻居 `op_peak` 的全部成果（QBMV3 三合一 2.196×、
> wo_a Triton 1.59×、M 补零）现在都可用；同时本节的融合算子也仍然有效
> （它砍的是 **DCP 专属**的 573 个图节点，对绝对时延同样有效）。

### 2.13.1 为什么做

真实 8 卡 profiler（run `dcpcap_1001_070358`，20 forwards）：DCP8 比 DCP1 多
**1213 个图节点/step**，其中 **573 个**属于 `_v41_dcp_merge_attention` 的逐元素
前/后处理（每层 ~15 个小算子 × 38 层）。实测**每个图节点值 ~3 µs 墙钟** ⇒ 约 1.72 ms/step。

### 2.13.2 交付物（全部在仓内）

| 文件 | 作用 |
|---|---|
| `experimental/v41-dcp/ascendc/merge/op_kernel/v41_merge.asc` | 两个 kernel：`pre`（打包 pack）、`post`（合并出输出） |
| `experimental/v41-dcp/ascendc/merge/op_host/v41_merge_host.cpp` | ctypes shim（走 torch_npu 当前 stream ⇒ **可被 ACL graph 捕获**） |
| `experimental/v41-dcp/ascendc/merge/CMakeLists.txt` | ASC 构建（`find_package(ASC)`，`--npu-arch=dav-2201`） |
| `experimental/v41-dcp/ascendc/merge/bench_merge.py` | 单卡数值验证 + 计时（判据：pre 逐位、post ≤1 ULP） |
| `experimental/v41-dcp/ascendc/merge/check_bufs.sh` | **静态自检**：每个 TQue/TBuf 必须有且只有一条 InitBuffer |
| `overlay/vllm_ascend/attention/v41_merge_kernel.py` | 运行时 wrapper（ctypes 加载 + tiling/pack/out 常驻缓存 + `available()` 安全门） |
| `overlay/.../dsa_v41.py` | 集成：`_v41_merge_kernel_on()`（env `V41_DCP_MERGE_KERNEL=1`）+ `_merge_kd` 早退分支 |

### 2.13.3 实测（chip6，`bench_merge.py`）

| T | pre | post | 合计/层 | ×38 层 | 数值 |
|---:|---:|---:|---:|---:|---|
| **1（生产 decode）** | 7.08 µs | 6.61 µs | **13.68 µs** | **0.520 ms/step** | **逐位一致** |
| 4 | 10.41 | 6.41 | 16.82 | 0.639 | 逐位一致 |
| 16 | 19.99 | 6.39 | 26.38 | 1.002 | 逐位一致 |
| 32 | 33.79 | 6.27 | 40.06 | 1.522 | 1 ULP |
| 96 | 79.02 | 8.97 | 88.00 | 3.344 | 1 ULP |

**数值判据**：T=1–16（`cudagraph_capture_sizes` 里的全部小尺寸）与 Python 参考
**逐位一致**（`max|d|=0`）；T≥20 差 **恰好 1 个 bf16 ULP**（0.000488281），
根因是**本 CANN 的 fp32 没有可用的除法指令**（`Divs` 链接期 `undefined symbol`；
`Div` 运行期 vector core exception）⇒ 只能用 `Muls(num, 1/den)`，而 torch 的
`x/den` 是真除法。生产单流 decode 是 **T=1 ⇒ 逐位一致**。

### 2.13.4 ★★★ 四个 AscendC 硬坑（全部实测踩过，务必规避）

1. **kernel 入口必须 `extern "C" __global__ __vector__`**（来自邻居）：写成 `__aicore__`
   会被当 AIC 派发，所有 `ASCEND_IS_AIV` 保护的向量指令**被静默编译掉**。
2. **fp32→bf16 必须 `RoundMode::CAST_RINT`**：`CAST_NONE` **静默不写**（dump 出来是旧数据）。
3. **任何 `TBuf`/`TQue` 用前必须 `InitBuffer`，且恰好一条**。我踩了**两次**：
   · 漏 `qOriF_` ⇒ `The GM address accessed by scalar exceeds 48 bits`；
   · 用脚本按行号改代码时**误删 `pre` 的 `qDen_` InitBuffer** ⇒ 同样 vector core exception，
     而且**换 chip 后仍复现**（我一度误判为"设备坏了"，在 chip7/chip6 上各浪费一轮）。
   ⇒ 已加 `check_bufs.sh` 静态自检。
4. **`aicore` 不支持 double**：`AscendC::Exp((double)delta)` 编译期报
   `cast to/from double precision floating variable is not allowed`。改用**矢量 Exp**
   （一次算 8 个车道再读回）。
   另：**不手写 `SetFlag/WaitFlag`** —— 第一版手写同步导致"**进程内首次 launch 完全不产出**、
   第 2 次起才正确"（连跑 6 次：第 1 次 4096/4096 未写入，2–6 次逐位一致）。
   改用标准 **TQue（AllocTensor/EnQue/DeQue/FreeTensor）** 后由框架管同步。

### 2.13.5 部署链路的两个改动

1. **`serve_a2.sh` 原来只挂 `*.py`**（`find . -type f -name '*.py'`）⇒ `.so` 进不了容器，
   `v41_merge_kernel.available()` 恒 False，融合算子**静默退回 Python 路径**
   （不报错、只是没加速，极难发现）。已改为同时挂 `.so`（备份 `serve_a2.sh.bak_dcpson`）。
2. `.so` 放到 `~/dcpw/vllm_ascend/attention/v41_merge_kernel.so`
   ⇒ 容器内 `/vllm-workspace/vllm-ascend/vllm_ascend/attention/v41_merge_kernel.so`。

### 2.13.6 当前状态与下一步

* **单卡验证已完成**（chip6，`available()=True`，pre 8/8 逐位、post 逐位一致）。
* **尚未在 8 卡实例上跑**：`dsv41-gen1` 是在本次改动**之前**起的（挂载清单里没有
  新增的 `.py`/`.so`），必须**重启一次**才能加载融合算子（约 7–10 分钟）。
  重启需要与「线 A（绝对性能）」的子代理协调（它正在用同一个实例）。
* **验收判据**（重启后必须全过）：T=904 针 = `Q7`、短问答 `17×23` = `391`、
  长针 2000/8000/16000 = 6/6、容量 = 6,082,458。
* **预期收益**：573 个图节点 → 2 个 kernel，按 3 µs/节点 估 **−1.2 ms/step**
  （1.72 ms 的节点成本 − 0.52 ms 的 kernel 时间）。

## 3. 性能分解（真实 8 卡 profiler，20 forwards 口径）

```
Computing                       20.2 ms/step
Communication (Not Overlapped)   4.23 ms/step   ← 0% overlapped
  ├ merge allReduce (DCP)        2.33 ms/step   47 次/step, avg 49 µs
  ├ q gather allGather (DCP)     0.72 ms/step   39 次/step, avg 18.5 µs
  └ 其它 allReduce (TP/MoE)      1.17 ms/step
```

* **集合通信是纯延迟 bound**：131 KB 的 allReduce 与 16 B 的 allGather 同价（~20–49 µs）⇒
  优化方向是**减少次数/相位**，不是减少字节。
* DCP 专属通信 ≈ **3.05 ms/step**，占 7.75 ms 差距的 39%。
* 逐算子 top（每 step）：HcPre 2.41 / GroupedMatmulSwigluQuantV2 2.12 /
  QuantBatchMatmulV3 1.97 / MatMulV2 1.64 / SparseFlashMla 1.43 / GroupedMatmul 1.34 …
  Cast **420 次/step**、Mul 186 次/step ⇒ merge 后处理是"多小算子"结构。

## 4. 下一步（按收益排序）

| # | 措施 | 预估收益 | 状态 |
|---|---|---|---|
| 1 | ~~merge 用 `reduce_scatter_tensor`~~ | ~~~1.0 ms~~ | **已实测证伪（无差异）** |
| 2 | 2nd SMLA（纯 ori，与 1st SMLA/merge 无依赖）放旁路流重叠 | ~0.7 ms | 待实施 |
| 3 | 融合 merge 后处理（sub/div/cast ≈10 个小算子/层 → 1 个 kernel） | ~0.5–0.8 ms | 待实施 |
| 4 | 结构性：只 gather「被选中的压缩 KV 行」+ 1 次 SMLA，替掉「q gather + 2 次 SMLA + merge allReduce + 后处理」（2 个集合通信 → 1 个） | ~2 ms | 评估中 |

1+2+3 合计约 2.2–2.5 ms ⇒ 32.1–32.4 ms ⇒ 比值 1.20–1.21（**边界**）；
要稳过 1.2 需要第 4 项。

## 5. 与邻居目录 `dsv41-refresh/op_peak` 的关系（重要）

那边做的是**单算子极致优化**（wo_a：Triton 9.43 µs = cube 类 24 核上限 99.9%，1.59×；
发现 vendor matmul M=8 vs M=16 差 2×；下一靶 QuantBatchMatmulV3 家族 1.3–1.6 ms/step）。

**但这些是"对称"优化**：DCP1 与 DCP8 同样受益，而本目标的判据是**比值**：

```
(DCP8 − X) / (DCP1 − X) 随 X 单调上升
X=0     → 1.289
X=1.5   → 1.306
X=3.0   → 1.324
```

⇒ 把它们接到 DCP 部署上会**抬高**这个比值。它们提升的是**绝对速度**，
要在"比值"口径下达标必须砍 **DCP 专属**开销（本文 §4）。
