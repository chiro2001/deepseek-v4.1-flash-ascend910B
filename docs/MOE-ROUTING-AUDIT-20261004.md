# MoE 路由三件套审计（MoeInitRoutingV3 / MoeTokenUnpermute / MoeGatingTopKHash）

**日期**：2026-10-04　**执行**：子代理 `/root/ced_die_budget`（die5；未碰 tp8k5/19210）
**产物**：`~/tmp/moe/`　**标注**：【实测】有运行数据；【实测·代码】有明确代码路径；【推断】由实测推导；【未确认】无证据

---

## 0. 结论速览

| # | 结论 | 判定 |
|---|---|---|
| 1 | `MoeInitRoutingV3` 的 17.8 µs **≈90% 是固定开销**（rows 102× ⇒ 时长仅 2.07×） | ✅【实测】 |
| 2 | 真正的计算（`aiv_vec_time`）只有 **0.30 µs = 1.7%** | ✅【实测】 |
| 3 | 固定开销的机制：**36 行数据被发射到 48 个 AIV 块**，且**跨核同步**；tiling 早已判定"单核排序"（`sortMode_=0`），但 `PostTileing` 仍无条件 `SetBlockDim(aivNum)` | ✅【实测·代码】 |
| 4 | `dispatch_ffn_combine` **要求 `weight1[]/weight2[]`** ⇒ 天生是 dispatch+FFN 全融合，**不存在"只融合 dispatch+combine"的形态** | ✅【实测·代码】 |
| 5 | `moe_gating_top_k` / `moe_gating_top_k_hash` **都没有权重入参** ⇒ gate matmul **无法融合**（旧结论成立） | ✅【实测·代码】 |
| 6 | **最小可落地改动：把 `MoeInitRoutingV3` 在小 batch 下的 BlockDim 从 48 降下来**（1 行 tiling + 可能的 kernel 配合） | 见 §3 |

---

## 1. 第 1 步：17.8 µs 的构成（固定 vs 可变）

### 1.1 服务真设备数据【实测】
`results/armF_meta2_1004_2010/.../PROF_000002_*/op_summary`（rank0，sleep=0 段）：

| n_tok | topk | rows | count | TaskDur p50 | aiv_time | **vec** | scalar | mte2 | mte3 | BlockNum |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3 | 15 | 1011 | **11.5** | 8.1 | **0.11** | 1.95 | 1.36 | 0.02 | **48** |
| 6 | 6 | **36** | **13400** | **17.8** | 12.7 | **0.30** | 1.54 | 3.83 | 0.04 | **48** |
| 256 | 6 | 1536 | 80 | **23.8** | 18.6 | 4.68 | 4.08 | 3.10 | 0.98 | **48** |

**读法（这是本轮最硬的一张表）**
* rows **15 → 36 → 1536（×102）**，而时长只 **11.5 → 17.8 → 23.8（×2.07）**
* 用后两点拟合：边际成本 = (23.8−17.8)/1500 = **0.004 µs/行**（几乎免费）
* ⇒ **固定部分 ≈ 17.7 µs，占总时长 ~90%**【实测】

### 1.2 进一步拆解（n=6 那一档，17.8 µs）
| 项 | 值 | 占 TaskDur |
|---|---:|---:|
| `aiv_vec_time`（真正的向量计算） | **0.30 µs** | **1.7%** |
| `aiv_scalar_time` | 1.54 µs | 8.7% |
| `aiv_mte2_time`（GM→UB） | 3.83 µs | 21.5% |
| `aiv_mte3_time` | 0.04 µs | 0.2% |
| `aiv_time`（AIV 内核自报跨度） | 12.7 µs | 71% |
| **Σ(各 AIV 分量)** | **5.71 µs** | 32% |
| **未解释（TaskDur − Σ分量）** | **12.1 µs** | **68%** |
| `aicore_time` | 0.0（合理：AI core 直接 return，见下） | — |
| `aiv_icache_miss_rate` | **0.156** | — |

### 1.3 机制（【实测·代码】）
`moe_init_routing_v3/op_kernel/moe_init_routing_v3.cpp`：
```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);   // 纯 AIV
if (g_coreType == AIC) { return; }                   // AI core 什么都不做 ⇒ aicore_time=0 合理
```
`op_host/moe_init_routing_v3_tiling.cpp`：
```cpp
void ...::Tiling4VBSCompute() {
    if (totalLength_ <= sortLoopMaxElement) { sortMode_ = 0; }   // ★ 单核排序
    else                                    { sortMode_ = 1; }
    ...
    if (sortMode_ == 0UL) { Tinlig4VBSOneCoreCompute(tilingData); return; }  // ★ 走单核分支
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void ...::PostTiling() {
    if (isEmptyTensor_) { context_->SetBlockDim(1); }
    else                { context_->SetBlockDim(aivNum); }        // ★★ 仍无条件 48 块
}
```
* workspace 里另有 `coreSyncWorkspaceSize = coreNum × SORT32_ALIGN × 2` ⇒ **跨核同步是显式成本**
* ⇒ **我们这一档（totalLength=36）的 tiling 已判定"排序只用 1 个核"，但内核仍以 48 块发射，48 个块全部参与同步**

### 1.4 【实测】隔离环境下**无法**用事件计时得到设备时间（负结果，重要）
用 `torch.npu.NPUGraph` 图内 replay 测同一算子：

| n_tok | rows | expandedX | 图内 µs/次 |
|---:|---:|---:|---:|
| 6 | 36 | 0.2 MB | 10.46 |
| 256 | 1536 | 7.9 MB | 10.46 |
| 1024 | 6144 | 31.5 MB | 10.60 |
| 4096 | 24576 | **125.8 MB** | 10.43 |

写 125.8 MB 只用 10.4 µs = **12 TB/s**，远超 HBM 带宽 ⇒ **该测量是 host enqueue 限制的，不是设备时间**。
⇒ **验证了你给的方法学提醒**：这个算子上事件计时给假数，必须用 msprof。

### 1.5 【未确认】隔离 msprof 采集未成功
尝试多种方式均未拿到隔离 op_summary：CLI 默认采集窗口只有 **312 ms** 且落在 Python import 期；
`--delay/--duration` 报 `Operation not permitted`；`torch_npu.profiler` 报 `ERR00100`。
⇒ 隔离设备时间**未取得**，§1.1–1.2 全部来自**服务 profile**（真设备数据，可信）。

---

## 2. 第 2 步：有没有便宜的替代路径

### 2.1 `dispatch_ffn_combine` —— **不能"只融合 dispatch+combine"**【实测·代码】
`csrc/torch_binding.cpp:2350`：
```
dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, Tensor expert_idx,
                     Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, Tensor[] bias2,
                     Tensor probs, str group, int max_output_size,
                     Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None,
                     float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)
```
* 它**接收两层的专家权重** ⇒ 天生包含 `gmm1 + swiglu + gmm2`，**没有"只做 dispatch+combine"的形态**
* 它还带 **`str group`（通信组）** ⇒ 属 **MC2 通信融合**系；而 `MC2=1` / `FUSED_MC2=1` 我们已实测**中性偏负**
* ⚠️ **只注册了一个变体**：`csrc/mc2/dispatch_ffn_combine_{,bf16,w4_a8}` 三个目录都在，但 `torch_binding.cpp` 里**只绑定 `dispatch_ffn_combine`**（`_bf16` / `_w4_a8` 无 torch 绑定）⇒ 想用别的变体还得自己加绑定

⇒ **要吃掉 686.9 + 242.0 = 929 µs/步，就得接受"整段 FFN 全融合 + MC2"**，不是小改动。

### 2.2 `MoeTokenUnpermute`（6.04 µs × 40 = 242 µs/步）
* 【实测】**Block Num = 6**（不是 48）⇒ 它的固定开销结构**与 `MoeInitRoutingV3` 完全不同，小得多**
* 服务实测：15 行 → 7.4 µs；36 行 → 6.1 µs；1536 行 → 15.7 µs
* 它已经是"小核数 + 轻量"的形态，**没有明显浪费**；要省只能靠"整体不被调用"（即全融合），单点优化空间小

### 2.3 `MoeGatingTopKHash`（3.98 µs × 40 = 159.7 µs/步）+ gate matmul
* `moe_gating_top_k(Tensor x, int k, ...)` 与 `moe_gating_top_k_hash(Tensor x, ...)` —— **都没有权重入参**
* ⇒ **确认无法融合 gate matmul**（审计旧结论成立）
* 【实测】Block Num = 6，`aiv_scalar_time` 1.91 µs / dur 4.9 µs ⇒ 也偏固定但绝对量小

---

## 3. 第 3 步：值不值得做 + 最小可验证方案

| 候选 | 预期收益 | 改动量 | 风险 | die5 先验 |
|---|---|---|---|---|
| **A. 降 `MoeInitRoutingV3` 小 batch 的 BlockDim（推荐）** | 【推断】**240–400 µs/步（0.9–1.5%）**（17.8 → 约 8–12 µs 量级）× 40 | **1 行 tiling**（`PostTiling`）+ **可能需 kernel 侧让 46 个核提前 return** | **中**：kernel 的 gather/scatter 阶段可能假设核数=48 | ✅ **可以**：vendor 覆盖 + 逐元素对拍（上轮 sentinel 已证 vendor 生效） |
| B. `dispatch_ffn_combine` 全融合 | 【推断】929 µs/步（3.5%）上限 | **>500 行**（加 torch 绑定 + MOE_AG 路径适配 + shared expert） | **高**：走 MC2 系，而 MC2/FUSED_MC2 已实测**中性偏负** | ⚠️ 只能小规模先验，全路径必须上服务 |
| C. gate matmul + topk 融合 | — | — | **不可行**（无权重入参） | 不需要 |

### 3.1 候选 A 的最小可验证方案【已把"关键未核实点"查清】

**查清的两处决定性事实**（决定改动量）：
1. `op_host/moe_init_routing_v3_tiling.cpp:304`：**`moeInitRoutingV3TilingData.set_coreNum(aivNum)`**
   ⇒ `coreNum` 被**硬编码为 48**，与 `SetBlockDim` 是**两个独立的东西**
2. kernel 侧：
   * **工作划分一部分用 tiling 的 `coreNum_`**（`moe_v3_cut_origin_t.h:164`、`moe_v3_expert_tokens_count.h:97`）
   * **另一部分用 `GetBlockNum()`**（`moe_v3_expert_tokens_count.h:134`）
   * 且**大量 `AscendC::SyncAll()`**（全网格栅栏，`moe_v3_*` 十余处）
   * 同时存在 `if (blockIdx_ < needCoreNum_)` 守卫（多余块会跳过工作）

⇒ **只把 `SetBlockDim` 改成 1 会静默算错**：代码按 `coreNum_ = 48` 切分工作，
但只有 1 块在跑 ⇒ 47 份工作没人做，且没有等待机制 ⇒ **结果不完整（静默错）**。

**因此最小改动是"两行、同一文件、必须成对"**：
```cpp
// op_host/moe_init_routing_v3_tiling.cpp
// ① PostTiling()
if (isEmptyTensor_)         context_->SetBlockDim(1);
else if (sortMode_ == 0UL)  context_->SetBlockDim(1);   // ★ 新增
else                        context_->SetBlockDim(aivNum);

// ② 第 304 行附近（目前是 set_coreNum(aivNum)）
moeInitRoutingV3TilingData.set_coreNum(sortMode_ == 0UL ? 1 : aivNum);   // ★ 必须与 ① 成对
```
⚠️ 注意 `sortMode_` 在 `Tiling4VBSCompute()` 里才被赋值；`set_coreNum()` 在第 304 行（较早）
⇒ **需要把 `set_coreNum` 移到 `Tiling4VBSCompute()` 之后**，或在 `PostTiling()` 里一并设。

**改动量更正：约 2–10 行（同一文件），不是"1 行"；但仍属小改动。**
2. **构建**：走 gmm1 线已打通的 AI core 重编链（`~/tmp/gmm1/tools/rebuild_v2_op.sh`），
   或用同一套 `V41_HC_OPP_PKG` vendor 覆盖（上轮已实测 vendor 优先）
3. **验证**（三层）
   * **正确性**：与 stock 逐元素对拍（`expandedX` / `expandedRowIdx` / `expertTokensCount` 必须完全一致）
   * **设备时长**：msprof 取 `Task Duration` + `Block Num`（应从 48 降到 1）
   * **端到端**：die5 上单独跑算子对拍；服务侧再跑一次 `decode ms/step`

### 3.2 一句话回答"哪些改动能先验 / 哪些必须上服务"
* **能先验**：A（单算子，die5 对拍 + msprof 即可）
* **必须先上服务**：B（涉及 MOE_AG / 通信组 / shared expert 全路径），只能先做小规模 smoke

---

## 4. ⚠️ 本轮测量环境的一个问题（需要协调）

`~/tmp/die5_lock.sh` 是协作式，**另一条线（trackB4）在我测量期间未取锁就跑了 `realization.py --mode mix_graph_wall`**：
* 【实测】我在 §1.4 的参数扫描（锁标签 `moe: initrouting scan`）期间，锁被他人 release 并占用；
* 之后我的隔离测试连续报 `ERR00100` 与 `copy_d2d_baseopapi` 错误，直到发现该进程仍在运行；
* ⇒ **§1.4 的隔离数值（10.4–10.5 µs 平）可能被污染**；
  但**结论不变**（125.8 MB / 10.4 µs = 12 TB/s 本身就不成立 ⇒ 必然是 host-limited），
  且 §1.1–1.2 用的是**服务 profile**，与 die5 无关。

---

## 5. 交付与卫生
* 工具：`scale_test.py`（图内扫描）、`scale2.py`（含 eager/输出核验）、`svc_shape.py` / `svc_shape2.py` / `kname.py`（服务 profile 只读分析）、`run_*.sh`（均带锁 + 唯一标签 + trap release）
* 每次测量：`acquire <标签>` → 测量 → `release <标签>`；本轮结束 `die5_lock.sh status = FREE`
* 未碰 `dsv41-tp8k5`/19210；未用 `rm -f`（一律 `mv` 到 `.bak<时间戳>`）
