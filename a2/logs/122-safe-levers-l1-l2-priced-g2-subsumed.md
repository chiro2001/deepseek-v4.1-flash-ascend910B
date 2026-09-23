# 122 — ★★★ `SAFE_LEVERS` 单卡定价：**L1 0.099–0.124** ｜ **L2 0.359（吞掉 G2）** ｜ 三条**负结果**（外提无对象 / 第3条 Cast 是算子侧硬约束 / 第4条改法会动数值）

> 2026-09-23 06:40–07:2x CST。执行：子代理 **`SAFE_LEVERS`**（单卡 **c1 / Phy-ID 6**，`tools/a3_chip.sh c1` 全程持锁）。
> 装置：**逐字复用 `GEOM_MICRO/scripts/g_micro.py`**（图内斜率 + 同图 Level1 profiler）+ 新增 **eager profiler** 格（生产里这两段是 eager 发射）。
> 产物：`a2/agents/SAFE_LEVERS/{REPORT.md,patches/,scripts/,out/}`（宿主 + A3 同名，md5 一致）。
> 标记：**【实测】/【推断】/【未确认】**。判据纪律：**全程不用 8 卡 `hp`**。

---

## 0. 一句话

两个"不改数值语义"的杠杆被定价，**且都是正的**：**L1 的 2× MoE 掩码 Cast = 0.099–0.124 ms/step**、**L2 的 `_slot_mapping_2d` 预备链单核融合 = 0.359 ms/step**（比 `GEOM_MICRO` 的 G2 大 2.1×，**因为它是同一对象的更完整版本** ⇒ **G2 不必再单独上臂**）。
★ 同时给出**三条负结果**，每条都有源码/算子侧的硬证据 —— 这些比正结果更重要（避免我们花机时做不动的改）。

---

## 1. 【实测】两格的定价

| 杠杆 | 唯一对象 | 原版 µs/次 | 候选 µs/次 | **ms/step** | 判据 |
|---|---|---:|---:|---|---|
| **L1** | `token_dispatcher.py:452-453` 的 MoE 掩码链（**2× `Cast INT32→INT64 "6,6"`**） | 9.48（op 口径/层） | 5.56 | **0.099（生产 prof 单价）×40** ～ **0.124（图内斜率）** | 重复率≥2 ⇒ 值得 |
| **L2** | `dsa_v41.py` 的 `_slot_mapping_2d` 预备链（`cols()` + 2×`ViewCopy 16384`） | **32.72**（eager/事件） | **2.82**（1 个 Triton 核，**就地写回同一缓冲**） | **0.359**（= 12 事件/步 × 29.9 µs） | 同上 |

---

## 2. ★★ L2 与 G2 是**同一对象**，而 L2 更安全（**这条改了排期**）

| 方案 | 收益/次 | ms/step | 是否换掉那个张量 |
|---|---:|---:|---|
| `GEOM_MICRO` 的 **G2**（`torch.stack` → 1 个 `Pack`） | 33.29 → 21.65 µs | 0.138–0.168 | ★ **换**（每步新建 tensor） |
| **`SAFE_LEVERS` 的 L2**（1 个 Triton 核**就地写回** `_slot_mapping_2d`） | 32.72 → **2.82 µs** | **0.359** | ★ **不换**（输出仍是同一个持久缓冲） |

★★ **为什么 L2 更安全**：`_slot_mapping_2d` 是**图输入**（`npugraph_ex` 捕获时它是 Placeholder，下游 scatter 直接吃它）。
G2 的 `torch.stack` 会把它换成每步新建的 tensor ⇒ **可能触发重捕获 / 图输入失配**（本仓反复栽过的 EE1016 家族）。
★ `SAFE_LEVERS` 的判据也绑得对：「**输出仍是同一个持久缓冲 `self._slot_mapping_2d`**（下游 scatter 图输入、结构未变）」。

★★★ **机理（本卷最硬的一条附带发现）**：**跨步列视图写在 Ascend 上退化为逐元素标量写** ——
6 个元素的跨步写要 **17.8–199 µs**，而**一条连续 store 只要 0.19–2.3 µs**。
⇒ 这同时解释了：G2 那 8 µs 的来源、"缩小缓冲"（G2 候选 B）**为什么无效**（形状本身是病根，不是 base 大小）、以及 L2 为什么能一次吃掉几乎全部。

---

## 3. 【★ 负结果】三条不该做的改（都有硬证据）

### 3.1 `L2` 的"循环不变量外提"**没有对象**（推翻 `GEOM_MICRO §1` 的附带发现）

三条证据：
1. 报告点名的 `arange/where/to(int32)` 表构在 **`VLLM_V41_KV8_FUSE=1` 下是死代码**（`_kv8_swa_table_kernel` 40 次/步已接管）；
2. 真身 `cols()` 的**层间重复早已被 `shared[slot_key]` 缓存消掉**（`ViewCopy 23.75/步` = **12 事件 × 2**，而**不是** 80/步）；
3. 那 12 次是 **12 个 cache group 的不同 slot_mapping**（旁证：runner 的 `_compute_slot_mapping_kernel` 也是 **12/步**）⇒ **没有可提的不变量**。

⇒ ★ **改法改为"同对象的单核融合"**（`GEOM_MICRO §6.2` 自己点名的方向），逐位等价，收益 **0.359**。
★ **教训**：`GEOM_MICRO` 那条"≈0.130 ms/step 的附带机会"是**对着一个已经不存在的对象**定价的 —— 与 `118`（`width` 同名不同物）/`089`（差第三个变量）**同族**。

### 3.2 L1 的第 3 条 Cast **卡在 CANN 算子侧**（不是我们代码）

`MoeGatingTopKHash` 的 `input_ids.to(int64)`：`moe_gating_top_k_hash.json` 的**三个 bin 里 `input_ids` 全声明 `int64`**
⇒ 去掉它**等于重编 CANN 算子** ⇒ **不改**。

### 3.3 L1 的第 4 条 `FLOAT→BF16` **改法会动数值**

`PROF_MINE §4.1` 说它的消费者是 `Abs "36"` —— ★ **那条归因是错的**（`Abs "36"` 实际是 `INT32;INT32`；
"同流下一条"这个启发式在这里失手）。真消费者是 **`MoeTokenUnpermute "36,5120;36;6,6"`**（`token_combine` 的 `probs=…to(hidden_states.dtype)`）。
⇒ **提前转 BF16 会把舍入移到掩码之前 ⇒ 动数值 ⇒ 不改**。

★ **这两条是子代理对既有结论的纠正，我已接受**（并已通知正在跑的 `G2G3_ARM`，防它引用 `PROF_MINE` 的第 4 条）。
⇒ **L1 的真值是 0.099–0.124，不是 0.147**。

---

## 4. 对 goal 账目的影响（**A 层第四次重算**）

| # | 项 | ms/step | 状态 |
|---|---|---:|---|
| 1 | `swa_table` @ width=1040 | **0.058** | ✅ 已落袋（`r8-tbl`） |
| 2 | **L2** `_slot_mapping_2d` 单核融合（**吞掉 G2**） | **0.359** | ✅ 已定价，**待上臂** |
| 3 | **L1** MoE 掩码 2× Cast | **0.099–0.124** | ✅ 已定价，**待上臂**（与 L2 打包） |
| 4 | G3 fp16 scale 融合 | 0.263–0.326 | ⏳ `G2G3_ARM` 在做 |
| 5 | `HcPost`+`HcPre` 跨支融合 | 0.30–0.64 | ⏳ 待派（**工作量大**，见 §5） |
| | **A 层合计** | **≈1.08–1.51** | ⇒ 27.905 − 1.51 ≈ **26.4** |
| 6 | **PGO**（数值无关） | **0 ～ 2.1【未确认】** | ⏳ `G2G3_ARM` 在测（`logs/121`） |
| | **合计（含 PGO 上限）** | | ⇒ **≈24.3** |

---

## 5. `HcPost`+`HcPre` 跨支融合：先例**存在但不是真核**（我核过）

| 事实 | 证据 |
|---|---|
| 仓内**有**这个抽象 | `graph_prep/ref/vllm/model_executor/layers/mhc.py:377` `class MHCFusedPostPreOp(CustomOp)`，docstring 逐字：*"Fused MHC post block followed by the next MHC pre block."* |
| CUDA 走**真融合核** | `:409` `torch.ops.vllm.mhc_fused_post_pre_tilelang(...)` ⇒ **TileLang kernel**（≈ 无 NPU 版） |
| ★ 但 NPU 的 `forward_oot` **只是 Python 组合** | `graph_prep/src/vllm_ascend/patch/worker/patch_triton.py:279-313`：`_mhc_post_torch(...)` 然后 `_mhc_pre_torch(...)` ⇒ **不省任何开销**，只是把两个 op 的调用点收进一个函数 |
| 我们的 V4.1 走的是**另一条实现** | 模型侧 `torch.ops._C_ascend.npu_hc_pre_v2` / `npu_hc_post`（CANN 算子），**不是** `MHCFusedPostPreOp` |

⇒ ★ **"参照 GLM5-next 的做法"这句话只对了一半**：**抽象层可以直接抄**（把 `hc_post` 与下一层 `hc_pre` 合成一次调用，调用数 86 → 46），
**但要让收益兑现，必须自己写一个真融合核**（AscendC 或 Triton），**TileLang 那份不能直接用**。
⇒ 工作量 = **C 层（周级）**，不是"改几行"；**先例给的是接口与语义契约，不是实现**。
★ 这也是本仓第一次把这条拆清楚：**`PROF_MINE §4.1` 排名 2 的 0.30–0.64 是"抽象层可抄、实现层要写"**。

---

## 6. 下一步（按"能否立刻跑"排序）

| # | 动作 | 说明 |
|---|---|---|
| 1 | **等 `G2G3_ARM`**（G3 + G2 + PGO A/B + `HcPre` 探针） | 它持 8051；它结束后我立刻跑 L1+L2 臂 |
| 2 | **L1+L2 组合臂** | `SAFE_LEVERS/scripts/run_arm_safe_levers.sh`（**DRY 冒烟已过**：组包+md5 门+端口门）；判据 = `check_safe_levers.py`（**已用基线 CSV 反向自测**：基线在 L1/L2 全 FAIL、回归闸全 PASS） |
| 3 | **最终组合臂**（L1+L2+G3+PGO） | 若 1/2 都通过，这是"生产候选"臂 |
| 4 | `HcPre` 定制核 / 跨支融合 | **C 层，周级**；需要单独立项与算子开发窗口 |
| 5 | AI_CPU 那 **1.41 ms/步**（4 类 metadata） | 同为 C 层（供应商算子） |

★ **打包原则不变**：单格 <0.2 ms 不单独上臂；**L1+L2（0.46–0.48）够一条臂，且两者 op 判据互不干扰**（L1 看 `Cast` 计数、L2 看 `ViewCopy` 计数）。
★ **最大未确认项（子代理自己标注）**：**host 侧成本** —— 生产这两段是 **eager 发射**，单卡只测了设备时间；若臂上 `hp` 反而升 >1 ms，应按"**host 变贵**"解释，**而不是判 FAIL**。

---

## 7. 附件与复现

```
patches/token_dispatcher_moemask.patched.py   md5 a91fbc48…
patches/dsa_v41.patched.py                    md5 7867da2a…（基线 30ecf49b…）
scripts/g_safe.py          # 装置（复用 g_micro + 新增 eager profiler）
scripts/l2diag.py
scripts/check_safe_levers.py   # 判据（已反向自测）
scripts/run_arm_safe_levers.sh # 8 卡臂（DRY 冒烟已过）
scripts/postcheck_safe_levers.sh
out/                       # 原始 JSON / profiler 对数
```
**唯一注意**：私有 PKG 必须落在 `S_graphfix/pkgs/` **命名空间**下（脚本已默认）——
理由：`run_arm_r8.sh:280` 的 `N_MNT_R8` 是按**挂载源路径**计数的（本仓同族第 4 次），放错命名空间会让自检门 `int8_mounts<7` 直接 FATAL。
