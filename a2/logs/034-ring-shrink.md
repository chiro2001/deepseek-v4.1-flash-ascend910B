# 034 — compressor state ring 降精度：**它是 KV8 的乘数，不是独立杠杆**（容量 ×1.000 / ×1.4655 / ×1.9133 三档实测）

> 2026-09-22 02:5x–03:4x CST。执行：子代理 **R_ringshrink**。机器：**A3（A3-node1）槽位 c1 = die 6**
> （`Ascend910_9382`，容器 `prbench-c1`；c2 当时被 `p3-c2-coex2`/`kv8pf-p26` 占，c0 被 `L3_8card` 占）。
> 全程只用 c1；没碰 `dsv41-a3` / `mooncake-master` / `jitpgo-*` / Phy-ID 8–15；没手设 `ASCEND_RT_VISIBLE_DEVICES`；
> **没写 `upstream-v41/`**；没动 `dsv41-release/`；没用 `/tmp`（`TMPDIR=~/tmp/20260922/R_ringshrink`
> 与容器内 `$TMPDIR`）；跨机传输全走 coscli；代码只写 `a2/agents/R_ringshrink/`。
> 产物：本日志 + `logs/raw/034-*` + `agents/R_ringshrink/`。
>
> **★ 与 `logs/033`（KV8_prefill）的关系**：033 用**测量用**改动量出了五个容量档，但其中三个 ring16 臂
> **死在 `Compressor requires contiguous projections, ring pages and controls`**，并把「把 state 视图做成连续的
> 再重跑三臂到 `/health`」明确交给了 034。**本任务的补丁把这条打通了**：§4 的六条臂**全部到 `/health`
> 并跑完两轮压测**，容量读数与 033 **逐格相同**。

---

## 0. 六句话结论（先给主代理）

1. **⛔【实测·推翻任务书 §1】ring 缩 FP16 单独用是 ×1.000，一个字节都不省。** 引擎臂
   `A3`（无 KV8，ring F32）与 `A2`（无 KV8，ring FP16）**都是 `GPU KV cache size: 22,719 tokens`**。
   机制**不是**任务书写的 "3×131072 全由 ring 顶着"，而是：ratio-2 槽的 capacity =
   `max(kv+index 73856, ring 131072, **SWA 别名 131072 ×10**)`——**ring 与 BF16 SWA 别名同值并列顶着**，
   ring 减半后 **BF16 SWA 页 131072 立刻接管**。⇒ 任务书的 `×1.571` 与 `×2.207` **都不成立**。
2. **✅【实测】它是 KV8 的乘数**：`KV8 双平面 + ring FP16` = **43,469 tokens = ×1.9133**，而
   `KV8 双平面 + ring F32` = **25,801 = ×1.1357** ⇒ **ring 单独贡献 ×1.6842**（= 1.9133/1.1357）。
   页几何 `540928 → 282880 B/block`，与 033 的读数**逐格相同（偏差 0%）**。
3. **✅【实测】"只量化 SWA + ring FP16"（不碰 long-KV）是真正可上线的中间档**：**33,295 = ×1.4655**，
   六条臂全部到 `/health`。而 `只量化 SWA + ring F32` = **22,719 = ×1.0000**
   （⇒ **纠正 033 §5.4 的"×1.135"**：×1.135 是"KV8 双平面 + F32 ring"，不是"SWA-only + F32 ring"）。
4. **★★【实测·真权重】误差不随序列长度累积，而且 FP16 的代价比现有 bf16 地板还小。**
   用**真 compressor 权重**（层 2/8/14 的 wkv/wgate）+ **真 kernel** + **fp64 golden**，
   在设备上跑 L = 1K/8K/32K 的 decode（每步喂新投影，B=8）：
   - 地板（输出被强制 bf16 的**表示误差**）= **1.6565e-3 ~ 1.6583e-3**；
   - **F32 ring 的总误差 = 1.6562e-3 ~ 1.6582e-3 ⇒ 现状本来就正好坐在地板上**；
   - **FP16 ring 的总误差 = 1.6641e-3 ~ 1.6663e-3 ⇒ 只比 F32 高 +0.41%~+0.51%**；
   - **L=1K → 32K，`rel_L2` 只从 7.679e-4 漂到 7.734e-4（+0.7%）；首/中/尾三段彼此差 <1%**。
   ⇒ **不存在"随 L 累积"**（机制：ring 存的是**原始投影**、每步覆写，**复用距离 ≤ 32 token**，
   不回读自己的输出；见 §3.2 的三条代码事实）。
5. **【实测·判据④】cast 免费**：40 步 NPUGraph 里 `compressor_from_triton` 每步
   F32 7.83~8.10 µs vs FP16（交错页）7.80~8.09 µs，**Δ ∈ [−0.11, +0.26] µs = 噪声**。
   ★ 且新寻址式在 **FP32 下逐比特 no-op**（原版包 vs 影子包 **int64 逐比特和相同**：`-477976451`）。
6. **★★【实测】改动面是 6 个逻辑改动点 / 21 组文本替换 / 29 处落点（含 3 个 kernel 签名 + 2 个 launch 点），不是 028 说的"两处常量/断言"。**
   最关键的一处 028/033 都没写：`reshape_cache` 用**槽页步长**建 state 视图 ⇒ ring 减半后
   **张量不再连续**（页步长 66560 B，而一页 32 行只占 65536 B）⇒ kernel 的
   `(cache_row*CACHE_SIZE + slot)*2*HEAD_DIM` 寻址**必须改成按页步长**（本任务的做法），
   否则不是"慢一点"而是**静默读错页**。
7. **⛔【实测+算术】`ring INT8` 整条支线关闭**：容量上它与 FP16 **完全相同（都是 ×1.9122）**；
   带宽上它每步只省 **0.041 µs**，比实测噪声地板（±0.26 µs/step）低 6 倍。
   ⇒ **容量零收益 + 时延零收益**，而代价是 scale 平面 + kernel 内多两阶段。
   **复活条件**（写死免得后人重复挖）：**只有当 SWA 页被压到 65536 B 以下**（如 SWA 上 int4）时，
   ring 才会重新 binding。**在此之前不开。** `logs/024` 的另两条缩法（`head_size` 减半 / `block_size` 32→16）
   **同样容量零收益、且要改算法 ⇒ 一并关闭**（§5.1/§5.2）。

---

## 1. ★ cannbot 对照（AGENTS.md §6：写 kernel / 做量化验证前必查）

在 A3-node1 只读查 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`。

| 查的地方 | cannbot 说什么 | 采纳 / 没采纳及理由 |
|---|---|---|
| **`model-infer-quantization/SKILL.md` §7（"验证真实生效与收益"）** | ★ 原文把**"等价性自检（接线正确性，非精度评测）"**与精度评测分开：*"量化算法的绝对精度由产物方报告，本 skill 不评测精度；这里只验证「接线对不对」"*；生效验证要求 **probe 优先、贴输出、不空打勾**；`references/quantization-fusion-and-benefit.md` §B.1 明确 *"**显存收益可以独立成立，不等价于低时延收益**"* 且 *"eager 数据不进收益结论，仅用于功能验证"* | ✅ **逐条采纳，并直接决定了本任务的证据结构**：①"接线对不对" ⇒ 用 **p7 隔离探针**（把 ring 页灌成已知值，反推 kernel 读的是哪一行）+ **CACHE_PAGE 的 int64 逐比特 no-op 对拍**，而不是只看容量变了；②"显存收益独立成立" ⇒ 容量（×1.000/×1.4655/×1.9133）与时延（+0 / +424 µs）**分栏报**；③"eager 不进收益结论" ⇒ 时延只用 **NPUGraph 内**的配对读数 |
| **`model-infer-quantization/references/quantization-fusion-and-benefit.md` A.2（算子映射）** | 把 *"`IndexerProlog` / `LightningIndexer` / `Sparse Flash Attention` / **模型专属 mega-kernel**"* 列为 **等级 D**：*"量化必须服从融合子图契约，尤其是**输出 dtype**、cache layout、必需 side tensor"* | ✅ **采纳为红线**：compressor 正是"模型专属 mega-kernel"（Triton 写的 3-kernel 链），所以我们**没有动它的输出 dtype**（`out` 仍是 BF16，`compressor_triton.py:661` 有硬校验）、没有动 slot 映射、没有动 frozen 的 mask 语义；只改**存储 dtype 与页步长** |
| **`ops/pypto-precision-compare/precision-verify/SKILL.md:145-171`（dtype→rtol/atol 表 + 双阈值）** | 给出 dtype 容差表：**FP16 rtol 1e-3 / atol 1e-3**、**BF16 rtol 5e-3 / atol 5e-2**；判定用双阈值（警告阈值 `abs_sum*rtol/2+atol`、失败阈值 ×128） | ⚠️ **部分采纳**：本任务的量是**逐元素舍入**（不是算子和），所以用 `rel_L2 / max_abs / 逐比特相等比例` 更贴题；但**采纳它的量级**——README 里 fp16 的 rtol=1e-3 与实测 `rel_L2=7.7e-4` **同量级且更小**，与"FP16 不是问题"的结论一致 |
| **`model-infer-kvcache/SKILL.md`（滑窗/长序列那一条）** | *"长序列 `KV_len > sliding_window` 的正确性必须靠**模型层**保证——环形 buffer 写 cache、或 `actual_seq_lengths_kv` 截断到窗口长度，不是 op 层负责"* | ✅ **采纳（与 020/033 一致）**：state ring 正是"**环形 buffer 写 cache**"的那类，所以 ring 的 dtype/环长都在**模型层**决定；我们**没有**碰 `seqused_*` / mask，只改模型层持有的 ring 页 dtype |
| **cannbot 有没有"环形状态缓存降 dtype"的专门条款？** | ❌ **没有**：`grep -rn "circular\|ring\|环形" model-infer-kvcache/ model-infer-quantization/` 无相关命中；KV cache 量化条款只覆盖 attention 的 K/V 平面 | **记录为文档空白**：本任务的判据是**我们自己定的**（地板对比 + L 扫描 + 页步长 no-op），不是抄来的 |

**一句话**：cannbot 给了两条决定性约束——**"量化必须服从模型专属 mega-kernel 的输出 dtype 契约"**
（⇒ 我们一个字没动池化输出的 BF16）与**"显存收益独立于时延收益"**（⇒ 容量与时延分栏）；
它**没有**环形状态缓存的降精度条款，所以精度判据是我们自己立的地板对比。

---

## 2. ★★ 容量：三档全部落到引擎（`GPU KV cache size`），并**推翻任务书 §1**

### 2.1 逐槽页构成（**镜像自己的** `plan_cache_slots` 现算，不是拿 `max()` 目测）

`agents/R_ringshrink/p1_pages.py`（容器内，**不占卡**）。自检两条：
**① 泛化式在 `scale_dim==0` 时与原版逐字节一致**（`diffs=[]`）；
**② 逐槽复算的 capacity == 镜像 `plan_cache_slots` 的输出**（`True`）。

现状（全 BF16 + FP32 ring，生产配置 head_dim 512 / index_head_dim 128 / block 128）：

| slot | 层 | ratio | long-KV | index | kv+index | **ring** | **SWA 别名** | capacity | binding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2 | 2 | 65536 | 8320 | 73856 | **131072** | **131072** | **131072** | ring 与 SWA 别名**同值并列** |
| 1 | 8 | 2 | 65536 | 8320 | 73856 | **131072** | **131072** | **131072** | 同上 |
| 2 | 14 | 2 | 65536 | 8320 | 73856 | **131072** | **131072** | **131072** | 同上 |
| 3 | 20 | 1 | 131072 | 16640 | 147712 | — | 131072 | 147712 | long-KV+index |

`pool_bytes_per_block = 540928`（逐字 = `logs/020` 实测），B/token = 4226。

### 2.2 【实测·算式重算】十一个变体（全部走镜像 `plan_cache_slots`，三条已知锚点 540928/476416/524288 全部复现）

| 变体 | slot 页 | pool B/block | ×vs 现状 | ring 页 |
|---|---|---:|---:|---:|
| **现状** bf16/bf16 + ring F32 | `[131072,131072,131072,147712]` | **540928** | ×1.0000 | 131072 |
| **ring→FP16（不动 KV8）** | 同上 | **540928** | **×1.0000** | 65536 |
| ring→BF16 / ring→INT8（不动 KV8） | 同上 | 540928 | ×1.0000 | 65536 / 32768 |
| **只 SWA int8 + ring F32** | 同上 | **540928** | **×1.0000** | 131072 |
| **只 SWA int8 + ring FP16** | `[73856,73856,73856,147712]` | **369280** | **×1.4648** | 65536 |
| 只量化 long-KV + ring FP16 | `[131072,131072,131072,131072]` | 524288 | ×1.0317 | 65536 |
| KV8 双平面 + ring F32 | `[131072,131072,131072,83200]` | 476416 | ×1.1354 | 131072 |
| **KV8 双平面 + ring FP16/BF16/INT8** | `[66560,66560,66560,83200]` | **282880** | **×1.9122** | 65536 / 65536 / 32768 |

★ **INT8 与 FP16 的容量完全相同（都是 ×1.9122）**：`ring ≤ 65536` 之后 binding 变成 SWA-int8 页 66560，
再缩 ring 一分钱不省 ⇒ **ring INT8 的额外收益 = 0，scale 那套活可以不做**。

### 2.3 【实测】引擎臂（tiny-dummy 单卡，六条臂**全部到 `/health`**）

`agents/R_ringshrink/run_engine_arms.sh`；影子包 = `agents/KV8_fuse/shadow` + 本任务的 ring 补丁；
KV8 平面由 env 门控；原始数据 `logs/raw/034-engine-arms/`（含每条臂的 `kv_size.txt` / `client.json` / `server.log`）。

| 臂 | 平面 | ring | **`GPU KV cache size`** | **×** | 到 `/health` | 两轮压测 |
|---|---|---|---:|---:|---|---|
| `A3` | 无 KV8 | F32 | **22,719** | ×1.0000 | ✅ 61 s | 4/4 ok，0 fail |
| **`A2`** | **无 KV8** | **FP16** | **22,719** | **×1.0000** | ✅ 60 s | 4/4 ok，0 fail |
| `B1` | 只 SWA int8 | F32 | 22,719 | ×1.0000 | ✅ 45 s | 4/4 ok，0 fail |
| **`B0`** | **只 SWA int8** | **FP16** | **33,295** | **×1.4655** | ✅ 60 s | 4/4 ok，0 fail |
| `A0` | KV8 双平面 | F32 | 25,801 | ×1.1357 | ✅ 90 s | 4/4 ok，0 fail |
| **`A1`** | **KV8 双平面** | **FP16** | **43,469** | **×1.9133** | ✅ 65 s | 4/4 ok，0 fail |

* **【实测】六格与 §2.2 的页几何逐格吻合（偏差 <0.05%）**；且与 **`logs/033` 的五格逐字相同**
  （22,719 / 25,801 / 33,295 / 43,469）⇒ **两条独立路径（我按页几何算 / 033 起真引擎量）互证**。
* ★ **033 挂在 `Compressor requires contiguous...` 的三个 ring16 臂，在本任务的补丁下全部起服成功**
  （`A1`/`A2`/`B0` 就是那三个臂），并**跑完了两轮 fill/replay**。
* ⚠️ **本表的 TTFT 不可用于时延结论**：六条臂的 `prompt_salt` 各不相同（18674/18757/18837/18915/18992）
  ⇒ **不是配对测量**；时延结论只用 §4.3 的 NPUGraph 配对读数。

---

## 3. ★★ 数值：真权重 + 真 kernel + fp64 golden，L=1K/8K/32K

### 3.1 为什么必须重做（028 的三条局限）

| 028 的做法 | 局限 | 本任务 |
|---|---|---|
| numpy CPU 仿真、合成 N(0,σ²) 分布、扫门控尖度 τ 五档 | **不是真权重、不是真 kernel**；门控尖度是**猜的** | 抽**真权重**（`p0_extract_weights.py`：层 2/8/14 的 wkv/wgate/norm，checkpoint 里是 BF16，加载时 upcast 到 FP32，两步都精确），跑**真 `compressor_from_projected`** |
| 只给 FP16 一档 | 缺 BF16 对照 | 给 F32 / FP16（连续页）/ FP16（交错页）三档 + 纯表示地板 |
| `rel_L2` 是"ring 往返误差" | **没说清"1.66e-3 地板"到底是哪个量** | 用 **fp64 golden** 把「地板」「F32 总误差」「FP16 总误差」三个量分开 |
| 没做长度扫描 | **ring 是状态，可能随 L 累积** | **L = 1K / 8K / 32K × 3 层**，且按位置分箱 |

**地板 1.66e-3 的来源（复核结论）**：它是
`‖bf16(fp32(g)) − g‖₂ / ‖g‖₂`，其中 `g` = 用 **fp64 按池化的数学定义**（逐维
`softmax([score₀,score₁])·[kv₀,kv₁]`）从**同一组 FP32 原始投影**算出来的输出。
即：**"池化输出被强制成 BF16"这件事本身的相对表示误差**，与 kernel 无关。
实测 **1.6565e-3 ~ 1.6583e-3**（逐字复现 028 的 1.6571e-3）。

### 3.2 ★ "会不会随 L 累积"：**不会**——三种独立证据

**(a) 【实测·代码事实】三条，缺一不可**：
1. ring 里存的是 `[kv|score]` 的**原始投影**（`_cache_update_kernel` 从投影大矩阵 `kv_ptr/score_ptr` 取，
   不是从 ring 自己取）⇒ **没有自反馈**；
2. 环长 32 行、ratio 2 ⇒ **复用距离 ≤ 32 个 token**（`tail_count = min(used_len,32)`，
   读只在 `seg_off < 0` 时发生，而 `slot = token_pos % 32`）⇒ 单次误差**不叠加**；
3. 池化输出本身强制 BF16（`out.dtype != bfloat16` 直接 raise）⇒ ring 的舍入与输出的舍入**同一量级或更小**。

**(b) 【实测】设备上跑真权重 + 真 kernel（B=8、每个 chunk 换种子，不是周期序列）**：

| 层 | L | 地板 | F32 ring vs golden | **FP16 ring vs golden** | FP16 vs F32 | 逐比特相等 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1,024 | 1.6583e-3 | 1.6582e-3 | 1.6651e-3 | 7.679e-4 | 95.57% |
| 2 | 8,192 | 1.6579e-3 | 1.6577e-3 | 1.6647e-3 | 7.728e-4 | 95.56% |
| 2 | 32,768 | 1.6572e-3 | 1.6570e-3 | 1.6645e-3 | 7.734e-4 | 95.56% |
| 8 | 1,024 | 1.6565e-3 | 1.6563e-3 | 1.6641e-3 | 7.733e-4 | 95.76% |
| 8 | 32,768 | 1.6565e-3 | 1.6562e-3 | 1.6661e-3 | 7.744e-4 | 95.75% |
| 14 | 1,024 | 1.6573e-3 | 1.6571e-3 | 1.6663e-3 | 7.778e-4 | 95.89% |
| 14 | 32,768 | 1.6582e-3 | 1.6581e-3 | 1.6661e-3 | 7.764e-4 | 95.88% |

* **关键读数**：**F32 ring 的总误差 == 地板**（1.6562e-3 vs 1.6565e-3）⇒ **现状 kernel 已经坐在地板上**；
  **FP16 ring 只把它抬高 +0.41%~+0.51%**（最大 1.6663e-3 vs 1.6573e-3）。
  这比 028 的说法更强：不只是"低于地板"，而是**在地板之上的增量只有万分之五**。
* **L 依赖**：L 从 1K 涨到 32K（32×），`rel_L2` 漂移 ≤ **+0.7%**，且**首/中/尾三段彼此差 <1%**
  （例：层 2 L=32K 三段 = 7.732e-4 / 7.744e-4 / 7.727e-4，**无单调趋势**）。
* `cos vs golden` 打印精度下 **= 1.0**，`max_abs` 1.60e-2 ~ 1.68e-2（与 028 的 1.56e-2 同量级）。
* F16 连续页（65536 B）与 F16 交错页（66560 B）**逐比特相同**（digest 与全部统计一致）
  ⇒ 页步长改动**不影响数值**，只影响寻址。

**(c) 逐槽复算 vs 引擎读数**：§2 的两条独立路径互证（<0.05%）。

### 3.3 两次"差点误判"的探针（值得进踩坑清单）

| 现象 | 根因 | 探针 |
|---|---|---|
| p3 的 fp64 golden 与 kernel 输出差 `rel_L2 = 0.77`（cos 0.64） | ★ NPU **没有 fp64**：`torch.double()` 被**静默降成 fp32**（有告警 `Device do not support double dtype now, dtype cast replace with float`）；我原以为 golden 对不上是语义问题 | 改成 **CPU 上 numpy fp64**（每 chunk 一次 D2H），并加 `p6_diag_golden.py` 逐候选对拍 |
| 三个 ring 臂的 `output_digest` **完全相同**（FP32 与 FP16 一字不差）⇒ 看着像"FP16 没生效" | ★ **我的补丁把 page 步长插错了位置**：`_pooled_blocked(..., CHUNK_ROWS, CACHE_PAGE, True)` 让形参 `HAS_RES=CACHE_PAGE`、`CACHE_PAGE=True(=1)` ⇒ cache 读到错页（全 0） | **`p7_isolate.py` 隔离探针**：把 ring 页**手动灌成已知值**（A–F 六 case），反推 kernel 读的是哪一行；修正插入位置后 **6/6 与"读 ring slot0 的 [kv\|score]"一致**，且 p5 自检也恢复 |

---

## 4. ★★ 补丁：**6 个逻辑改动点 / 21 组文本替换 / 29 处落点**，其中 **3 个 kernel 签名 + 2 个 launch 点**

生成器 `agents/R_ringshrink/make_ring_fp16_patch.py`（**整包拷贝 + PYTHONPATH 影子包**，
不写镜像、可复现、可 diff；产物含 `ring_fp16_manifest.json` 与 unified diff）。

（下表是 **6 个逻辑改动点**；生成器实际做了 **21 组文本替换 / 29 处落点**，逐条列在
影子包的 `ring_fp16_manifest.json` 的 `edit_sites` 里。）

| # | 文件 | 位置 | 改什么 | 为什么必须改 |
|---|---|---|---|---|
| 1 | `core/deepseek_v41.py` | `DeepseekV41CompressorStateSpec.__post_init__` | `dtype != float32` → `not in RING_STATE_DTYPES`；加 `RING_STATE_DTYPES` / `ring_state_dtype()`（`VLLM_V41_RING_FP16` 门控，默认关） | spec 层的硬守卫（028 已指认） |
| 2 | `core/deepseek_v41.py` | `reshape_cache` | `sum(plane_sizes) != block_stride` → **`> block_stride` 且 `block_stride % dtype.itemsize == 0`** | ★ **"ring 必须填满整个 slot"** 这条断言在 ring 缩后必然为假（65536 ≠ 66560） |
| 3 | `models/deepseek_v41/compressor.py` | `DeepseekV41CompressorStateCache.__init__` | 同 #1 的 dtype 守卫 | 第二道守卫 |
| 4 | `models/deepseek_v41/compressor.py` | state spec 构造点 | `dtype=torch.float32` → `dtype=ring_state_dtype()` | 真正把 dtype 传下去的地方 |
| 5 | `ops/triton/compressor/compressor_triton.py` | `compressor_from_projected` | 拆开 dtype 校验（**kv/scores 必须 FP32**；ring 允许 FP16）+ **新增页步长校验** + **把 `is_contiguous` 收窄到 kv/scores/metadata/out** | ring 页不再连续（见 #6） |
| 6 | `ops/triton/compressor/compressor_triton.py` | **`_pool_kernel` / `_pooled_blocked` / `_cache_update_kernel`** | 三个 kernel 各加 `CACHE_PAGE: tl.constexpr`；寻址 `(cache_row*CACHE_SIZE + slot)*2*HEAD_DIM` → **`cache_row*CACHE_PAGE + slot*2*HEAD_DIM`**；cache 读出显式 `.to(tl.float32)`、写入显式 `.to(cache_ptr.dtype.element_ty)`；两个 launch 点传 `CACHE_PAGE=state_cache.stride(0)` | ★★ **028 与 033 都漏掉的那一处** |

### 4.1 ★【实测·代码事实 + 探针】为什么必须动 kernel：`reshape_cache` 给 state 用的是**槽页步长**

```
slot capacity = max(kv+index, state_ring, *swa_aliases)
             = max(73856, 65536, 66560) = 66560        # ring 缩 FP16 之后
reshape_cache:  as_strided(..., stride=(block_stride // itemsize, 32→…, 1))
                                      ^^^^^^^^^^^^ = 33280 fp16 元素 = 66560 B
```
而一页的 32 行只占 `32 × 1024 × 2 = 65536 B` ⇒ **`stride(0) != 32*2*HEAD_DIM` ⇒ 张量不连续**，
而 kernel 的旧寻址**隐含假设两者相等**。→ **不是"慢一点"，是静默读错页**。

**两条路，我们选了 (a)**：

| 路 | 做法 | 代价 |
|---|---|---|
| **(a) 本任务** | 保持 ring 的自然几何（65536 B），**把页步长变成 kernel 参数** | 3 个 kernel 签名 + 1 个 constexpr + 1 处连续性校验；**不改几何、不需要魔数** |
| (b) 033 §4.3-⑤ 的建议 | 把 state 视图做成"连续的"（等价于让一页的行宽填满 slot，即 `head_size 1024 → 1040`） | 仍要改 kernel 的**行宽**（`2*HEAD_DIM` → 1040）；且引入**与 SWA 页几何耦合的魔数**（SWA 页一改，1040 就错） |

★ **路 (a) 在 FP32 下是逐比特 no-op**：那时 `stride(0) == 32*2*HEAD_DIM`，
`cache_row*CACHE_PAGE + slot*2*HEAD_DIM` 与旧式是**同一个表达式**。
实测：原版包 vs 影子包、同一条 F32 臂、`output_digest`（bf16 位模式的 int64 和）
**都是 `-477976451`**，`vs_golden rel_L2` 都是 `1.6569540260565829e-3`。

### 4.2 副作用检查（逐条核过；除注明者外均为【实测·代码事实】）

| 假设 F32 的地方 | 核过没有 | 结论 |
|---|---|---|
| `reshape_cache` 的 `view()` 对齐 | ✅ | `block_stride % dtype.itemsize == 0`（66560 % 2 = 0），已加显式校验 |
| `_cache_plane_sizes(state_spec)` | ✅ | `rows * head_size * dtype.itemsize` 自动跟随 dtype |
| `AscendCircularBufferSpec.real_page_size_bytes` | ✅ | 同上，dtype 泛化 |
| `view/reshape/squeeze(-2)` | ✅ | `[N,32,1,1024]` → `[N,32,1024]`，stride(1)=1024=2*width、stride(2)=1，已加校验 |
| 算子 dtype 校验 | ✅ | 两处守卫 + `compressor_from_projected` 的 dtype 检查（kv/scores 仍要求 FP32） |
| `torch.empty(..., dtype=float32)` | ✅ | ring 的页**不是** `torch.empty` 建的，来自 `reshape_cache(uint8 缓冲)`；只有 `kv`/`scores` 是 FP32（未改） |
| 卸载层（`NPUOffloadingSpec`） | ✅ | state 组 `prefix_cacheable=False`，D2 修复后已被**排除在卸载之外**（`logs/009/022`）⇒ 它不认识 ring dtype 也没关系 |
| 前缀缓存 / hash | ✅ | 同上，state 组不参与 |
| MTP/DSpark 的推测解码守卫 | ⚠️ | `validate_cache_runtime` 里那句注释是 *"preserve FP32 ring residuals"*，**只是注释**；语义（S < 32 行）与 dtype 无关。**本任务没跑 DSpark 臂 ⇒【未确认】** |

### 4.3 判据④ 时延（配对、图内）

40 步进一个 NPUGraph，生产形状 B=8，层 2/8/14 各一次（`p3_ring_numerics.py` 的 `perf`）：

| 层 | L | F32 连续页 | **FP16 交错页** | Δ |
|---|---:|---:|---:|---:|
| 2 | 1,024 | 7.842 | 7.984 | **+0.142** |
| 2 | 8,192 | 7.815 | 7.856 | **+0.041** |
| 2 | 32,768 | 8.029 | 7.922 | **−0.107** |
| 8 | 1,024 | 7.932 | 7.860 | **−0.072** |
| 8 | 32,768 | 7.938 | 7.952 | **+0.014** |
| 14 | 1,024 | 7.833 | 8.089 | **+0.256** |
| 14 | 32,768 | 8.095 | 8.034 | **−0.061** |

**Δ ∈ [−0.107, +0.256] µs/步**（在 7.8–8.1 µs 上），**符号随机 ⇒ 是噪声不是成本**。
⇒ **【实测】cast 免费，判据④通过**。

### 4.4 【实测】"只量化 SWA + ring16"的时延（**主推档**）

用 `agents/KV8_gate/p5_e2e.py`（同一 harness、同一 die）**复跑**（原始数据
`logs/raw/034-p5-e2e-latency.json`，028 的原始文件已按位还原）：

| 量 | 028（原） | **034（复跑）** |
|---|---:|---:|
| SWA 层的增量 `swa_fused` | +13.93 µs/层 | **+10.61 µs/层** |
| cmp 读路径 `cmp_fused` | +31.64 µs/层 | +35.68 µs/层 |
| 整层 `full_fused` | +45.57 µs/层 | +46.29 µs/层 |
| 整步（40 SWA + 4 cmp） | +0.684 ms | **+0.567 ms** |

⇒ **【实测】"只量化 SWA"这一档的时延 = 40 × 10.61 µs = `+424 µs/step ≈ +1.34%`**
（028 的分层数字给的是 40 × 13.93 = **+557 µs ≈ +1.75%**；033 独立测得 **+12.05 µs/层 × 40 = +466 µs ≈ +1.55%**）。
**三组读数落在 +424 ~ +557 µs（+1.34% ~ +1.75%）之间** ⇒ 主代理问的"是否远低于 +1.75%"：
**是，本次配对复跑落在下沿 +1.34%**，但**跨次方差约 ±15%**，所以**建议按 +1.3% ~ +1.6% 记账**，
不要按单次最优值 +1.34% 写死。

---

## 5. ★ 给主代理的决策表（容量 × 时延）

> 标注约定：本表的**容量**列 = §2.3 引擎实测（六臂）× §2.2 页几何复算；**时延**列 = §4.3/§4.4 的配对实测
> 与 033 独立复跑；**prefill** 列 = 033 §3.3 实测。表内无【推断】项。

| 档 | 容量 | 时延 | prefill 代价 | 依赖 |
|---|---:|---|---|---|
| 现状（全 BF16 + F32 ring） | ×1.0000 | — | — | — |
| 只缩 ring FP16 | **×1.0000** | ~0 | ~0 | **白做，不要单独上** |
| **SWA int8 + ring FP16** | **×1.4655** | **+424~466 µs（+1.34~1.55%）** | **≈0（不碰压缩面）** | 本任务补丁（已通 `/health`） |
| KV8 双平面 + ring FP16 | **×1.9133** | +644 µs（+2.15%，033） | +0.00~0.58 ms（033 已解） | 本任务补丁 + 033 的 prefill wiring |

**⇒ 建议顺序：`① ring 缩 FP16（本任务补丁，它是所有容量档的前提）→ ② 按"要不要 long-KV 那 +30.5% 容量"
决定选哪一档`**。两条都成立才轮到 033 的决策树；"只缩 ring"不要单独上。

### 5.1 ★ INT8 ring 的独立复核（主代理点名要的）：容量零收益 + 时延零收益 ⇒ **支线关闭**

**结论：`ring INT8` 这条支线整条关闭。** 两条独立理由，各自都足以否掉它：

| 轴 | 证据 | 结论 |
|---|---|---|
| **容量** | §2.2 逐变体：`KV8 双平面 + ring FP16` 与 `+ ring INT8` **都是 `282880`**（ring ≤65536 之后 binding 变成 SWA-int8 页 66560） | **【实测·算式重算】零收益** |
| **时延/带宽** | 见下面的算术 | **【推断·算术，但被实测夹住】零收益** |

**带宽算术（为什么 INT8 连带宽都省不出来）**：decode 下每个请求每步在每個 ratio-2 层上
**写 1 行**（`min(used_len,32)` 行，decode 时 = 1）**读 ≤1 行**（残余组），一行 = 1024 元素：

```
FP16 ring：每层每请求 2 KB 读 + 2 KB 写 = 4 KB
INT8 ring：每层每请求 1 KB 读 + 1 KB 写 = 2 KB       ⇒ 每层每请求省 2 KB
B=8、3 个 ratio-2 层         ⇒ 每步省 8 × 3 × 2 KB = 48 KB
A3 连续拷贝上限 1161 GB/s（015 §5.1）⇒ 48 KB / 1161 GB/s = 0.041 µs/step
```

**而实测的噪声地板是 ±0.26 µs/step**（§4.3：F32→FP16 整个减半的 Δ ∈ [−0.11, +0.26] µs）。
⇒ INT8 要省的 **0.041 µs** 比"整个减半"能省的量**还小一个数量级、比噪声低 6 倍**
⇒ **测不出来，也不该去测**。

**另外两条"反向成本"**（都是 INT8 独有、FP16 没有的）：
1. **score 半边要参与 softmax**（`_pooled_blocked` 的逐维 `exp(score - max)/sum`）⇒
   反量化必须在 softmax **之前**、而且必须逐组/逐行带 scale ⇒ 多一个 scale 平面 + 多一遍 dequant；
2. **量化点落在 kernel 内部**（`_cache_update_kernel` 写完 ring 就完事，没有 host 侧机会）⇒
   要在 Triton 里再加"算 scale + 量化"两个阶段，等于把 028 花大力气压下去的
   "**算子个数 × 每核延迟**"（40 层图里每个设备算子 4–6 µs）又加回去。

**★ 复活条件（写清楚，免得后人重复挖）**：ring INT8 的容量收益 = 0 的**前提是 binding 别名 ≥ 65536**。
当前两个竞争别名是 **SWA-int8 页 66560** 与 **kv+index 41600**（KV8 下）。
⇒ **只有当 SWA 页被压到 65536 以下**（例如 SWA 也上 int4：`128×(512×0.5+scale)` ≈ 33792 B）
**且** kv+index 仍 < 65536 时，ring FP16 才会重新 binding，INT8 才重新有意义。
**在此之前：不开。**

### 5.2 ★ `logs/024` 那"三条缩法"的最终状态（A/B/C 逐条了结）

| # | 024 的提法 | 最终状态 | 理由 |
|---|---|---|---|
| **A** | `dtype: float32 → bfloat16` | ✅ **已实装，但形式改了：用 FP16 而不是 BF16** | 028 §P1 已证 FP16 的舍入比输出自己的 bf16 舍入**还小 8×**（本任务 §3 用真权重复核：FP16 只把地板抬高 +0.41%~+0.51%，而 BF16 是地板的 1.3×）⇒ **同样减半，数值白送** |
| **B** | `head_size: 2*width → width`（"那 2 里可能有一半可省"） | ⛔ **否决（两条独立理由）** | ① **算法上不能省**：`[kv \| score]` **两半都在用** —— `_pooled_blocked` 读 `+offs_h` 拿 kv、读 `+HEAD_DIM+offs_h` 拿 score 做并列 softmax，`_cache_update_kernel` 两半都写；`state_cache.shape[1:] != (32, 2*width)` 还是**硬断言**。② **容量上零收益**：砍掉一半 ⇒ ring 页 65536→32768，**远低于 66560 的 binding** |
| **C** | `block_size: 32 → 16`（环长减半） | ⛔ **否决（容量零收益 + 语义风险）** | ① **容量上零收益**：65536→32768，**仍在 66560 之下**，binding 不动。② `STATE_RING_ROWS=32` 是**三处断言的不变量**（`__post_init__` / `CompressorStateCache.__init__` / `reshape_cache`），且环长与"段内残余可回读的最大距离"耦合 ⇒ 改它属于**改算法**，而收益是 0 |

**一句话**：**024 的三条缩法里，只有 A 值得做（且要做成 FP16）；B/C 在"容量"这一维上是零收益，
在"算法"这一维上是要付代价的 ⇒ 两条都关掉，`block_size` 与 `head_size` 不要动。**

---

## 6. 没做的事 / 未确认

| # | 项 | 状态 |
|---|---|---|
| 1 | **8 卡真权重的权威 `GPU KV cache size`（c0）** | ⛔ **未做**：c0 全程被 `L3_8card`（`l3_l3-a-16d-16x128k`）占，c2 被 `p3-c2-coex2`/`kv8pf-p26` 占。本任务的引擎读数全部是 **tiny-dummy 单卡 TP1** |
| 2 | **端到端文本 sha256（改前 vs 改后）** | ⚠️ **未做**：L1_dummy 的压测客户端不落 `out_sha256`（只有 `requests_ok` + TTFT）。**替代证据**：① §3 的真权重 + 真 kernel 数值（比文本 sha256 更直接针对 ring dtype）；② 六条臂全部到 `/health` 且两轮压测 **4/4 ok、0 fail** |
| 3 | **`VLLM_V41_RING_FP16` 与 KV8 prefill wiring（033）的联动** | ⚠️ **未做**：本任务的影子包是 `KV8_fuse/shadow` + ring 补丁，**不含** 033 的 `VLLM_V41_KV8_PREFILL`。而 §2.3 的臂是 decode 为主的 2048-token prompt，prefill 路径**已走过**（`--enable-prefix-caching` + 2048 token 预填）但没做 033 那种 chunked-prefill 专项 |
| 4 | **DSpark / 推测解码下的 ring** | ⚠️ **未做**（见 §4.2 末行） |
| 5 | **`σ` 量程扫描** | ⚠️ **只跑了 σ=1**：`rel_L2` 是尺度无关的（round-to-nearest 的相对误差不随 σ 变），所以量程检查另有其法——见下表 |
| 6 | **FP16 量程/次正规数** | ⚠️ **只做了权重侧核算**：真权重 `max|wkv| = 0.148 ~ 0.168`、`max|wgate| = 0.672 ~ 1.523`（各层 RMS 0.024~0.036）⇒ 投影量级 `O(σ)`。FP16 上限 65504，要溢出需 σ ≈ 4e4（残差流不可能）；下溢只在 \|x\| < 6.1e-5（次正规）——那对 `rel_L2` 的影响是 1e-8 量级 ⇒ **不是风险**。**但真实残差流的 σ 没在真模型上量过 ⇒【未确认】** |

---

## 7. 取证与产物

```bash
# 不占卡：逐槽页构成 + 十一变体（容器内）
docker exec prbench-c1 python3 /work/agents/R_ringshrink/p1_pages.py

# 生成补丁（两条基线：镜像原版 / KV8_fuse 影子包）
python3 make_ring_fp16_patch.py --base /vllm-workspace/vllm-ascend \
  --out /work/agents/R_ringshrink/shadow --diff /work/agents/R_ringshrink/ring-fp16.patch
python3 make_ring_fp16_patch.py --base /work/agents/KV8_fuse/shadow \
  --out /work/agents/R_ringshrink/shadow-kv8 --diff /work/agents/R_ringshrink/ring-fp16-on-kv8.patch

# 真权重 + 真 kernel + fp64 golden（需 die）
bash tools/a3_chip.sh c1 --timeout 900 --name r-ring-p3 -- bash /work/agents/R_ringshrink/run_p3.sh

# 六条引擎臂（需 die）
bash tools/a3_chip.sh c1 --timeout 600 --name r-eng2 -- bash /work/agents/R_ringshrink/run_engine_arms.sh
```

| 类别 | 文件 |
|---|---|
| 不占卡探针 | `p1_pages.py`（逐槽页构成 / 11 变体）、`p0_extract_weights.py`（真权重抽取） |
| 设备探针（诊断链） | `p3_ring_numerics.py`（主：数值+时延）、`p5_probe_read.py`（尖门控反推）、`p6_diag_golden.py`（候选 golden）、**`p7_isolate.py`（ring 页灌已知值）** |
| 补丁 | `make_ring_fp16_patch.py` + `ring-fp16.patch`（原版基线）+ `ring-fp16-on-kv8.patch`（KV8 基线）+ 两份 `ring_fp16_manifest.json` |
| 臂驱动 | `run_p3.sh`、`run_engine_arms.sh`、`p4_show.py`、`p9_digest.py`、`p10_arms.py` |
| 原始数据 | `logs/raw/034-p1-pages.json`、`034-p3-numerics.json`、`034-p3-noop-stock.json` / `-shadow.json`、`034-p5-probe-read.json`、`034-p7-isolate.json`、`034-p5-e2e-latency.json`、`logs/raw/034-engine-arms/`（65 文件：六臂的 `server.log` / `kv_size.txt` / `client.json` / `driver.log` / `offload_cfg.txt`） |

* 真权重（31,460,352 B）不进仓：`p0_extract_weights.py` 在 A3-node1 宿主上从
  `~/models/DeepSeek-V4.1-Flash/model-0000{5,11,17}-of-00048.safetensors` 抽出
  `layers.{2,8,14}.attn.compressor.{wkv,wgate,norm}` 到 `/work/agents/R_ringshrink/weights/`（含 `manifest.json`）。
* ★ **一处需要声明的数据事故（已修复，含复核与防再发规则）**：合并 tar 时把本任务复跑的
  `028-p5-e2e.json` 解到了 `logs/raw/`，**覆盖了 028 的原文件**。处理三步：
  1. **还原**：用 `agents/KV8_gate/raw/028-p5-e2e.json` 拷回 `logs/raw/028-p5-e2e.json`；
  2. **★ 逐位复核（不是"看起来对"）**：还原后 `md5sum` = **`c7e96e1d55a94a6b415aa6fe960c32b5`**，
     与 KV8_gate 目录里的原件 **md5 完全一致**；且**逐值核对**了 028 正文引用的四格
     （`swa_fused = 13.93`、`cmp_fused = 31.64`、`full_fused = 45.57`、`step fused = 0.684`）
     —— 四格与 `logs/028` §0/§4 的正文数字**逐字相同**；
  3. **复跑结果另存**为 `logs/raw/034-p5-e2e-latency.json`（`swa_fused = 10.61` 等第五组读数），
     **不覆盖任何人的文件**。
  ★ **防再发规则（建议进 `AGENTS.md`）**：**跨机回来的 tar 一律先解到 `raw/<本次编号>-import/`
  这种新目录再按需 `mv`，绝不直接 `tar xzf ... -C logs/raw`**；若必须原地解，
  用 **`tar --keep-old-files`**（GNU tar 会给"已存在"警告而不是覆盖），
  并在解包后 `md5sum` 对一遍本次新增以外的文件。

---

## 8. ★★★ 推荐配置（可直接抄进 `publish/`）

```
推荐 = L5(per-group bpc) + L1(池行数) + SWA-quant + ring16 + [KV8 双平面 可选]
容量 = ×2.87（不含 KV8）/ ×3.75（含 KV8）  相对 L5-only 基线
代价 = decode +1.34%~1.6%（不含 KV8）/ +2.15%（含 KV8）；prefill ≤+0.015%（含 KV8）
前置 = ① kv8_scratch_plane 按 role 分键（033）
       ② ring 页步长传给 kernel：CACHE_PAGE（034，3 个 kernel 签名）
       ③ ring dtype 守卫 ×2 + reshape_cache 的 "fill" 断言 + spec 构造点（034）
       ④ VLLM_V41_RING_FP16 / VLLM_V41_KV8[_SWA] / VLLM_V41_KV8_PREFILL 四个门控
```

### 8.1 逐项复核（我这条线上能独立核的，**逐格核过**）

| 项 | 主代理给的数 | 我的复核 | 结论 |
|---|---|---|---|
| `ring16` 的不含 KV8 倍数 | `×1.4655`（只量化 SWA） | §2.3 引擎臂 `B0 = 33,295 / 22,719 = ×1.4655`；§2.2 页几何 `540928/369280 = ×1.4648` | ✅ **采纳**（两路径差 0.05%） |
| `ring16` 的含 KV8 倍数 | `×1.9133` | §2.3 引擎臂 `A1 = 43,469 / 22,719 = ×1.9133`；§2.2 页几何 `540928/282880 = ×1.9122` | ✅ **采纳** |
| `×2.87` | `1.96 × 1.4655` | `1.96 × 1.4655 = 2.8724` | ✅ **算术对**（`1.96` 来自 `logs/030`，**不是我量的**） |
| `×3.75` | `1.96 × 1.9133` | `1.96 × 1.9133 = 3.7501` | ✅ **算术对**（同上） |
| decode 代价（不含 KV8） | `+1.34%~1.6%` | §4.4：本任务配对复跑 `+10.61 µs/层 ×40 = +424 µs = +1.34%`；028 的分层数 `+13.93 ×40 = +557 µs = +1.75%`；033 独立 `+12.05 ×40 = +466 µs = +1.55%` | ⚠️ **区间采纳**：实测三组落在 **+424 ~ +557 µs（+1.34% ~ +1.75%）**，`+1.34%~1.6%` 覆盖了下沿，**但没有覆盖 028 的 +1.75% 上沿** |
| decode 代价（含 KV8） | `+2.15%` | §4.4 复跑 `full_fused` 整步 `+0.567 ms`（028 是 `+0.684 ms`）；033 独立测得 `+644 µs` | ⚠️ **区间采纳**：`+0.567 ~ +0.684 ms` = **+1.80% ~ +2.15%**，主代理给 `+2.15%` 是**最保守端** |
| prefill 代价（含 KV8） | `≤+0.015%` | 033 §3.3 的生产形状设备饱和口径（`+0.00~0.06 ms/step`） | ✅ **采信 033**（本任务没测 prefill 专项） |
| 前置 ①②③④ | — | ①（`kv8_scratch_plane` 按 role 分键）是 **033 独占的发现**，**本任务没有碰那段代码、也没复核**（只在 §4.2 核过它与 ring dtype 无交互）；②③ 是本任务的 21 组替换（② 对应 §4.1 的 `CACHE_PAGE`）；④ 是我加的门控 | ⚠️ **①②④ 采信/转述，③ 是我自己的** |

### 8.2 ★ 我**要改**的一处（不是数字，是用法）

**把 `+1.34%~1.6%` 改成 `+1.3%~1.8%`（不含 KV8）/ `+1.8%~2.2%`（含 KV8），并在 `publish/` 里注明"单次配对读数跨次方差约 ±15%"。**

理由：三组**同一 harness、不同运行**的读数分别是 +1.34% / +1.55% / +1.75%（不含 KV8）
和 +1.80% / +2.15%（含 KV8）。主代理给的区间**只从最优点起算**，
而 `+1.75%` 这一格是 **028 自己测的、被 033 独立复现过的**——
按"不许用相邻数字顶替缺的那格"的规矩，**不能把它排除在区间外**。
⇒ 写 `+1.3%~1.8%` 才是"三组读数都落在里面"的区间；
含 KV8 同理（`+0.567 / +0.684 / +0.644 ms` ⇒ `+1.8%~+2.2%`）。

**如果 `publish/` 需要一个单点值**：取**中位**——不含 KV8 `+1.55%`（= 033 那一格）、含 KV8 `+2.15%`（028/033 一致）。

### 8.3 ★ 一处必须写进 `publish/` 的**依赖顺序**（否则容量拿不到）

```
ring16 必须先于（或与）SWA-quant / KV8 同时生效 —— 单独上 ring16 = ×1.0000（§2.3 臂 A2 vs A3 实测）
但 ring16 也**不能单独上**：它只是"解除 SWA/KV8 页的上限"，自己一格都不省
⇒ 顺序：① per-group bpc + 池行数（L5+L1，与 ring 正交，随便先后）
        ② ring16（本任务补丁）
        ③ SWA-int8（→ ×1.4655）或 KV8 双平面（→ ×1.9133）
```
