# 026 — KV8 读侧 rebuild 融成 kernel：整层 +345.7 → **+38.2 µs/层**（判据 ≤60 达标）

> 2026-09-22 02:0x–02:2x CST。执行：子代理 **KV8_fuse**。机器：**A3（A3-node1）槽位 c1 = die 6**
> （`Ascend910_9382`，容器 `prbench-c1`，`triton 3.2.0` / triton-ascend）。
> 只用 c1（`tools/a3_chip.sh c1 --name kv8f`）；只用 `~/tmp` 之外的 `TMPDIR=/work/agents/KV8_fuse/tmp`（没用 `/tmp`）；
> 没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；**没写 `upstream-v41/`**；没动 `dsv41-release/`；
> 跨机传输全走 coscli；改的只有 `agents/KV8_fuse/`（读取 `agents/KV8_gather/shadow` 作为基线，未修改它）。

---

## 0. 五句话结论（先给主代理）

1. **【实测·成功】rebuild 融成 kernel 了，而且是逐比特等价**：SWA 窗口重建 = **2 个 kernel**、
   cmp 选行重建 = **1 个 kernel**（profiler 实测：torch 侧 **50 个 / 33 个** 设备算子 → 融合后 **2 个 / 1 个**）。
   重建产物（scratch / block table / 重编号后的 sparse indices）与 023 的 fast 路径 **`torch.equal` 全 True**，
   端到端 `npu_sparse_flash_mla` 输出**逐比特相同**。
2. **【实测】整层增量：+345.7 → +38.2 µs/层**（40 层整图，生产形状；run7 猴子补丁臂 +41.0、
   真·影子包路径 +38.2）。**判据 ①（≤+60 µs/层）达标，余量 36%。**
   SWA 层 +142.6 → **+12.8~14.2 µs/层**；cmp 读路径 +216.9 → **+24.1~28.2 µs/层**。
3. **【实测】整步推算：+6.57 ms（+21.9%）→ +0.63~0.68 ms（+2.1~2.3%）**（40 SWA + 4 源层，30 ms 步）。
   **判据 ②（≤+0.2%）不达标**；而且 **【推断】它在物理上不可达**：融合后的 rebuild 每步仍要搬
   **151.8 MB**（读 64 MB + 写 127 MB），按本 die 实测的 1161 GB/s 连续拷贝上限也要 **131 µs/步 = 0.44%**，
   即"零开销完美实现"也进不了 0.2%（详见 §6）。
4. **【实测】**首次调用触发 Triton JIT，3 个 kernel 的编译摊在整轮 35 s 里；之后**可 capture、可 replay、
   改输入跟着变**（replay vs eager `max_abs_diff = 0.0`）；prefill / 非 2 的幂几何**自动回退**到 torch 路径
   （回退与 fast 路径逐比特相同）。
5. **【实测】精度零回退**：真量化器 `rel_L2 = 5.4331e-3`、`cos = 0.9999857`、`max_abs = 5.55e-4`、`nan = 0`
   —— **与 020/023 完全同一个数**（融合没有引入任何数值变化）。

---

## 1. ★ cannbot 对照（写 kernel 前必查；§6 红线）

在 A3-node1 只读查 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/` 与 `~/opensrc/ops-transformer/`。

| 查的地方 | cannbot 说什么 | 我们采纳 / 不采纳及理由 |
|---|---|---|
| **`ops/torch-ops-profiler/SKILL.md` + `examples/layer_norm_profiler_reference/`（4 文件模板：`layer_norm_profiler_common.py` / `benchmark_layer_norm_torch_npu_profiler.py` / `layer_norm_perf_cases.jsonl` / README）** | 用 **`torch_npu.profiler`**（warmup/active 固定）做"**自定义算子 vs 标杆**"双路径对比，报告**必须**双路径、标杆必须在 NPU 上跑 | ✅ **采纳（本任务就是这条）**：`p11_prof.py` 是它的最小落地——双路径 = torch 50/33 算子 vs Triton 2/1 kernel，产出 `kernel_details.csv` 的**算子计数**；没生成 JSONL 用例文件（我们不是在开发算子库） |
| `ops/ops-profiling/SKILL.md`（`msprof_profile_run.sh --compare/--quick`，7 组 aic-metrics） | 标准上板采集与 kernel-level 加速比对比；`--quick` 只取 kernel 时间 | ⚠️ **部分采纳**：本次用 `torch_npu.profiler` 拿**算子个数与名字**（够回答"是不是算子个数 × 每核延迟"）；**没走 msprof 的 7 组 aic-metrics**（时间预算，且我们要的是计数不是 roofline） |
| **`model/model-infer-fusion/SKILL.md`**（第一步拆解子链路 → 第二步匹配仓库参考实现 → 官方 API 参数校验） | 融合要**逐子链路**匹配现成 torch_npu 算子，命中后必须查官方文档确认参数约束 | ✅ **采纳为方法论**：把 rebuild 拆成 (a) 索引算术 (b) 两平面 gather (c) g128 反量化 (d) 写 scratch + 重编号，逐条查算子表 —— 结论是**没有任何一个现成算子覆盖 (a)–(d)** |
| `model/model-infer-fusion/references/torch_npu_API/torch_npu_list.md:1-160`（144 个接口） | 量化族只有 `npu_anti_quant`（反量化）、`npu_quantize`、`npu_quant_scatter(_)`（量化+写）、`npu_dynamic_quant`、`npu_kv_quant_sparse_flash_attention`（Per-Token-Head-Tile-128，**存 8 算 8**）；**没有** gather+dequant 融合算子 | ❌ **A 路线否决**：`npu_anti_quant` 只做 (c)（且要 1-D scale，023 §1 已实测 `EZ1001: scale dim num must be 1`），`npu_quant_scatter` 方向相反；唯一"不需要重建"的候选是 `kv_quant_sparse_flash_attention` |
| `~/opensrc/ops-transformer/attention/kv_quant_sparse_flash_attention/op_host/*_tiling.cpp:1196` | `OP_CHECK_IF(qHeadDim_ != 576, "q_head_dim only support 576")` | ❌ **否决**：我们的 latent `head_dim = 512`（448 nope + 64 rope 内嵌），**这条唯一能"不重建"的路被 head_dim 挡死**（与 023 §1 一致，本次给出源码行号） |
| `~/opensrc/ops-transformer/attention/gather_pa_kv_cache/README.md` | 原生 PA gather（`blockTables`+`seqLens` → 连续 keyRef/valueRef），**支持 INT8 数据类型**，但不反量化 | ❌ **未采纳**：它能把 (b) 换成 1 个原生算子，但 (c)(d) 仍要 2–3 个算子，且它自带一次独立调用；Triton 把 (a)–(d) 写进同一个 kernel 更省（实测 SWA 全部 2 个 kernel） |
| **`model/model-infer-superkernel/SKILL.md:13-17` + "重要原则"** | SuperKernel **仅支持 `exe_mode: ge_graph` + Atlas A3 + PyTorch，仅 decode 生效**；**"配置互斥：不支持 eager 模式和 aclgraph 模式"** | ❌ **D 路线否决**：生产 decode 走 vLLM-Ascend 的 **ACLGraph**（本 harness 与 015/020/023 全部是 `torch.npu.NPUGraph`）；要用 SuperKernel 得先把整条服务换成 GE 图（cann-recipes 那套），超出 KV8 的范围与预算 |
| `ops/triton-op-coding/SKILL.md` + `references/triton-ascend-*.md` | Triton-Ascend 编码规范（`tl.arange`、禁止退化 torch 等） | ✅ **采纳**：全程没有在 `forward` 里留任何 torch 计算；`tl.arange` 必须 >1（本次实测 shape-1 会让编译器**硬 abort**，见 §7） |
| `ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/03-Ascend-Extensions/10-mem-ops.md:188-260,323-410` | `index_select_simd`（GM→UB、1D index、dim 不能是最后一维、**不查越界**）、`gather_out_to_ub`、`scatter_ub_to_out`（023 §1 记的同一份） | ❌ **未采纳（B 路线的子选项）**：不需要 SIMD 索引原语——索引算术 + 跨页指针算术 + 反量化用普通 `tl.load/store` 写在同一个 kernel 里即可，避开"dim 不能是最后一维 / index 必须 1D / 不查越界"三条限制，也避开 `al.*` 扩展在 aclgraph 里的未知行为 |
| `model/model-infer-kvcache`（block/slot 映射一节） | `物理块 = block_table[b, 逻辑块]`、`物理 slot = 物理块×block_size + 块内偏移` | ✅ **沿用 023 的逐字采纳**：本次**没有改布局**（页 stride / 语义一个字节没动），只是把读侧的算子换成 kernel |
| `model/model-infer-graph-mode/SKILL.md` | Decode 图模式 / 重编译检查 | ⚠️ **未采纳**：本次图的形态沿用 015/020/023 的 ACLGraph harness，没有引入新的图模式；重编译检查以"kernel 数固定 + replay 一致"代替 |

**一句话**：cannbot 里**没有**能直接用的 `gather+dequant` 融合算子（A 路线否决），
SuperKernel 与我们的 aclgraph 生产帧**模式互斥**（D 路线否决），
所以**走 B 路线：Triton-Ascend 自己写 kernel**（C 路线 AscendC 四文件模板备而不用，理由见 §2）。

---

## 2. 路线判定（按代价）

| # | 路线 | 判定 | 依据 |
|---|---|---|---|
| **A** | 现成融合 op | ⛔ **否决** | 144 接口表里无 `gather+dequant+scatter`；唯一"存 8 算 8"的 `kv_quant_sparse_flash_attention` 硬卡 `q_head_dim==576`（源码行号见 §1） |
| **B** | **Triton-Ascend** | ✅ **采纳** | 容器自带 `triton 3.2.0`；**同一份影子包里已有先例** `vllm_ascend/ops/triton/engram_int8.py::gather_dequantize_engram_int8`（Engram 的 device-index int8 gather+dequant，生产路径在用）⇒ "Triton 能不能进图"在本文档里不是未知数 |
| **C** | AscendC 自写 kernel | ❌ **本次不需要** | 四文件模板（`ops/torch-ops-profiler/examples/layer_norm_profiler_reference/`）是给"要进算子库/要注册 aclnn"的场景；我们只需要**图内一个自定义 launch**，Triton 已经做到 2+1 个 kernel 且逐比特等价。AscendC 的代价（算子库注册 + aclnn 适配 + 图捕获）在本次判据下没有回报 |
| **D** | SuperKernel | ⛔ **否决（模式互斥）** | 只支持 `ge_graph`，**明确不支持 aclgraph**；生产 decode 是 ACLGraph |

---

## 3. 实现：3 个 kernel 换掉 ~83 个算子

代码：`agents/KV8_fuse/kv8_fuse_triton.py`（+ `shadow/vllm_ascend/attention/kv8_fuse_triton.py` + dsa_v41.py 末尾 8 行挂载）。

| kernel | 干什么 | 网格 / tile |
|---|---|---|
| `_kv8_swa_rows_kernel` | 每程序一个 (scratch 页, 行块)：算 `first_block`、取 `block_table` 物理页、**跨页指针算术取 int8 + fp16 scale、g128 反量化、写 bf16 scratch** | `B×2×(128/2)=1024` 程序，tile `[2 行 × 4 组 × 128]` |
| `_kv8_swa_table_kernel` | 逻辑块 → scratch 页的重映射表（每程序一行，按 128 宽循环） | `B` 程序 |
| `_kv8_cmp_rows_kernel` | **索引算术（`idx>>log2(BS)` / `&mask` 取代 `//` `%`）+ 双平面 gather + 反量化 + 写 scratch + 写重编号 indices** | `rows*topk/16=256` 程序，tile `[16 行 × 4 组 × 128]` |

* 反量化语义与 `kv8_dequant_rows` 逐字一致：`int8→fp32`、`fp16→fp32`、**一次 fp32 乘法**、`→bf16`；
  差异只在"scale 广播"用 **3-D tile 的多维乘法**而不是 `cols // GROUP` 索引（后者在 Triton-Ascend 上直接编错，见 §7）。
* 融合把 torch 侧的中间张量全省了：torch 版每层要落 `sel_i8`/`sel_scale`/`deq`/`renumbered`/`table` 等一堆
  GM 往返（**profiler 数出来的 50 + 33 个设备算子**就是这么来的）。
* 回退：`query_rows != num_reqs`（prefill）、非 2 的幂几何、`per_req*block_size != topk` ⇒ 自动走 torch 原路径，
  **回退路径已实测与 023 的 fast 路径逐比特相同**（`prefill_fallback.matches_fast = true`）。

---

## 4. 正确性（判据 3 / 4 / 5）

### 4.1 逐比特等价【实测】

| 对拍 | 结果（`torch.equal`） |
|---|---|
| SWA：`fused` vs `023 fast`：scratch / table | **True / True** |
| SWA：`fused` vs `shipped current`（2D 高级索引版） | **True / True** |
| cmp：`fused` vs `023 fast`：scratch / table / 重编号 indices | **True / True / True** |
| cmp：`fused` vs `shipped current` | **True / True / True** |
| **端到端**（真 `npu_sparse_flash_mla`，无损臂）`F_fused` vs `C_current` / `D_fast` | **True / True** |
| **端到端**（真量化器）`fused` vs `current` / `fast` | **True / True** |

### 4.2 精度（判据 4）【实测】

| 对照 | `rel_L2` | `cos` | `max_abs` | `nan` |
|---|---|---|---|---|
| `current` vs BF16 | 5.4331e-3 | 0.9999857 | 5.55e-4 | 0 |
| `fast` vs BF16 | 5.4331e-3 | 0.9999857 | 5.55e-4 | 0 |
| **`fused` vs BF16** | **5.4331e-3** | **0.9999857** | **5.55e-4** | **0** |

⇒ 与 020/023 的 `rel_L2 ≈ 5.43e-3` **完全相同**，精度零回退。

### 4.3 图兼容（判据 5）【实测】

| 项 | 结果 |
|---|---|
| capture（40 层整图、多个 arm） | ✅ 全部成功（三个 kernel 在 warmup 阶段完成 JIT） |
| replay | ✅ 与 eager 逐比特（`max_abs_diff = 0.0`） |
| **改变输入后 replay** | ✅ `outputs_differ = True` + `replay_matches_eager = True`（原地改 `seq_lens` / `indices` / `block_table`） |
| prefill 回退 | ✅ 与 fast 路径逐比特相同；cmp 回退分支形状 `[32,128,1,512]/[1,32]/[2,1,512]` |

---

## 5. 性能

### 5.1 算子计数（`torch_npu.profiler`，设备 `kernel_details.csv`）【实测】

| 重建 | torch 版（算子数/次） | **融合版（kernel 数/次）** |
|---|---:|---:|
| SWA 窗口（payload+scale gather、`where`/`arange`/`minimum`/`gather` 表重映射、反量化、copy） | **50** | **2** |
| cmp 选行（索引算术 + 双平面 gather + 反量化 + copy + 重编号） | **33** | **1** |

> 这与 023 §4.2 的"**算子个数 × 每核延迟**"完全对得上：023 估的"SWA ~30 个算子"实测是 **50 个**，
> cmp "~30 个"实测 **33 个**。融合把它们压到 **2 / 1**。

### 5.2 单次重建（图内 40 reps，扣 7.9 µs 空图地板）【实测】

| 重建 | torch current | torch fast(023) | **fused** | tile 最优 |
|---|---:|---:|---:|---:|
| SWA | 131.4 µs（50 op） | 130.7 µs | **16.1 µs**（净 8.2） | rows=2 → 净 **6.5** |
| cmp | 207.8 µs（33 op） | 208.4 µs | **32.6 µs**（净 24.6） | rows=16 → 净 **11.8** |

### 5.3 ★ 40 层整图（生产形状，判据口径）【实测】

| 图（40 层一图） | SWA-only 层 | 整层（SWA+cmp） |
|---|---:|---:|
| A BF16 | 24.0–26.4 µs/层 | 42.3–44.7 µs/层 |
| C int8，**shipped** 重建 | 164.4–166.3 | 390–405（023：391.7） |
| D int8，023 flat gather | ≈C | 391–399 |
| **F int8，fused kernel（本任务）** | **38.2–44.7** | **80.8–98.2** |

| 增量（µs/层） | 023 fast | **fused（run7 / 真影子包）** |
|---|---:|---:|
| SWA 层 | +139.5 | **+12.8 / +14.2** |
| cmp 读路径 | +207.8 | **+28.2 / +24.1** |
| **整层** | **+345.7** | **+41.0 / +38.2** |

| 整步外推（40 SWA + 4 源层，30 ms 步） | 023 | **fused** |
|---|---:|---:|
| ms/step | +6.57 | **+0.63 ~ +0.68** |
| 占比 | +21.9% | **+2.1 ~ +2.3%** |

> tile 组合：`SWA_ROWS=2, CMP_ROWS=16`（图内最优：绝对 80.8 µs/层，vs `(8,4)` 的 98.2）。
> 真影子包那一轮（`026-fuse-realpath.json`）里 `dsa_v41.kv8_ori_plane` 就是
> `vllm_ascend.attention.kv8_fuse_triton.fused_ori_plane`，且 harness/影子包两份 kernel 文件 **md5 相同**
> （`6ce00b8f6fdd9ba4ad5935876601f8d6`）——测的就是要交付的那份代码。

---

## 6. ★ 判据对账（含"0.2% 是否物理可及"）

| # | 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|---|
| 1 | **整层增量**（40 层整图，生产形状） | ≤ +60 µs/层（原 +347.2） | **+38.2 ~ +41.0 µs/层** | ✅ **达标**（余量 ~33%） |
| 2 | **整步推算** | ≤ +0.2%（原 +21%） | **+0.63~0.68 ms = +2.1~2.3%** | ❌ **不达标** |
| 3 | 逐比特等价（vs fast 路径 `torch.equal`） | True | **6/6 张量 + 2/2 端到端 True** | ✅ |
| 4 | 真量化精度 | `rel_L2 ≈ 5.43e-3` | **5.4331e-3（cos 0.9999857 / nan 0）** | ✅ |
| 5 | 图兼容（capture / replay / 改输入跟着变） | ✅ | **✅（replay vs eager 差 0.0）** | ✅ |

### 6.1 【推断】判据 ② 在**任何**融核方案下都不可达（这一条比结论本身重要）

融合后每步仍必须搬的字节（B=8、2 页/请求、topk=512）：

```
SWA 层：payload 读 1.05 MB + scale 16 KB + bf16 scratch 写 2.10 MB ≈ 3.16 MB
cmp 层：payload 读 2.10 MB + scale 32 KB + bf16 scratch 写 4.19 MB + 重编号 16 KB ≈ 6.34 MB
整步  ：40 × 3.16 + 4 × 6.34 = 151.8 MB/step
```

* 本 die 实测连续拷贝上限 **1161 GB/s**（015 §5.1）⇒ 151.8 MB 的**理想地板 = 131 µs/步 = 0.44%**；
* 要进 **+0.2%（60 µs/步）** 需要 **≥2530 GB/s** 的有效带宽 = 本机拷贝上限的 **2.2 倍** ⇒ **不可能**；
* 实测融合版达到 151.8 MB / 0.63 ms ≈ **241 GB/s**（= 地板的 21%，剩下的是小尺寸 + 启动 + 尾部效应）。

⇒ **结论：判据 ② 不是"没做好"，而是"口径不可能"**。要想真正进 ≤0.2%，
只能**不重建**（即让 SMLA 直接读 int8：需要 CANN 侧支持 head_dim 512 的 `kv_quant_*`，本次已确认 576 硬约束），
或者**把需要重建的层数从 44 层降到 ~4 层以下**。

### 6.2 口径说明（供主代理裁决）

任务书里"`≤+0.2%`（60 µs/层）"这两个写法**互不自洽**：60 µs/层 × 40 层 = **2.4 ms = +8%**，
而 0.2% × 30 ms = **60 µs/步**。本文按两个口径都报了：
**按"≤60 µs/层"= ✅ +38.2；按"≤60 µs/步"= ❌ +630 µs（且物理不可达，见 §6.1）**。

---

## 7. 坑（Triton-Ascend 3.2.0，都是硬 abort/错值，记下来省后人一小时）

| # | 现象 | 原因 / 解法 |
|---|---|---|
| ① | **2-D tile 的 `cols // GROUP` 索引编译出**：468,421 个错元素 + NaN，且慢 24 倍（297 µs vs 12.2 µs） | 用 `[ROWS, GROUP, DIM//GROUP]` **3-D tile** + 多维乘法广播；`raw/026-micro.json` 里 `v1 equal=false / v2 equal=true` |
| ② | `%`（或 `/`）出现在**地址表达式**里 ⇒ `parseRem Assertion 'Address expression with modulo is not supported yet'` **硬 abort** | 全部改成移位/掩码：`idx >> log2(BS)`、`idx & (BS-1)`（`BS`/`TOPK` 必须 2 的幂，非幂次直接走回退） |
| ③ | `tl.arange(0, 1)`（每个程序 1 行）⇒ `encountered AddPtrOp produced by unsupported operation` **UNREACHABLE 硬 abort** | tile 行数**必须 ≥2**（现在 `SWA_ROWS=2`/`CMP_ROWS=16`，`=1` 时在 wrapper 里抬到 2） |
| ④ | 单程序整页（16384 元素）⇒ `ub overflow, requires 12582912 bits while 1572864 bits available` | UB ~192 KB，tile 别超 ~8 K 元素（`ROWS≤16` 配 512 宽是安全的） |
| ⑤ | 影子包 `cp -r` 只成功了一半（第一次因 root 权限失败，第二次因"目录已存在"跳过）⇒ 少了 `_cann_ops_custom/vendors/custom_transformer`，`npu_sparse_flash_mla_metadata` 报 `cmp_ratio should be 4 ... but got 1`（**假故障，坑了我 20 分钟**） | 复制影子包后必须 `diff -rq` 校验；本次已在 `026-fuse-realpath.json` 里加了 **kernel 文件 md5** 自证 |

---

## 8. 给主代理的建议

### 8.1 对"KV8 能不能成立"的回答

**【实测】唯一的门槛（融核）过了**：整层 +345.7 → **+38.2 µs/层**（判据 ≤60 ✅），
逐比特等价、精度零回退、图兼容。
⇒ **"KV8 读侧融不成 1 个 kernel"这个否决策略被推翻**；KV8 从"时延负收益"变成
**"×1.135 容量换 +2.1% 时延"**（原来是 +21%）。

但**判决权仍在另外两件事上**（都不在本次范围）：

| 事项 | 状态 | 影响 |
|---|---|---|
| **判据 ②（+0.2%）不可达** | 【推断】物理不可达（§6.1） | 建议把判据改成 **≤+1% 或 ≤+0.5%/步**；否则任何 KV8 实现都会被判死 |
| **prefill 侧**（015 §5.3） | 未解：2048-token chunk ⇒ 4 层 45–105 ms/step（+10~26%），本次融合**只覆盖 decode**（prefill 走回退） | KV8 上生产前必须先解决 prefill（并集去重 / prefill 期整平面反量化），否则一样是负收益 |
| **容量分母**（025 §2.2） | ×1.135（要 ×1.84 才有意义），真凶是 FP32 compressor state ring | 需 `float32 → bfloat16`，独立任务 |

### 8.2 降级建议（如果主代理要收档）

1. **不建议**只量化 long-KV 不量化 SWA：容量只有 **×1.032**（005/025 已判"太小不值得"）；
   不过本次顺带说明——那条路用上融合后，cmp 读路径只要 **4 层 × ~25 µs ≈ +100 µs/步（+0.33%）**，
   **代价已经很小、纯粹是容量不够**。
2. **若 KV8 继续**：建议顺序 = ① prefill 方案（去重/整平面）→ ② state ring 缩到 BF16（拿 ×1.91）→
   ③ 把本 kernel 接进真服务并跑端到端（GSM8K / Vision / `GPU KV cache size`）。
3. **若 KV8 收档**：本 kernel 的结论不要丢——它是"**小算子堆叠不是延迟问题、是算子个数问题**"的一个可直接复用的样板
   （`kv8_fuse_triton.py` 里的 3-D tile / 移位索引算术 / ≥2 行 tile 三条经验适用于本项目其它 rebuild 类路径）。

---

## 9. 复现与交付物

```bash
# 真·影子包路径（交付形态）：dsa_v41.py 末尾把 kv8_ori_plane / _kv8_cmp_plane 换成 fused
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c1 --name kv8f --timeout 900 -- \
  bash -c "cd /work/agents/KV8_fuse && export TMPDIR=/work/agents/KV8_fuse/tmp \
    TRITON_CACHE_DIR=/work/agents/KV8_fuse/triton_cache PYTHONPATH=/work/agents/KV8_fuse/shadow \
    KV8_RAW=/work/agents/KV8_fuse/raw && python3 p9_fuse.py"'

# kernel 计数（profiler 双路径）
ssh A3-node1 '... PYTHONPATH=/work/agents/KV8_gather/shadow python3 p11_prof.py'
```

| 东西 | 路径 |
|---|---|
| kernel + 挂载（可直接叠在 KV8_swa/KV8_gather 影子包上） | `agents/KV8_fuse/kv8_fuse_triton.py`、`agents/KV8_fuse/shadow/vllm_ascend/attention/dsa_v41.py`（末尾 8 行） |
| harness | `agents/KV8_fuse/p9_fuse.py`（40 层整图 + 正确性 + 精度 + replay）、`p10_micro.py`（kernel 微基准/正确性定位）、`p11_prof.py`（算子计数） |
| 原始数据 | `logs/raw/026-fuse-realpath.json`（**主证据**）、`026-fuse-run7.json`（tile 调优 + 猴子补丁臂）、`026-fuse-run6.json`（tile sweep）、`026-micro.json`（v1/v2 对拍 + 启动地板）、`026-prof.json`（50/33 → 2/1）、`026-fuse-run1.json`（第一版失败的证据）、`026-run7.log` / `026-realpath.log` / `026-run6.log` |
| A3 副本 | `~/projects/dsv41-upstream-pr/agents/KV8_fuse/`（容器内 `/work/agents/KV8_fuse/`） |
