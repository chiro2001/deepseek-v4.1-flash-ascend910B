# 028 — KV8 裁决：融 kernel 的可行性（P0）+ state ring 的精度与容量（P1）

> 2026-09-22 02:0x–02:3x CST。执行：子代理 **KV8_gate**。机器：**A3（A3-node1）槽位 c1 = die 6**
> （`Ascend910_9382`，容器 `prbench-c1`）。全程只用 c1；没碰 `dsv41-a3` / `mooncake-master` /
> Phy-ID 8–15；没用 `/tmp`（`TMPDIR=/work/agents/KV8_gate/tmp`）；没写 `upstream-v41/`；
> 跨机传输全走 coscli。代码 → `a2/agents/KV8_gate/`；原始数据 → `a2/logs/raw/028-*`。

---

## 0. 五句话结论（先给主代理）

1. **【实测】融 kernel 做成了，而且是逐比特精确的。** 两个 Triton-Ascend kernel
   （`kv8_swa_fused2` / `kv8_cmp_fused2`）各自把「索引算术 → 页/行 gather → INT8 反量化 →
   BF16 scratch → scratch block table」整条链压进 **1 个 launch**。真量化器端到端
   `fused_vs_current` **`torch.equal` = True**，`rel_L2 = 5.4331e-3`（与 023 **逐位相同**，零回退）。
2. **【实测】整层增量：+357.86 → ≤ +45.57 µs/层**（40 层整图、生产形状）。
   拆开：**SWA 层 +145.42 → +13.93 µs/层**，**cmp 读路径 +212.44 → +31.64 µs/层**。
   **⇒ 任务书判据 1（整层 ≤+60 µs）达标 ✅。**
3. **【实测】整步推算：+6.667 ms → +0.684 ms（+21% → +2.15%）**。
   **⇒ 判据 2（整步 ≤+0.2% ≈ 63 µs）不达标 ❌，差 10.8×。**
4. **★【实测+推断】但 +0.2% 这个判据靠 rebuild 路线是达不到的，而且不是"还没调好"：**
   rebuild 的定义就是「把 INT8 物化成 BF16」，每层 SWA 必读 1.03 MB + 写 2.10 MB、
   cmp 必读 2.13 MB + 写 4.26 MB；实测最好 = **684 µs/step**，而**把 SWA 整条抹成 0，
   仅 4 层 cmp 也还要 127 µs = +0.4%**（判据的 2×）。⇒ 唯一能达标的形态是
   **让 attention 算子自己读 int8**（`npu_kv_quant_sparse_flash_attention` 那种），
   **那是"要新算子"，不是"融 kernel"**。
5. **【实测·算式重算】P1：容量分母解开了，而且是"白送"的。**
   ring 存的**不是累加器**，是 `[kv(512) | score(512)]` 的**原始 FP32 投影**，
   而池化输出**本身就是 BF16**。⇒ 缩到 **FP16**（不是 BF16）时，池化结果
   `rel_L2 = 7.6e-4`，**低于现有的 bf16 输出地板 1.66e-3**（"精度免费"）。
   容量：**`pool_bytes_per_block` 476416 → 282880，即 ×1.135 → ×1.912**
   （判据 ×1.84 **达标 ✅**；三个已知锚点 540928/476416/524288 全部复现）。

---

## 1. ★ cannbot 对照（AGENTS.md §6：必做，先做，不占卡）

只读查 `~/projects/dsv41/src/cannbot/vendor/cannbot-skills/`。**注意**：
`model-infer-fusion/scripts/torch_npu_query.py` 需要 torch_npu 环境，在**宿主机上跑会静默降级**
到 `_FALLBACK_DOCS`（本次踩到）；要查真实签名必须**在容器里**跑。

| 查的地方 | cannbot 说什么 | 采纳 / 没采纳 | 证据 |
|---|---|---|---|
| ★★ `model/model-infer-superkernel/SKILL.md:13-17`、`重要原则`、`配置检查清单` | **"SuperKernel 仅支持 ge_graph 模式、Atlas A3 硬件、PyTorch 框架，且仅在 decode 阶段生效"**；**"配置互斥：不支持 `eager` 模式和 `aclgraph` 模式"**；检查清单要求 `exe_mode != eager/aclgraph` | ⛔ **否决路线 A** | KV8 的**全部**性能实测（020/023/本次）都在 **ACLGraph**（`torch.npu.NPUGraph`，40 层一图）口径下。SuperKernel 要求把整条链换成 `ge_graph`（cann-recipes-infer 的 `exe_mode`），且它在 vLLM-Ascend 里**没有对应 hook**。**不是配置能解决的** |
| `model/model-infer-fusion/SKILL.md` 全节 + `references/torch_npu_API/torch_npu_list.md:1-160` | 144 个接口目录；`npu_gather_sparse_index`、`npu_anti_quant`、`npu_quant_scatter(_)`、`npu_kv_quant_sparse_flash_attention` 在列 | **逐个核对（见下）** | `raw/028-p2-docs.json` |
| `ops/torch-ops-profiler/examples/layer_norm_profiler_reference/` | **四文件模板确实存在**（`README.md` / `layer_norm_profiler_common.py` / `LAYER_NORM_PROFILER_PERF_GUIDE.md` / `layer_norm_perf_cases.jsonl` / `benchmark_*`） | **路线 D 备用，本次没走到**（Triton 先成了） | `find` 命中 |
| `ops/triton-latency-optimizer/references/docs_triton_IR/docs_triton_ascend/03-Ascend-Extensions/10-mem-ops.md` | Triton-Ascend 内建 `index_select_simd`（GM→UB 零拷贝，1D index）、`gather_out_to_ub`、`index_put`、`scatter_ub_to_out` | ✅ **实机确认符号存在**（`triton.language.extra.cann.*`），但**最终没用**：直接算地址的 `tl.load` 比走 mem_ops 更省事且已达标 | `raw/028-p1-probe2.json` |
| `model/model-infer-graph-mode/SKILL.md` | 图模式仅 decode、Prefill 保持 eager；**固定 tensor 图外预创建**；`kv_len` 每步变要防重编译 | ✅ **逐条采纳** | scratch plane / scratch block table / 重编号索引张量全部**图外预分配 + 复用**，图内零分配 ⇒ 40 层捕获、改输入原地 replay 正常 |
| `model/model-infer-quantization/SKILL.md:424-451` | 量化改造必须证明"真实生效"（对象级/权重级/算子级 probe + 等价性自检） | ✅ **采纳其口径**：本任务的"生效证明"= 真量化器端到端 `torch.equal` + `rel_L2` 复现 023 的 5.4331e-3（**不是**只看代码 diff） | §3.2 |

### 1.1 ★ 逐个核对 cannbot 列出的"现成融合算子"（路线 B）【实测：在本机 torch_npu 上查签名】

`raw/028-p2-docs.json`（`torch_npu 2.10.0.post4`，容器内 `_op_plugin_docs.py` docstring）：

| 算子 | 实际签名/约束 | 能不能替掉 rebuild |
|---|---|---|
| `npu_gather_sparse_index(input, index)` | `input` 维度 + `index` 维度 − 1 ≤ 8；**int8 支持**；`out.shape = index.dim + input.dim − 1` | ⛔ **不能**：它只做 gather，**不含反量化**。rebuild 的 30 个算子大部分在反量化链 + 索引算术上 |
| `npu_anti_quant(x, scale, offset?, dst_dtype)` | 反量化；023 实测 2D scale 报 `EZ1001 scale dim num must be 1` | ⚠️ **只能省掉 dequant 那 4 个算子**（按 023 §4.2 那一臂 ≈15 µs/层），gather/索引算术/表格重编号照旧 |
| ★ `npu_kv_quant_sparse_flash_attention` | `key` = **int8 的 k_nope + 同 dtype 的 k_rope + float32 量化参数按 D 维度拼接**；`key_quant_mode=2` = per-tile-128 | ★ **唯一能真正解掉 rebuild 的形态**，但：拼接后 **D = 512+64 = 576**（与 023 记的 "576" 吻合）；它是 **SparseFlashAttention** 形态，**没有 ori/cmp 双平面 + SWA band mask** 的对应物 ⇒ **不是"换一个算子"，是"新增算子"**（给 `npu_sparse_flash_mla` 开 int8 KV 入口） |
| `npu_scatter_pa_kv_cache` / `npu_quant_scatter(_)` | 写侧 | 无关（写侧已按 C1 在 scatter 处量化，本次一行没改） |

### 1.2 ★ 路线 C 的额外证据：**生产路径本来就在用 Triton**

`vllm_ascend/ops/triton/compressor/compressor_triton.py:644 compressor_from_projected`
——**FP32 state ring 的读写就是一个 Triton kernel**。所以「Triton 在 Ascend 上能进生产」
不是推断，是现状。容器里 `triton 3.2.0`、`backends = ['ascend']`、
`triton.language.extra.cann` 在位（`raw/028-p0-env.json`）。

---

## 2. P0 第 1 步：四条候选路线的逐条判定

| # | 路线 | 判定 | 依据 |
|---|---|---|---|
| **A** | **SuperKernel** | ⛔ **否决【实测·文档】** | `model-infer-superkernel/SKILL.md`：**只支持 `ge_graph`，明确不支持 `aclgraph`**；而 KV8 的全部性能口径就是 ACLGraph。要改也改不动（vLLM-Ascend 无 `exe_mode` 概念） |
| **B** | **现成融合算子** | ⚠️ **部分可用，但不能替代 rebuild【实测·签名】** | `npu_gather_sparse_index` 只是 gather；`npu_anti_quant` 只省 dequant；能真正解掉的 `npu_kv_quant_sparse_flash_attention` **是另一种 attention 形态**（无 ori/cmp 双平面），用它 = 新增算子 |
| **C** | **Triton-Ascend** | ✅ **采纳，且已做成【实测】** | 见 §3。容器里 triton 3.2.0 + ascend backend；生产路径已在用 Triton（§1.2）；两个 kernel 直接进 NPUGraph 捕获/replay |
| **D** | **AscendC 四文件模板** | ⏸️ **不必走（本次）** | C 已达标；D 留作"要在算子内部做 tiling 下沉"时的备选（模板路径见 §1） |

---

## 3. P0 第 2 步：实现 + 硬判据

### 3.1 两个 kernel（`a2/agents/KV8_gate/p5_e2e.py`）

| kernel | grid | 一个 launch 里做完的活 | 关键设计 |
|---|---|---|---|
| `kv8_swa_fused2` | `(8, 2, 4)` = 64 programs | 读 `lens` → 算 `first_block`/窗口命中页 → 读 `swa block_table` → **页 gather**（页 stride ≠ payload，普通 `.view()` 非法）→ 逐 group 标量 scale 反量化 → 写 BF16 scratch 页 → 写 scratch block table | 宽 tile `[BR=32, DIM=512]`；**scale 用 `rows[:,None]*ROW_C + cols//128` 直接寻址**（p8 实测这个"非仿射"写法反而最快） |
| `kv8_cmp_fused2` | `(8, 16)` = 128 programs | 读 TopK 索引 → `>=0` 合法化 → `//`、`%` 算 (block, offset) → 读 `cmp block_table` → **行 gather** → 反量化 → 写 scratch → 重编号 sparse indices + scratch block table | **`tl.static_range(GROUPS)` 把 4 个 group 展开在一个 program 里**（grid 缩 4×）；直接算行地址（不再需要 023 的 `chunk = gcd(...)` 拼索引）|

两处都保持 **C1–C5 五条红线**：写入侧（`scatter_cache_sk` / `kv8_store_rows` / `kv8_swa_store`）、
spec、页几何、`seqused_*`/mask 参数、scratch 布局 **一行没改**（只换"取数算子"）。

### 3.2 判据 3 / 4：逐比特 + 真量化精度【实测】

（`raw/028-p5-e2e.json`，真量化器、真实 spec + 真实分配器建页、B=8、topk 512、窗口 128，
走真实 `_native_attention` → `npu_sparse_flash_mla`。）

| 对照 | 结果 |
|---|---|
| **fused vs 现状（真量化）逐比特** | **`torch.equal` = True** ✅ |
| fused vs BF16：`rel_L2` | **5.4331e-3**（023 是 5.4331e-3 ⇒ **逐位复现，零回退**）✅ |
| 同上：`cos` / `max_abs` / `nan_total` | **0.9999857 / 5.55e-4 / 0** ✅ |
| 无损臂（loose，非真实量化）fused vs 现状 | **`torch.equal` = True** ✅ |
| 独立正确性（`p3_fused.py`：scratch / block table / 重编号索引） | **5/5 `torch.equal` = True，`max_abs = 0.0`** ✅ |

### 3.3 判据 1 / 2 / 5：40 层整图（生产形状）【实测】

口径与 020/023 完全一致：**一图装 40 层**，`reps=40`，中位数，所以 ~320 µs 的 replay 开销在差值里自动对消。

| 臂 | µs/层 | 增量（µs/层） |
|---|---:|---:|
| `swa_bf16`（现状） | 24.51 | — |
| `swa_int8_current`（018 的 rebuild） | 169.93 | +145.42 |
| **`swa_int8_fused`** | **38.44** | **+13.93** |
| `full_bf16`（现状） | 44.68 | — |
| `full_int8_current` | 402.54 | +357.86 |
| `full_int8_fast`（023 的 flat gather） | 402.86 | +358.18 |
| **`full_int8_fused`** | **90.25** | **+45.57** |

| 增量 | 023（flat） | **本次（fused）** | 降幅 |
|---|---:|---:|---:|
| SWA 层 | +145.4 | **+13.93** | **−90%** |
| cmp 读路径 | +212.8 | **+31.64** | **−85%** |
| 整层 | +358.2 | **+45.57** | **−87%** |

**整步推算（40 SWA 层 + 4 源层）【推断·按实测外推】：+6.667 ms/step → +0.684 ms/step
（+21% → +2.15%）。**

### 3.4 ★ 判据总表

| # | 判据 | 目标 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | 整层增量（40 层整图） | ≤ +60 µs/层 | **+45.57 µs/层** | ✅ **达标** |
| 2 | 整步推算 | ≤ +0.2%（≈63 µs） | **+684 µs/step（+2.15%）** | ❌ **差 10.8×** |
| 3 | 逐比特等价 | `torch.equal` = True | **True** | ✅ |
| 4 | 真量化精度 | ≈ 5.43e-3 | **5.4331e-3**（逐位复现） | ✅ |
| 5 | 图兼容 | capture / replay / 跟输入变 | 40 层 `NPUGraph` 捕获 + 原地 replay + 改输入跟变；图内零分配 | ✅ |

---

## 4. ★★ 下限标定：**"融成 1 kernel"能达标吗？—— 不能，而且这是可算的**

（`raw/028-p6-best.json`、`raw/028-p8-tune.json`，同一 40-rep 图口径，含 ~7.9 µs/call 的图地板）

### 4.1 先标定"地板"和"纯搬数据"的价格

| 标定 | 值 |
|---|---:|
| 空图 replay（`lambda: None`，40 reps） | **7.86 µs/call**（p8 那次）/ **8.07**（p6 那次）——两次进程间的差就是噪声下限 |
| 连续 3 MB `copy_` | **8.29 µs/call** |
| 连续 6 MB `copy_` | **8.35 µs/call** |

⇒ **6 MB 的连续 DMA 与空图同价 ⇒ 单次调用的设备侧只要 ≲4 µs 就完全藏在地板下。**
（地板是 replay 的固定开销，不是带宽；**别看 3 MB→6 MB 只涨 0.06 µs 就以为带宽无限**。）

### 4.2 再量"没有 gather、没有索引算术"的反量化下限

| 形状 | 纯顺序 INT8→BF16 反量化（grid 与真 kernel 同形） |
|---|---:|
| cmp 形状（32 页 × 128 行，2.13 MB 入 / 4.26 MB 出） | **43.57 µs** |
| SWA 形状（16 页 × 128 行，1.03 MB 入 / 2.10 MB 出） | **40.48 µs** |

### 4.3 最后量融合 kernel 本身（调形状的过程也在这）

| 形状 | SWA 融合 | cmp 融合 |
|---|---:|---:|
| 每 program 一个 group（grid ×GROUPS） | 65.6 µs | 51.8 µs |
| **group 用 `tl.static_range` 展开（grid ÷GROUPS）** | **8.26 µs** | **28.46 µs** |
| **纯顺序**反量化对照（per-group grid，§4.2） | 40.48 µs | 43.57 µs |
| **纯顺序**反量化对照（static grid，与融合同 grid） | — | 32.98 µs |

★ **这一层的成本是「程序数 × 调度」，不是字节**（跟 023 §4.2"算子个数 × 每核延迟 ≈5 µs"同源，
只是把"算子"换成"program"）。SWA 融合 kernel 的 **8.26 µs ≈ 地板 7.86 µs**，
**设备侧真实代价 ≈ 0.4 µs**；cmp 因为它必须做 512×8 次**离散 512 B 行读**，落在 28.46 µs。
注意最后两行：**连"一个 gather 都不做、纯顺序读"的反量化都要 33–44 µs** ——
`28.46 µs` 的融合 cmp kernel 已经比同 grid 的纯顺序版更快，说明它基本跑在形状下限上了。

### 4.4 ⇒ 可达性论证【推断，但每一步都锚在实测上】

1. rebuild 的**定义**就是「把 INT8 物化成 BF16」：SWA 每层必读 1.03 MB + 写 2.10 MB，
   cmp 每层必读 2.13 MB + 写 4.26 MB。**这部分流量不能省**（省了就等于让算子读 int8，那是另一件事）。
2. 一 step 有 **84 次 rebuild**（40 SWA + 4 cmp），**每次至少 1 个 kernel launch**。
3. 实测最好：**13.93 µs/层 × 40 + 31.64 µs/层 × 4 = 684 µs/step**。
4. **把 SWA 整条抹成 0**（kernel 已经在 0.4 µs 量级；剩下的 13.93 里绝大部分是
   40 层**共享同一块 scratch** 造成的 RAW 串行【推断，未分离测量】），
   **仅 4 层 cmp 仍要 127 µs = +0.4%** —— 已是判据（63 µs）的 **2×**。

⇒ **结论：`≤+0.2%` 不是"融 kernel 还没做好"，而是"rebuild 形态做不到"。**
能达标的唯一形态是 **attention 算子自己读 INT8**（在 `npu_sparse_flash_mla` 上开
per-tile-128 的反量化入口，沿 `npu_kv_quant_sparse_flash_attention` 的形态）——
**那是新增算子需求，按 cannbot §6 的流程移交，不是本任务能收的。**

---

## 5. P1：state ring 的精度（不占卡）+ 容量精确重算

### 5.1 ★ 先纠正一个前提：**它不是累加器**【实测·代码事实】

`logs/024` 担心"ring 是累积状态 ⇒ 精度影响随步数累积"。查代码后**这个前提不成立**：

* `vllm_ascend/ops/triton/compressor/compressor_triton.py:644 compressor_from_projected`
  的 docstring：*"Pool C2 into token-aligned BF16 rows, then update a private FP32 ring"*；
  ring 行 = `[kv_row(512) | score_row(512)]`（`HEAD_DIM` 偏移两半）；
* python 参考（`tests/deepseek_v41_utils.py:625-626`）：`state_cache[b, pos % 32, :H] = kv[t]`、
  `state_cache[b, pos % 32, H:] = scores[t]` —— **存的是原始 FP32 投影**，不是累加量；
* 读回条件是 `seg_off = token_pos - start_pos < 0`，即**组内那一半跨段 token**；
  decode（1 query token/step）下**每个 ratio-2 组固定有一半来自 ring**，所以它在热路径上；
* **池化输出本身就被强制成 BF16**（`compressor_triton.py:661`，`out.dtype != bfloat16` 直接 raise）。

⇒ 精度问题是"**一次 bf16 往返**"，不是"每步累积"。

### 5.2 【仿真·CPU，不占卡】ring 存 BF16 / FP16 的代价

`a2/agents/KV8_gate/p9_ring_sim.py`（numpy，round-to-nearest-even bf16 模拟；
按 kernel 语义做 ratio-2 组内 per-lane 2 行 softmax 池化；raw/028-p9-ring-sim.json）：

| 量 | `rel_L2` |
|---|---:|
| 单行 ring 值 **BF16** 往返 | 1.66e-3 |
| 单行 ring 值 **FP16** 往返 | **2.08e-4** |
| ★ 池化输出本身 `.to(bf16)` 的**现状噪声地板** | **1.66e-3** |
| ring 存 **BF16** 时的池化输出 vs FP32 ring | **2.17e-3**（= 地板的 **1.3×**；73–92% 元素逐比特不变） |
| ring 存 **FP16** 时的池化输出 vs FP32 ring | **7.6e-4**（= 地板的 **0.46×**，**低于地板**） |

* 扫 gate 尖度 τ ∈ {0.05, 0.25, 1, 4, 16}，结论**稳定**（±20%）；
* ‖值域尺度不变**（round-to-nearest 的相对误差与 σ 无关），所以没有"分布未知"的致命敏感性。

★ **⇒ 建议的缩法不是 024 的 A（FP32→BF16），而是 A′（FP32→FP16）**：
**同样把页减半（131072 → 65536 B），但数值上白送**——FP16 的舍入比池化输出
本来就要吃的 bf16 舍入**还小 8×**，落在现有地板以下。
【未确认】唯一前置：ring 里的 FP32 投影不能超过 fp16 量程（|x| < 65504）——
按 residual stream 的量级不可能，但没在真权重上量过。

### 5.3 ★ 容量：精确重算（三个已知锚点全部复现）【实测·算式重算】

`a2/agents/KV8_gate/p7_capacity.py`，走 `plan_cache_slots` + `_cache_plane_sizes`
现算（不是拿 `max()` 目测），raw/028-p7-capacity.json：

| 配置 | `pool_bytes_per_block` | 容量 ×（vs 现状） | 与已有实测对账 |
|---|---:|---:|---|
| 现状（全 BF16 + FP32 ring） | 540928 | ×1.000 | ✅ = `logs/020` 的 540928 |
| KV8 双平面 + FP32 ring | 476416 | ×1.1354 | ✅ = `logs/020/023` 的 ×1.135 |
| KV8 只量化 long-KV + FP32 ring | 524288 | ×1.0317 | ✅ = `logs/023` 的 ×1.032 |
| **KV8 双平面 + ring 减半（BF16 或 FP16）** | **282880** | **×1.9122** | —（= `logs/020` 推算的 ×1.91） |
| KV8 只量化 long-KV + ring 减半 | 524288 | ×1.0317 | **无收益**（该配置下 ring 不是绑定别名） |

逐槽页（现状）：`[131072, 131072, 131072, 147712]` —— 前 3 槽（ratio-2，有 state）
**正好被 ring 131072 顶住**；第 4 槽（ratio-1）由 `long_kv+indexer = 147712` 决定。
ring 减半后前 3 槽落到 `max(kv+index 41600, ring 65536, SWA 66560) = 66560`。

⇒ **判据"容量 ×1.84"达标（×1.912）**，但**前提是 KV8 双平面 + ring 减半同时成立**。

### 5.4 ★ 缩小 ring 的真实障碍【实测·代码事实】

**FP32 不是 dtype 偏好，是被断言的不变量**，要改两处代码：

1. `vllm_ascend/core/deepseek_v41.py:96-98` —
   `DeepseekV41CompressorStateSpec.__post_init__`：
   `if self.dtype != torch.float32 or self.block_size != 32 or self.compress_ratio != 1: raise ValueError("Aurora state requires a 32-row FP32 uncompressed ring")`；
2. `vllm_ascend/ops/triton/compressor/compressor_triton.py:652` —
   `if kv.dtype != float32 or scores.dtype != float32 or state_cache.dtype != float32: raise ValueError("Aurora projections and ring state must be FP32")`。

**两处都是"改常量 + 改断言"，不涉及算法**（ring 的语义、槽位映射、读出条件全不变）。
本次**没有改**（只读分析 + 算式重算），符合 P1 的"不必真实改代码"。

---

## 6. ★ 裁决

### 6.1 两条硬指标的最终状态

| 硬指标 | 原状态 | **本次之后** | 判据 | 判定 |
|---|---|---|---:|---|
| **时延**（整步） | +6.41~6.67 ms（+21%） | **+0.684 ms（+2.15%）** | ≤+0.2% | ❌ **差 10.8×，且经证明 rebuild 路线不可达（§4.4）** |
| **时延**（整层） | +347~358 µs/层 | **+45.57 µs/层** | ≤+60 µs/层 | ✅ **达标** |
| **容量** | ×1.135 | **×1.912**（KV8 双平面 + ring 减半） | ×1.84 | ✅ **达标（前提：改 ring 两处 dtype）** |
| **精度** | rel_L2 5.4331e-3 | **5.4331e-3（零回退，且逐比特）** | ≈5.43e-3 | ✅ |

### 6.2 结论：**KV8 不收档，但必须"降级定位 + 改判据"；A2 上线仍走线 1**

**依据（按重要性）：**

1. **【实测】融 kernel 不是"推断可行"，是"已经做到且逐比特"。** 任务书 P0 的判据 1、3、4、5
   全达标；差的是判据 2（整步 ≤+0.2%）。
2. **【实测+推断】判据 2 靠 rebuild 形态不可达**（§4.4）：最好 +2.15%，本设计下限 ~+0.4~0.5%。
   ⇒ "把 KV8 时延做进 +0.2%" 这个命题**已经有答案：不能**（除非新增算子）。
   **所以 025 §3.1 的决策前提（"时延必须先进 ≤+0.2%，否则不该继续"）需要按这条重写。**
3. **【实测·算式重算】容量这一侧已经从"只赚 13.5%"变成"赚 91%"**（×1.135 → ×1.912），
   而且**代价几乎为零**（FP16 ring 的数值误差低于现有的 bf16 输出地板）。
   ⇒ 025 §3.1 那张账要重算：

   | 情景 | 容量 | 时延 | 净效果 |
   |---|---:|---:|---|
   | 现状 | 3.50M | 基线 | — |
   | KV8（**融 kernel + ring 减半**） | **6.69M（×1.912）** | **+2.15%** | ✅ **赚 91% 容量，付 2.15% 时延** |
   | KV8（只融 kernel，ring 不动） | 3.97M（×1.135） | +2.15% | ⚠️ 仍是亏 |

4. ⇒ **建议**：
   * **A2 主线仍然走线 1（DRAM 卸载）**（零时延代价、已实测 17.9×）——**不动**；
   * **KV8 从"时延零代价的容量手段"降级为"容量第二级"**：只在
     **（a）融 kernel +（b）ring 缩 FP16** 两件一起做的前提下上，交换比是
     **+2.15% 时延 ↔ +91% 容量**，这在"长上下文/多并发"目标下是**赚的**；
   * **不要再投入去追 +0.2%**：那需要的是
     **在 `npu_sparse_flash_mla` 上开 INT8 KV 入口（per-tile-128 反量化）**，
     属于"新增融合算子"，按 cannbot `model-infer-fusion` 的流程移交
     （本日志 §1.1 已把形态、D=576、缺失的 ori/cmp 双平面列清）。
   * 若主代理**坚持 +0.2% 是硬门**：那 KV8 就该按 025 的 P2 **收档为"未来候选"**，
     **并且日志要写清"不是融 kernel 失败，是 rebuild 形态与判据不兼容"**。

### 6.3 为什么融不进 +0.2% —— 一段话讲清（给主代理直接用）

> KV8 的读侧要**把 INT8 物化成 BF16** 才喂得动 `npu_sparse_flash_mla`。一 step 有 84 次
> 这样的物化（40 SWA + 4 cmp），每次至少一个 kernel launch。我们把它压到 **1 launch/次、逐比特精确**，
> 整层从 +358 µs 压到 **+45.6 µs**（−87%），但整步仍是 **+0.684 ms = +2.15%**。
> 而且这是**地板**：把 SWA 整条抹成 0，光 4 层 cmp 的离散行 gather 就还要 127 µs（+0.4%），
> 已是判据 63 µs 的 2 倍。要进 +0.2%，只能不再物化 —— 也就是**让 attention 算子自己读 int8**。

---

## 7. 复现与坑

```bash
# 只用 c1（退出码 75 = 没抢到锁）
ssh -o ControlPath=none A3-node1 'cd ~/projects/dsv41-upstream-pr && \
  bash tools/a3_chip.sh c1 --name kv8gate --timeout 1200 -- bash -lc \
  "cd /work/agents/KV8_gate && export TMPDIR=/work/agents/KV8_gate/tmp \
   KV8_RAW=/work/agents/KV8_gate/raw PYTHONPATH=/work/agents/KV8_gather/shadow && python3 p5_e2e.py"'
```

| 脚本 | 作用 |
|---|---|
| `p0_env.py` | 环境探针：triton/backend、`_C_ascend` 算子清单、`index_select`/Triton 的 NPUGraph 捕获、空图地板 |
| `p1_probe2.py` | 深挖：`tl.extra.cann.*` 符号、triton 进图 + 改输入跟变、`npu_anti_quant` 与 dequant 链的图内价 |
| `p2_docs.py` | 容器内 dump 12 个候选算子的 docstring（cannbot 第 3 步的"查官方文档"） |
| `p3_fused.py` | 第一版融合 kernel + **逐比特对拍**（scratch / table / 重编号索引） |
| `p4_sweep.py` | cmp 取数写法扫描（找到"每 program 一个 group"把 207.7 → 52.2 µs） |
| `p5_e2e.py` | **主 harness**：40 层整图性能 + 真量化精度（本次判据来源） |
| `p6_best.py` | 下限标定（空图 / memcpy / 纯顺序反量化）+ BR 扫描 |
| `p8_tune.py` | 最后调形状（`tl.static_range` 展开把 grid 缩 4×） |
| `p7_capacity.py` | P1 容量精确重算（三锚点校准） |
| `p9_ring_sim.py` | P1 ring 精度仿真（**numpy，CPU，不占卡**） |

**坑（都是本趟实际踩到的）**
① `model-infer-fusion/scripts/torch_npu_query.py` 在**宿主机**上跑会静默降级到 `_FALLBACK_DOCS`
  并报"未找到 API"——必须在容器里跑，否则会得出"这个算子不存在"的错误结论；
② `torch.ops._C_ascend.npu_scatter_nd_update_sk` **光 import 不够**，必须显式
  `enable_custom_op()`（`vllm_ascend.utils`），否则 `AttributeError`；这也要求
  `PYTHONPATH` 指向影子包（它自带 `_cann_ops_custom/vendors`）；
③ **Triton JIT 的 `tl` 必须是定义 `@triton.jit` 那个文件的模块级 global**：
  在函数里 `import triton.language as tl` 会在编译期报 `NameError('tl is not defined')`（踩两次）；
④ **每 program 一个 group（grid ×GROUPS）把成本抬 4×**（SWA 8.26 → 65.6 µs）——
  `tl.static_range(GROUPS)` 展开后 grid 缩回 1×；
⑤ 宽 tile 的 `BR=64/128` 会 `MLIRCompilationError ... PlanMemory Failed`（UB 放不下）⇒ BR 上限 32；
⑥ `p7_capacity` 必须设 `VLLM_V41_KV8*` 环境变量，否则"int8 变体"**静默**全给同一个页大小
  （我第一次就是这样拿到 6 个相同的 540928）；
⑦ `DeepseekV41CompressorStateSpec.__post_init__` 是 FP32 断言 ⇒ `dataclasses.replace(dtype=...)`
  直接抛；要重算容量得先 mutate 实例，再走 **`plan_cache_slots`**（不能走 `group_cache_specs`，
  它内部会再 `replace` 一次触发断言）；
⑧ 图内微基准**必须和空图地板比**：3 MB 与 6 MB 的 `copy_` 都是 ~8.3 µs ⇒ 地板是 replay 固定开销。

**纪律**：只用 c1（die 6）；没写 `/tmp`；没碰 `dsv41-a3` / `mooncake-master` / Phy-ID 8–15；
没写 `upstream-v41/`；没改任何别人的影子包；跨机全走 coscli。

---

## 8. 交付物

| 东西 | 路径 |
|---|---|
| 融合 kernel + 40 层 harness（**可直接叠在 KV8_gather 影子包上**） | `a2/agents/KV8_gate/p5_e2e.py` |
| 逐比特对拍 harness | `a2/agents/KV8_gate/p3_fused.py` |
| 下限标定 / 形状调优 | `a2/agents/KV8_gate/p6_best.py`、`p8_tune.py` |
| P1 容量重算 / ring 精度仿真 | `a2/agents/KV8_gate/p7_capacity.py`、`p9_ring_sim.py` |
| cannbot 查询产物 | `a2/agents/KV8_gate/p0_env.py`、`p1_probe2.py`、`p2_docs.py` |
| 原始数据 | `a2/logs/raw/028-p0-env.json` … `028-p9-ring-sim.json`（10 份） |
| A3 副本 | `~/projects/dsv41-upstream-pr/agents/KV8_gate/`（容器内 `/work/agents/KV8_gate/`） |

**结论标注汇总**：【实测】= 本机 A3 c1 上跑出来的数（§2、§3、§4.1-4.3、§5.3 的重算、§5.4 的代码事实）；
【推断】= 按实测外推（§3.3 的整步、§4.4 的可达性、§4.4 第 4 条的串行解释）；
【仿真】= §5.2 的 CPU 数值实验；【未确认】= §5.2 的 fp16 量程前置。
