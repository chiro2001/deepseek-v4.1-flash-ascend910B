# 117 — ★★★ 两条**改变判读**的结果：① 1M 长前缀「冷算 == 取回」**逐字节成立**（用户点名的未验证项已结）② 设备侧**逐算子瓶颈定性**：我们不是在"修 int8"，是撞在 **decode 原生 M=6/36 形状 + 供应商 tiling** 上

> 2026-09-23 05:4x–06:1x CST。① 来自**子代理 `ARM_1M` 的 D 阶段**（8 卡 `r8-r8-1m-de`，A2 同规模 DRAM 卸载）；② 来自**主代理**在 8 卡 profiler 产物上的**纯离线**重算（`op_summary` 的 ratio 列，**零占卡**）。
> 标记：**【实测】/【推断】/【未确认】**。凡引用仓内原文给 `路径:行号`。

> ★★★ **2026-09-23 07:0x 更正（见 `logs/121`）**：本卷 §4/§4.3 引用的"历史基线 **30.45/31.52/30.92**"与
> "**128K 慢 3.3%**"用的是**PGO 开/关两臂的合并中位**（错口径）。同类比（都 `PYTHON_PGO=0`）应为
> **31.950 vs 31.626 ⇒ 慢 1.02%**（对官方推荐配置 PGO 开是 **+5.7%**）。
> ★ §4.3 的**推断方向不但不变、而且更强**（同口径重算）：
> **`faA` 的 8K→128K 增长 = 31.626 − 31.391 = +0.235 ms**，而**我们 = 31.950 − 27.905 = +4.045 ms**
> ⇒ 我们引入的"随上下文增长成本" **≈ +3.81 ms**（原写 3.6，量级同）；**结论"我们引入了上下文相关成本"成立**。
> ★ 另：本卷 §5 的下一步清单**已被 `logs/119` + `logs/121` 取代**（`swa_table` 已由 `r8-tbl` 关闭、PGO 成为新杠杆）。

---

## 0. 一句话

① **用户点名"从未被验证"的那一条，现在有了逐字节证据**：3 并发 × **1,048,416 token** 前缀，`fill`（**冷算**）TTFT p50 = **868 s**，`replay1/2`（**DRAM 取回**）TTFT p50 = **5.57 s / 5.44 s**，而**按 prompt 的 sha256 三条三次全部相同**、`mismatched=[]`。同时 `CPU_to_GPU` 累积量 74.1 → 146.6 GB ⇒ **取回真的在搬 KV，不是"其实还在 HBM 里的假命中"**。取回把 TTFT 降了 **156×**。
② **设备账的"为什么"第一次被查到算子内部**：把 `op_summary` 的 `aic_scalar_ratio / aic_mte2_ratio / cube_utilization` 按 decode 窗加权后，
**前三名大算子全是"cube 在做数学的时间极少"**：`GroupedMatmulSwigluQuantV2` `mac=0.038 / scalar=0.501`、`GroupedMatmul` `scalar=0.561`、`HcPre` `mac=0.069 / scalar=0.309+0.229`；而 `MatMulV2` 是 `mte2=0.875`（**纯权重带宽**）。
⇒ **【推断·强】那 3.9 ms 的缺口不是"int8 没修干净"，而是 decode 原生小 M（6/36）+ w4a8 分组 GEMM 的 tiling 开销**：既不在我们可改的代码里，也不是加开关能拿的。

---

## 1. 【实测】`ARM_1M` D 阶段 —— **长前缀口径下"冷算 == 取回"**

产物：`~/projects/dsv41-upstream-pr/agents/ARM_1M/phaseD/D.{client.log,client.json}`（8 卡臂 `r8-r8-1m-de`）。
配置（从 `D.client.json` 逐字读）：`replay_prompt_tokens = 1048416`（= `max_model_len − 128 − 32`）、`max_tokens = 128`、`concurrency = 3`、`rounds: fill → reset → replay1 → reset → replay2`。

| 轮次 | 口径 | TTFT p50 | tok/s | failed | **输出 sha256（三条合起来）** |
|---|---|---:|---:|---:|---|
| `fill` | **冷算**（KV 不在 DRAM 里） | **868,126 ms** | 0.2 | 0 | `9063f0134e8c271039b19e6c6e26db18525478f71392e58243329e271e0fe764` |
| `replay1` | **取回**（前缀已落外部 KV） | **5,567.5 ms** | 9.6 | 0 | `9063f0134…`（**同上**） |
| `replay2` | **取回**（再来一遍） | **5,444.6 ms** | 9.5 | 0 | `9063f013…`（**同上**） |

**逐条比对**（不是只看合计）：

| prompt | fill sha256 | replay1 | replay2 |
|---|---|---|---|
| 0 | `7adec2ae…` | `7adec2ae…` | `7adec2ae…` |
| 1 | `939ccef6…` | `939ccef6…` | `939ccef6…` |
| 2 | `bfec3aab…` | `bfec3aab…` | `bfec3aab…` |

`sha256_common_prompts = 3`、`sha256_mismatched_prompts = []`、`replay_matches_fill_sha256 = true`。

### 1.1 为什么这不是"假命中"（★ 关键反证）

`vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` 的**累积量**：

```
fill 起点   1.5496e9   (1.55 GB)
fill 结束   1.5496e9   ← 冷算这一轮**几乎没取回**（符合预期：第一次是纯写入）
replay1 后  7.4076e10  (74.1 GB)
replay2 后  1.4660e11  (146.6 GB)
```

⇒ 每轮 replay 取回 **≈72.5 GB**，量级与"3 × 1M token 的 KV 从 DRAM 搬回 HBM"**自洽**；
且 **`fill` 与 `replay` 的差别恰好就是这 72.5 GB 的搬运** ⇒ 唯一变量是"算还是取"。

### 1.2 这条结掉了什么（对照历史）

| 历史口径 | 状态 |
|---|---|
| `HANDOVER` §：**"长前缀口径下『冷算 == 取回』从未被验证"** | ✅ **已结**（本条 §1） |
| 短前缀（8K/128K）下的同族判据 | 早已通过（`phaseCp` 的 `replay==fill`、题库 10/10） |
| **仍未验证**的口径 | **"取回后的输出与『重算一遍』在每个 token 上都一致"** 需要 `128 token` 的逐 token 比对（本条只到**整段 sha256**；整段相同已经蕴含逐 token 相同，但**生成路径若中途换页、事后拼接则会掩盖**）⇒【推断】风险低，但**未单独证** |

---

## 2. 【实测】逐算子瓶颈定性（新表；`op_summary` 的 ratio 列）

方法：`a2/agents/HC_PROBE/op_bottleneck.py` —— 流向解析 `op_summary_*.csv`（104 MB），
**只留 decode 稳态窗** `[1790111383115698.5, 1790111385955563.5]`（96 步），
ratio 按**该行自身的 duration 加权**（否则 1 个 1500 µs 的 prefill 会和 8000 个 30 µs 的 decode 同权）。
产物：`a2/agents/HC_PROBE/out_bottleneck.csv`（85 种算子 / 289,482 task / 窗内 2992.8 ms ⇒ 31.18 ms/步）。

| OP | 次/步 | ms/步 | `aic_mac` | `aic_mte2` | **`aic_scalar`** | `aiv_vec` | `aiv_scalar` | `cube_util` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GroupedMatmulSwigluQuantV2 | 42.99 | 3.298 | **0.038** | 0.282 | **0.501** | 0.003 | 0.160 | 86.8 |
| MatMulV2 | 110.94 | 2.903 | 0.087 | **0.875** | 0.114 | 0.000 | 0.000 | 77.9 |
| **HcPre** | **86.00** | **2.709** | **0.069** | 0.177 | **0.309** | 0.034 | **0.229** | 59.9 |
| QuantBatchMatmulV3 | 225.98 | 2.349 | 0.069 | 0.460 | **0.445** | 0.016 | 0.137 | **50.9** |
| GroupedMatmul | 42.99 | 1.583 | 0.043 | 0.357 | **0.550** | 0.008 | 0.382 | 77.9 |
| SparseFlashMla | 40.00 | 1.336 | 0.041 | 0.175 | 0.340 | 0.015 | **0.489** | **46.3** |
| QuantLightningIndexerV2 | 8.00 | 0.884 | 0.050 | 0.070 | 0.310 | 0.058 | 0.100 | **16.0** |
| RmsNorm | 139.99 | 0.736 | — | — | — | 0.154 | **0.397** | — |
| MatMulV3 | 48.99 | 0.726 | 0.151 | 0.709 | 0.251 | — | — | 83.5 |
| DynamicQuant | 180.99 | 0.648 | — | — | — | **0.425** | 0.241 | — |
| HcPost | 85.99 | 0.641 | — | — | — | **0.607** | 0.310 | — |
| MoeInitRoutingV3 | 42.99 | 0.615 | — | — | — | 0.030 | 0.157 | — |
| Cast | 387.18 | 0.579 | — | — | — | 0.032 | 0.297 | — |
| ScatterNdUpdateSk | 98.00 | 0.538 | — | — | — | 0.020 | 0.321 | — |
| InplacePartialRotaryMul | 145.00 | 0.533 | — | — | — | 0.111 | **0.429** | — |
| `_kv8_swa_rows_kernel`（我们的） | 40.00 | 0.423 | — | — | — | 0.059 | 0.123 | — |
| ViewCopy | 30.73 | 0.250 | — | — | — | 0.002 | **0.499** | — |

### 2.1 三条立刻可读出的结论

1. ★★ **`aic_scalar` 在大算子上一律 0.25–0.70，而 `aic_mac` 只有 0.04–0.15**。
   `GroupedMatmul` `scalar=0.550` / `GroupedMatmulSwigluQuantV2` `0.501` / `QuantBatchMatmulV3` `0.445` —— 这三个加起来 **7.2 ms/步**。
   对照 §3 的形状（M=6 或 36、48 个分组）⇒ **【推断·强】这是"小 M + 多分组"固有的 AIC 标量开销**（寻址/tiling 逐组重算），不是数据量问题。
2. ★ **`MatMulV2` 是 `mte2=0.875` 的纯权重带宽 bound**（2.903 ms/步）。它读的是 **bf16 权重**（见 §3 的形状），
   即 **w4a8 量化没覆盖到的那部分矩阵** ⇒ 想降它只能"再量化"，属模型侧，**不在本轮范围**。
3. **`ViewCopy` 的 `aiv_scalar=0.499`** —— 一个"纯拷贝"算子 50% 时间花在标量上，是 `cannbot` CASE-002「跨 dtype cache view 阻断 reinplace」的同形症状（`CANNBOT_DOC` §1.3）。量小（0.25 ms/步）但**是唯一"看起来就不该存在"的项**。

---

## 3. 【实测】大 GEMM 的**形状分布**（decode 窗，`a2/agents/HC_PROBE/op_shapes.py`）

产物：`a2/agents/HC_PROBE/out_shapes_wide.csv`（4 种算子 / 27 种形状 / 40,598 task / 972.8 ms）。

| op | 次/步 | 单次 µs | ms/步 | cube_util | aic_mte2 | aic_scalar | 形状（`Input Shapes` 原文） |
|---|---:|---:|---:|---:|---:|---:|---|
| GroupedMatmulSwigluQuantV2 | 40.00 | 78.7 | 3.149 | 87.0 | 0.288 | **0.503** | `36,5120;36;48;48,72,320,16,64;48,4608;48,4608` |
| GroupedMatmul | 40.00 | 36.6 | 1.465 | 78.7 | 0.373 | **0.561** | `36,2304;48,2304,5120;48,5120;48,1,5120;;;;48;36` |
| MatMulV2 | 40.00 | 24.1 | 0.963 | 73.5 | **0.890** | 0.059 | `6,4096;4096,1024` |
| MatMulV2 | 2.00 | 354.0 | 0.708 | 82.0 | **0.991** | 0.011 | `6,6144;25600,6144` |
| QuantBatchMatmulV3 | 40.00 | 13.5 | 0.542 | 66.7 | 0.483 | 0.264 | `6,5120;40,320,16,32;1280;6` |
| MatMulV2 | 40.00 | 13.4 | 0.534 | 82.4 | 0.763 | 0.017 | `6,1024;5120,1024` |
| QuantBatchMatmulV3 | 48.00 | 10.0 | 0.479 | **51.9** | 0.486 | **0.482** | `6,1280;128,80,16,32;4096;6` |
| QuantBatchMatmulV3 | 40.00 | 10.4 | 0.414 | 63.5 | 0.388 | **0.697** | `6,288;160,18,16,32;5120;6` |
| QuantBatchMatmulV3 | 43.00 | 8.3 | 0.356 | **26.8** | 0.542 | 0.385 | `6,5120;16,320,16,32;512;6` |
| QuantBatchMatmulV3 | 40.00 | 8.2 | 0.327 | **31.5** | 0.527 | 0.315 | `6,5120;18,320,16,32;576` |
| MatMulV2 | 4.95 | 31.3 | 0.155 | 89.5 | 0.805 | **0.632** | `1,256;129280,256` ← LM head |
| MatMulV2 | 8.00 | 6.0 | 0.048 | **7.3** | 0.779 | 0.088 | `6,5120;32,5120` |

**读法**：
· `QuantBatchMatmulV3` 的 `40,320,16,32` / `16,320,16,32` 这类是 **w4a8 的 int4 打包权重**（4-bit ⇒ 权重面比 bf16 小 4×）
  ⇒ 所以它 `mte2` 只有 0.46–0.54、**不是带宽 bound**，而是被 **scalar 0.26–0.70 + cube_util 27–67%** 卡住。
· **`26.8%` / `31.5%` 那两格（合计 0.68 ms/步）是全场 cube 利用率最低的大块** ⇒ 若要"找供应商 op 的靶子"，就是它们。
· `MatMulV2 "6,6144;25600,6144"` 单次 354 µs、`mte2=0.991` ⇒ 读 `25600×6144×2 B = 315 MB`，**1.0 TB/s 级别，已饱和**。

---

## 4. 【判读】"离 ≤24 ms 还有多远"——**分三层，如实写**

### 4.1 硬账（**都不含"再量化/换供应商 op"这类模型级改动**）

| 层级 | 项 | ms/步 | 前提 |
|---|---|---:|---|
| **A. 我们自己的代码（包已就绪或已定价）** | `swa_table` 列并行（`MERGED_TBL`，`9bdcdaf5`） | **0.40–0.53** | 待 8 卡臂（卡一空就跑） |
| | fp16 scale 写并入 int8 K 核（`PROF_MINE` §4.1） | 0.263 | 需改自研核 + 8 卡臂 |
| | MoE gating 的 3×int32→int64 + 1×float→bf16 | 0.147 | 需改框架侧 + 8 卡臂 |
| | `HcPost`+`HcPre` 跨支融合 | 0.30–0.64 | 需参照 GLM5-next 变体；**未确认** |
| | **小计** | **1.11–1.58** | |
| **B. 平台/配置（A2 专属）** | MC2（`MatMul+AllReduce`，`hccl-matmul` 路线） | ≤1.153 | **仅 A2**、**不支持入图**、**本期不支持 quant** ⇒ 与我们的 aclgraph 冲突，需评估 |
| **C. 结构（周级工程，非开关）** | `HcPre` 定制核（`hc_mult=4` 专用；仓内有 `dsv4_hc_pre` 的 golden/生成/调优全链） | ≤2.0 **【未确认】** | 需 AscendC/PyPTO 开发窗口 |
| | AI_CPU → AICore（`SparseFlashMlaMetadata` 等 4 类） | ≤0.70 | 需供应商替换 |
| | MoE 分组 GEMM 的 tiling（`scalar=0.50` 那两格） | ？ | **供应商 op**，非我们可改 |

### 4.2 结论

| 判读 | 结论 |
|---|---|
| **A 层全上** | 27.905 − 1.58 ≈ **26.3 ms/步**（8K 服务端口径） |
| **A + C 里最乐观的一条（HcPre 定制核拿到 2.0）** | ≈ **24.3 ms** —— **贴着线，且依赖"未确认"的 kernel 工程** |
| ⇒ **≤24 ms（8K 服务端 quote）** | **【推断·强】不是"再改几行"能到的**：必须动**供应商算子**或**再量化**，或**明确口径** |
| ⇒ **≤24 ms（≈1K、进程内，即 goal 里 24 ms 的真实出身）** | 我们用三档服务端 quote 线性外推 **≈27.67** ⇒ **【推断】同样不达**；但**该口径从未在服务上实测**（缺 `TOKENS=1024` 那一格） |
| **已达成、无争议的部分** | ✅ 四轴同开 ｜ ✅ 保精度（题库 10/10、冷算==取回逐字节） ｜ ✅ **8K/32K 服务端 quote 超越加 int8 之前的基线**（27.905<30.45、29.226<31.52） |
| **明确未达成** | ❌ 128K（31.950 > 30.92，慢 3.3%） ｜ ❌ 服务端 8K ≤24 |

### 4.3 关于 128K 那 3.3%（**独立的一条，值得单修**）

历史基线 **8K→128K 只涨 0.47 ms**（30.45→30.92），我们 **涨 4.05 ms**（27.905→31.950）
⇒ **【推断】我们引入了 ≈3.6 ms 的"随上下文增长"的成本**，候选：池 7938 页下的 per-step 记账 / 128K 时取页更多导致 scale 面 gather 变多。
★ 这条**与 ≤24 无关**（8K 上不体现），但它决定"能不能在 A2 生产的长上下文场景不退步" ⇒ **建议单独立项**，用同一条臂跑 **8K vs 128K 两档 profile 对差**（约 2×30 min）。

---

## 5. 下一步（按"卡一空就能跑"排序）

| # | 动作 | 占卡 | 判据 |
|---|---|---|---|
| 1 | `ONLY=tbl bash ~/tmp/run_perf_fix_arms.sh`（`MERGED_TBL`） | 8 卡 / 8051 / 25 min | `[bneck] hp` 降 ≥0.40 ms；`GPU KV cache size` 仍 **427,643**；容器内 kernel md5 = `9bdcdaf5` |
| 2 | 同一条臂上补 **`TOKENS=1024`** 与 `TOKENS=8192` 两格 quote | 复用活的臂 | 得到"1K 服务端 ms/步"，与 23.9/24.9 **同口径**对照 |
| 3 | 精度复跑（题库 10/10 + `prefix-pair` + `same_run_replay`） | 复用活的臂 | 与 `r8-merged` 逐项不退化 |
| 4 | 8K vs 128K **两档 profile 对差**（§4.3） | 8 卡 ×2 | 找出那 3.6 ms 的归属 |
| 5 | `HcPre` 单卡拆解（本卷已写好探针 `a2/agents/HC_PROBE/hc_probe.py`） | **单卡 c1/c2** | ★ 见 §6 —— 当前**被"生产镜像才有 `_C_ascend.npu_hc_pre_v2`"挡住** |

---

## 6. 【未确认/受阻】`HcPre` 单卡探针为什么还没跑出数

写了 `a2/agents/HC_PROBE/hc_probe.py`（扫 `hc_sinkhorn_iters ∈ {1,2,5,10,20,40}` 拆开那 30.5 µs/次），
在**空闲单卡** `prbench-c1`（走 `tools/a3_chip.sh c1`，**不占 8 卡**）上跑，**失败**：

```
AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'npu_hc_pre_v2'
⇒ torch.ops._C_ascend 里 n_ops = 2（空壳）
```

原因【实测】：`npu_hc_pre_v2` 是 **vllm-ascend 自己 `csrc/torch_binding.cpp:1424/2894` 注册的私有算子**，
只在**生产镜像**（8 卡臂用的 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`）里注册；
`prbench-c1` 里 `.so` 字符串**有** `npu_hc_pre_v2`，但**没被 dlopen 注册**（`import vllm_ascend` 后 `_C_ascend` 仍是空壳）。
替代路径（`torch.ops.npu.npu_mhc_pre` 等 6 个 op 在 `prbench-c1` 里存在）是 **torch_npu op-plugin 的另一族实现**，**不是同一条 kernel** ⇒ **不能拿它当判据**。
⇒ 想跑这个探针，必须在**生产镜像**里起一个**单卡**容器（当前 c0/c1/c2 都是 `prbench-*` 镜像；8 卡全被 8051 臂占着）**或**等 8 卡臂拆掉后在它的容器里 exec。
★ 这是"单 chip 尽快踩坑"在本条上的**真实阻塞点**，登记备查，**不硬凑数**。

---

## 7. 复现配方（全部只读 / 纯离线）

```bash
# ① 逐算子瓶颈定性（104 MB，流式，O(1) 内存）
RP=~/projects/dsv41-upstream-pr/shadow-pkg/results/r8_r8-prof_20260923_045235/prof/dp0_pp0_tp0_dcp0_ep0_rank0_1444_20260922210922991_ascend_pt/PROF_000004_*/mindstudio_profiler_output
cd ~/projects/dsv41/a2/agents/HC_PROBE
ssh A3-node1 "python3 - $RP/op_summary_*.csv '1790111383115698.5 1790111385955563.5 96'" < op_bottleneck.py > out_bottleneck.csv
# ② 大 GEMM 的形状分布
ssh A3-node1 "python3 - $RP/op_summary_*.csv '1790111383115698.5 1790111385955563.5 96' 'QuantBatchMatmulV3,MatMulV2,GroupedMatmulSwigluQuantV2,GroupedMatmul' 40" < op_shapes.py > out_shapes_wide.csv
# ③ D 阶段判据（ARM_1M 产物）
python3 -c "import json;d=json.load(open('~/projects/dsv41-upstream-pr/agents/ARM_1M/phaseD/D.client.json'));print(d['replay_matches_fill_sha256'], d['sha256_mismatched_prompts'], d['fill_out_sha256_all'])"
```

**产物**：`a2/agents/HC_PROBE/{op_bottleneck.py,op_shapes.py,hc_probe.py,out_bottleneck.csv,out_shapes_wide.csv,out_bottleneck.err,out_shapes_wide.err}`；
`ARM_1M/phaseD/D.{client.json,client.log}`（子代理侧）。
