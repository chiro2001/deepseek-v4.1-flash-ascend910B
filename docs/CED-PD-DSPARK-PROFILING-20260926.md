# CED + DSpark 的 decode 时延剖析（2026-09-26）

> 目标：**降低 decode 时延**。本文只讲 decode，prefill 不在范围内。
> 数据来源：A3-21，CED 交付口径 + `SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1`，
> D 侧 torch_npu profiler 捕获（rank0），一次 144K prompt + 128 输出 token。

## 0. 口径：以 **step** 为准（不是 ms/token）

开了推测解码后，**引擎的工作量单位是 step**，不是 token：
一个 decode step = 1 次 40 层 target forward（M = 1 + SP_TOKENS = **8** 行）
\+ 1 次 3 层 draft forward + 验证。

| 量 | 32K | 144K | 说明 |
|---|---:|---:|---|
| **decode step 墙钟（无 profiler）** | **34.2 ms** | **36.6 ms** | `tools/ced_pd_bench.py`，step 数由服务端计数器反推 |
| 同上，profiler 开启时 | — | 39.5 ms | **profiler 本身约 8% 开销**；下文算子占比取自这一组 |
| 其中 device core 时间 | — | 38.5 ms | 累加所有流的 kernel 时间 |
| **设备利用率** | — | **97.5 %** | ⇒ **没有气泡可挖** |
| A（平均接受长度） | 3.43 / 3.10 | 3.25 / 3.15 | `1 + 7 × accepted/draft`，与 vLLM 的 Mean acceptance length 同式 |
| ms/token | 10.0 / 11.3 | 11.5 / 11.8 | = step ÷ A。**会被 A 稀释，不能单独当判据** |

**A 与 A2 单实例的对比**：A2 上 `DRAFT_GRAPH=0` 的实测 A = 2.7–3.0
（`scripts/serve_a2.sh` 的注释）；本方案 CED 的 D 侧是 **3.1–3.4**。
⇒ **CED 的 128-token 重放把草稿窗口喂满了，接受长度不低于单实例**。
这一点很重要：草稿的可见范围就是 128（`DeepseekV41DraftSWASpec.sliding_window`
必须等于 target SWA 的 128），而 CED 的重放长度**恰好也是 128**，
所以重放步 `num_scheduled_tokens=128` ⇒ `_dflash_num_context=128` ⇒ 窗口一次写满，
不需要靠 decode 逐步预热。若这条链断了（例如重放没走
`build_model_inputs_first_pass`），A 会掉回 ≈1.0。

关系式：`ms/token = ms/step ÷ A`。A≈1.0 时 ms/token 会显示成"变快"，
其实每步只出一个 token（`reports/draft-graph-negative-control.md` 的负控就是这么骗人的）。

**step 数怎么数**：每个 decode step 会 draft 恰好 SP_TOKENS 个 token，所以
`steps = Δvllm:spec_decode_num_draft_tokens_total / SP_TOKENS`。
`tools/ced_pd_bench.py` 已按这个口径输出 `ms/step` 与 `A`（`--metrics-url` 指向 D）。

## 1. 与 A2 单实例基线对比：CED 的 D 侧**没有**引入额外开销

| | 步墙钟 |
|---|---:|
| A2 单实例 TP8+EP8 aclgraph（`reports/scalar-bound-op-audit.md`，74 步中位） | **38.5 ms** |
| 本方案 CED-PD 的 D 侧 + DSpark | **39.5 ms** |

同样的算子清单、同样的次数（QLI 8/步、HcPre 86/步、MoE 43/步）。
⇒ **CED 的 128-token 重放只影响 prefill，decode 侧与单实例同构**；
要降时延就是优化模型本身的 decode，没有 CED 特有的浪费。

## 2. decode 段的算子构成（rank0，39.5 步，累加 1519.8 ms core 时）

| 算子 | 次/步 | **ms/step** | 占比 | 现状 |
|---|---:|---:|---:|---|
| **QuantLightningIndexerV2**（稀疏注意力索引） | 8.0 | **5.72** | 14.5% | 已开 `QLI_NO_CANDIDATE`（真权重下 577→715 µs/op，已优化过一轮） |
| GroupedMatmulSwigluQuantV2（MoE gmm1+swiglu+quant） | 43 | 3.40 | 8.6% | 已融合 |
| aclnnMatmul / MatMulV2（含 o_lora 两跳） | 113 | 3.17 | 8.0% | `wo_a` 已用 F3（2D 化） |
| AivKernel（TP 通信） | 96 | 3.17 | 8.0% | 已开 MoE AllGather |
| **HcPre**（HyperConnection 前处理） | 86 | 3.16 | 8.0% | **无优化，未在历史审计中出现** |
| QuantBatchMatmulV3（q_a/wkv/wq_b/共享专家） | 226 | 3.15 | 8.0% | 已拆分为多小 matmul（结构原因） |
| GroupedMatmul（MoE gmm2） | 43 | 1.95 | 4.9% | |
| SparseFlashMla（稀疏注意力主体） | 40 | 1.73 | 4.4% | |
| aclnnMatMulV3（router gate 等） | 49 | 0.87 | 2.2% | |
| RmsNorm | 140 | 0.74 | 1.9% | ⬅ 可融合（见 §4 机会 2） |
| HcPost | 86 | 0.72 | 1.8% | **无优化** |
| DynamicQuant | 181 | 0.69 | 1.7% | |
| 其余（RotaryMul / MoeInitRouting / Scatter / Add / Cast …） | — | ~3.5 | 9% | |

**读法**：这张表的"占比"是在**设备已经 97.5% 忙**的前提下分的。
所以任何一项的削减都近似**直接**转化为 step 时延的削减（不是"把气泡填上"）。

## 3. 已经开启的优化（都已核对生效）

`patches/PATCHES.md` 的 14 个补丁里，与 decode 相关的 6 个**全部已在 A3 生效**
（容器 env 逐项核对）：

| 补丁 | 收益（A2 实测） | A3 状态 |
|---|---|---|
| MoE AllGather（#7） | 128K −4.25 ms | `V41_MOE_COMM_ALLGATHER=1` ✅ |
| QLI no-candidate（#11） | −0.49 ms/pass | `V41_QLI_NO_CANDIDATE=1` ✅ |
| o_proj 2D（#8） | −0.31~0.76 ms/step | `V41_O_PROJ_2D=1` ✅ |
| moe-mask-range（#9） | −0.51 + 0.096 ms | `V41_MOE_MASK_RANGE=1` ✅ |
| rope-idxsel（#10） | −0.45~0.62 ms/pass | `V41_ROPE_IDXSEL=1` ✅ |
| engram JIT + 分块 gate（#2/#5） | −0.35 / −1.56 ms | `ENGRAM_JIT=1 GATE_CHUNK=0` ✅ |

⇒ **A2 上已知的所有 decode 优化都已在跑**，继续降时延必须做**新的**优化。

## 4. 优化路线图（按 收益/风险 排序）

### 机会 1：HcPre + HcPost（**3.88 ms/step，9.8%**）—— 最大未开发区

HyperConnection 是 V4.1 的架构特性（`hc_mult` / `hc_attn_fn` / `hc_ffn_fn`），
每层在 attention 前与 ffn 前各一次 `HcPre`（86 次/步 = 40×2 + 3×2 层）、
每次 36.7 µs，`HcPost` 每次 8.3 µs。

历史审计（`reports/scalar-bound-op-audit.md`）**只记了次数、没记耗时**，
所以这项一直没被当成优化目标。

**下一步**：读 `model.py` 里 `hc_*` 的调用链，判断能否把
`HcPre` 与相邻的 `RmsNorm`/`DynamicQuant` 融合（同一段代码里连续做三件事）。

### 机会 2：`rms_norm_dynamic_quant` 解锁到 W4A8（**0.25~0.40 ms/step**）

融合算子现成、仓库已在 W8A8 路径使用（`attention/dsa_v1.py:1688/:1815`），
W4A8 因为 `_is_w8a8_dynamic(...)` 判断（`dsa_v1.py:1677`）与 V4.1 自写的
multistream 路径（`dsa_v41.py:352` 单独 `quantize`）没接上。

改动小、可回退，**建议作为第一个落地实验**。

### 机会 3：共享专家并入 routed 路径（**0.8~1.6 ms/step**）

`dispatch_ffn_combine_w4_a8` 已随包下发但**仓库内零调用点**。
收益中高、置信度中低（需要打通整段融合算子）。

### 机会 4：QLI 的 kernel 级优化（**5.72 ms/step，14.5%**）

已经是最大单项，且已经吃过一轮 no-candidate 优化（99.3 → 50.3 µs dummy /
1527 → 577 µs 真权重）。剩下的空间在 kernel 内部（candidate 池 2048 块 × 8 token
对 1 个 query token，属于典型的 M=1 低效形态）。
**工作量最大，不建议先动。**

### 已实测排除的方向

#### 排除 1：`V41_SLOT_MAP_FUSED`（host 优化对 device-bound 无效）

这个开关原来是个**死开关**：`serve_a2.sh` 挂了 `patches/files/block_table.py`，
却**没有 `-e V41_SLOT_MAP_FUSED=...` 透传**，所以容器里根本没有这个 env
（已用 `/proc/<pid>/environ` 核对）。本轮补上了透传并做了单变量实验。

它的机制是：decode 稳态每步 `_compute_slot_mapping_kernel` 启动
**KV 组数次**（D 侧 13 组），单次 device 只有 2.5–3.2 µs，但每次要付
~65–70 µs 的 host/排队代价 ⇒ 每步约 0.8 ms 的 host 串行；融合成一次
二维 grid 启动后 host 时间 1.774 → 1.048 ms/step（−41%）。

**实测（只改这一个变量，DRAFT_GRAPH=1 不变，144K 四针 4/4 PASS、答案逐字节相同）**：

| ctx | 融合前 ms/step | 融合后 ms/step |
|---|---:|---:|
| 32K | 34.20 / 34.10 | 34.22 / 34.22 |
| 144K | 36.60 / 36.52 | 36.43 / 36.56 |

**没有任何变化**。原因是 §0 已经给出的那条：**设备利用率 97.5%**，
host 在 device 忙的时候异步准备下一步，那 0.8 ms 的 host 时间**完全被隐藏**，
根本不落在关键路径上。

⇒ 这条实验的价值是**证伪了"host 还有 0.8 ms 可省"的直觉**，
并把"必须减 device 工作量"这个方向钉死。
（透传的修复保留：它是真 bug，且在 prefill 主导的场景下 host 时间未必被隐藏。）

### 已排除的方向

* **SP_TOKENS 调优**：per-position 接受率 0.653/0.297/0.208/0.079/0.040/0.020/0.020，
  第 4 位后已接近 0。按 `step(M) = 25.0 + 1.81×M` 外推：SP=4 ⇒ 15.2 ms/token，
  SP=7 ⇒ 12.1，SP=9 ⇒ 13.1。**当前 SP=7 已接近最优**。
* **draft 入图**：实测 144K 上图臂 11.2 ms/token vs eager 臂 11.1（在噪声内），
  32K 上反而更差（10.7 → 11.3/11.7）。draft 只有 3 层，图 replay 的固定开销
  盖过了省下的 launch。保留它是为了与 A2 口径一致，**不要指望它降时延**。
* **同步/调度优化**：设备利用率 97.5%，没有气泡。

## 4.5 下一步该做什么（按可行性）

### 已定位：长上下文 decode 变慢的**唯一主因就是 QLI**

同口径采了两份 profiler（32K 与 144K，都是 128-token 输出、同一实例配置），
decode 段的算子对比：

| 算子 | 32K µs/op | 144K µs/op | Δ ms/step |
|---|---:|---:|---:|
| **QuantLightningIndexerV2** | **313.6** | **715.1** | **+3.13** |
| AivKernel（通信） | 28.1 | 33.0 | +0.38 |
| aclnnMatmul | 24.8 | 25.0 | −0.16 |
| GroupedMatmulSwigluQuant | 82.4 | 79.1 | −0.31 |
| HcPre | 39.1 | 36.7 | −0.36 |
| QuantBatchMatmulV3 | 14.1 | 13.9 | −0.19 |
| GroupedMatmul（MoE gmm2） | 46.0 | 45.4 | −0.12 |
| | | **净** | **≈ +2.4** |

**净 +2.4 ms/step，与端到端实测的 34.2 → 36.6 完全吻合。**

也就是说：**从 32K 到 144K，decode 变慢的钱全部花在 QLI 上**
（QLI 次数不变、恒为 8 次/步，变的是**单次耗时**）。

机制上说得通：QLI 从 candidate 池里选 top-512（`candidate_topk_blocks=2048`、
`candidate_block_size=8`），池子随上下文增长直到 2048 块的上限
（32K ≈ 250 块、144K ≈ 1125 块），所以单次耗时 ∝ 候选块数。

**含义**：长上下文（尤其 1M，池子到上限）的 decode 时延由 QLI 主导，
而砍候选集会直接动模型精度。**这不是配置能解决的，属 kernel / 算法层。**

**能立即做、但现在还没做的**：

1. **草稿路径的 slot-mapping 融合**。`compute_slot_mapping_draft()`
   （`patches/files/block_table.py:759`）**没有走融合路径**，仍是逐组循环，
   而 DSpark 下它每步都会被调用。按排除 1 的结论，这大概率也是 host-only、
   被 device 隐藏 —— 但它是同一族代码里的明显不一致，值得对齐。

**需要算子/内核开发（收益大、周期长）**：

2. **HyperConnection 融合**（3.88 ms/step，9.8%）：把 `npu_hc_pre_v2`
   与紧随其后的 `input_layernorm` / `rms_norm_cast` 合并。
3. **QLI**（5.72 ms/step，14.5%）：candidate 池 2048 块对 1 个 query token，
   典型的 M=1 低效形态。
4. `rms_norm_dynamic_quant` 解锁到 W4A8（0.25–0.40 ms/step）、
   共享专家并入 routed 路径（0.8–1.6 ms/step）。

**已经到头的**：SP_TOKENS（7 最优）、draft 入图（无收益但保留）、
host 侧优化（被隐藏）。

## 5. 复现方式

```bash
# 1) 采一段 profiler（D 侧）
TAG=xxx CONTEXT=4096 TOKENS=768 bash experiments/dspark/prof_capture.sh
#    注意：144K + 128 token 的窗口里 decode 只占 1.5 s，要分析 decode 就用短上下文 + 长输出

# 2) 解析（容器内）
docker exec dsv41-ced-d4b python3 -c "
from torch_npu.profiler.profiler import analyse; analyse('<...>_ascend_pt')"

# 3) step 级切分：每步多久、计算占多少、有没有气泡
python3 tools/ced_prof_steps.py <...>/ASCEND_PROFILER_OUTPUT/task_time.csv --gap-ms 0.5

# 4) decode 段的算子排行
python3 tools/ced_prof_ops.py <...>/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
    --t0 0.5 --t1 2.046 --top 25 --by-stream

# 5) step 口径的端到端数字（ms/step + A）
python3 tools/ced_pd_bench.py --base-url http://127.0.0.1:18992 \
    --tokenize-url http://127.0.0.1:18990 --model deepseek-v41-ced-pd \
    --contexts 32768,144000 --max-tokens 128 --ignore-eos --repeat 2 \
    --metrics-url http://127.0.0.1:18991 --out results/step_bench.json
```

**坑**：
* `kernel_details.csv` 的时间基准与 `task_time.csv` **不同**（前者只覆盖有 kernel 的
  2.0 s，后者是 16.5 s 含空转）。切窗口要用 `kernel_details` 自己的 origin。
* 大桶（768 token decode）的 `analyse()` 会以
  `Failed to get acl to npu flow events` 失败并且**不生成 task_time.csv**；
  分析 decode 用 128 token 的桶即可。
* profiler 目录属主是容器内 root，宿主 `ls` 会 Permission denied。
