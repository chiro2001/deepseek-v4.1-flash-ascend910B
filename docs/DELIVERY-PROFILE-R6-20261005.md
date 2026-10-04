# ★ 交付口径的第一份 decode 画像（devidx=1）：N=1 与 N=8（2026-10-05）

> 为什么重要：此前所有"独占贡献 / 空闲归因"分析用的都是 **`ENGRAM_DEVICE_INDEX=0`**
> 的历史 profile（`k6full_1004_100156`），而交付口径是 **`=1`**。
> 本轮 armF（交付基线）采到了交付口径的 profile，并**手工补跑了 msprof 导出**
> （原来的自动导出没跑完就被清容器了，见 §5 的流程）。
>
> 数据：`results/armF_r6_base/prof/`，rank0；N=1 捕获 58 步、N=8 捕获 55 步。
> 标注：【实测】。

## 0. 一句话

1. **步内几乎全满**：N=1 并集 94.1%（真空闲 0.152 ms / 0.58%），N=8 并集 **96.0%**（0.203 ms / 0.50%）
   ⇒ **没有任何"填气泡"的空间**，收益只能来自"少干活"。
2. **主图独占 56~57%**：N=1 14.73 ms / 步（步长 25.96 ms），N=8 23.05 ms（步长 40.81 ms）。
3. ★ **metadata（AICPU）在交付口径下只有 0.16 ms/步（0.6%），N=8 更降到 0.09 ms（0.2%）**
   —— 我之前基于 devidx=0 profile 喊的"1.09 ms/步、4%"**不成立于交付口径**，该线取消。
4. 主图的账与 devidx=0 一致：**MAC 3.5%、向量算术 4.8%，而标量 18% + AIV 跨核等待 ~44%**
   ⇒ "同步/标量受限"是**交付口径下的**结论，不是历史 profile 的假象。

## 1. N=1（① 单流）逐 stream 独占贡献

| stream | 身份 | 任务/步 | busy | **独占** | 占比 |
|---:|---|---:|---:|---:|---:|
| **146** | 主模型 40 层（图） | 1487 | 16.561 | **14.726** | **56.7%** |
| **142** | DSpark draft | 250 | 2.121 | **1.941** | **7.5%** |
| **47** | 采样 / 输入准备 | 420 | 1.698 | **1.575** | **6.1%** |
| 144 | KV·wkv 路径 | 160 | 0.984 | 0.395 | 1.5% |
| 35 | AICPU metadata ×7 | 37 | 1.402 | **0.163** | **0.6%** |
| 143 | 共享专家 | 160 | 1.199 | 0.144 | 0.6% |
| 147 / 145 | rejection 采样 / allreduce | 93 / 81 | 0.809 / 2.207 | 0.003 / **0.000** | 0% |
| 239/240/241 | 小副流 | 12/12/8 | — | 0.033/0.041/0 | ~0.3% |

* 步长 25.96 ms；并集 24.422（94.1%）；**真空闲（≥20 µs）合计 2.429 ms / 16 步 = 0.152 ms/步**。
* 145/147 号流**独占为 0** ⇒ 通信与 rejection 完全被盖住，动它们不影响墙钟。

## 2. N=8（② 多流）逐 stream 独占贡献

| stream | 身份 | busy | **独占** | 占比 |
|---:|---|---:|---:|---:|
| **109** | 主模型 40 层 | 26.753 | **23.052** | **56.5%** |
| **105** | DSpark draft | 3.399 | **3.116** | **7.6%** |
| **47** | 采样 / 输入准备 | 2.319 | **2.153** | **5.3%** |
| 107 | 共享专家 | 2.307 | 0.965 | 2.4% |
| 106 | KV 路径 | 2.060 | 0.122 | 0.3% |
| 35 | metadata | 1.613 | **0.088** | **0.2%** |
| 108 | allreduce | 3.873 | **0.000** | 0% |

* 步长 40.81 ms；并集 39.190（**96.0%**）；真空闲 0.203 ms/步（0.50%）。
* **draft 与采样两项在 N=8 反而变大**（3.12 + 2.15 = 5.27 ms，12.9%）
  ⇒ ② 的"越并发越亏"来自这两条，而不是主图。

## 3. ★ 交付口径的主图资源账（N=1，主图任务时长合计 16.681 ms/步）

| 单元 | ms/step | 占比 |
|---|---:|---:|
| AICore 总 | 9.321 | 55.9% |
| ├ **AIC MAC** | **0.578** | **3.5%** |
| ├ **AIC scalar** | **3.002** | **18.0%** |
| ├ AIC MTE1 | 1.659 | 9.9% |
| ├ AIC MTE2（载入） | 4.063 | 24.4% |
| └ AIC fixpipe | 0.358 | 2.1% |
| AIV 总 | 10.571 | 63.4% |
| ├ **AIV vec** | **0.794** | **4.8%** |
| ├ AIV scalar | 2.524 | 15.1% |
| ├ AIV MTE2 | 1.393 | 8.3% |
| └ AIV MTE3 | 0.335 | 2.0% |
| **└ 未归因（≈跨核同步等待）** | **≈5.5** | **≈33%** |

按算子族的"同步/其它"（= `aiv总 − max(子分量)`，MIX 算子里的等待）：

| 算子族 | dur | **同步/其它** | 同步占比 | 调用/步 |
|---|---:|---:|---:|---:|
| **GroupedMatmulSwigluQuantV2（gmm1）** | 2.781 | **2.236** | **86%** | 40 |
| **HcPre** | 2.465 | **1.527** | **83%** | 80 |
| GroupedMatmul（gmm2） | 1.668 | **0.920** | 63% | 40 |
| QuantBatchMatmulV3 | 0.953 | **0.684** | **85%** | 88 |
| SparseFlashMla | 1.335 | 0.300 | 53% | 40 |
| MoeInitRoutingV3 | 0.605 | 0.294 | 70% | 40 |

⇒ **交付口径下，主图里"可动的量"依旧是"同步/标量"，不是算术**（MAC+vec 合计 8.3%）。
候选顺序：**gmm1 2.24 > HcPre 1.53 > gmm2 0.92 > QBMV3 0.68**（N=1）。

N=8 时同一张表（主图 26.68 ms/步）：MAC 1.488（5.6%）、vec 1.577（5.9%）、
**AIC MTE2 8.628（32.3%）**、AIV 未归因 ~9.4（35%）⇒ 与 N=1 同构，只是载入占比抬头。

## 4. 本轮的三个"取消/降级"

| 之前的说法 | 现在 | 依据 |
|---|---|---|
| metadata 1.09 ms/步（4%），值得合并 7 次调用 | **取消**：交付口径 0.163（N=1）/ 0.088（N=8） | 本文 §1/§2 |
| 尾巴"~1000 个小算子 × 4 µs = 4 ms"（devidx=0） | **降级**：交付口径 s47 只有 1.58 ms 独占 | 本文 §1 |
| host-bound（acl 33.9 ms/步） | **取消**：其中 65% 是同步阻塞 | `DECODE-EXCLUSIVE-HOST` §4 |

## 5. ★ 新流程：手工补跑 msprof 导出（本轮踩坑）

**现象**：`/stop_profile` 之后，`torch_npu` 的自动导出**不在 stop 时同步完成**；
如果随后很快清掉容器，`mindstudio_profiler_output/`（或本次的 `ASCEND_PROFILER_OUTPUT/`）
就**不会生成**，只剩 6.4 GB 原始数据。

**补救（在任何带 CANN 的容器里，不需要 NPU）**：

```bash
cat > /tmp/prof_analyze.py <<'PY'
import sys, torch_npu
from torch_npu.profiler.profiler import analyse
analyse(sys.argv[1], max_process_number=32)
print("ANALYSE_DONE")
PY
docker run --rm -v <run>/prof:/pf -v /tmp/prof_analyze.py:/a.py:ro \
  quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
  bash -lc "python3 /a.py /pf/<rank>_..._ascend_pt"
# 产物：<rank>_..._ascend_pt/ASCEND_PROFILER_OUTPUT/{kernel_details,api_statistic,...}.csv
```

**列名不同**（`kernel_details.csv` vs `op_summary*.csv`），需要归一化：

| 新（`kernel_details.csv`） | 旧（`op_summary*.csv`） |
|---|---|
| `Name` | `Op Name` |
| `Type` | `OP Type` |
| `Start Time(us)` | `Task Start Time(us)` |
| `Duration(us)` | `Task Duration(us)` |
| `Accelerator Core` | `Task Type` |

`tools/*.py` 已支持两种布局（找不到 `op_summary*.csv` 时自动用 `kernel_details.csv`）；
本轮的归一化脚本见 `tools/normalize_kernel_details.py`。

## 5.5 非主图两条流的具体构成（N=1，交付口径）

### draft（stream 142，占 7.5%）—— 是"小 M 下的权重载入"

| 每步 | ms/步 | 算子 | 说明 |
|---:|---:|---|---|
| 3 | 0.241 | `SparseAttnSharedkv` | 3 层注意力 |
| 6 | 0.227 | `HcPre` | 每层 2 次（38 µs/次） |
| 1 | **0.161** | `MatMulV2 "5,5120;16160,5120"` | draft 的 LM head；16160×5120×2 B = 165 MB ⇒ 载入受限 |
| 5 | **0.156** | `MatMulV2 "1,256;129280,256"` | **vocab 映射，每次读 66 MB 权重 × 5 次** |
| 5 | 0.083 | `ArgMaxV2 "1,129280"` | 同上链 |
| 3 | 0.136 / 0.109 | `gmm1` / `gmm2` | MoE（M=15） |
| 3 | 0.062 / 0.042 / 0.026 | `QBMV3` | q_a / q_b / kv |
| 3+3 | 0.075 / 0.050 | `MatMulV2` o_proj | |
| 18 | 0.054 | `GatherV3`（RoPE 表） | 与 decode 同源 |

**可动的点**：5 次 `[1,256]×[129280,256]` 每次都要把同一份 66 MB 权重读一遍
（合计 330 MB ≈ 206 µs 的理论载入）。若把 5 次合成 1 次 `[5,256]×[129280,256]`，
权重只读一遍 ⇒ **理论省 ~0.12 ms/步（0.5%）**。这条在 vllm 的
`v1/spec_decode/vocab_mapping.py` 里（我们有源码），属"中低成本"。

### 采样 / 输入准备（stream 47，占 6.1%）

| 每步 | ms/步 | 算子 | 归属 |
|---:|---:|---|---|
| 1 | 0.213 | `SparseAttnSharedkvMetadata` | 这一次走的是 s47（AICPU） |
| 1 | 0.144 | `MatMulV2 "6,5120;16160,5120"` | target LM head |
| **24** | **0.194** | **`ViewCopy "16384;1;1;1;6;1;1;1"`** | **与 `Fill "1;"` 严格交替的 24 组**（见下） |
| 46 | 0.063 | `Fill "1;"` | 其中 24 次属于上面那组 |
| 53 | 0.062 | `Cast "6"` | dtype 提升（`DivMods`/`GeScalar`） |
| 2 | 0.062 | `ZerosLike "8192,6144"` | padding 底噪（`PAD_SKIP` 的靶点） |
| 18 | 0.059 | `GatherV3`（RoPE） | |
| 25 | 0.048 | `SelectV2 "6;6;"` | rejection 采样 |
| 15–16 | ~0.09 | `FloorMod/FloorDiv/GreaterEqual` | 位置/槽位链 |

**24 组 `Fill+ViewCopy` 的现场**（时间线上严格交替，每次 ~8 µs）：
它**在 devidx=0 与 devidx=1 两份 profile 里都在**、形状完全相同
（`16384;1;1;1;6;1;1;1`），且在 rejection 采样链之后 ⇒ 与 engram host 路径无关。
24 = ?（未定位到源码行；`16384 = 2 × 8192` 与 `max_num_batched_tokens` 同阶）。
【推断】是某个"逐项写回 + 逐项 fill"的循环（24 项）。**定位方法已备**
（`tools/find_op_context.py` 给出前后文；下一步可在容器里对 `tensor.copy_` 打栈）。
价值：**0.19 ms/步（0.75%）**，且是纯 Python 侧可控的循环。

### 另一条候选：`hc_sinkhorn_iters`（0.3 ms/步，需真实权重验证）

`npu_hc_pre_v2` 的 `hc_sinkhorn_iters` 是**可选属性**（op-proto 默认 20，Python 侧可传）。
tiny 上实测设备时长：

| iters | 设备时长（p50） |
|---:|---:|
| 20 | **33.76 µs** |
| 1 | **29.98 µs** |

⇒ 每次调用约 **0.2 µs/迭代**；20→1 可省 **3.8 µs/次 ⇒ 80 次 = 0.30 ms/步（1.2%）**。
⚠️ **未做数值验证**：本次夹具用的是 `scale=base=0` 的退化输入（任何迭代数都逐位相同），
真实权重下 Sinkhorn 未必收敛到同一点 ⇒ **要拿真实 `hc_fn/hc_scale/hc_base` 复测**，
再决定是否值得（收益 1.2%，风险是数值语义变化）。

## 6. 复现

```bash
BASE=$HOME/cedpd-repo/results/armF_r6_base/prof
# 1) 导出（若还没导出）
docker run --rm -v $BASE:/pf -v ~/tmp/prof_analyze.py:/a.py:ro <IMAGE> \
  bash -lc "python3 /a.py /pf/dp0_pp0_tp0_dcp0_ep0_rank0_1434_20261004195319855_ascend_pt"
# 2) 归一化 + 分析
python3 tools/normalize_kernel_details.py <...>/ASCEND_PROFILER_OUTPUT/kernel_details.csv ~/tmp/n1
python3 tools/excl_steps.py ~/tmp/n1 146 16
python3 tools/resource_budget.py ~/tmp/n1 146 58
```
