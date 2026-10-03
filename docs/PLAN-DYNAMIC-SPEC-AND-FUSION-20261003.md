# 计划：动态 K 接线修复 + 融合算子（2026-10-03）

> 用户指令：「目前动态 k 之前实现过但是 **req>2 的时候性能掉到 10 tps**。如果可以解决也是好的。
> 此外**融合算子也需要做**，可以先用 **tiny 环境子代理并行**。你规划一下，然后 create goal。」

---

## 0. 资源现状（已核实，2026-10-03 06:0x）

| 资源 | 位置 | 状态 | 用途 |
|---|---|---|---|
| **独立 TP8** | a3-21 **dies 8–15**（NPU4–7） | 我们的交付档在跑（`dsv41-tp8k5` @19210，K=5） | **Track A 主线** |
| **tiny 实例** | a3-21 **dies 2–3**（NPU1） | `dsv41-tinyspark` @19310，health 200；模型 `/home/l00886679/models/out/v41-tiny-dspark`（6.1 MB dummy） | **Track B**（子代理） |
| **空闲 die** | a3-21 **dies 4–7**（NPU2–3） | 无进程、HBM 仅 ~3 GB 残留 | 需要第二个小实例时的备份 |
| 邻居算子脚手架 | `~/projects/dsv41-workspaces/wt-graph/op_peak/` | `ascendc/{smoke_add,wo_a}`、`kernels/{qbmv3,wo_a}_kernels.py`、13 篇 docs | Track B 直接复用 |

---

## 1. Track A：动态 K 在独立 TP8 上接线，并查清"req>2 掉到 10 tps"

### 1.1 已知事实（全部有出处）

| 事实 | 出处 |
|---|---|
| 动态 K **在 CED-PD 的 D 侧跑通过**：K=7 ↔ K=0 按请求数切、切得回来、144K/1M 正确性全过 | `docs/CED-PD-DYNAMIC-SPEC-20260926.md` §11 |
| 默认表 `1,1,7;2,8,0`（N=1 走 K=7，N≥2 走 K=0） | `scripts/serve_a3_ced_pd.sh:199` |
| **独立 TP8 用不了**：`serve_a2.sh:1645` 有门 `[ "$V41_CED_ROLE" = "decode" ] \|\| die` | 已核实 |
| 还有第二道门：`V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1` 必须显式给（否则上游 PIECEWISE 降级 + V4.1 只支持 eager/FULL_DECODE_ONLY ⇒ 模型构造期炸） | `serve_a2.sh:1659` |
| **CED 档的并发 ≥2 从来没有吞吐基线**：§11.4 只记了 ms/step（两流 32.2/32.5），`decode tok/s` 那一列是空的 | `CED-PD-SPEC-MODE-20260928.md` §7.2 |
| 今天在独立 TP8 上实测的**静态**包络：N=16 时 SPEC=0 **434.1** vs SPEC=1 K=5 **398.6**（+8.9% 可拿） | `docs/V41-TP8-DSPARK-MULTISTREAM-PERF-20261003.md` §4 |

⇒ **"req>2 掉到 10 tps"这条既没有文档记录、也没有解释**。它是用户的口头报告，
**必须先复现**，不能先入为主。

### 1.2 量级判断（为什么 10 tps 很可疑）

10 tok/s = **100 ms/token**。参照点：

| 路径 | ms/token |
|---|---:|
| 静态 SPEC=0（纯自回归） | 19.4 |
| 静态 SPEC=1 K=5 | 9.2 |
| **报告值** | **~100** |

⇒ 比最慢的合法路径还慢 **5×**。这不像"K 选错了"，更像**每步都在跑重路径**：
候选是 ① 每步降级 eager ② padding 到远大于真实 batch 的桶
③ 图键不命中导致每步重新捕获/派发 ④ 与 admission gate 的交互。

### 1.3 步骤

| # | 做什么 | 判据 |
|---|---|---|
| **A1** | 放宽 `serve_a2.sh` 的门：`SP_SCHEDULE` 允许非 CED 角色，但**强制要求**同时给 `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`（保持 fail-closed） | 只给 `SP_SCHEDULE` 仍被拒；两个都给才放行 |
| **A2** | 独立 TP8 上起 `SP_SCHEDULE='1,1,5;2,32,0'`（K=5 与当前交付档对齐），跑 1/2/4/8/16 曲线 | 拿到完整三元组 `(ms/step, A, tok/s)`；**若复现 ~10 tps，进入 A3** |
| **A3** | 定位：同时抓 ① 容器内 `[DYNAMIC-SPEC]`/降级日志 ② `torch_npu` profiler ③ `/metrics` 的 `num_draft_tokens` 增量（判 K 是否真切成 0）④ 图键/桶命中 | 能指名道姓说出"哪一步在做什么"，而不是"感觉慢" |
| **A4** | 修 + 验收 | 正确性：**K=7 与 K=0 两条路径都要覆盖**（144K/1M 四针 + 并发 2 各带不同针 + `regress2.py`）；性能：N=16 ≥ **430 tok/s**、N=1 ≥ 105 tok/s |

### 1.4 若 A3 查不出（备用路线）

不押注单一方案。若动态 K 修不动，**同样的包络可以静态拿到**：
N≥12 的流量走一个 `SPEC=0` 的实例，N<12 走 `SPEC=1`，前面加一个极简按并发路由
（**不是**鉴权/配额那套，只是一个 byte-level 的端口选择）。
代价是多占 8 个 die；判据同上。

---

## 2. Track B：融合算子（子代理，tiny 环境并行）

### 2.1 目标算子（按收益排序，全部来自今天的设备画像）

| 目标 | 现状 | 省 |
|---|---|---|
| **HcPre + HcPost 融合** | `torch.ops._C_ascend.npu_hc_pre_v2` / `npu_hc_post`，各 **86 次/步**；HcPre 48 µs/次、HcPost 21.8 µs/次 | **5.8 ms/step（7.5% @N=8）** |
| **RmsNorm + DynamicQuant 接进 W4A8** | 融合算子**已存在**（`dsa_v1.py:1688/:1815` 已在 W8A8 用），W4A8 因 `_is_w8a8_dynamic()` 判断没接上 | 0.25–0.40 ms/step |

### 2.2 为什么 HcPre 值得动（关键判据）

HcPre 的实测输入形状 `[96,4,5120]` + 权重 `[24,20480]`，单次算术量
≈ 96×4×20480 = **7.9 MFLOP**，却花 **48 µs** ⇒ **约 164 GFLOPS**，
对这个硬件是**纯 overhead bound**（launch + 小算子调度），不是算力 bound。
⇒ 融合/合并的价值高于"优化它的算术"。

### 2.3 子代理的交付物与步骤

| # | 做什么 | 判据 |
|---|---|---|
| **B1** | 找 `npu_hc_pre_v2` / `npu_hc_post` 的**源**：CANN 已开源（`cann/ops-transformer`、`cann-recipes-infer`、`deepseek-ai/TileKernels`），先确认是"能读源码"还是"必须自写" | 写出结论 + 源码链接/路径；不允许猜 |
| **B2** | 在 tiny 上做**逐算子基准**：HcPre/HcPost 单算子的耗时、shape、与相邻 `RmsNorm`/`DynamicQuant` 的依赖关系 | 拿到单算子 baseline（不重启主服务） |
| **B3** | 实现第一个融合 kernel：**HcPre+RmsNorm**（`hc_pre → input_layernorm` 是相邻且都作用在同一个 `[T,4,5120]` 上） | tiny 上端到端不崩、数值与参考逐元素对齐（容差按 fp16/bf16 惯例） |
| **B4** | 接进主服务的 overlay（`~/cedpw` 那套 mount），在**独立 TP8** 上验收益 | 省 ≥3 ms/step；正确性 `regress2.py` 5/5 |

**子代理纪律**（沿用本仓的既有约定）：
- **长等待**：起服/编译这类长任务用长超时，不要频繁轮询；
- **产物放 `~/tmp/`**，不用 `/tmp`；**禁用 `rm -f`**（用 `mv` 到 `.bak`）；
- 每次改动后跑回归；
- 遇到需要**复杂决策**（改架构、动交付默认值）时**停下来汇报**，不要自己拍板。
- **不要动 dies 8–15**（Track A 在用）；tiny 用 dies 2–3，如需更多用 dies 4–7。

---

## 3. 产出与验收总表

| 项 | 判据 |
|---|---|
| Track A 代码 | `serve_a2.sh` 的门放宽（保持 fail-closed）+ 根因修复补丁 |
| Track A 数据 | 独立 TP8 上 1/2/4/8/16 五档三元组；N=16 ≥430、N=1 ≥105 tok/s；K=7/K=0 两条路径的正确性证据 |
| Track B 代码 | 融合 kernel（AscendC 或 Triton）+ overlay 接线 |
| Track B 数据 | tiny 上的单算子 + 端到端对比；独立 TP8 上 ≥3 ms/step |
| 共同 | 全部推到 GitHub `feat/v41-dcp8`；负结果也要写（本仓纪律） |

---

## 4. 风险与预置

| 风险 | 预置 |
|---|---|
| 动态 K 的"静默算错"（豁免了上游 PIECEWISE 降级门） | 只认**探针证据**，不认"起来了 + ms/step 正常"；K=0 路径必须单独覆盖 |
| dies 4–7 被别人抢 | 子代理优先用已在跑的 tiny（dies 2–3）；需要新实例时先 `npu-smi` 确认 |
| 融合 kernel 数值不达标 | 先做"融合但不改数值域"的版本（同 dtype、同累加顺序）；不达标就退回并记录负结果 |
| 主线服务被误动 | Track A 每次重启前确认 `dsv41-tp8k5` 无人使用；子代理不碰 dies 8–15 |
