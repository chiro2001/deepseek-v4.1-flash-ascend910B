# TP8+DCP8 + DSpark 的 decode 绝对性能：设备预算与两条优化（2026-10-02）

> 用户判据：「虽然精度看起来好，但是这个性能就很差了」。
> 本文给出**当前构建的绝对账**、与 A2 参照的差距拆解，以及两条已实现优化的实测结果。
> 所有结论标 **【实测】**/**【推断】**/**【未确认】**。

## 0. 口径（先看这节，否则所有数字都会被误读）

| 项 | 值 |
|---|---|
| 实例 | a3-21 chip 8–15，TP8 + DCP8，`MAX_SEQS=16 BAT_TOKENS=2048` |
| 模型 | `/home/l00886679/models/out/v41-flat-verify3` |
| 投机 | `SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1`（DSpark K=7） |
| 并发 | 1 / 8 / 16，长 prompt（≈1800 字符，取自《红楼梦》语料） |
| 计量 | `~/tmp/noprof.py`：差减法隔离 prefill；`ms/step = Δwall / (Δdraft/K/N)` |
| 桶列 | 稠密化后的 `1,2,3,4,8,12,16,20,24,32,40,48,56,64,72,80,88,96,128` |

**★ 重要口径纠正**【实测】：line-A 报告用 **×0.857** 把 profiled 折算成无 profile
（依据是那台实例 32.58 → 35.88）。**本实例不适用**：同一配置
profiled 的 decode span = **43.39 ms/step**，无 profile 的墙钟 = **43.33 ms/step**
⇒ 本实例 profiling 开销 ≈ 0。**本文件一律用实测绝对值，不做 ×0.857 折算。**

## 1. 与 A2 生产参照的差距【实测】

| 配置 | ms/step | A | ms/token | 聚合 tok/s |
|---|---:|---:|---:|---:|
| A2 生产 `SPEC=0`（N=1） | 24.35 | 1.0 | 24.35 | 41.06 |
| line-A **DCP8** `SPEC=0`（N=1，无 profile） | **32.58** | 1.0 | 32.58 | 30.69 |
| A2 生产 `SPEC=1`（N=1） | 32.68 | 3.10 | 10.54 | **94.79** |
| **本构建 DCP8 `SPEC=1`（N=1）** | **43.33** | 3.03 | 14.30 | **64.9** |

拆解：
* **DCP 本身的税**：`24.35 → 32.58` = **+8.23 ms/step（+34%）**【实测，line-A 同配置】
* **DSpark 的税**：A2 上 `+8.33`（24.35→32.68）；本构建上 **+10.75**（32.58→43.33）
  ⇒ DSpark 在 DCP8 上比在 A2 上贵 **+2.4 ms/step**（merge 包随 T=1→8 变大）【推断】

⇒ **要追平 A2 的 94.79 tok/s，主要战场是 DCP 的 +8.23，不是 DSpark。**

## 2. ★ 设备预算：43.39 ms/step 里 13.72 是空闲【实测】

方法：line-A 的 `tools/idle_report.py`（区间并集求补集 → 归因到"空闲前最后结束的算子"），
在**干净 decode 步**上统计。本部署（DSpark）的结构常量是
`QuantBatchMatmulV3 == 226 条/簇`（不是 SPEC=0 的 208），66 个干净步。

| | ms/step |
|---|---:|
| span | 43.39 |
| busy（区间并集） | 29.67 |
| **idle** | **13.72（31.6%）** |
| idle ≥ 50 µs | 7.89（占全部空闲 57.5%） |

### 2.1 归因 TOP（idle≥50 µs，按"前算子"排序）【实测】

| 前算子 | ms/step | 条/step | 次均 µs | 该算子自身 kernel |
|---|---:|---:|---:|---:|
| **ViewCopy** | **1.940** | 9.5 | 203.8 | 0.639 |
| **Add** | **1.620** | 8.1 | 200.3 | 0.348 |
| **_compute_slot_mapping_kernel** | **1.302** | 10.8 | 120.7 | **0.032** |
| Fill | 0.469 | 3.3 | 141.9 | — |
| Sub | 0.431 | 1.6 | 265.7 | 0.422 |
| FloorMod | 0.332 | 2.0 | 164.9 | — |
| GatherV3 | 0.284 | 1.7 | 170.6 | — |
| ClipByValueV2 | 0.218 | 2.1 | 103.4 | — |

* **前 3 项合计 4.86 ms/step = 11% of span**，且它们的 kernel 时间都很小
  ⇒ 是**下发/依赖停顿**，不是算力。
* 空闲直方图的 0–5 µs 桶有 **1839.7 条/step**（次均 1.02 µs）——这是每个 kernel
  边界的下限（line-A 是 1521 条/step，本构建**多 320 条/step**）【实测】

## 3. 通信账（N=1）【实测】

去重后（`AivKernel` 与 `hcom_*` 是同一条的两种记账）：

| | 无优化 | **开融合 merge kernel** |
|---|---:|---:|
| `allReduce gid=097`（DCP merge） | 78.34 条/步 / 2162.6 µs | **39.13 条/步 / 1650.2 µs** |
| `allReduce gid=503`（TP/EP） | 91.72 条/步 / ~3340 µs | 91.64 条/步 / 3340.2 µs |
| `allGather gid=374`（q gather） | 39.60 条/步 | 39.58 条/步 / 664.6 µs |
| 合计 | ~212 条/步 | **173 条/步** |

⇒ 融合 kernel 把 merge 的**第二次 allreduce（权重列，256 B）**消掉了：
**每层少 1 次集合通信 × 38 层 = 38 次/步**（与代码注释"16 B 与 1 MB 同价"一致）。

## 4. 优化 1：AscendC 融合 merge kernel（`V41_DCP_MERGE_KERNEL=1`）

**实现**：`v41_merge_kernel.py` 的 `merge_pre` + all_reduce + `merge_post`，
替换原来的「pack 两次 ViewCopy → allreduce → 切片/减法/除法/Cast」整条链。
（该路径的**三个缺陷**——`.so` 同名遮蔽、`weights=None` 撞守卫、裸指针收非连续切片——
已于 2026-10-01 修好并验收，见 `V41-DSPARK-OPT-MERGEKERNEL-SHADOW-20261001.md`。）

**实测（同脚本同配置，无 profile）**

| 并发 | 关闭 | 开启 | 变化 |
|---:|---:|---:|---:|
| 1 | 43.33 | **42.47** | **−2.0%** |
| 8 | 95.75 | **93.06** | **−2.8%** |

**生效确认**【实测】：profile 里 `v41_merge_pre` / `v41_merge_post` 各 **2622 次 ÷ 67 步 = 39.1/步**
（= 层数），且 `allReduce gid=097` 恰好减半。

### 4.1 ⚠️ 但它在 N=16 崩【实测】

`noprof.py 1,8,16` 跑到 N=16 阶段时引擎崩（前 87 个请求 200 OK，之后 16 路 500）：

```
model.py:884 → dsa_v41.py:3762 _v41_dcp_merge_attention
  → dsa_v41.py:985  _mk.merge_pre(output, lse, _ori_lse_f32, _pk)
  → v41_merge_kernel.py:181  s = torch.npu.current_stream().npu_stream
RuntimeError: ... ERR00100 PTA call acl api failed
```

（该行是**报丧点**：设备侧先抛错、下一次主机侧流查询才报出来 —— 本文件已多次记录这一模式。）

⇒ **【未确认】** 崩溃是否由 T=128 触发（融合 kernel 已知 T≥20 有 1 ULP 级偏差，
`V41-DSPARK-OPT-MERGEKERNEL-SHADOW-20261001.md` §10）。**当前状态：默认关，不建议在
N≥16 的生产配置上打开**；若要转正必须先加 T 上限门控或修 kernel。

## 5. 优化 2：slot-mapping 融合的"部分融合"（`V41_SLOT_MAP_FUSED=on`）

**原实现的问题**【实测+源码】：`_v41_fused_precheck` 用**配置值** `bt.dcp_world_size > 1`
做判据 ⇒ DCP8 下**任一组不合格就整步回落**。但实际上：
* 复制态组（SWA / compressor）的 `bt.effective_dcp_world_size == 1`，
  走的**就是**融合 kernel 所复刻的那条 Triton 分支（`TOTAL_CP_WORLD_SIZE == 1`）；
* 结果：10.8 次/步的 kernel 启动一次都没省，profiler 里紧随其后 **1.302 ms/step** 空闲。

**改动**（`worker/block_table.py`，`[V41-SLOT-MAP-PARTIAL]`）：
1. 逐组判据移到 `_v41_fused_group_reason()`，**改用 `effective_dcp_world_size`**；
2. `_v41_fused_groups()` 只收合格组，新增 `_v41_fused_rest()`；
3. `_v41_try_fused()` 对合格组发一次 2D grid，**不合格组用原路径补算**（不再整步回落）；
4. `max_num_batched_tokens` 取合格组的 **max**（原实现固定用 `idx[0]`）。

**正确性门禁**【实测】：以 `V41_SLOT_MAP_FUSED=verify` 起服（融合结果与原逐组路径
**逐元素**比对），跑含 16000 token prefill 的回归 →
**零不一致、零回落日志**（无 `回落到逐组路径`、无 `部分融合` ⇒ 全部组都进了 grid）。
回归：`17×23→391`、`count n=2000/16000`、长针 `904/8000→Q7`、
**并发 8 路与单流参考逐字一致 8/8**。

**性能【实测】**（同脚本同配置，无 profile；表内为 `noprof.py` 的差减法 ms/step）：

| 并发 | 只开 slot-map | 基线 | 变化 |
|---:|---:|---:|---:|
| 1 | **42.33** | 43.33 | −2.3% |
| 8 | **93.66** | 95.75 | −2.2% |
| 16 | **131.24** | 130.80 | ±0（且**不崩**） |

**设备侧证据（比墙钟更硬）**【实测】：同一实例再采一次 profile 做归因对比
（`tools/idle_report.py`，结构常量 `QuantBatchMatmulV3==226`）：

| | 基线 | 开 slot-map |
|---|---:|---:|
| span | 43.39 | **42.10** |
| idle | 13.72 | **12.00** |
| idle ≥ 50 µs | 7.89 | **6.42** |
| `_compute_slot_mapping_kernel` 归因空闲 | **1.302（TOP-3）** | **已从 TOP-18 消失** |

### 5.1 默认值

`serve_a2.sh` 现在**只在 `KV_ARGS_EXTRA` 含 `--decode-context-parallel-size N`（N∉{0,1}）时**
把 `V41_SLOT_MAP_FUSED` 默认成 `on`；DCP=1 / 无 DCP（A2 生产）行为不变。
显式传 `V41_SLOT_MAP_FUSED=verify|on|0` 优先。

## 6. 尝试过但**无收益**的一项：DCP slot-mapping 的按步缓存

**假设**：`_compute_dcp_slot_mapping` 的链里，前 9 个算子（`positions // vpbs`、
`% vpbs`、`//interleave % world == rank`、`local_off`）**只依赖 positions 与
(vpbs, interleave, world, rank)**，与 group 无关 ⇒ 同一步内多个 DCP group 可只算一次；
`req_indices`（`arange` + 差分 + `repeat_interleave`）同理。

**实现**：`[V41-DCP-SLOT-HOIST]`，按 `(步序号, positions.data_ptr(), 参数)` 缓存。

**实测**（N=1 三次）：43.13 / 43.15 / 41.96 —— **与基线 43.33 在噪声内**（N=1 的
run-to-run 离散就有 ±1.5%）。
⇒ **已回退**（不留未验证的复杂度）。

## 7. 关于测量精度的一个诚实说明

本机 N=1 的 run-to-run 离散约 **±0.6 ms/step（±1.5%）**。因此：
* 「−2.0%」「−2.3%」这类**墙钟**结论都**处在噪声边缘**，单次测量不足以定论；
* 本文对 slot-map 的结论建立在**设备侧归因**上（span −1.29 ms、idle≥50 −1.47 ms、
  目标算子从 TOP-18 消失），这是可复现的、更强的证据；
* 融合 merge kernel 的结论同样有设备侧支撑（`v41_merge_pre/post` 39.1 条/步、
  `allReduce gid=097` 78.34 → 39.13 条/步）。

## 8. 下一步（按收益排序）

| # | 目标 | 依据 | 预期 |
|---|---|---|---|
| 1 | **把 `_compute_dcp_slot_mapping` 的逐元素链也做成 Triton kernel**（在 `_compute_slot_mappings_multi_kernel` 里加 `TOTAL_CP_WORLD_SIZE > 1` 分支） | 归因显示 ViewCopy 1.739 + Add 1.454 + Cast 0.572 + Fill 0.439 + Sub 0.396 + FloorMod 0.283 ≈ **4.9 ms/step 的 ≥50 µs 停顿**，上下文全是这条链的算子（FloorDiv/Cast/Mul/Sub/ZerosLike/SelectV2/BroadcastTo） | **~10%** |
| 2 | 修融合 merge kernel 在 T≥128 的崩溃（或加 T 上限门控） | §4.1；T≤64 已验证可用 | ~2%（N≤8） |
| 3 | `hcom_allReduce gid=503`（TP/EP）92 条/步、3.34 ms/step | §3；与 DCP 无关，A2 上也有 | 需上游配合 |
| 4 | `allGather gid=374`（q gather）39.6 条/步、0.66 ms/step | §3 | DCP 结构自带 |



## 9. 发布版最终验收（2026-10-02，`dcpcap_1002_123902_ship`）

配置 = 本机 A3 DCP8 默认（`serve_a2.sh` 因 `--decode-context-parallel-size 8`
自动把 `V41_SLOT_MAP_FUSED` 置 `on`），`V41_DCP_MERGE_KERNEL` **未开**。

| 并发 | ms/step | A | ms/token | 聚合 tok/s | 基线同口径 |
|---:|---:|---:|---:|---:|---:|
| 1 | 42.37 | 2.76 | 15.35 | 61.3 | 43.33 |
| 8 | 96.32 | 2.25 | 6.35 | 157.4 | 95.75 |
| 16 | 133.81 | 2.24 | 4.74 | 211.0 | 130.80 |

**★ 诚实结论**：N=1 的 −2.2% 与基线同向；**N=8/N=16 的墙钟差异（+0.6%/+2.3%）
完全落在本机 ±1.5% 的 run-to-run 离散内**，因此
「墙钟层面这条优化省了多少」**无法用单次测量判定**。
可以判定的是**设备侧**：`_compute_slot_mapping_kernel` 的 ≥50 µs 停顿
**1.302 → 0 ms/step**、该 kernel 的 10.8 次/步启动被合成 1 次 2D grid、
decode span 43.39 → 42.10 ms/step。这三项都是同实例同配置的可复现差异。

**精度回归（发布版，全 PASS）**：`17×23→391`、`count n=2000/16000`、
长针 `904/8000→Q7`、**并发 8 路与单流参考逐字一致 8/8**。

### 9.1 一个测试陷阱（已修）

`regress2.py` 最初用 `ignore_eos=True` 做并发一致性判据，实测在真权重下
**给出 3–4/8 的假失败**（强制越过 EOS 后，续写对批内数值差异极其敏感）。
改用自然停止的短输出后稳定 **8/8**。**不要再用 `ignore_eos` 做一致性判据。**

---

# 第三部分：DSpark 的 13.2 ms 落在哪（2026-10-02 深挖）

## 10. DSpark 增量的定标（用我们自己的历史实测）

| 配置 | ms/step | 出处 |
|---|---:|---|
| 我们 DCP8 `SPEC=0` | **29.21** | `docs/V41-DSPARK-BATCH1-ROOTCAUSE-20261001.md:224`（今日基线口径） |
| 我们 DCP8 `SPEC=1` | 42.37 | §9 实测 |
| **⇒ DSpark 增量** | **+13.16 ms/step（+45%）** | |
| A2 DCP1 `SPEC=0 → SPEC=1` | 24.35 → 32.68 = **+8.33（+34%）** | `CED-PD-DYNAMIC-SPEC-20260926.md` §11.4 |

⇒ 主战场是 **DSpark 的 13.16 ms**，其中约 4.8 ms 是 DCP×DSpark 的交互
（DCP 特有部分），其余是 DSpark 固有开销。

## 11. 步的结构：真正的洞在**尾部 6 ms**（不是散布的气泡）

把干净 decode 步按开始时间切 20 桶（本文件 `~/tmp/timeline.py`、`~/tmp/eagerphase.py`）：

| 阶段 | 设备占用 | 说明 |
|---|---:|---|
| 0–20% | 17–19% | 输入准备（eager） |
| **25–90%** | **121–128%** | 多流并行、**算力饱和**（不是瓶颈） |
| 80–100% 的 stream 47 | — | **197 个微型算子的爆发** |

**stream 47 的算子分布【实测】**（单步 203 条）：

| 桶（时刻%） | 80 | 85 | 90 | 95 |
|---|---:|---:|---:|---:|
| stream 47 条数 | **98** | 19 | 29 | **51** |

构成：`Cast 33 / Fill 24 / IndexCheck 19 / Index 19 / Add 10 / Sub 9 /
BroadcastTo 8 / SelectV2 7 / Mul 7 / GatherV3 6 / FloorDiv 6 / FloorMod 6`。

**尾部序列实测**（`~/tmp/tailseq.py`，从 90.5% 起）清楚地显示这是 **DSpark 的 draft 循环**：

```
IndexCheck → Index → MatMulV2 → Transpose → GatherV2(Embedding)
  → Add → ArgMaxV2 → Cast → GatherV2 → MatMulV2 → Add → ArgMaxV2 → Cast → …
```

即 **K=7 个 draft step 的 Python 循环**，每轮 ~8–10 个微型算子，
外加 verify/sample 的收尾。

## 12. ★ 定量：host 下发是瓶颈，不是算力

| 指标 | 值 | 出处 |
|---|---:|---|
| 每步算子总数 | **3972 条** | `~/tmp/tinyops.py` |
| 其中 ≤64 元素的**微型算子** | **459.5 条/步（12%）** | 同上 |
| 这些微型算子的设备时间合计 | **~2.4 ms/步** | 同上 |
| `host,node,launch` | **46 081 次 / 64 步 = 720/步**，共 4.59 ms/步 | `api_statistic` |
| `host,acl,aclrtLaunchKernelWithHostArgs` | 45 725 次 = 714/步，3.44 ms/步 | 同上 |
| 设备空闲（并集补集） | **12.0 ms/步（28.5%）** | `idle_report.py` |

⇒ 459 个微算子 × ~25 µs/条的 Python+下发代价 ≈ **11.5 ms/步**，
与实测空闲 12.0 ms 吻合。**瓶颈是"把成千上万个微型算子逐个交给设备"，不是算力。**

## 13. 两条被排除的路径（不要重走）

### 13.1 `TASK_QUEUE_ENABLE=2` —— 与图捕获不兼容【实测】

`TASK_QUEUE_ENABLE=2`（二级流水，本可缓解下发瓶颈）在本 CANN/torch_npu 上
**直接拒绝图捕获**（容器内子进程冒烟测试，未动服务）：

```
TASK_QUEUE_ENABLE=1 → CAPTURE_OK, REPLAY_OK 288.5 us/replay
TASK_QUEUE_ENABLE=2 → CAPTURE_FAIL:
  RuntimeError: Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture,
  please export TASK_QUEUE_ENABLE=1/0.  ERR00007 PTA feature not supported
```

而我们的 decode 是 `FULL_DECODE_ONLY` 图 ⇒ **不能用**。
（复现：`docker exec <ctr> bash -lc "TASK_QUEUE_ENABLE=2 python3 /tmp/tq_smoke.py"`）

### 13.2 `--no-async-scheduling` —— 破坏长上下文精度【历史实测，再次确认】

`docs/V41-DSPARK-BATCH1-ROOTCAUSE-20261001.md` §6.4：加回该 flag 后长上下文精度立即崩
（`17×23`/T=904 仍 PASS，但 4 个长上下文用例全 FAIL）⇒ **交付配置必须不带**。

## 14. 下一步（唯一量级足够的杠杆：把 draft 循环搬进图/内核）

| # | 动作 | 依据 | 预估 |
|---|---|---|---|
| 1 | **把 DSpark 的 K 轮 draft 循环从 Python 循环改成图内/单内核** | §11/§12：197 条/步的尾部爆发、~30 µs/条 | **6–10 ms/step** |
| 2 | 把 `_prepare_inputs` 的 eager 段一并图化 | §11 的 0–20% 也偏低 | 2–3 ms/step |
| 3 | 减少 `ScatterNdUpdateSk`/`MoeTokenUnpermute` 等每层 1 次的微型算子 | 各 40–58 条/步 | 需层内融合 |

**共同前提**：`V41_CED_DYNAMIC_SPEC_FULL_GRAPHS` 那套「每个 K 各捉一组图」的基础设施
（`experimental/ced/core_config_dynamic_sd_gate.patch` +
`core_model_runner_dynamic_spec.patch` + `patch_cudagraph.py`）**已经存在**，
把 draft 循环纳入它的图是关键路径。

---

# 第四部分：一个被实测否证的归因（2026-10-02 晚）

## 15. 假设：`dsa_v41.build()` 的"每组各算一遍"

**发现**：`dsa_v41.py::build()` 里读两个 kwarg 当"按步缓存"用 ——
`kwargs["common_v41_metadata"]` / `kwargs["common_v41_batch_metadata"]`，
而**全仓没有任何地方传它们**（`grep` 只在本文件命中）⇒ 每次都拿到新建的空 dict ⇒
`_build_batch_metadata`、`get_cos_and_sin_dsa`(rope)、compressed lengths、
`_publish_task` 的常驻缓冲、`slot_key` 槽位映射 **全部每组各算一遍**（约 11 组/步）。

**旁证**：一步的**前 8 ms** 里 stream 47 上有 **381 条**微型算子
（设备合计仅 0.729 ms、覆盖 9%），且 381 ≈ 35 × 11 组 —— 与"每组各算一遍"吻合。

**实现**：把这两个 dict 挂在 `common_attn_metadata`（runner 每步新建）上，
键含 `id(common)` ⇒ 天然按步作用域、跨请求必然失效。开关 `V41_DCP_STEP_CACHE`。

## 16. ★ 实测结果：**证伪**

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 缓存命中（`[V41-STEPCACHE]` 日志） | — | ✅ 8/8 worker 各命中一次 |
| 前 8 ms 窗口算子数 | 381 | **395（没减少）** |
| 该窗口设备时间 | 0.729 ms | 0.755 ms |
| idle 中位 | 12.78 ms | **12.83 ms** |
| span 中位 | 42.92 ms | 43.14 ms |
| N=1 / N=8 / N=16 ms/step | 42.37 / 93.06 / 130.72 | 42.70 / 93.70 / 135.67 |

⇒ **缓存命中是真的，但那 381 条算子不是来自这里。归因错误。**

**处置：已回退。**

## 17. 这个负结果告诉我们什么（对下一步很重要）

1. 那 381 条（前 8 ms、stream 47、纯 elementwise）**来自 `dsa_v41.py` 之外的代码**
   —— 最大候选是 **`vllm_ascend/worker/model_runner_v1.py::_prepare_inputs`**
   （该文件**不在我们的 overlay 里**，我们没有它的挂载）。
2. **"减少算子数 ⇒ 变快"这个因果链在本实例上未经证实**：
   前 8 ms 的 381 条算子设备只占 0.729 ms，把它们全部删掉最多省 0.7 ms ——
   9% 的覆盖率意味着**设备在等，但等的不一定是这些算子的下发**。
   下一步必须先证明"谁在让设备等"（例如用 `aclrtSynchronizeEvent` 的
   调用栈/profiler 的 host 时间轴），再动手。
3. 仍然成立的事实：`common_v41_metadata` 缺失是**上游语义未生效**的真缺陷
   （4 处缓存全部失效），只是它的代价被别的东西掩盖了。若将来这些缓存成为
   热点，修法是现成的（本节的 diff）。

## 18. 工具清单（本次新建，已放在 a3-21 `~/tmp/`）

| 脚本 | 用途 |
|---|---|
| `tinyops.py` | 微型算子（≤N 元素）条数与设备时间 |
| `timeline.py` / `cover.py` | 一步按桶/按 ms 的设备覆盖率（找洞） |
| `eagerphase.py` / `headops.py` | 某 stream / 某窗口的算子构成 |
| `idledist.py` | 逐步 idle 分布（均值 vs 中位） |
| `waitattr.py` / `streambd.py` / `streamgap.py` | 前等待归因 / 按 stream 拆解 |
| `opcount.py` / `stepgaps.py` / `seqstep.py` | 算子计数 / 单步 gap / 单步序列 |
| `tq_smoke.py` | `TASK_QUEUE_ENABLE` 与图捕获的兼容性冒烟 |
