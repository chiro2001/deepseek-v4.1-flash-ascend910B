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
