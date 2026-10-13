# TP8 下一个性能前沿：瓶颈画像与可做的六件事（2026-10-03）

> 承接 [`V41-TP8-DSPARK-MULTISTREAM-PERF-20261003.md`](V41-TP8-DSPARK-MULTISTREAM-PERF-20261003.md)
> （8 并发 315.3 tok/s = A2 基线 1.54×，目标已达成）。本文回答**"接下来还能提什么"**，
> 用**设备侧实测**而不是猜测来排序。所有数字标 **【实测】**/**【推断】**。

---

## 0. 先看画像：瓶颈到底是什么

用 `~/tmp/devbusy.py`（本次新建，区间并集求真空闲）对 **TP8 + SPEC=1 K=7、N=8** 的
torch_npu profile 做整体统计【实测】：

| 量 | 值 |
|---|---:|
| 窗口内算子数 | **2750 条/step**（有非零耗时） |
| 窗口 span | 65.1 ms/step |
| **设备并集 busy** | **53.7 ms/step（82.5%）** |
| **真空闲** | **11.4 ms/step（17.5%）** |

### 空闲是**弥散**的，没有单一元凶【实测】

把每段空闲归因给"它前面最后结束的那条算子"，TOP-10 加起来才 4.6 ms：

| 空闲前的算子 | ms/step | 条/step |
|---|---:|---:|
| QuantBatchMatmulV3 | 1.4 | 50.8 |
| DynamicQuant | 0.8 | 83.1 |
| hcom_broadcast_ | 0.8 | 2.0 |
| Cast | 0.6 | 27.0 |
| InplacePartialRotaryMul | 0.6 | 67.7 |
| MoeInitRoutingV3 | 0.5 | 42.2 |
| 其余（RmsNorm / alltoallv / MoeTokenUnpermute / MatMulV2 / …） | 各 ≤0.4 | — |

空档直方图：`50–100 µs` 段 2.31 ms/step、`100–500 µs` 段 5.25 ms/step，
**没有 >1 ms 的整块洞**。

> **★ 这条直接否掉了"DCP8 那套 host 下发优化可以平移过来"的想法。**
> DCP8 是 `idle 12.0 ms/step（28.5%）` 且 TOP-3 占 4.9 ms（slot-mapping 链），
> 抄掉那一条就赚 1.3 ms；**TP8 只有 17.5%，且 TOP-10 分摊**。
> ⇒ **TP8 要提速，只能"减少设备工作量"，不能"填气泡"。**

---

## 1. 设备工作量花在哪（N=8、K=7，各流累加，按 HcPre/86 归一到每步）【实测】

| 项 | ms/step | 占累加 | 说明 |
|---|---:|---:|---|
| **`hcom_allReduce`（gid=503，TP/EP）** | **17.5** | **21.6%** | **87.8 次/步**、次均 199 µs；= 43 层 × 2 |
| GroupedMatmulSwigluQuantV2（MoE gmm1） | 7.5 | 9.3% | 43 次/步，次均 174 µs |
| QuantBatchMatmulV3 | 4.0 | 4.9% | 226 次/步，次均 17.7 µs |
| GroupedMatmul（MoE gmm2） | 3.9 | 4.8% | |
| MatMulV2 | 3.6 | 4.5% | 117 次/步 |
| **HcPre** | **4.1** | **5.0%** | 86 次/步，次均 47 µs |
| SparseFlashMla（主注意力） | 2.8 | 3.5% | |
| RmsNorm | 1.9 | 2.3% | 140 次/步 |
| **HcPost** | **1.7** | **2.1%** | 86 次/步 |
| InplacePartialRotaryMul | 1.3 | 1.6% | |
| ScatterNdUpdateSk | 1.3 | 1.6% | |
| DynamicQuant | 1.0 | 1.2% | |
| MoeInitRoutingV3 | 0.9 | 1.1% | |
| MatMulV3 / SparseFlashMlaMetadata / Cast / Add / ViewCopy / … | 各 0.4–1.0 | | |
| **累加合计** | **81.0** | 100% | 并集 busy 只有 53.7 ⇒ 约 27 ms 是跨流重叠 |

**读法**：`allreduce` 一项 = 并集 busy 的 **32–41%**。
**它是唯一"体量够大到值得动结构"的目标。**

---

## 2. ★ 已实测证伪的一条：TP8 → DP2TP4（不要重走）

§1 的推论是"allreduce 占并集 busy 的 32–41% ⇒ 把 TP 域从 8 缩到 4 能砍掉它"。
**这条在 2026-10-03 实测被证伪，而且是双重证伪**【实测】：

| 项 | TP8 / DP1（交付档） | **TP4 / DP2** |
|---|---:|---:|
| KV 总容量 | 2,823,516 | **4,612,730（1.63×，是加分项）** |
| 单流 tok/s | **109.5** | **60.1 / 64.3 / 67.8（−40%）** |
| 4 并发总吞吐 | **225.4** | 128.4 / 209.4 / 215.5（离散极大） |
| 8 / 16 并发 | 315.3 / 398.6 | **没测到 —— 跑压测时引擎崩了** |
| 起服耗时 | ~12 min | **14 min+**（136 个 static kernel × 2 个 DP 引擎） |

崩溃签名（DP0/TP2 rank）：

```
RuntimeError: npuSynchronizeDevice: NPUStream.cpp:714
  AclrtSynchronizeDeviceWithTimeout, error code is 507011
[ERROR] ERR00100 PTA call acl api failed → Model execution failed
→ ApiServer_0 died with exit code None → 整个进程树退出
```

**为什么会输（机理）【推断】**：
1. **DP 让单请求只用一半硬件** —— TP4 时 attention/embedding/稠密权重只切 4 份，
   每个 rank 要算 2× 的量；而请求只会落到**一个** DP 副本上 ⇒ 单流 −40% 是必然。
2. **省下的不是那个 allreduce** —— 本实例的 allreduce 是 **gid=503 的 TP/EP 组**，
   MoE 的 EP 仍然是 8（= DP2 × TP4）⇒ 通信域根本没变小，
   变小的只是 attention/dense 那部分。§1 那条"缩 TP 域"的推论**把两个域混为一谈**了。
3. **KV 容量反而变大（1.63×）** 说明"DP 复制权重会挤掉 KV"这个先验**在本模型上是错的** ——
   真正决定总容量的是"每 rank 覆盖 1M 上下文的 KV 需求"（TP4 与 TP8 相同），
   所以这个形态**值得留给容量受限的场景**，但**不是性能路线**。

> **附带的真缺陷**【实测】：`VLLM_ENGINE_READY_TIMEOUT_S` **从来没有被 `serve_a2.sh` /
> `serve_a3.sh` 透传进容器**（全仓 `grep` 命中 0 处），而 vLLM 在超时时明确提示用这个变量。
> 后果：DP 多引擎 + 冷 static kernel 编译（>600 s）**必然**被默认超时杀掉，
> 且错误信息指向"weight loading 慢"而不是真正原因。已在 a3-21 的 `serve_a2.sh` 里
> 补一行透传（`${VLLM_ENGINE_READY_TIMEOUT_S:+-e ...}`），**待并入 main**。

---

## 3. 六件可做的事（按"收益 ÷ 代价"排序）

| # | 做什么 | 预期 | 依据 | 代价 | 风险 |
|---:|---|---|---|---|---|
| **1** | **动态 K：N≥12 时关 spec** | **+8.9% @N=16** | 同机实测 SPEC=0 **434.1** vs SPEC=1 K=5 **398.6**；收益随并发单调衰减（2.12/1.54/1.36/1.14/**0.92**×），交叉点 N≈12 | 接线 + 2 次重启 | 中（dynamic-spec 路径要求全量正确性探针） |
| **2** | **MoE 通信模式 A/B**（`V41_MOE_COMM_ALLGATHER` 等） | 未知，上限 ~10 ms/step | AllGather 模式下 MoE 输出**必须** allreduce（这正是 87.8 次里的一半）；换 alltoall/combine 可把求和折进通信 | 2 次重启 | 中 |
| **3** | **HcPre + HcPost 融合** | **5.8 ms/step（7.2%）** | §1；且历次审计**从未优化过这两项**（`CED-PD-DSPARK-PROFILING-20260926.md` §4 把它列为"最大未开发区"） | 写融合 kernel（AscendC/Triton） | 中 |
| **4** | **RmsNorm + DynamicQuant 融合到 W4A8 路径** | 0.25–0.40 ms/step | 融合算子现成（`dsa_v1.py:1688/:1815` 已在 W8A8 用），W4A8 因 `_is_w8a8_dynamic()` 判断没接上 | 小改 | 低 |
| **5** | **缩小 TP 域（TP8→TP4）** | **−40%（已证伪）** | §2 | — | — **不要做** |
| **6** | 追 DCP8 那套 host-下发优化 | **≈ 0** | §0：TP8 idle 只 17.5% 且弥散 | 大 | — **不要做** |

---

## 4. 为什么剩下几条里"动态 K"最值钱

1. **它是唯一"零风险 + 收益已被同机实测锚定"的一条**：434.1 vs 398.6 是同一天、
   同一台机、同一套脚本测出来的，不需要新假设。
2. **它同时修掉一个用户体验问题**：N≥12 时 spec 在数学上就是净亏的
   （`T = N(1+K)` 涨得比产出快），关掉它既提速又省算力。
3. **基础设施已经存在**：`SP_SCHEDULE`（按请求数切 K）+ 两个补丁文件
   （`core_model_runner_dynamic_spec.patch` / `core_config_dynamic_sd_gate.patch`）
   已在 `experimental/ced/`，目前只被 `V41_CED_ROLE=decode` 的脚本门挡住。

**已知的坑**：dynamic-spec 路径要显式承担"关掉上游 PIECEWISE 降级保护"的风险，
**主要失效模式是静默算错** ⇒ 必须跑完整正确性探针（144K/1M 四针 + 并发一致性），
不能只看"起来了 + ms/step 正常"。

---

## 4. 一条不能忘的纪律

本文所有 device 数字都来自 **SPEC=1 K=7、N=8** 的那次 profile。K=5 的
`T` 从 64 降到 48 行，各算子的**占比会变**（allreduce 与 MoE 的按行成本下降，
固定开销项如 HcPre 的占比上升）⇒ **改配置后要重采一次再排序**，
不要把这张表当成 K=5 的比例。

---

## 5. 复现

```bash
# 设备画像（任何一次 profile 之后都能跑，只读）
docker run --rm \
  -v ~/cedpd-repo/results/<run>/prof:/prof:ro \
  -v ~/tmp/devbusy.py:/p.py:ro \
  --entrypoint bash quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
  -lc 'F=$(ls /prof/*rank0*/PROF_*/mindstudio_profiler_output/op_summary_*.csv|head -1); python3 /p.py $F'

# 空闲归因（需先把 QBMV3_PER_STEP 改成该配置的结构常量：SPEC=0→208、K=7→226）
python3 ~/tmp/idle_report.py <mindstudio_profiler_output 目录> --thresh 50 --qbmv3 226
```
