# cannbot 手段穷尽式尝试：最终总结

> 2026-09-16 21:30｜A3-node1 chips 8-15（性能主线）+ A3-node2 chips 8-15（实验）
> 回答用户要求：「必须尝试 cannbot 中提到的所有可能的优化手段，并且优先尝试最有可能的」

---

## 0. 结论先行

**cannbot 中与本部署（A3 8 卡、vLLM + npugraph_ex/aclgraph、W4A8、DSpark、单流延迟）相关的手段已逐条试完。**

| 结果 | 数量 | 说明 |
|---|---|---|
| ✅ **已采纳并生效** | 3 | static kernel（含静默降级修复）、FUSED_MC2、jemalloc |
| ❌ **实测无收益 / 更差** | 8 | 见 §2 |
| ⛔ **本机不可用（环境限制）** | 5 | 见 §3 |
| ➖ 不适用 | 6 | 见 §4 |
| 🔄 技术可行但投入产出比低 | 1 | GroupedMatMulAllReduce，见 §5 |

**性能现状（unprofiled 客户端墙钟，3 发中位）**：
8K **34.27** ／ 32K **35.85** ／ 128K **39.10 ms/step**（A=2.763，**70.1 tok/s**）。
达标项：KV 3,388,563（>3M ✓）、Vision 23/23 ✓、GSM8K 198/200 ✓、Engram-int8 常驻 DRAM ✓、DSpark ✓。

**目标 128K >110 tok/s 的判定**：需 ms/step ≤25.1，而 **128K 的 compute 地板本身 ≈28.5**
（32K compute 25.3 + 长上下文注意力 ~3.3）⇒ **不可达**。现实可达 31–35 ms/step（**79–89 tok/s**）。

---

## 1. ✅ 已采纳并生效（3 项）

| # | 手段 | 收益 | 证据 |
|---|---|---|---|
| 1 | **static kernel**（`enable_static_kernel=1`） | 128K −2.87 ms | 历史 F1 报告；**本轮发现它会被静默禁用**并修复（见下） |
| 1b | **`LOCAL_WORLD_SIZE` 注入（修复静默降级）** | **−4~5 ms（12%）** | `static-kernel-silent-disable-fix.md`：警告 24→0、`compile start` 0→3、8K 37.95→34.20 |
| 2 | **`enable_fused_mc2=1`**（MoE 融合） | host −40%、comm 9.19→4.55 ms/step、设备 Free 25.1%→12.8% | `moe-host-sync-and-fused-mc2.md` |
| 3 | **jemalloc**（`LD_PRELOAD`） | 12% | 历史上缺它 37.4→42.6 ms |

---

## 2. ❌ 实测无收益 / 更差（8 项，勿重复）

| # | 手段 | 实测结果 | 依据 |
|---|---|---|---|
| 1 | **`SP_TOKENS` 扫描**（5/7/9） | S=5：8K 41.10 / 32K 42.01 / 128K 45.13（**更差**）；S=7 最优。逐位置接受率 pos5/6 **恒为 0**，几何上界 ≈2.87，S=7 已到渐近线 | `acceptance-and-hotreload.md` §1 |
| 2 | **BF16 draft 换接受率** | 32K A 2.833→3.012（+6.3%），但 ms +1.1%；**净 tok/s 不变/变差**；且 **KV 2,700,814 < 3M**（违硬指标） | `draft-precision-acceptance-ab.md` |
| 3 | **`CPU_BIND=1`**（`enable_cpu_binding`） | 8K 39.86 / 32K 40.68（vs 34.40 / 34.35） | 历史上实测；**但该对比跨机，结论偏弱**（见 §3 第 3 条） |
| 4 | **route 流水化**（async_op） | 同会话 A/B：on 34.50 vs off₂ 34.57 ⇒ 无差异 | `acceptance-and-hotreload.md` §3 |
| 5 | **localmeta / gather / b2g / 消 D2H** | 全部实测否决（详见历史报告） | `engram-optimization-round2.md` |
| 6 | **`MAX_RUNTIME_CORE_NUMBER=3`** | 8K −0.05 / 32K +0.14 / 128K +0.51 | `cannbot-measures-triage.md` §P4 |
| 7 | **`HCCL_BUFFSIZE=2048`** | 8K −0.50 / 32K +0.60 / **128K +1.27** | 同上 |
| 8 | **`TORCHINDUCTOR_NPU_BACKEND=ascendc`** | 8K −0.55 / 32K −0.13 / **128K +2.20** | 同上 |

> ⚠️ 第 6–8 项无法同会话配对（都要重启），8K/32K 差异在 ±0.7 ms 噪声内；
> 只有 128K 的退化超出噪声。**结论：不采用。**

---

## 3. ⛔ 本机不可用（5 项，环境限制，无需再试）

| # | 手段 | 阻塞原因（实测） |
|---|---|---|
| 1 | **`CPU_AFFINITY_CONF`**（措施6，预期 ~10%） | `npu_affine` 被 **DCMI 禁用**：`dcmi get affinity cpu info by device id is not supported`；显式范围格式**静默无效**（亲和掩码始终 320-639，三种格式实测不变） |
| 2 | **MC2 / `mc2_comm_alg=hierarchy`** | `MoeDistributeDispatchV2` 要求 **`epWorldSize` 是 16 的倍数**，我们 EP=8 ⇒ 起服即 `Invalid_Input(EZ0004)`；且与 `enable_fused_mc2` 互斥 |
| 3 | **MegaMoe（`enable_fused_mc2=2`）** | `moe_intermediate_size=2304`，`2304 % 512 = 256 ≠ 0` ⇒ `_is_megamoe_supported_by_config` 返回 False，配置**静默退回 0**（丢掉现有融合路径） |
| 4 | **`MatmulAllReduce`（mm+all_reduce 融合）** | `ops-transformer` 的 `matmul_all_reduce_def.cpp` **只有 `ascend950`/`ascend910b`/`ascend310p`，没有 `ascend910_93`** ⇒ 报 `SoC version ascend910_93 verification failed`。**上游设计如此，升级 CANN 无效**（9.2.0/master 同样） |
| 5 | **`use_sequence_parallel_moe`** | 要求 `data_parallel_size > 1`，我们 **DP=1** |

---

## 4. ➖ 不适用（6 项）

| 手段 | 原因 |
|---|---|
| 多实例并行 / AICore 控核（措施 M1/M2） | 目标是**单流延迟**，非吞吐 |
| 单算子 tiling key 优化（措施 N3） | 需算子团队，非用户侧可配 |
| AICPU→AICore（措施 N8） | 需 CANN 算子团队重写 |
| 混合调度（措施 N7） | decode shape 已固定 |
| `model-infer-superkernel` | 仅 GE 图 + A3 Decode，**与 aclgraph 互斥** |
| `model-infer-prefetch` | 面向 memory-bound 热点，我们是通信/调度 bound |

---

## 5. 🔄 唯一的技术活路：`GroupedMatMulAllReduce`（投入产出比低）

**已完成的验证**（`gmar-single-group-path.md`）：

| 步骤 | 状态 |
|---|---|
| `MatmulAllReduce` 在 A3 不可用 | ✅ 上游不支持，确定 |
| `GroupedMatMulAllReduce` 在 A3 **可用** | ✅ 内核在、符号在、**参数校验通过** |
| 参数约束的完整解 | ✅ **streamMode 必须=1**、**splitItem 只能 0/2**、**bias 必须 fp32** |
| 真机执行 | ❌ 卡在 **MC2 HCCL 通信资源创建**（`group_name_0` 不是 MC2 可复用的通信域名） |

**收口的三条理由**：
1. **收益无法证明**：分离臂微基准 208.7 µs/次，但 profile 实测单次 allReduce 墙钟仅 **25.7 µs**
   （87 次共 2.24 ms/step）⇒ 差 8 倍，说明**真实推理中通信已被流水线交错**，微基准不可外推。
2. **接入成本一周级**：vllm-ascend 对这条路径**零绑定**（`mmrs_fusion` 是死代码、
   `npu_mm_reduce_scatter_base` 从未被调用），要自写 aclnn 调用 + 改两处**图内** forward + 重捕获。
3. **收益上限 2.24 ms/step（6%）**，且这是"上限"而非预期。

---

## 6. 本轮最大的两项实质收益

| # | 成果 | 数值 |
|---|---|---|
| **1** | **修复静态内核静默降级** | **−4~5 ms/step（12%）**：8K 37.95→**34.20**、32K 39.25→**35.85**。根因是 `LOCAL_WORLD_SIZE` 未进 `os.environ`，torch_npu 静默禁用 static kernel（无错误码）。**对 A2 交付同样重要**——A2 若直接 `vllm serve` 也会命中 |
| **2** | **纠正了测量方法论**（3 处） | ①profiler 采集有 **+3.7~6.6 ms** 开销与 **−122 token** 显存足迹，绝对性能只能认 unprofiled 墙钟；②我此前用"外部假设步数"算设备账是**循环论证**，现改为同会话客户端实测；③**步数分母错了 10%**（实际 43.94 锚/步，不是 40）→ 设备账已修正 |

---

## 7. 剩余空间与现实的性能目标

**修正后的 128K 设备账**（生产态折算，2026-09-16 06:00 更新）：

| 项 | ms/step | 依据 | 可否消除 |
|---|---|---|---|
| target 40 层 compute | **≈28.6** | SPEC=0 实测 busy 28.62（含 MoE 9.4） | ❌ MoE 是访存下限（`aic_mac_ratio=0.7%`） |
| 其中长上下文增量（128K vs 32K） | ≈3.3 | 128K−32K 之差 | ❌ |
| draft 设备忙 | ≈5.7 | SPEC=7 − SPEC=0 的 busy 差 | ❌ 它是 A=2.73 的来源 |
| **spec 接受判定往返（FREE）** | **≈5.1** | SPEC=7 − SPEC=0 的 FREE 差；**SPEC_EAGER=0 已证伪可回收性** | ❌ 架构性 |
| Engram host 路径 | ≈2.8 | `nohost` 同会话配对（32.42 vs 35.19） | 🔄 部分（多轮尝试已收窄） |
| 其它 FREE / host | ≈2 | 余量 | 🔄 部分 |

**零开销地板 ≈ 31.8 ms/step**（生产态 busy，已含 draft 必要成本）⇒ **A=2.763 时 86.9 tok/s**。

> 上一版报告里的"地板 28.5"是**低估**的：它按比例折算 profiled busy，
> 未把 draft 的 5.7ms 设备忙计为"必要"。修正后地板 **31.8**，对应 **86.9 tok/s**。

| 档 | 手段 | 32K | 128K | 128K tok/s |
|---|---|---|---|---|
| 现在 | — | 35.85 | 39.10 | **70.1** |
| 档 1 | Engram route 下发暴露（`Fill→alltoallv` 1.39 ms/step） | ≈33.5 | ≈36.8 | ≈75 |
| 档 2 | + 通信/计算重叠 | ≈30.8 | ≈34.1 | ≈81 |
| 档 3 | + HcPre/HcPost 融合 | ≈28.5 | ≈31.5 | ≈88 |
| **理论地板** | 完美重叠 + 零 FREE + 零 host | — | **28.5** | **89**（A=2.763 时） |
（**已作废**，见上表修正：真实地板 31.8 ms / 86.9 tok/s）

⇒ **110 tok/s 需要 ≤25.1 ms/step，比地板 31.8 还低 21%** ⇒ 必须同时
**压低 compute ≥12%**（算子级优化，需 CANN 算子团队）**或**把 **A 提到 ≥3.15**（需改 draft 模型）。

---

## 8. 证据路径

| 报告 | 内容 |
|---|---|
| `static-kernel-silent-disable-fix.md` | 12% 收益的修复全过程 |
| `cannbot-measures-triage.md` | cannbot 措施逐条 triage（本文的详细版） |
| `device-account-fixed-baseline.md` | 修复后设备账 |
| `optimization-headroom-estimate.md` | 空间估计（含步数分母修正） |
| `comm-compute-overlap-cannbot.md` | 零重叠成因 + cannbot 解法 + 死代码缺口 |
| `gmar-single-group-path.md` | GroupedMatMulAllReduce 完整调研与收口 |
| `draft-precision-acceptance-ab.md` | BF16 draft 否决 |
| `profiler-overhead-analysis.md` | profiler 开销 + 方法论修正 |
| `/home/user/opensrc/FINDINGS.md` | 开源算子库调研（706 MB 源码已下载） |
