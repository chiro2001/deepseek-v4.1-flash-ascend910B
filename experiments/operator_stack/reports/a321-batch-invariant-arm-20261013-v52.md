# 批不变（batch-invariant）质量臂：四处厂商缺陷与修复（2026-10-13 v52）

目标不变：正式 TP8/40 层 DSpark 全开销 ≤20ms/step，且最优服务须通过
GSM8K 100/100 + Vision 23/23 + 模型归属。当前状态：性能最优 **26.44–26.61 ms/step**
（A=2.9395，clean 7/8，token 与冻结基线逐位一致）；质量侧 Vision 23/23 通过、GSM8K 99/100。
本报告说明 GSM 缺口的根因、厂商正解以及为跑通它必须修复的四处厂商缺陷。

## 1. 缺口根因（承 v51 分叉审计）

- 同一 checkpoint/同一官方编码器/同一 canonical 8-shot：**1 行 decode（A1）答 13 → 100/100**；
  **6 行 verify（DSpark，K=5）答 12 → 99/100**；两 DSpark 臂（async+mls 与冻结 sync）逐字节一致。
- 分叉点=回答第 19 个 token，为 verifier 的 bonus token；该步 DSpark 上报分布
  top-1/top-2 **精确并列**（671 与 58559 同为 −0.918505），A1 侧 top-1 领先 0.25。
  两条路径的 logprob 均落在同一 **0.125 量化网格**上 ⇒ 批形状数值差异把"2 步领先"变成"0 步并列"。
- 逐轮 accept/rollback 收据一致（发出 token 均为自身上报 top-1；accepted 位置=draft 值、
  bonus 位置=回退采样），**不存在接受语义错误** ⇒ 需要的是数值一致性，而不是改接受逻辑。

## 2. 厂商正解：`VLLM_BATCH_INVARIANT=1`

vLLM-Ascend 在 worker 初始化时提供 `init_batch_invariance()`（`vllm_ascend/batch_invariant.py`）：
当 `VLLM_BATCH_INVARIANT=1` 且 AscendC 批不变算子可用时，覆盖 `weight_nz_mode`/确定性环境、
并注册 **批不变实现**：`aten::mm`/`aten::matmul`/`aten::softmax`/`npu_fused_infer_attention_score`/
`npu_add_rms_norm`/`torch.sum`。这正是让"6 行 verify"与"1 行 decode"数值一致的机制。
配套 CANN 算子包 `cann-ops-batch_invariant-<soc>-1.0.0-linux.aarch64.run`（提供
`libopapi_batchinvariant.so`）由 vllm-ascend `csrc/build_batch_invariant_ops.sh` 安装。

## 3. 跑通它必须修复的四处厂商缺陷

| # | 现象 | 根因 | 我们的修复（快照内、env 门控） |
|---|---|---|---|
| 1 | `TypeError: reduce_sum() got an unexpected keyword argument 'axis'`（权重后处理期） | `batch_invariant` 影子 `torch.sum`/`Tensor.sum`，而厂商自家 W4A8 加载器调用 `.sum(axis=1)` | `bi_axis_compat.py`：`axis`→`dim` 映射；**非末维回退原生 sum**（厂商批不变 reduce 只实现最后一维，且这些调用只在加载期发生，不在服务前向） |
| 2 | `aclnnReduceSumBatchInvariant ... not in libopapi_batchinvariant.so` | 容器只有 torch 扩展，缺 CANN 侧算子包；且首装误用 **910b** 变体 | 用 `get_soc_version()=253 / Ascend910_9382` 判定本机为 **910_93（A3）**，卸载 910b 后安装 `cann-ops-batch_invariant-A3-1.0.0`（`--quiet --nox11 --install-for-all`），以 `ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant` 接入。（不可改 `LD_LIBRARY_PATH`：会破坏 `set_env.sh` 对 `libhccl.so` 的解析。）设备验证：**6 行与 1 行 reduce 逐位一致** |
| 3 | Dynamo：`mul` 假张量形状失配（engram gate `.sum(-1)` 保持输入形状） | A3 算子包的 **Meta（fake tensor）内核形状错误**（`npu_reduce_sum_batch_invariant` 返回输入形状） | `bi_axis_compat.install_meta_kernels()` 为 7 个算子注册正确 Meta 形状（reduce_sum/reduce_mean/softmax/log_softmax/mm/matmul/add_rms_norm）；容器自检 `[BI-META] installed=7 skipped=`，fake 张量形状正确 |
| 4 | `aclnnGroupedMatmulSwigluQuantWeightNzV2`：weight storageShape 需 5 维，实得 1 维 | `override_envs_for_invariance()` 强制 `weight_nz_mode=0`，使 W4A8 MoE 权重留在 ND 布局，而本栈按 NZ 融合算子派发 | 包装 `override_envs_for_invariance`：调用原函数后恢复 `weight_nz_mode=1`（本栈默认；BI 的算子注册与权重布局无关），stderr 打 `[BI-NZMODE]` 回执 |

四处修复后，批不变臂已能通过：引擎初始化 → 权重加载 → Dynamo 图捕获 → **首次前向**
（`[BI-META] installed=7`、0 次 Dynamo/形状错误、8/8 worker 启用批不变）。

## 4. 启动环境的现实约束（外部租户抖动）

- physical 8–15 被多个外部租户的高频短作业反复占用（qwen3-30B 测试、moe_compute_exp 等），
  多次出现"看守检查通过后数秒内被抢"→ 加载期 OOM。
- 已采用**看守脚本**（每 20–30s 检查"4 个 NPU 块无进程 + 8 卡 HBM 占用 <8GiB"，
  命中即以固定参数启动一次并在成功后退出；失败自动递增 job-id/port 规避残留目录）。
  本轮看守确实捕获过两次真实空闲窗口并成功启动（v9x2、v11x1）；
  看守自身曾因成功判据写错（匹配 `performance_receipt_granted`，而回执字段是
  `formal_receipt_granted`）未退出而重复尝试，已修正为匹配
  `"service_launch_requested": true` 且单次成功后退出。

## 5. 判据与后续

- 判据：批不变臂若生效，6 行 verify 与 1 行 decode 数值一致 ⇒ `gsm8k-12` 应给出与 A1 相同的
  **13** ⇒ **GSM8K 100/100**；同一作业的 orchestrator 会随后跑 Vision 23/23。
- 性能代价：已备妥 `src_dspark_native_graph_api_bi_v1`（API 快照 + 同一 shim），
  待质量臂通过后用同一 2K/256 负载测 `VLLM_BATCH_INVARIANT=1` 的 ms/step，
  判断"质量修复"与"≤20ms"能否在同一服务上共存。

## 6. 证据（本地仓库）

- `evidence/formal_a321/quality_bi_prep/`：`bi_axis_compat.py`（shim 全文）、
  `build_bi_seed_v9.py`（快照/种子构建器）、`patch_v9_manifest.py`/`patch_v9_seed.py`
  （seed 与 manifest 双钉一致性）、`NOTE.md`（本报告的精简版+文件说明）。
- `evidence/formal_a321/quality_async_prep/`：v1 质量候选合同与快照清单（对照基线）。
- `evidence/formal_a321/quality_gate_runs/`：两 DSpark 臂的 GSM/Vision 原始结果（99/100、23/23）。
- `evidence/formal_a321/divergence_audit_20261013/`：分叉审计（两侧 top-8 logprobs 等）。

## 7. 诚实边界

- 批不变臂**尚未**产出实际的 100/100 结果（芯片抖动导致未跑到评测完成）；
  本报告只声明"机制已跑通到首次前向"与"四处缺陷的修复及证据"。
- `VLLM_BATCH_INVARIANT` 的**性能代价未知**（待测）；若代价过大，质量与 ≤20ms 的共存
  将需要额外权衡（届时如实报告，不用 A1 结果替代）。
- 本报告不构成 ≤20ms 或质量门的通过声明；两者仍是目标。
