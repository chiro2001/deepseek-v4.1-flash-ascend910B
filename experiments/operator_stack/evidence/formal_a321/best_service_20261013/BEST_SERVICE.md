# 最优正式服务套件（可复现）：26.5524 ms/step + GSM100/100 + Vision23/23

本目录把 **2026-10-13 的最优正式服务**固定为"一条命令就能复现"的配方：
快照、环境、命令、期望结果、以及诚实的边界。所有数字都来自本仓库内冻结的收据。

## 1. 服务的定义（两个互补窗口，同一配置）

| 窗口 | job | 快照（容器内） | 结果 |
|---|---|---|---|
| 性能窗口（2K/256 × 8 串行、审计/profiler 关） | `formal_dspark_native_graph_api_v11_bi` | `/work/src_dspark_native_graph_api_bi_v1` | **(26.5523755957162 ms/step, A=2.7309236947791167, 102.85044684362282 tok/s)**，747 actual rounds / 2040 tokens，clean 8/8、复读 0 |
| 质量门（GSM8K 严核 + Vision） | `formal_dspark_quality_async_v14_bi` | `/work/src_dspark_general_quality_v11` | **GSM8K 100/100**（空答 0、错误 0）、**Vision 23/23 PASS**；`actual_client_quality_passed=True` |

两者配置等价：同一模型路径（`/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`）、
同一 40 层/hidden5120/384 专家 top6/W4A8_DYNAMIC/Engram int8/MTP3 权重、`HCCL_DETERMINISTIC=strict`、
`VLLM_BATCH_INVARIANT=1`（批不变，质量关键）、async 调度、两个多流重叠开关。

## 2. 关键环境（完整 35 项见 `env_manifest.json` 与 `env_exports.sh`）

```
ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant   # 批不变 CANN 算子包（A3 变体）
VLLM_BATCH_INVARIANT=1            # 6 行 verify 与 1 行 decode 数值一致（质量门关键）
DSPARK_ASYNC_COUNTER=1
DSPARK_ASYNC_SCHEDULING=1         # async 调度（把宿主段藏起来）
V41_MLS_SHARED_EXPERT=1           # 共享专家 gate_up 走辅助流
V41_MLS_DSA_OVERLAP=1             # DSA MLA prolog/compressor 走辅助流
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
HCCL_DETERMINISTIC=strict  HCCL_BUFFSIZE=1024  HCCL_OP_EXPANSION_MODE=AIV
ASCEND_MAX_OP_CACHE_SIZE=-1  TASK_QUEUE_ENABLE=1  PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
V41_ENGRAM_DEVICE_INDEX=1  V41_ENGRAM_DEVICE_FALLBACK=0  V41_ENGRAM_JIT=1  V41_ENGRAM_HOST_RESIDENT=1
V41_QLI_NO_CANDIDATE=1  V41_MOE_COMM_ALLGATHER=1  V41_MOE_MASK_RANGE=1  V41_ROPE_IDXSEL=1  V41_O_PROJ_2D=1
（其余 17 项为 V41_* 默认/门控项，逐字见 env_exports.sh；不要手工改写，直接 source）
```

## 3. 复现步骤

前置：a3-21 上容器 `dsv41-real-stack-tp8-20261010-a321` 已启动；physical 8–15 全部空闲
（宿主 `npu-smi` 的 NPU 4–7 都显示 "No running processes"；容器内 npu-smi **看不到其他租户**，
所以占用判断必须用宿主口径）。

```bash
# 0) 环境（逐字来自冻结收据）
source evidence/formal_a321/best_service_20261013/env_exports.sh

# 1) 性能窗口（快照 src_dspark_native_graph_api_bi_v1，2K/256 × 8 串行）
PYTHONPATH=$R/offline_tools/native_api_launcher_v3 \
  python3 $R/offline_tools/native_api_launcher_v3/launch_native_api_diagnostic_v4.py \
  --root $R --source-dir /work/src_dspark_native_graph_api_bi_v1 \
  --container dsv41-real-stack-tp8-20261010-a321 --job <new_job> \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --workload /work/dspark_api_workload_cpu_20261011_v1/workload.json \
  --port <free_port> --draft-graph \
  --extra-env DSPARK_ASYNC_COUNTER=1 --extra-env DSPARK_ASYNC_SCHEDULING=1 \
  --extra-env V41_MLS_SHARED_EXPERT=1 --extra-env V41_MLS_DSA_OVERLAP=1 \
  --extra-env VLLM_BATCH_INVARIANT=1 \
  --extra-env ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant
# 随后用同目录 run_native_api_client_when_ready.py 跑冻结客户端（8×2K/256 串行）

# 2) 质量门（快照 src_dspark_general_quality_v11；GSM8K 严核 + Vision）
#    见 evidence/formal_a321/quality_gate_pass_bi_v14/launched.json 的原 argv（含 --stop-after）
```

## 4. 期望结果与核验锚点

- 性能：`acceptance_small.json.aggregate` 的三元组应落在 **(26.55 ± 0.18) ms/step、A≈2.73**、
  clean 8/8；原始 `measured_0..7_http.json` 与聚合必须一致（本仓库已冻结 8 份）。
- 质量：`strict_quality_receipt.json` 的 `actual_GSM100_passed=True`、`actual_Vision23_passed=True`；
  `gsm100.json` summary = `n=100, correct=100, empty=0, errors=0`。
- 锚点（SHA256 见 `env_manifest.json`）：两个 `launched.json`、`acceptance_small.json`、
  `strict_quality_receipt.json`。

## 5. 诚实边界（必读）

- **本服务 26.55 ms/step，未达 ≤20ms**；缺口 6.55ms 的定量分析见
  `reports/a321-remaining-gap-budget-20261013-v54.md`（1 行实测 19.23ms ⇒ 固定成本≈17.7ms、
  边际≈1.5ms/行；6 行要 ≤20ms 需厂商内核级融合）。
- 批不变模式会改变 token 流（与旧非 BI 窗口 0/8 相同），质量以官方 100/100 + 23/23 判定。
- 本套件不构成 formal receipt；所有 `*_granted` 字段保持 false（见各收据）。
