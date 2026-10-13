# DCP v6 单容器启动候选：CPU-only 预备计划

**状态：仅生成和核对，未执行。** 本计划没有启动 service、访问 API/port18960、查询 NPU/IPC 运行态或发 generation。候选运行脚本含 `--execute`，只供 root 实读审核；site 是唯一执行者，须等待 root 明确放行。

后续 root 已实读 launcher，并以当前 `service.make_plan` 做独立 CPU 参数审计：
把远端 overlay 路径仅在**文件验证函数**中映射到本机冻结 v6 原件，其他计划字段保留
预定远端路径；不是替代 SDK/device。KV4GiB、TP8/DCP8、MAXLEN196608/BAT8192/SEQ1、
BI/Engram0/async/best/spec5真实图配置、det-reduce1、8–15且无0–7派生、清除CED污染均通过。
CPU plan fingerprint=`e1277a56ecf760c436cdcb6819c98763bf8b353b538f0634bba238fc1a592c90`，
不得作为已运行实例的 fingerprint/SID。当前外部 fyf_1009 已轮换占到physical9，
因此不是可执行资源窗口；**未授权启动**，下一实际窗口仍须source/OPP/write/IPC/device门。

## 候选绑定

- served model：`dsv41-delivery-dcp8-v6`
- DCP8 form：`a3-tp8-dcp8`
- D API port：`18960`
- source：`/work/src_dspark_general_quality_v11`
- generic v7 package：`/work/delivery_matrix_generic_20261013_v7/delivery_matrix`
- 新 D job root：`/home/l00886679/tmp/dsv41-dcp-v6-runtime/d6`
- base candidate source fingerprint：`8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`
- candidate 文件 SHA-256：`5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`

这里仅绑定未部署的 base candidate source fingerprint；未沿用 A2 runtime identity、SID/runtime fingerprint，也未沿用 DCP v3/v5 service runtime identity。新服务的 runtime fingerprint/SID 只能由 site 后续在 D6 服务计划/启动时生成并交 root 审核。

## 从当前 CLI 确认的配置

| 项目 | 预定值与来源 |
|---|---|
| Physical chips | `8,9,10,11,12,13,14,15`；base candidate 已是 8–15，生成命令不传 `--physical-worker`，因此不选 0–7 派生 worker |
| Worker class | 原候选 `parallel_tiny.dspark_formal_prep.worker.FormalDSparkWorker`；服务层仍用通用 `delivery_matrix.worker.DeliveryWorker` 包装 |
| Device index | `0`，由 `prepare_launch.py --device-index 0` 覆盖 candidate 默认值 |
| DCP | `decode_context_parallel_size=8`；`V41_DCP=1`；replicate indexer `1` |
| Ordered reduction | `--dcp-det-reduce 1` → `V41_DCP_DET_REDUCE=1`，只选择 DCP 有界确定性候选 |
| Overlay / draft | `dcp_runtime_20261013_v6` / `dcp_draft_20261013_v2`，路径见下方 CLI；不使用旧 overlay |
| Model length / sequence / batch | `196608` / `1` / `8192`。`prepare_launch.py` 没有 `--batched-tokens` 参数；生成命令省略该项后，当前 `service.py` 默认 `8192` |
| KV cache | `4294967296` bytes，沿用 base candidate |
| Prefix / EnGram | prefix caching `false`；EnGram enabled、int8、device index `0`，route probe 空字符串 |
| Quality/performance mode | BI=`1`、async scheduling=`1`、DSpark draft graph与metadata=`1`、DSpark5 (`num_speculative_tokens=5`, `enforce_eager=false`)，best multistream候选保留 |

## CPU 证据与源码 hash

- `service.py --help` 与 `prepare_launch.py --help` 均从当前源码读取并成功退出；没有执行服务。
- DCP v6 overlay 13 个 manifest payload 全部逐文件 hash 校验；官方 `build_dcp_stability.verify_candidate()` 返回所有五个 reduction gate 共享有界 helper、128 MiB staging cap、64 chunk rows、512-byte alignment，fixed pack address 保留。`actual stability/quality/performance` 仍为 false。
- DCP best draft v2 的 2 个 manifest payload 均逐文件 hash 校验。
- CPU-only `test_dcp_stability.py`：8/8 通过；测试使用 `/home/chiro/tmp/dsv41-dcp-v6-prep` 作为 `TMPDIR`，没有 NPU/网络调用。
- 用官方 `prepare_launch.py` 生成的 review-only launcher 为 `/home/chiro/tmp/dsv41-dcp-v6-prep/generated/d6_run_service.sh`，exit 0，3961 bytes，mode `0644`，SHA-256 `b84f9f32cbacaff3e47c5e62171a2dc1a46e7c373e5ab20fb5927ebf33a111ed`；其 `service_launch_requested=false`。文件未执行。

| 输入文件 | SHA-256 |
|---|---|
| `delivery_matrix/service.py` | `c464aceae5fa789027334dc6222cfa47112bcf20ae700f1d4ac8c95d6c6f13c9` |
| `delivery_matrix/prepare_launch.py` | `700e2a86ff9af5b9f16d937fbf4369f86babfcc4a8ec5d8ffe8116c088eacdfa` |
| `delivery_matrix/run_service.sh` | `cc5bcd644c78e5259b82f2cfc8aff3627ab6482b52fb36e8bd7f197ec5a1cdd9` |
| `delivery_matrix/build_dcp_stability.py` | `a47d32a0eb2ca7de6c75349829217e9ba116e370f0b614637853494f095644d0` |
| `dcp_runtime_20261013_v6/manifest.json` | `b0c1d36e44707c872707477677a5ef60a4d3d53f86c5df0d1092e47e41706a89` |
| `dcp_draft_20261013_v2/manifest.json` | `abb1a60cd24f8040a9f87a3a69471a0e38e2a166aed6d073f43c6229316d5074` |
| `evidence/a2sim_tp8_v4/source_gate/package_files.sha256` | `e38cab6bed9b0b53512142d8f350e7d39fc2f02dc30ea7f2da73db81519df0e1` |
| Base candidate contract file | `5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2` |

The v7 package file manifest binds the DCP v6 overlay, draft v2, `prepare_launch.py`, `service.py`, and `run_service.sh` hashes above. Site must still verify that its staged `/work/delivery_matrix_generic_20261013_v7/delivery_matrix` files match this manifest before execution.

## CPU-generated launcher recipe

This is the exact local CPU-only generator invocation. Its `--output` is a local review artifact; `--job-root` is only the future D-container path embedded in that artifact.

```sh
TMPDIR=/home/chiro/tmp/dsv41-dcp-v6-prep PYTHONDONTWRITEBYTECODE=1 python3 /home/chiro/projects/dsv41/operator-stack-tp8/experiments/delivery_matrix/prepare_launch.py --candidate /home/chiro/projects/dsv41/operator-stack-tp8/experiments/delivery_matrix/evidence/a2sim_tp8_v4/source_gate/candidate_contract.json --source /work/src_dspark_general_quality_v11 --package /work/delivery_matrix_generic_20261013_v7/delivery_matrix --job-root /home/l00886679/tmp/dsv41-dcp-v6-runtime/d6 --output /home/chiro/tmp/dsv41-dcp-v6-prep/generated/d6_run_service.sh --port 18960 --served-model dsv41-delivery-dcp8-v6 --form a3-tp8-dcp8 --device-index 0 --physical-chips 8,9,10,11,12,13,14,15 --dcp-overlay /work/delivery_matrix_generic_20261013_v7/delivery_matrix/dcp_runtime_20261013_v6 --dcp-draft /work/delivery_matrix_generic_20261013_v7/delivery_matrix/dcp_draft_20261013_v2 --dcp-replicate-indexer 1 --dcp-det-reduce 1 --max-model-len 196608
```

The generated launcher invokes `bash /work/delivery_matrix_generic_20261013_v7/delivery_matrix/run_service.sh ... --execute` only if a human later runs the artifact. This plan does not run that launcher. The generated `prepare_launch.py` JSON contains `VLLM_CACHE_ROOT` and `STACK_DSPARK_FORMAL_OUTPUT` candidate placeholders under `/work/results/q4_placeholder`; current `service.py.make_plan()` overwrites these from the actual `--output` to `service/cache` and `service/native`. Site must inspect the final D6 `service/plan.json` and stop if either effective path escapes D6.

## Writeability and IPC checks for the future site operator

- D-container must see `/home/l00886679/tmp/dsv41-dcp-v6-runtime/d6` at the same path and be able to write its root receipts (`runner.pid`, `run.log`, `run.exit`), `tmp`, `service`, and effective `service/cache` + `service/native` trees. `run_service.sh` explicitly sets `TMPDIR` from `DELIVERY_TMPDIR=/home/l00886679/tmp/dsv41-dcp-v6-runtime/d6/tmp`.
- The planned D6 TMPDIR prefix is 47 bytes; a UUID-shaped ZMQ IPC path under it is 84 bytes (`<107`). This is a static estimate only; site must verify the actual container IPC socket path remains below Linux `sun_path` 107-byte limit before proceeding.
- Site must recheck 8–15 physical ownership/health and full required IPC/device visibility immediately before launch. The “8–15 available” note is from root; this worker did not probe NPU or container state.
- Port 18960 must be unowned by any competing service. No service/port ownership was queried in this preparation.
- Keep the job root unique and writable by the container UID. Do not reuse the DCP v3/v5 output path or any CED job; do not stop/restart another service.
- Root must read this plan and the review-only launcher, then authorize site as the only operator. No generation, API, canary, or quality/performance result is authorized by this plan.
