# DCP v3 attention merge reduction 只读审查

审查范围：`dcp_runtime_20261013_v3/vllm_ascend/attention/dsa_v41.py` 中 DCP8 attention partial merge 的求和顺序、`V41_DCP_DET_REDUCE` 覆盖范围，以及将来独立 candidate 的 CPU 正负控。未改源码、未重启服务、未运行设备测试。

## 证据与运行边界

- 当前候选源：仓库 `/home/chiro/projects/dsv41/operator-stack-tp8`，HEAD `cf16be3821b91f0f66ce87e46b84b534677a9f4c`；该文件在该 HEAD 是未跟踪 overlay，路径 `experiments/delivery_matrix/dcp_runtime_20261013_v3/vllm_ascend/attention/dsa_v41.py`，SHA-256 `8f2f11fc2c097579040957b3eb42c2f2d97ac9213199282cca1a7f2ec45a7d76`。overlay manifest `experiments/delivery_matrix/dcp_runtime_20261013_v3/manifest.json:30-36` 的 delivery 与 historical source SHA 相同，并列 historical source root `/home/l00886679/dcpw`。
- 服务计划上限是 `max_num_batched_tokens=8192`（`experiments/delivery_matrix/evidence/dcp8_v3/service/service_selection.json:27-30`）；runner/config 设置 `HCCL_DETERMINISTIC=strict`（runner `:26-27`，service plan `:62-69`）。运行时 flag 复核 `experiments/delivery_matrix/evidence/dcp8_v3/quality/dcp_runtime_flags_readonly_20261013.md:12-26` 确认 `/tmp/v41_perf_flags` 不存在，且 `V41_DCP_DET_REDUCE`、`V41_DCP_ORI_OWNER` 均未设置；那次 `/proc/<pid>/environ` 定向查询只读取这两个变量，不能据此声称其他 DCP selector 的进程环境也已核验。
- `GSM100` receipt 是 99/100，唯一错误 `gsm8k-12` 在 512-token 输出预算下自相矛盾并于 `before year` 处截断；原件未记录 finish_reason / usage（`experiments/delivery_matrix/evidence/dcp8_v3/quality/gsm100_failure_diagnostic.md:3-9`）。A2 同题正确是跨配置观察，不是同环境 DCP reducer A/B。因此以下是源码路径审查，不是这一个 GSM 错题的因果证明。

## 结论

Python 源码的普通分支调用 `torch.distributed.all_reduce`，没有规定 rank 顺序的累加；从 DCP merge 代码自身看，它不是 `_v41_ordered_allreduce` 所实现的逐 rank 定序求和。代码注释也明确记录 HCCL 在特定 shape/T 下的求和顺序可变（`dsa_v41.py:339-360,1165-1174`）。不过这不等于已证明当前请求的实际 HCCL reduction 非确定：该服务计划设置了 `HCCL_DETERMINISTIC=strict`，这是独立于 `V41_DCP_DET_REDUCE` 的后端开关；本仓历史证据认为 strict 可提高单批次稳定性，但重复同一 ctx/请求的 spread 仍会跨批次变化（`CORRECTNESS_STATUS.md:51-75,206-231`）。故不能把 GSM 99/100 归因于 all-reduce 顺序。

`V41_DCP_DET_REDUCE=1` 是**部分覆盖**：它覆盖融合 merge kernel 的 pack 和常规 Python pack；不覆盖默认折叠路径上独立的 denominator all-reduce，也不覆盖 `perf_no_pack` 与 legacy pack 分支。`/tmp/v41_perf_flags` 中 `det_reduce=1` 的文件开关覆盖面更广，但本次现场文件缺失。

## 代码路径与覆盖矩阵

`_v41_ordered_allreduce()` 的定义是 `all_gather_into_tensor` 后从 rank 0 到 rank N−1 逐个相加（`dsa_v41.py:339-360`）。模块导入时将 `V41_DCP_DET_REDUCE` 读取为常量 `_DCP_DET_REDUCE`（`:1888-1895`）；需在进程导入模块前设置。`_refresh_perf_flags()` 遇到 `/tmp/v41_perf_flags` 不存在会清空文件 flag map（`:99-145`）。

| 分支 / collective | 源码位置 | 文件 `det_reduce=1` | env `V41_DCP_DET_REDUCE=1` |
|---|---|---:|---:|
| 融合 AscendC `_merge_kd` 的 pack | `:977-997`，归约 gate `:986` | 覆盖 | 覆盖 |
| 常规 folded Python pack | `:1150-1182`，归约 gate `:1171` | 覆盖 | 覆盖 |
| folded + 默认 `sepw` 独立 `weights`/denominator 归约 | `_want_contigw` 与 `weights` all-reduce：`:1425-1442` | 覆盖 `:1438` | **不覆盖**；`:1438` 只检查文件 flag |
| `perf_no_pack` 的 4 次分开归约 | `:1454-1470` | 覆盖 `:1457-1461` | **不覆盖**；gate 只查文件 flag |
| 非 folded legacy pack | `:1471-1486` | 覆盖 `:1475` | **不覆盖**；gate 只查文件 flag |
| `reduce_scatter` 实验分支 | `:1183-1231` | 文件 det flag 先命中，绕过 RS | env det flag 先命中，绕过 RS |

代码默认（`V41_DCP_MERGE_KERNEL`、`V41_DCP_RS_MERGE` 未启用；`/tmp/v41_perf_flags` 缺失；`perf_no_pack=False`）在正常 compressed+ori-ref 路径下会先将 `scaled` 与 `weights` pack，再走普通 `all_reduce`；由于 `sepw` 没有设为 `0`、`V41_DCP_CONTIGW` 没有启用，之后还会对 `weights` 单独再做一次普通 all-reduce（`:911-929,1150-1231,1425-1453`）。服务候选的计划文件没有列出这几个可选 selector；运行时 flag 复核也没有逐个读它们，所以这是**代码默认路径推导**，不是对活动进程每个环境变量的完整快照。

`sepw=0` / `V41_DCP_CONTIGW=1` 会避免独立 denominator collective、改从 pack weight column 取值，但代码记录该 contigw 路径在图捕获 decode 上可能读到垃圾值，且“永久默认关”（`dsa_v41.py:1414-1447`）；不能把它当作无风险的定序修复。

## all-gather 定序的内存约束

`_v41_ordered_allreduce()` 当前一次性分配 `(dcp, *t.shape)` 的完整暂存张量（`dsa_v41.py:351-360`）。常规 folded pack 的宽度是 `D+1` 向上按 128 对齐；V4.1 `D=512` 时为 640（`:217-252`、`:904-905`）。按 `DCP=8, T=8192, H=64, W=640, FP32` 上限估算，单 rank gather buffer：

`8 × 8192 × 64 × 640 × 4 = 10,737,418,240 bytes = 10 GiB`。

原始本地 pack 另约 1.25 GiB/rank。若走宽度 `2D+2=1026` 的 unfurled 路径，完整 gather 暂存约 16 GiB/rank。该数值是按服务计划最大 T 的容量推导，不是已观测到的运行分配；但足以说明不能只为一题就全局打开全尺寸 gather 定序路径。

未来 candidate 可实现 **沿 T 分块、所有 rank 同步同一组 chunk 边界** 的定序 gather：每块仍按固定 rank 顺序加和，避免改变数学分组；例如 `chunk_T=64/128` 时，典型 `[DCP=8,H=64,W=640]` gather 暂存约 80/160 MiB/rank，另加既有输出/pack。代价是 `8192/chunk_T` 次 collective（128/64 次），必须在新 candidate 单独测正确性与延迟，并选定明确的 per-rank memory cap；不要在正在运行的 v3 上试改。

## ORI_OWNER 与 HCCL 开关的边界

- `V41_DCP_ORI_OWNER` 未设置时 `_v41_dcp_ori_owner()` 返回 `seqused0`（`:321-336`）。它控制 SMLA 的 ori 输入归属，不控制求和顺序。需注意源码注释与赋值不完全一致：函数 docstring 说默认时 rank>0 把 `seqused_ori_kv` 设为 0（`:327-330`），但当前执行体 `_ori_seqused = seq_lens` 是无条件赋值（`:3112`），随后 main/second SMLA 都传该值（`:3424,3747`）；builder 在非 `rank0` 模式下对所有 rank 给 `has_ori_kv=True`（`:4434-4438`）。把这个独立的 ownership/参考项行为记作待澄清点，不把它混同 reducer 确定性，也不推定它导致 GSM 单题失败。
- `HCCL_DETERMINISTIC=strict` 是启动计划/runner 中的环境设置，不等于 `V41_DCP_DET_REDUCE=1`。历史 `CORRECTNESS_STATUS.md:206-231` 显示 `true` 曾把 GSM100 降到 91/100，strict 曾过 100/100；同文 `:51-75` 随后更正 strict 并不保证跨测量批次逐位稳定。不要将 strict 解释成当前 Python 代码已执行 rank-ordered gather。

## 后续新 candidate 的最小稳定性改动与 CPU 正负控

建议新 candidate 中引入一个统一的 reduction helper，显式覆盖 pack、独立 denominator、`perf_no_pack`、legacy pack 与融合路径；`V41_DCP_DET_REDUCE` 若作为总开关，应与文件 flag 在所有分支语义一致。保留 HCCL strict 作为原服务基线，不能通过 `HCCL_DETERMINISTIC=true` 替代逐分支修复。对 8192-token 预填充使用受内存上限约束的分块定序 gather，并在新 candidate 评估 collective 数带来的延迟。

可先做不触碰 NPU 的纯 CPU harness（stub collective，不 import worker/runtime）：

1. **正控**：8 个 rank 的 synthetic FP32 numerator/weight 张量，分别驱动主 pack、独立 denominator、no-pack、legacy-pack 的定序 helper；重复运行应逐位一致，并与同一 rank 顺序的 reference 相同。再由 spy 验证 `V41_DCP_DET_REDUCE=1` candidate 的每条 DCP 归约分支都不落到裸 `all_reduce`。
2. **负控**：使用 `[1e20, 3, -1e20]` 这类 FP32 cancellation rank values，模拟两种归约树，证明裸 HCCL/all-reduce 抽象允许的不同加法顺序可改变结果，而固定 rank 顺序稳定；另保留一个“主 pack 有序、denominator 裸 all-reduce”的 mutant，期望 coverage spy 失败，以防当前 env 覆盖缺口回归。
3. **边界控**：分别模拟 `V41_DCP_DET_REDUCE` env-only 与 `det_reduce` file-only；env-only 必须覆盖 denominator/no-pack/legacy branch，file-only保持兼容。ORI ownership 独立测试默认值与 `has_ori_kv`/`seqused_ori_kv` 语义，先解决注释—代码不一致再形成数值结论。

最终因果裁定仍需在冻结的新 candidate 上做重复同题/同 token 前缀对照，且请求记录 `finish_reason`、`completion_tokens` 与原始文本。现有 GSM 单题答复截断疑似较强，但不足以证明或否定 DCP merge reducer 是影响因素。
