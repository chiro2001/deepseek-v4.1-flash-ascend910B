# LOG_INDEX.md —— 原始日志路径清单（A3-node1 / A3-node2）

> 用户要求：**只给 reports/*.md + 原始日志路径**，不要塞大文件。
> 所以本目录只放「路径索引 + 关键样本（每条 JSON 1–2 KB）」，原始日志仍留在开发机上。
> 主机前缀 `A3-node1:` 表示 `ssh A3-node1`；`$P` = `/home/user/projects/dsv41`，`$H` = `/home/user`。

---

## 1. 本包内的关键 samples（**逐字节复制**，可直接复算）

| 文件 | 内容 | 用途 |
|---|---|---|
| `samples/p42_t4_quote_131072_faA_128k_r{1..8}.jsonl` | 128K 单流 × 8 发，**PGO 关**，全补丁 | `EXPECTED_PERF.md` §1 的原始数据 |
| `samples/p42_t4_quote_131072_faB_128k_r{1..8}.jsonl` | 128K 单流 × 8 发，**PGO 开**（推荐配置） | `EXPECTED_PERF.md` §1.2；峰值 **110.5 tok/s**（= `A×1000/ms` 口径；同文件 `decode_tok_s` 字段写 110.935，算法不同） |
| `samples/p42_t4_quote_32768_fa{A,B}_32k_r{1,2}.jsonl` | 32K 单流 | `EXPECTED_PERF.md` §3 |
| `samples/p42_t4_quote_8192_fa{A,B}_8k_r{1,2}.jsonl` | 8K 单流 | `EXPECTED_PERF.md` §3 |
| `samples/p42_t4_quote_8192_faA_w.jsonl`、`..._faB_w.jsonl` | 8K warmup 发 | 同上 |

复算命令：`python3 tools/analyze_samples.py "logs_meta/samples/p42_t4_quote_131072_faB_128k_r*.jsonl"`

## 2. 性能 / 起服日志（A3-node1）

| 内容 | 路径 |
|---|---|
| 起服日志目录（约 200 份） | `A3-node1:$P/logs/perf/` |
| 128K/32K/8K 测量 jsonl 目录 | `A3-node1:$P/logs/perf/a21/` |
| faA（PGO 关）/ faB（PGO 开）起服日志 | `A3-node1:$P/logs/perf/faA_serve.log`、`faB_serve.log`（同目录另有 `*faA*`/`*faB*` 变体） |
| 全补丁臂的早期起服日志 | `A3-node1:$P/logs/perf/a21_final_0734_serve.log` |
| quote 测量脚本（口径权威） | `A3-node1:$P/delivery_20260914/assets/p42_t4_quote.sh`（= 本包 `tools/p42_t4_quote.sh`） |
| 容量测量脚本 | `A3-node1:$P/delivery_20260914/assets/p36_capacity15.py` |
| profiler 目录（py-spy / torch profiler） | `A3-node1:$P/logs/prof_a21*/`（另有 `prof_ag`/`prof_se0`/`prof_nospec` 等 20+ 个） |

## 3. 各优化项的验证证据

| 项 | 报告（已逐字复制到本包 `reports/`） | 原始数据路径 |
|---|---|---|
| MoE AllGather | `moe-allgather-breakthrough.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_ag*.jsonl`、`logs/perf/a21_sweep_ag_*.log` |
| SP_TOKENS=5 | `sptok-sweep-allgather.md`、`real-weight-128k-5shot.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_sptok5_128k.jsonl`、`..._s5cap_128k.jsonl` |
| F3 `wo_a` 2D | `f3-wo-a-2d.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_{f3a,f3a2,f3b}_*.jsonl` |
| moe-mask-range | `moe-mask-range-verified.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_cm{B,C}_*.jsonl` |
| rope-idxsel | `rope-idxsel-verified.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_{vmA,vmB,vrA,vrB}_prof.jsonl` |
| Engram JIT | `engram-jit-verified.md`、`numba-landing-plan.md` | `A3-node1:$P/logs/prof_a21*/`、`logs/perf/a21/` |
| QLI no-candidate | `qli-no-candidate-verified.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_{vqA,vqB}_prof.jsonl` |
| CPython PGO+LTO | `cpython-pgo-verified.md` | `A3-node1:$H/cpython_pgo/`、`A3-node2:$H/cpython_pgo/REPORT.md`、`A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_fa{A,B}_128k_r*.jsonl` |
| **MOE_ZERO（v4：已判负结果）** | **`session-attractor-and-clean-rate.md`**（v4 新增）、`uninit-output-audit.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_mz_{on,off}_r*.jsonl`、`A3-node1:/tmp/iab.log`（S2 同会话交错）、`A3-node2:/tmp/freq3.log`（A/B/A） |
| **MOE_NONFINITE（v4：已判无差异）** | **`session-attractor-and-clean-rate.md` §6.4**、`uninit-output-audit.md` | `A3-node1:/tmp/iab_s4.log`（N=24/臂） |
| **DSpark draft 入图（v4：负控已确认，正控待验）** | **`draft-graph-negative-control.md`**（v4 新增）、`draft-graph-numinput-fix.md`、`dspark-graph-enable-plan.md` | `A3-node1:$P/logs/perf/a21/p42_t4_quote_131072_rgnif_*.jsonl`、`A3-node1:$P/logs/prof_dgv{A,B}/`、`$P/probe_draft/SELFCHECK.txt` |
| **多 batch / 多轮对话（v4 新增，生产臂已过、`PREFIX=0` 待跑）** | **`multibatch-and-mixed-load.md`**（v4 逐字复制，88 行）、`a-basin-and-acceptance-shape.md` §7 | `A3-node1:/tmp/multibatch_session.log`、`A3-node1:$P/logs/perf/mbgP/{summary,multiturn,concurrency_c8,mixed_long_short}.json`、`A3-node1:$P/logs/perf/prod_p{1,0}_*` |
| **数值确定性 / clean 判据（v4 新增）** | **本包 `CORRECTNESS_STATUS.md`**、`A3-node2:wt-graph/reports/correctness-line.md` | `A3-node2:/tmp/{freq3,spec0,spec0_rep2,cvq2,strict_thr,strict_bisect,strict_ctl}.log`；`A3-node2:wt-graph/logs/corr/` |
| 静态内核静默禁用 | `static-kernel-silent-disable-fix.md` | `A3-node1:$P/logs/`（含 `static_kernel.py:650` 的会话日志） |
| 非确定性 / A 的第三源 | `nondeterminism-rootcause.md`、`uninit-output-audit.md`、`ctx-nondeterminism.md` | `A3-node1:$P/logs/prof_*` |

## 4. 量化链证据（A3-node1 / A3-node2）

| 内容 | 路径 |
|---|---|
| 量化复现指南（69 KB） | 本包 `quant/REPRO_W4A8_QUANT.md`（源：`A3-node1:$P/docs/REPRO_W4A8_QUANT.md`） |
| 量化实施日志 | `A3-node1:$P/logs/perf/msmodelslim_repro_impl.md` |
| 基准 manifest（供结构 diff） | `A3-node1:$P/logs/perf/quant_manifest_ref.json` |
| 量化补丁说明 | 本包 `quant/patches/MSMODELSLIM_PATCHES.md`（源：`A3-node1:$P/patches/PATCHES.md`） |
| 发布冻结规格（三件交付物 + 红线 + 指纹） | `A3-node1:$P/RELEASE_TASK_SPEC_V1.md` |
| 发布指南 | `A3-node1:$P/PUBLISH_GUIDE_QUANT_V1.md`（⚠️ §4.1 的 checkout commit 写错，以 `RELEASE_TASK_SPEC_V1.md` 为准） |

## 5. A2 历史复现（上一轮，供参考）

| 内容 | 路径 |
|---|---|
| A2 复现报告（2026-09-14） | 本仓库 `$WORKSPACE/A2_复现报告.md`（未逐字复制：体积大且含已被本轮覆盖的旧结论） |
| A2 起服日志 | 本仓库 `$WORKSPACE/a2_repro_serve.log` |
| 上一版包骨架（= v2） | 本仓库 `a2_manual_pkg/`（v3 沿用其双命令骨架） |
| A3-node1 上的旧交付 | `A3-node1:$P/delivery_20260914/`（`assets/`、`evidence/`、`analysis/`）、`A3-node1:$P/a2_package/`（`configs/`、`data/`、`expected/`、`logs/`、`patches/`） |
