# CED-PD 上下文交接（2026-09-26 04:00）

**一句话**：CED-PD（P 只跑前 20 层 + 层 20 全局源；D 做 128-token 有界重放 +
全 40 层 decode，BF16 KV）已在 A3 单机 16 卡上**完成全部验收项**，
prefill 相对全 40 层基线 **1.99×（144K，正好等于层数比 40/20）**。上一份交接见
[`CED-PD-HANDOVER-20260925-1345.md`](CED-PD-HANDOVER-20260925-1345.md)（历史，别当现状）。

## 0. 恢复时的第一件事

```bash
cd ~/projects/dsv41/ced-pd-release
git log --oneline -1                 # 应为 8fbf6ce
git status -sb | head -3             # 只应有未跟踪的 tools/rssh.sh
ssh a3-21 'docker ps --format "{{.Names}}|{{.Status}}" | grep -iE "dsv41-ced|proxy"'
ssh a3-21 'for p in 18990 18991; do curl -s -o /dev/null -w "$p=%{http_code} " $p; done'
# 期望：dsv41-ced-p2b / dsv41-ced-d4b / proxy 三个容器 Up，两个 health 都是 200
```

## 1. 现状（2026-09-26 03:56 核）

| 角色 | 容器 | 芯片 | 端口 | 口径 |
|---|---|---|---|---|
| P（CED prefill） | `dsv41-ced-p2b` | 0–7 | 18990 / KV 19090 | 只跑层 0–19 + 层 20 全局源 |
| D（CED decode） | `dsv41-ced-d4b` | 8–15 | 18991 / KV 19091 | 128-token 重放 + 全 40 层 |
| proxy | `dsv41-ced-metadata-inline-repeat8-proxy-20260924-1450` | — | 18992 | 官方 load_balance_proxy |

共同参数：`MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128`、
`KV_DTYPE=bfloat16`、`ENGRAM=1 ENGRAM_STORAGE=int8 ENGRAM_DEVICE_INDEX=0`、
`CPU_BIND=0`、`SPEC=0 DRAFT_GRAPH=0`、`PREFIX=0`、`NPUGRAPH_EX=1 STATIC_KERNEL=0`、
`MULTISTREAM=0 DSA_OVERLAP=0`、两侧 `num_blocks=29076`。

### 1.1 本轮的验收结果（全过）

| 项 | 144K | 1M |
|---|---|---|
| 四针 A/B/C/D | 4/4 | 4/4 |
| 流式（TTFT） | 10.95 s | 101.2 s |
| 多轮（三轮，**真实长度**） | 3/3（144,105 tok） | 3/3（**999,510 tok**） |
| **缓存命中**（实验臂 `PREFIX=1`） | 常规/整池/交错全过 | 常规/整池全过，**≈18×** |

**性能对照**（同机同请求，唯一变量是 P 的层数）：

| 上下文 | CED prefill | 基线 prefill | 比 | CED decode | 基线 decode |
|---|---:|---:|---:|---:|---:|
| 32K | 12,536 tok/s | 7,219 | 1.74× | 25.3 ms/step | 25.4 |
| 144K | **13,676** | 6,599 | **2.07×** | 26.8 | 26.8 |
| 1M | 9,897 | 3,557 | 2.78× | 37.9 | 37.4 |

TTFT：144K **21.8 → 10.95 s**；1M **280.3 → 101.2 s**。
吞吐（CED 臂，2K prompt、并发 1/2/4）：**41.4 / 70.6 / 114.5 tok/s**。

## 2. 代码位置（分支 `feat/ced-pd-a3`，远端已同步）

| 文件 | 作用 |
|---|---|
| `scripts/serve_a3_ced_pd.sh` | CED 的 P/D 角色入口；含 `PREFIX`/`SPEC`/`DRAFT_GRAPH` 硬门、`CED_EXPERIMENTAL_GRAPH` 的两个前提、`[CED-POOL-GUARD]` |
| `experimental/ced/mooncake_hybrid_connector.py` | CED 连接器：P→D 的 group 列表、D 侧上半层 SWA 预清零、`[CED-32BIT-GUARD]`、`[CED-KVGEOM]` 探针 |
| `experimental/ced/dsa_v41.py` | `[CED-SWA-CLIP]`（replay 首 query 越界读修复）、`[CED-KVGEOM]` 算子级探针 |
| `experimental/ced/core_scheduler_replay.patch` | **D 侧**：128-token 重放调度（含 `[CED-KVRECV]` 空接收分支） |
| `experimental/ced/core_scheduler_prefill_hit.patch` | **P 侧**：整段命中时回退到上一个 128 对齐边界 |
| `experimental/ced/core_model_runner_prompt_tail.patch` | 单 token prompt 尾步强制 eager（图模式必需） |

## 3. 文档地图

| 文档 | 内容 |
|---|---|
| [`CED-PD-ACCEPTANCE.md`](CED-PD-ACCEPTANCE.md) | **先读这个**：启动命令、验收矩阵、判据、结果 |
| [`CED-PD-PERF-20260925.md`](CED-PD-PERF-20260925.md) | prefill/decode/吞吐的逐项实测与两臂对照 |
| [`CED-PD-BLOCK-BOUND-20260925.md`](CED-PD-BLOCK-BOUND-20260925.md) | 32 位页步长回绕（`num_blocks ≤ 29076`）的完整根因链 |
| [`CED-PD-CACHE-HIT-PLAN-20260925.md`](CED-PD-CACHE-HIT-PLAN-20260925.md) | 缓存命中：从"阻断"到"可用"的四轮过程 |
| [`CED-PD-PROFILING-20260925.md`](CED-PD-PROFILING-20260925.md) | 流级 profiler 分析方法与 2×2 结果 |
| [`CED-PD-GRAPH-PREREQ-20260925.md`](CED-PD-GRAPH-PREREQ-20260925.md) | 图模式的两个前提（其中一个 env 是死的） |
| [`CED-D-1M-LAYOUT-BUG-20260924.md`](CED-D-1M-LAYOUT-BUG-20260924.md) | SWA-clip 修复的算子源码级核实 |
| `evidence/ced_prefix_hit_20260926/` | 缓存命中的原始探针与结论 |
| `evidence/ced_ms_switch_20260925/` | `MULTISTREAM` 开关的单变量隔离与流级证据 |
| `evidence/ced_acceptance_20260925/` | 验收矩阵的原始证据 |

## 4. 三条必须知道的经验（都是踩过的）

### 4.1 起服前必须清干净**所有**自有容器

`docker rm -f <显式名字>` 会漏掉新命名的容器 → 卡被占 → 新实例拒绝启动 →
**而 health 会被旧容器应答**，看起来"起来了"。本轮踩过三次。
现在所有 `launch_*.sh` 都改成按前缀清：`^dsv41-(ced|pfx|base)` 且 `grep -v proxy`
（proxy 的名字也以 `dsv41-ced` 开头，第一版按前缀清把它一起删了）。
并且**只有新 run 的 `serve.log` 真的出现预期标志**时才报就绪。

### 4.2 sha 门对应的是「镜像原始 + admission_gate.patch」

`serve_a2.sh` 里的 `_base = 533eed493cb...` **不是**镜像原始文件
（原始是 `c67bda2886...`），而是打完 `patches/admission_gate.patch` 之后的内容。
生成补丁必须先打 admission gate 再 diff；在容器里 `git checkout` 会把
admission gate 一起抹掉。

### 4.3 判据要用「服务端计数器」，不要用响应字段

`usage.prompt_tokens_details.cached_tokens` 经 PD 代理**恒为 0**，即使真的命中也一样。
正确判据：`vllm:prefix_cache_hits_total`、
`vllm:prompt_tokens_by_source_total{source="local_cache_hit"}`。

## 5. 已知边界与未完成项

### 5.1 **DSpark（投机解码）在 CED 下是关的**（两层强制）

| 层 | 位置 | 行为 |
|---|---|---|
| 启动器 | `scripts/serve_a3_ced_pd.sh` | `SPEC` / `DRAFT_GRAPH` 非 0 → `exit 2` |
| 模型 | `patches/files/model.py:907` | `V41_CED_ROLE=prefill/decode requires SPEC=0 during the replay prototype` |

实测核对（两个容器）：`--speculative-config` 出现 **0** 次、
`speculative_config=None`、`/metrics` 里投机解码指标 **0** 条。
容器里的 `DSPARK_*` env 是 `serve_a2.sh` 无条件导出的模板值，**不是启用证据**。

→ **完整分析见 [`CED-PD-DSPARK-ANALYSIS-20260926.md`](CED-PD-DSPARK-ANALYSIS-20260926.md)。**
+结论一句话：**P 侧带 DSpark 是架构性不可行**（DSpark 要目标层 37/38/39 的残差，
+而 CED 的 P 在第 20 层 break）；正确路线是 **P 保持 `SPEC=0`、D 开 `SPEC=1`**，
+再补 D 侧的四处代码 + 验证一个前提。

### 5.2 缓存命中只是**实验臂**

`PREFIX=1` 需要显式 `V41_CED_ALLOW_PREFIX=1`；交付口径是 `PREFIX=0`。
要转正需把两条修复（P 侧对齐回退、D 侧空接收）视为正式改动并重跑完整矩阵。

### 5.3 基线自身的缺陷（与 CED 无关，已划到范围外）

**全 40 层基线的 D 侧 `MULTISTREAM=1 DSA_OVERLAP=1` 在长上下文下会静默乱码**
（HTTP 200、`completion_tokens` 打满、无 `finish_reason`、含 `<|box|>`）。
已单变量定位到"就是这组开关"（只重启 D、只改它：0/4 乱码 → 4/4 通过），
但**具体缺哪条同步未定**。
⚠️ A2 生产环境正在用 `MULTISTREAM=1 DSA_OVERLAP=1`，这条值得单独开任务。

### 5.4 两个性能数字的未归因部分

* 1M 的 2.78× 大于层数比 2.0×：已部分归因（两臂 prefill 随上下文衰减不同：
  基线 7,219→3,557 即 −51%，CED 12,536→9,897 即 −21%），**不打算再往算子级拆**。
* 冷 prefill 的 wall 在 5.6 s 与 22.9 s 之间波动（同机同长度），**未归因**。

## 6. 资源与工具

* a3-21：chip0–15 全部为本项目所用（P 0–7、D 8–15）。
* a3-22：本轮未使用；**chip0/1 曾由用户划给 1+1 实验**，恢复前先确认。
* 关键工具（`tools/`）：
  `ced_pd_acceptance.py`（验收 runner，含 `--selfcheck`）、
  `ced_pd_bench.py`（prefill tok/s 与 decode ms/step，流式逐块计时）、
  `bench_concurrency.py`（并发吞吐；**走代理时必须 `--tokenize-url` 指到 P**）、
  `ced_prof_streams.py`（profiler 流级分析）、
  `ced_pair_dump_verdict.py` / `ced_maxid_sweep.py`（块号取证）。
* 探针（a3-21 `~/tmp/20260924/ced_numeric/`）：`pfx_probe.py`、`pfx_probe2.py`、
  `pfx_fullhit.py`、`pfx_interleave.py`、各 `launch_*.sh`。
* 大文件走 COS（`bash scripts/cos-put.sh` / `cos-pull.sh`），不要起临时 http server。

## 7. 环境坑

* **git push**：`~/.gitconfig` 里 `http.proxy=https.proxy=http://127.0.0.1:14514`
  是**死的**，会让 push 报 `TLS connect error: unexpected eof`（而 `curl github.com`
  返回 200）。本仓库已加 `git config --local http.proxy ""` 覆盖，**未动全局配置**。
* 容器内 profiler 目录属主是 root，宿主 `ls` 会 Permission denied，
  要用 `docker exec <容器> bash -lc ...` 读。
* `docker exec` 的离线分析（`torch_npu.profiler.profiler.analyse`）可以在一次性容器里
  对只读挂载的捕获做，不需要停服务。
