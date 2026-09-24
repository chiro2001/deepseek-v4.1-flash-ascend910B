# CED-PD（P 只跑前 20 层 + D 128-token 有界重放）启动与验收

本文给出 **可复现的启动命令**、**验收矩阵与判据**、以及结果口径。适用于 A3
单机 16 卡（P 用 chip0–7，D 用 chip8–15，proxy 同机）。目标定义见仓库根
`AGENTS`/目标条目；本文只描述"怎么起、怎么判、结果怎么看"。

> **当前状态（2026-09-24）**：正确性根因已定位并修复（见
> [`CED-D-1M-LAYOUT-BUG-20260924.md`](CED-D-1M-LAYOUT-BUG-20260924.md)），
> 修复已合入 `experimental/ced/dsa_v41.py`（`V41_CED_SWA_CLIP`，默认 `1`）。
> 已验证：算子源码级前提（7 条）、离线逐长度判据、修复不变量的静态 lint、
> **tiny 单卡真机在 `GRAPH=1 EAGER=0` 下裁剪分支按设计执行**（见
> [`../evidence/ced_swa_clip_tiny_ab_20260924/README.md`](../evidence/ced_swa_clip_tiny_ab_20260924/README.md)）。
> **尚未验证**：8+8 真权重下的碎片形态重复通过率、`V41_CED_SWA_CLIP=1/0` 单变量
> A/B、以及第 4.1 节的受控性能对照。在补齐之前，本文的启动配置标记为"实验臂"，
> 不能当交付口径；按用户 2026-09-24 的"先不动"指示，这部分实验已挂起。

## 1. 拓扑与前置

| 角色 | 芯片 | 本文口径端口（脚本默认是 18550/18551，需显式覆盖） | KV 端口 | 容器名前缀 |
|---|---|---|---|---|
| P（CED prefill） | 0–7 | 18990 | 19090 | `dsv41-ced-*` |
| D（CED decode） | 8–15 | 18991 | 19091 | `dsv41-ced-*` |
| proxy | — | 18992 | — | `dsv41-ced-*-proxy` |

前置：模型为完整 W4A8 权重目录；镜像 `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`；
起服前用 `bash tools/list_chips.sh` 确认 16 张卡没有别人的进程（**不要 kill 非本项目的进程**）。

## 2. 启动

### 2.1 P（前 20 层 + 层 20 全局源投影）

```bash
cd <repo>
export MODEL=<完整模型目录>
export NAME=dsv41-ced-p-$(date +%m%d_%H%M%S) RUN_ID=ced_p_$(date +%m%d_%H%M%S)
export DEVS="0 1 2 3 4 5 6 7" PORT=18990 KV_PORT=19090
export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0
export STATIC_KERNEL=0 NPUGRAPH_EX=1 MULTISTREAM=1 DSA_OVERLAP=1
bash scripts/serve_a3_ced_pd.sh prefill
```

### 2.2 D（128-token 有界重放 + 完整 40 层 decode）

```bash
export MODEL=<完整模型目录>
export NAME=dsv41-ced-d-$(date +%m%d_%H%M%S) RUN_ID=ced_d_$(date +%m%d_%H%M%S)
export DEVS="8 9 10 11 12 13 14 15" PORT=18991 KV_PORT=19091
export MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 GPU_UTIL=0.92 BLOCK=128
export KV_DTYPE=bfloat16 ENGRAM=1 ENGRAM_DEVICE_INDEX=0 CPU_BIND=0
export STATIC_KERNEL=0 NPUGRAPH_EX=1 MULTISTREAM=0 DSA_OVERLAP=0
# 图模式 + prompt 尾步 eager + metadata 主流；SWA clip 修复默认开。
export CED_EXPERIMENTAL_GRAPH=1
export V41_CED_GRAPH_PROMPT_TAIL_EAGER=1 V41_CED_METADATA_INLINE=1
export V41_CED_SWA_CLIP=1
bash scripts/serve_a3_ced_pd.sh decode
```

`scripts/serve_a3_ced_pd.sh` 对 decode 角色**要求显式选择诊断臂**
（`CED_DIAGNOSTIC_EAGER=1` 或 `CED_EXPERIMENTAL_GRAPH=1`），这是防止把
eager 或未验收的图模式误当交付的 fail-closed 设计，不要绕过。

### 2.3 proxy

```bash
export NAME=dsv41-ced-proxy-$(date +%m%d_%H%M%S)
export PROXY_PORT=18992 PREFILL_PORT=18990 DECODE_PORT=18991
bash scripts/serve_a3_pd_proxy.sh
```

### 2.4 起服硬门（每条都要看）

```bash
curl -sf http://127.0.0.1:18990/health && curl -sf http://127.0.0.1:18991/health
curl -sf http://127.0.0.1:18992/v1/models | head -c 200
# D 侧必须看到（缺任何一条都说明跑的不是本文口径）：
#   [CED-META] inline metadata group=...            （metadata 主流）
#   [CED-GRAPH] one-token prompt tail forced eager  （prompt 尾步 eager）
#   Replaying aclgraph                              （生成仍走图）
```

## 3. 验收矩阵

用仓库内的 runner 一次跑完（判据与 A3-21 既有证据同一套四针）：

```bash
python3 tools/ced_pd_acceptance.py \
  --base-url http://127.0.0.1:18992 \
  --tokenize-url http://127.0.0.1:18990 \
  --model deepseek-v41-ced-pd \
  --corpus data/hongloumeng.txt \
  --mode all --context-tokens 144000,1000000 \
  --repeat 2 \
  --metrics-urls http://127.0.0.1:18990,http://127.0.0.1:18991 \
  --out results/ced_acceptance_$(date +%m%d_%H%M%S).json
```

工具本体可用 `--selfcheck` 离线自证（内置 mock 服务 + 负控）：`python3 tools/ced_pd_acceptance.py --selfcheck`。

| # | 项 | 判据 | 证据 |
|---|---|---|---|
| 1 | 22-token 短针 | 精确答 `ZQ7K-3341` | `short22.result.json` |
| 2 | 144K 四针 A/B/C/D | 4/4 命中（`ZQ7K-3341/VX2M-8890/HT4P-5527/RB9N-6014`），`u_fffd=0` | `needle144k_*.result.json` |
| 3 | 1M 四针 A/B/C/D | 同上，且**重复 ≥2 轮全过** | `needle1000k_*.result.json` |
| 4 | 流式 | 答案正确 + 记录 TTFT + `finish_reason` | `stream*.result.json` |
| 5 | 多轮 | 同会话三轮问不同针，逐轮正确（状态未被带坏） | `multiturn*.result.json` |
| 6 | 缓存命中 | 第二次同前缀 `cached_tokens > 0`（`PREFIX=0` 时此项按"不适用"记录） | `prefix*.result.json` |
| 7 | 性能 | prefill / TTFT / TPOT / 吞吐 / KV 占用 | 结果的 `wall_s`/`ttft_s`/`tpot_ms` + `metrics_before/after` |
| 0 | runner 自证 | `--selfcheck` 通过（内置 mock + 负控） | 退出码 0 |
| 0b | 修复不变量 | `python3 tools/ced_swa_clip_verify.py --lint-code` 退出码 0 | 见 3.1 |

### 3.1 修复专项（`V41_CED_SWA_CLIP` 单变量 A/B）

同一 P、同一请求模板，只切 D 的 `V41_CED_SWA_CLIP`：

```bash
# A 臂（旧行为，应能复现失败）
V41_CED_SWA_CLIP=0 bash scripts/serve_a3_ced_pd.sh decode
# 跑 8 次 1M；记录 [CED-BLOCKS] 的 descents 与首 token
# B 臂（修复）
V41_CED_SWA_CLIP=1 bash scripts/serve_a3_ced_pd.sh decode
```

判据：**碎片形态（`descents>0`）下 ≥10 次全过**；连续形态不退化；仍是
`GRAPH=1 EAGER=0`。注意 1M 路径本身不是逐位可复现（通过-通过之间 digest 也有
差异），所以看**失败率**，不要看逐位相等。

## 4. 性能口径

### 4.0 已有的指示性数字（**不是受控对照**，缺 4.1 的同条件矩阵）

同一台 A3-21、同为 1M 级 prompt（1,019,847 token）、`temperature=0`、非流式、
`max_tokens=64`、实际输出 7 token：

| 配置 | 1M 单请求 wall | 来源 |
|---|---:|---|
| CED（P 只跑层 0–19 + 层 20 全局源投影） | **~102.7 s**（8 次：102.1/104.8/102.8/103.4/102.8/101.8/101.7/102.7） | 2026-09-24 A3-21 实测（`metadata-inline` 臂） |
| 全 40 层（direct full40，无 CED role、无 KV connector） | **~288.8 s**（291.1/289.6/287.7/286.8） | [`../evidence/ced_8x8_length_scan_20260924/README.md`](../evidence/ced_8x8_length_scan_20260924/README.md) |

比值约 **2.8×**，方向与"P 跳过 20 层 decoder 计算"一致。**但两臂的连接器、
DSpark 配置与代码版本不同，不能当受控结论**；要写进交付必须按 4.1 重测。
另外要明确：CED P 仍然加载**完整权重**（实测约 39.5 GB/rank），省的是**计算**
不是显存；论文那部分显存收益来自 global/SWA 分层 TTL，我们尚未实现（见第 6 节）。

### 4.1 受控对照流程（交付前必须跑）

两臂用**同一 prompt 字节、同一 `max_tokens`、同一 `MAX_SEQS/BAT_TOKENS/GPU_UTIL`**，
每臂重复 ≥3 次取中位数：

```bash
# A 臂：全 40 层双 TP8 PD 基线
MODEL=$MODEL MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 \
  DEVS="0 1 2 3 4 5 6 7" PORT=18990 KV_PORT=19090 \
  bash scripts/serve_a3_pd.sh prefill
MODEL=$MODEL MAX_LEN=1048576 MAX_SEQS=4 BAT_TOKENS=8192 \
  DEVS="8 9 10 11 12 13 14 15" PORT=18991 KV_PORT=19091 \
  bash scripts/serve_a3_pd.sh decode
# （+ proxy 18992），然后跑第 3 节 runner 的 --mode needle

# B 臂：CED（启动命令见 2.1/2.2），同一 runner、同一 prompt
```

要记录的对照量：`wall_s`（非流式）、`ttft_s`（流式）、`tpot_ms`、completion
tokens、以及 `--metrics-urls` 抓到的 P/D KV 占用与请求计数。**报告时必须同时给
接受长度**，否则 tok/s 会被投机解码的文本可预测性放大 3× 以上
（见 [`BENCH-METHODOLOGY.md`](BENCH-METHODOLOGY.md)）。

- **prefill 时间**：以 `needle1000k` 的 `wall_s`（非流式）或流式 TTFT 近似；
  要与全 40 层双 TP8 PD 基线**同 prompt、同 max_tokens、同并发**对照。
- **TTFT**：流式首 token 延迟（`stream*.result.json` 的 `ttft_s`）。
- **TPOT**：`1000 * (wall_s - ttft_s) / completion_tokens`（runner 已算 `tpot_ms`）。
- **吞吐**：固定 prompt/输出长度下的 completion tokens/s；注意文本可预测性会通过
  投机解码把吞吐拉高 3× 以上（见 [`BENCH-METHODOLOGY.md`](BENCH-METHODOLOGY.md)），
  所以必须同时报接受长度，不能只报 tok/s。
- **资源**：`--metrics-urls` 抓到的 KV 占用与请求计数，P/D/proxy 各一份。

## 5. 证据布局

`--out results/xxx.json` 汇总；`--out-dir`（默认同名 `_evidence/`）逐请求落
`<tag>.request.json`（实际发送字节，含 SHA-256）、`<tag>.result.json`
（状态/时延/答案/判据/usage）。归档时对证据目录跑 `sha256sum` 汇总，
不要只留汇总 JSON。

## 6. 与论文的差距（验收范围之外，供解释结果用）

对齐情况见 [`ARCHITECTURE_PD_ANALYSIS.md`](../../pd_single_a3/ARCHITECTURE_PD_ANALYSIS.md)：
global/SWA 分层 TTL、CSA2 候选池跨节点一致性、vision/EPD 独立扩缩（**已明确不做**）
与 4+4 拓扑都尚未实现。因此本文验收的是 **CED 计算切分 + 有界重放**这条主线，
不是论文的完整系统。
