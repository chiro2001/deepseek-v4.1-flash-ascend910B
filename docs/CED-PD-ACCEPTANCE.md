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

**投前必做：先跑长度校准**，确认 `/tokenize`、语料长度与目标长度都能对上，再发推理：

```bash
python3 tools/ced_pd_acceptance.py --calibrate-only \
  --base-url http://127.0.0.1:18990 --tokenize-url http://127.0.0.1:18990 \
  --model deepseek-v41-ced-pd --corpus data/hongloumeng.txt \
  --context-tokens 144000,1000000
```

2026-09-24 在 A3-21 的真实 P 上实测（`max_model_len=1048576`）：

```text
[calibrate] 探针 '你好，这是一次 tokenize 自检。' -> 10 tokens
[calibrate] 目标   144000  针 A/B/C/D  含提问=143993/143991/143992/143990  偏差≈0.006%  OK
[calibrate] 目标  1000000  针 A/B/C/D  含提问=999401/999399/999400/999398  偏差≈0.060%  OK
[calibrate] 不达标项 0（需 0）
```

语料用 `data/hongloumeng.txt`，SHA-256
`a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578`（826,651 字符），
**与早期 144K/256K/520K/1M 扫描使用的是同一份语料**，所以新旧结果可直接对照。
注意它只有约 82 万字符，1M token 的目标会把语料重复约 1.2 遍。

### 3.0 判据为什么不是"包含即通过"

`judge()` 要求：答案里出现期望串**且**长度 ≤200 字符**且**不含针文本特征词
（`运维备忘`/`校验码是`/`请只回复`/`只给`）。只做子串匹配会被"把题面复述一遍"
骗过——模型照抄含针的原文，期望串自然出现。`--selfcheck` 里有三条负控专门钉住这点：
答错、复述题面、拖沓长答案（含码但 >200 字符），三者都必须判 FAIL。

| # | 项 | 判据 | 证据 |
|---|---|---|---|
| 1 | 22-token 短针 | 精确答 `ZQ7K-3341` | `short22.result.json` |
| 2 | 144K 四针 A/B/C/D | 4/4 命中（`ZQ7K-3341/VX2M-8890/HT4P-5527/RB9N-6014`），`u_fffd=0` | `needle144k_*.result.json` |
| 3 | 1M 四针 A/B/C/D | 同上，且**重复 ≥2 轮全过** | `needle1000k_*.result.json` |
| 4 | 流式 | 答案正确 + 记录 TTFT + `finish_reason` | `stream*.result.json` |
| 5 | 多轮 | 同会话三轮问不同针，逐轮正确（状态未被带坏）。**必须核对证据里的 `usage.prompt_tokens`**：2026-09-25 之前 runner 把语料按 `target//8` 构造，声称 144K/1M 实际只有 18K/125K（见 [`../evidence/ced_acceptance_20260925/README.md`](../evidence/ced_acceptance_20260925/README.md) §2）；现已改为按 `max_model_len` 反推并在缩水时打警告 | `multiturn*.result.json` |
| 6 | 缓存命中 | 第二次同前缀 `cached_tokens > 0` | `prefix*.result.json`。**CED 配置下记 N/A**：原型硬门禁止 `PREFIX=1`，见第 7 节 |
| 7 | 性能 | prefill / TTFT / TPOT / 吞吐 / KV 占用 | 结果的 `wall_s`/`ttft_s`/`tpot_ms` + `metrics_before/after` |
| 0 | runner 自证 | `--selfcheck` 通过（内置 mock + 负控） | 退出码 0 |
| 0b | 修复不变量 | `python3 tools/ced_swa_clip_verify.py --lint-code --lint-server --sweep-max 2999` 退出码 0 | 见 3.1 |
| 0c | 长度校准 | `--calibrate-only` 不达标项 0 | 见上方示例 |

> `--metrics-urls` **只列 P 和 D**：官方负载均衡 proxy 没有 `/metrics`（实测 404），
> 列进去只会得到一条被记录的 error 条目；D 未启动时同理（connection refused）。

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

## 7. 目标里「缓存命中」这一项目前**无法验收**（是缺口，不是待测）

目标要求「144K 与 1M 上下文、流式/多轮/**缓存命中**正确性验证」。今天 CED 原型
**在启动层就禁止**了缓存命中，所以这一项不是"还没跑"，而是"跑不了"：

```bash
# scripts/serve_a3_ced_pd.sh 开头的硬门
for setting in "SPEC:${SPEC:-0}" "PREFIX:${PREFIX:-0}" "DRAFT_GRAPH:${DRAFT_GRAPH:-0}"; do
  ...
  if [ "$value" != 0 ]; then
    echo "[a3-ced][FAIL] $key=$value；当前 CED replay 原型要求 $key=0" >&2
    exit 2
```

即 `PREFIX=1` 会被直接拒绝。要把它变成可验收项，至少要先处理两处代码级前提：

1. **上半层 SWA 的"清零"不变量会被缓存块绕过。**
   `experimental/ced/mooncake_hybrid_connector.py` 的 D 侧预清零目标是
   `blocks.get_unhashed_block_ids_all_groups()`，即**本次新分配**的块。
   开启前缀缓存后，被复用的命中块是 hashed 的、**不在**这个集合里；而 CED 的 P
   在层 20 截断、上层（G7–G11）SWA 从未由 P 计算过，所以那些缓存页里是**上一次
   请求的残留**。这与 9 月 24 日定位的那类"读到不属于本请求的页"是同一族问题，
   只不过这次来源是缓存块而不是空块。要开缓存，必须先决定：缓存块参与清零，
   还是把 G7–G11 整体排除在前缀缓存之外（后者更省事，但要动 cache 分组配置）。
2. **调度器的边界断言假定"恰好从 P 装载 N−1 个 token"。**
   `experimental/ced/core_scheduler_replay.patch` 里
   `if replay_end != prompt_len - 1 or request.num_computed_tokens != replay_end: raise`。
   有前缀命中时 D 本地已算 K 个 token、只从 P 取 `N−1−K` 个；虽然装载完成后
   总数仍等于 `N−1`（断言可能仍成立），但 `replay_start` 之前的区间是否真的能由
   缓存提供、以及"末 token 必重算"的语义在有缓存时是否仍成立，都需要单独推导与
   真机验证。**这一条是分析结论，未在真机上证明会失败。**

因此本文的验收矩阵里，「缓存命中」一项在 CED 配置下记 **N/A（原型未支持）**，
不要用 `PREFIX=0` 的结果去填这一格。要覆盖它，需要一个 `PREFIX=1` 的独立原型臂，
并按上面两条先做单变量验证。

> 相关的历史口径：全 40 层 `AscendStore` 臂曾经跑通 1M 且四针 4/4 正确
> （见 [`../evidence/ced_8x8_length_scan_20260924/baseline_pdstore_bf16_needle1m.json`](../evidence/ced_8x8_length_scan_20260924/baseline_pdstore_bf16_needle1m.json)），
> 但那是**另一套连接器与 DSpark 配置**，不能当作 CED 的缓存命中证据。
