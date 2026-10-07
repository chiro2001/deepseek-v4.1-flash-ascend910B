# ★ 更正 S13：conc=2 的"只落后 7%"是空结果（DBO 根本没触发）（2026-10-07）

> 承接 `S13-CBP-STEP2-DBO-2STREAM`。本文**撤回 S13 的核心结论**，并给出真正有效的测量。
> 环境：tiny（`dsv41-tinyspark`），DCP=1，graph 模式，dummy 权重，SPEC=1 SP_TOKENS=7。全部【实测】。

---

## 0. 一页纸

| 项 | S13 结论 | **更正后** |
|---|---|---|
| DBO 在 conc=2 有效吗 | "0.93×，只落后 7%" | ❌ **根本没触发**（16 token < 阈值 32） |
| 真正的 DBO @ conc=2 | — | **38.5 tok/s vs 基线 67.0 ⇒ 0.575×** |
| 与历史一致吗 | "远好于 −42%" | ✅ **与历史 −42% 完全一致**（本轮 −42.5%） |
| 两条 ubatch 流并行吗 | 未测 | ❌ **几乎不并行**：200ms 窗口内两主流同时活跃仅 **0.4ms** |

---

## 1. 根因：`check_ubatch_thresholds` 的判定

```python
# vllm/v1/worker/ubatch_utils.py:38
def check_ubatch_thresholds(config, num_tokens, uniform_decode) -> bool:
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold   # 默认 32
    else:
        return num_tokens >= config.dbo_prefill_token_threshold  # 默认 512
```

**在本项目的参数下**（SPEC=1、SP_TOKENS=7 ⇒ 每请求每步 8 token）：

| 场景 | token 数 | 阈值 | 触发？ |
|---|---:|---:|---|
| conc=2 decode | 2×8 = **16** | 32 | ❌ **不触发** |
| conc=4 decode | 4×8 = **32** | 32 | ✅ 触发 |
| prefill（1024 prompt ×2） | 2048 | 512 | ✅ 触发 |

**日志佐证**：所有 `[DBO-CAT0]` 的输入形状都是 `Tensor(1024, 5120)`
—— **1024 正是 prefill 的 token 数**，说明观测到的 ubatching 活动全部来自 prefill。

⇒ **S13 测的 conc=2 "DBO 臂"实际跑的是"带 DBO 环境变量但 decode 不 ubatch"的配置**，
那个 −7% 只能来自 prefill ubatch 的残留/重启波动，**不是 DBO 的 decode 效果**。

---

## 2. 决定性实验：把阈值降到 8，强制 conc=2 触发

```bash
--enable-dbo --all2all-backend=deepep_low_latency --dbo-decode-token-threshold 8
```

| conc | DBO(thr=32，不触发) | **DBO(thr=8，触发)** | **DCP=1 基线** | **触发后 vs 基线** |
|---:|---:|---:|---:|---:|
| 2 | 62.1 | **38.5**（38.0/38.5/38.5） | **67.0** | **0.575×** |
| 4 | 64.4 | 63.6（62.9/63.6/64.2） | **107.5** | **0.592×** |

**⇒ 真正的 DBO**：conc=2 **−42.5%**、conc=4 **−40.8%**，
**与 DBO 线的历史记录（−42%）完全吻合。**

---

## 3. Profiler：两条 ubatch 流几乎不并行

`prof_dbo_thr`（threshold=8，conc=2，含 prefill）device 最忙 200ms 窗口：

| 指标 | 值 |
|---|---|
| kernel 数 | 24875（**124 个/ms**） |
| 有 kernel 在跑 | 93.5% |
| **≥2 kernel 同时** | **6.3%** |
| AIC 忙 | 40.1% |
| AIV 忙 | 40.5% |

**流分布**：

| stream | kernel 数 | 活跃时长 |
|---|---:|---:|
| **178** | 7615 | **76.2 ms（38.1%）** |
| **179** | 7202 | **71.8 ms（35.9%）** |
| 175 | 3619 | 10.6 ms |
| 47 | 2807 | 10.1 ms |

**并发时刻的流组合**：

| 组合 | 时长 |
|---|---:|
| (176, 178) | 3.6 ms |
| (176, 179) | 3.3 ms |
| (175, 178) | 2.3 ms |
| (175, 179) | 2.2 ms |
| **（178, 179）** ← 两条 ubatch 主链 | **0.4 ms** |

**⇒ 两条 ubatch 各自占用一整条流（178/179，各 72~76ms），但它们之间只重叠 0.4ms。**
　 并发全部来自**辅助流（175/176）与主链**的重叠。

**这与 `_run_ubatches_graph` 的代码不符**（那段代码显式把 ubatch1 放 side 流、ubatch0 放 root 流）。
⇒ **运行期很可能走的是 threading 路径（`_run_ubatches`），而它所有 ubatch 共用一条 `compute_stream`**
（`_make_npu_ubatch_contexts` 只接受单个 `compute_stream` 参数）。
　 日志里 `[DBO-GRAPH] capturing=False -> thread` 支持这个判断。

---

## 4. 更正后的 CBP 账

**判据**：`tax < S(N)`，其中 `S(2) = 1.137`（DCP=1 基线）。

| 方案 | 实测 tax(2) | 相对门槛 |
|---|---:|---|
| **DBO（阈值强制触发）** | **2 × 38.1 / 38.5 = 1.98** | ❌ **超门槛 74%** |
| S4 微基准（两条独立链） | 1.166 | ⚠️ 略超 2.5% |
| 理论同相位（S2） | 1.670 | ❌ |

**⇒ DBO 这条实现路径距离门槛还很远（1.98 vs 1.137），不是"只差 8%。"**

---

## 5. 下一步的候选

| # | 动作 | 理由 |
|---:|---|---|
| 1 | **确认运行期走的是哪条路径**（thread vs graph） | 若走 thread ⇒ 改成 graph 路径可能拿到 `_run_ubatches_graph` 的真并行 |
| 2 | **量化 DBO 开销的构成**（算子数翻倍 vs 元数据 vs 同步） | 决定是否可回收 |
| 3 | **对照真 DP**（2 个独立实例，各 1 请求） | 这是"不拆批、独立前向"的天然形态，vLLM 原生支持；若 DP 赢 ⇒ 直接用它 |

---

## 6. 复现

```bash
# 强制触发的 DBO 臂
ssh a3-21 'bash /tmp/launch_dbo_thr.sh 8'
# 基线（DCP=1，无 DBO）
ssh a3-21 'bash /tmp/launch_armMcp1.sh'
# 测量
ssh a3-21 'cd ~/cedpd-repo && python3 ~/tmp/bench_conc.py --base-url http://127.0.0.1:19310 \
  --model deepseek-v41 --concurrency 2,4 --prompt-tokens 1024 --output-tokens 96 --repeats 3'
```

配置台账：`~/cedpd-repo/results/TINY-CONFIG-LOG.md`（**不再每次都恢复 tiny；只记录**）
