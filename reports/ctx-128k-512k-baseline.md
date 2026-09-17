# 128K / 512K 输出 tok/s 基线（AllGather 生产配置）

> 2026-09-16 08:21–08:24 CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`（port 8020）
> 配置：`static_kernel=1 npugraph_ex=1 MOE_AG=1 FUSED_MC2=1 MULTISTREAM=0 SP_TOKENS=7 GPU_UTIL=0.94`
> Engram(int8, gate CHUNK=0, local-owner, hash fast) + Vision + DSpark(num_spec=7, SPEC_EAGER=1)
> KV cache size: **4,161,011 tokens**（Available KV 17.13 GiB）
> 口径：T4 逐字引用 quote、单流独占、`exclusive_ok=True`、unprofiled 客户端墙钟

## 1. 同会话实测（本次）

| 上下文 | ms/step | A（接受长度） | **输出 tok/s** | TTFT | prefill tok/s |
|---|---|---|---|---|---|
| **128K** | 34.809 | 2.793 | **79.94** | 29.71 s | 4412 |
| **512K** | 46.001 | **1.503** | **32.55** | 125.93 s | 4163 |

证据：`logs/perf/a21/p42_t4_quote_131072_agbase_131072.jsonl`、
`logs/perf/a21/p42_t4_quote_524288_agbase_524288.jsonl`、`/tmp/a21_wait_measure_0816.log`

## 2. 128K 的重复性（三次独立会话，各 3 发中位）

| 会话 | ms/step | A | tok/s |
|---|---|---|---|
| AllGather #1 06:24 | 35.32 | — | ~78 |
| AllGather #2 06:52 | 35.10 | 2.763 | 78.4 |
| AllGather #3 07:44 | 35.14 | 2.763 | 78.6 |
| 本次 08:21 | 34.81 | 2.793 | 79.9 |

⇒ **128K 78–80 tok/s 稳定**（±1.5%）。

## 3. 512K 的关键发现：瓶颈在**接受率**，不在设备时间

- ms/step 只从 34.81 涨到 46.00（**+11.2 ms，+32%**，长上下文注意力成本）
- 但 A 从 **2.793 塌到 1.503（−46%）** ⇒ tok/s 掉 **59%**
- 按 `tok/s = A × 1000 / ms`：512K 若维持 A=2.79，应为 60.7 tok/s；实际 32.5

⇒ **512K 的优化优先级是"恢复接受率"，而不是继续压 ms/step。**

### 3.1 历史对照（不同配置，仅作量级参考）

| 配置 | 512K ms/step | A | tok/s | 出处 |
|---|---|---|---|---|
| gear A（Engram-**off**、BF16 KV、static=0） | 49.56 | 5.149 | 103.9 | `HANDOVER.md` §4 |
| R2b（Engram-on + vision） | ~49.6 | ~2.44 | 49.2 | `EXEC_PLAN_VB.md:63` |
| **本次（Engram-on + static + AllGather）** | **46.00** | **1.503** | **32.55** | 本报告 |

同配置下 512K 的 ms/step 已是历史最好，**差距全在接受率**。

## 4. 附带发现：一个不能用的开关组合（landmine）

`MOE_AG=0 FUSED_MC2=0 MULTISTREAM=1`（本次为做 MC2 对照而起的 `a21_mc2ck_0801`）
服务能 READY，但**输出完全崩溃**：GSM8K-200 仅 **1/200**，回复为乱码
（如 `"Let,s fris3zg,2.ama IN smaller is"`）。
⇒ 该组合 **禁止用于任何 A/B 或验收**；是 `MOE_AG=0`（走 MC2 选择逻辑）还是
`FUSED_MC2=0` 导致，尚未定位（下一轮单独二分）。
证据：`~/lmeval_out/mc2_ck_gsm8k200.json`、`logs/perf/a21_mc2ck_0801_serve.log`
