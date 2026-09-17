# 多 batch / 多轮对话 / prefill-decode 混合负载 验证

> 2026-09-16 21:12 CST｜A3-node1 chips 8-15｜容器 `dsv41-a21-perf`｜**生产口径** `MAX_SEQS=32 + PREFIX=1`
> 工具：`exp_tools/multibatch_gate.py` + `exp_tools/multibatch_session.sh`｜原始日志 `/tmp/multibatch_session.log`、`logs/perf/mbgP/`

---

## 0. 为什么做这个（此前的空白）

用户提问「我们是否进行过多 batch 对话的正确性验证」「是否测过 prefill+decode 一起推理」。
查证结果：

* 历史并发测试（GSM8K-200、C-Eval）用的是 `--conc 4 --serialize-prefill 1`，
  而该开关的语义是 **"hold a global lock until first token (avoid concurrent prefills)"**
  ⇒ **decode 并发、prefill 被故意串行化** ⇒ **prefill 与 decode 同时跑从未测过**；
* **多轮对话从未测过**（GSM8K 的 chat 8-shot 是单轮里的 few-shot，不是增长历史）；
* 而 A2 生产配置是 `max-num-seqs 32` + **prefix caching ON** —— 前缀缓存的本质就是
  "让 decode 队列里随时插入新请求"，**该边界在生产中天然存在**。

---

## 1. 结果（全部通过）

**起服**：`MAX_SEQS=32 PREFIX=1`，READY，`static_kernel.py:650` 降级 = 0，
`CAPTURE_SIZES` 自动扩到 15 个桶（`1,2,3,4,6,8,12,16,20,24,32,40,48,96,192`）。

### [A] 多轮对话（8 轮增长历史 + 针召回）

第 1 轮埋 3 个随机 10 位串，之后每轮考一个；最后从全长历史再各问一次。

| 项 | 结果 |
|---|---|
| 轮内召回 | **7/7** |
| 末次全长召回 | **3/3** |
| 每轮 `uniq2` | **1.00**（无复读） |
| 最终 prompt_tokens | 409 |
| 每轮延迟 | 0.2–0.3 s |

逐轮：`turn2 k0 OK / turn3 k1 OK / turn4 k2 OK / turn5 k0 OK / turn6 k1 OK / turn7 k2 OK / turn8 k0 OK`，
最后 `recall k0/k1/k2` 全 OK。

### [B] 并发 batch（`conc=1` vs `conc=8`，**逐 item 比对**）

同一组 16 道算术题，两种并发度各跑一遍，逐题比对：

| 臂 | 正确 | 墙钟 |
|---|---|---|
| `conc=1`（串行基线） | **16/16** | 3.7 s |
| `conc=8`（并发） | **16/16** | 8.2 s |
| **逐项不一致** | **0 项** | — |

### [C] ★ prefill + decode 真正同时跑（此前的空白）

1 条 **128K** 请求先启动，**2 s 后**并发 6 条短请求（`conc=3`）。

| 臂 | 正确 |
|---|---|
| 基线（短请求单独跑） | **6/6** |
| **与 128K prefill 并发** | **6/6** |
| **逐项不一致** | **0 项** |

长请求本身也正常返回（131,072 prompt tokens，输出为连贯的红楼梦原文：
`，什么没看过的戏，我不去。"凤姐道："他们那里凉快，两边又有楼…`）。

⇒ **结论：在"128K prefill 与多条短请求 decode 同时进行"这一此前从未覆盖的边界下，
未观察到任何正确性退化。**

---

## 2. 局限（必须一并写进交付）

1. **难度天花板**：B 块的 16 题是简单算术，两种并发度都 16/16 ⇒ 该测试能抓"整体性损坏"，
   **抓不到细微退化**。真要提灵敏度需换成"容易算错/需要长推理"的题，或加大样本。
2. **多轮历史偏短**：8 轮只到 **409 tokens**，没有把长历史（几 K）+ 前缀缓存命中逼出来。
3. **单次测量**：每块只跑 1 遍，未重复；真实置信度有限。
4. **只测了 `PREFIX=1` 这一臂**（生产对齐）。`PREFIX=0`（每请求都真 prefill、混合更凶）
   那一臂**尚未跑** —— 已排入队列。
5. 未与正确性线的 `spread`（数值确定性）判据交叉 —— 本报告只覆盖**功能性正确性**，
   不涉及"是否逐位确定"。

## 3. 复现命令

```bash
ssh A3-node1
TAG=mbgP MSEQS=32 PREFIX=1 CONC=8 ROUNDS=8 \
  bash ~/projects/dsv41/exp_tools/multibatch_session.sh
# 结果：~/projects/dsv41/logs/perf/mbgP/{summary.json,multiturn.json,concurrency_c8.json,mixed_long_short.json}
```
