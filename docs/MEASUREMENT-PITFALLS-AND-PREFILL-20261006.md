# 测量陷阱（prefix cache / 重复退化 / 冷启动）+ 更正后的 prefill 曲线（2026-10-06）

> 本轮先是在测 tp8k5 时**踩中三个测量陷阱**，随后发现 tp8k5 被外部停掉并**恢复**了它。
> 三处此前的数字必须更正。全部为【实测】。

## 0. 更正清单

| 此前结论 | 更正 |
|---|---|
| 「并发 prefill 比单请求快 **5.5×**（95,730 tok/s）」 | ❌ **prefix cache 污染**。干净实测：conc=8 只有 **0.59×**（并发更慢） |
| 「单请求 prefill **17.4K tok/s**」 | ❌ 同样是缓存污染。干净实测：**7.0~7.5K tok/s** |
| `docs/TP8-CURVE-CORRECTED-20261006.md` 的 decode 曲线（conc=32 → 1413 tok/s） | ✅ **仍然成立**（decode 段不受 prefill 缓存影响；见 §1.3 的核验） |

## 1. 三个测量陷阱

### 1.1 prefix cache（tp8k5 的 `PREFIX=1`）

**同一个 8208-token prompt 的 TTFT**：

| 次序 | TTFT |
|---|---:|
| 冷（首次） | 约 **0.59 s** |
| 热（第二次，命中缓存） | **0.172 s** |

conc=4 的内部对照更直观：`min TTFT 0.176s`（缓存命中）vs `max 0.370s`（未命中）。

⇒ **任何 prefill/TTFT 测量必须用唯一 nonce 破缓存**，并用
`vllm:prompt_tokens_cached_total` 的增量**核验命中率为 0**。
（我此前的 4K/16K/64K 三点用的是**嵌套前缀**，后面的点自动命中了前面点的缓存。）

### 1.2 重复退化（同一 prompt 反复生成）

同一 prompt 反复生成会让模型进入**逐字重复**：

```
rep(20-gram 最高频占比) = 0.092
尾部: '用一句话概括上文。 请用一句话概括上文。 …' 反复
```

退化文本对投机解码是"白送"：**接受长度被虚高到 4.97/5**（≈ SP_TOKENS 上限）。

**判别实验**：conc=1、**完全相同**的 prompt 连跑两次 ⇒ 接受长度 **4.97 vs 1.95**，
吞吐 **236.7 vs 119.6 tok/s**。同一输入、同一并发、`temperature=0`，差 2×
⇒ 不是并发效应，是**内容退化与否**。

⇒ 吞吐测量必须用**互不重叠的切片 + 轮换任务后缀**，并抽样检查重复度。

### 1.3 冷启动（重启后的首批请求要编译）

服务刚重启时，首批请求会触发 kernel 编译。实测**同一个脚本、同样的长度**：

| 时点 | 4096 tok 的 TTFT |
|---|---:|
| 刚重启、无预热 | **8.180 s** |
| 预热之后 | **0.553 s** |

⇒ 测量前必须**先发几条预热请求**（跨过若干长度档），否则数字完全不可用。

（decode 曲线为何不受影响：它读的是**窗口期**的 drafts/generation 增量，
窗口在 5 s 稳定期之后才开，编译已完成——这次我补了 `num_requests_running == conc`
的等待条件与 `prompt_tokens_cached` 核验。）

## 2. 更正后的 prefill 曲线（单请求，nonce 破缓存，预热后）

```bash
python3 ~/tmp/tp8_prefill_clean.py http://127.0.0.1:19210 4096,16384,65536,131072 8
```

| prompt tokens | TTFT | **prefill tok/s** | 缓存命中 |
|---:|---:|---:|---:|
| 4,153 | 0.553 s | **7,504** | 0.0% |
| 16,414 | 2.228 s | 7,368 | 0.0% |
| 65,820 | 9.300 s | 7,077 | 0.0% |
| 131,317 | 19.805 s | **6,631** | 0.0% |

⇒ **单请求 prefill ≈ 7.0~7.5K tok/s**，随长度轻度衰减（131K 时 −12%，符合 chunk 内 O(n²)）。

## 3. ★ 新发现：prefill **不随并发获益，反而在 conc=8 崩塌**

整批 8 条 ≈8252 token 的 prompt 同时提交（每条带唯一 nonce，零缓存命中）：

| 并发 | 聚合 prefill | 相对单请求 | 各请求 TTFT（min / 中位 / max） |
|---:|---:|---:|---|
| 1 | **7,010 tok/s** | 1.00× | 1.177 / 1.177 / 1.177 s |
| 2 | 6,450 | 0.91× | 1.410 / 2.557 / 2.557 s |
| 4 | **7,607** | 1.07× | 1.176 / 4.159 / 4.340 s |
| 8 | **4,173** | **0.59×** | 1.435 / 13.644 / 15.817 s |

（服务端 `prompt_tokens` 增量与请求 token 数逐一对上，缓存 0.0% ⇒ 数据干净。）

**两条读法**：

1. **prefill 没有"合批红利"**：conc=2/4 与单请求基本持平（0.91× / 1.07×），
   说明它**不是带宽型**（不像 decode 的 MoE 权重流），多序列并排不会摊薄固定成本；
2. **conc=8 明显恶化（0.59×）**，且 TTFT 分布显示请求几乎被**串行化**（max 15.8 s ≈ 8 × 1.98 s）。

【推断】一个可测的次要因素：我的 prompt 是 8252 token，**刚好越过 BAT_TOKENS=8192 的块边界**
⇒ 每条要拆成 2 个 chunk，而**第二个 chunk 只有 60 token 却要付一整步**。
conc=8 时变成 16 步而不是 8 步。

**这是目前看到的唯一"明显低于设计点"的环节**，而且它带来的行动很便宜：
**控制 prefill 的并发准入**（宁可排队也不要同时 prefill），在 conc=8 场景下潜在收益可达 **1.7×**。

## 4. tp8k5 停服事件与恢复记录

### 4.1 事件

测量过程中 tp8k5 的 APIServer 于 **12:46:39 UTC 收到 SIGTERM** 并优雅退出
（日志显示 `Shutting down / Application shutdown complete`，**无任何报错**；
其前最后一行是常规 `/metrics` 轮询）。容器随后只剩僵尸进程。

### 4.2 恢复（第一次尝试失败 → 找到原因）

第一次恢复用 `serve_a2.sh` 的**默认** `DEVS="0 1 2 3 4 5 6 7"` ⇒ 失败：

```
ValueError: Free memory on device (8.53/61.27 GiB) on startup is less than desired
```

原因：**chips 0–3 已被别的负载占用**（`npu-smi` 实测）：

| chips | 占用者 | 每 chip |
|---|---|---:|
| 0–1 | `hlz-dsv4-dp2`（另一个用户，Up 6 days）的 `VLLMWorker_DP` | 53.6 GiB |
| 2–3 | **我们自己的 `dsv41-tinyspark`**（TP=2） | 27.5 GiB |
| 4–7 | 空闲 | ~3 GiB（驱动基线） |

而原始 tp8k5 用的是 **`devs='8 9 10 11 12 13 14 15'`**（不是 0–7）。

### 4.3 恢复结果（核验通过）

用原始设备列表与全部原参数重启后：

```bash
MODEL=/home/l00886679/models/out/v41-flat-verify3 IMAGE=local/dsv41-a3-tp8:20261005-1001 \
NAME=dsv41-tp8k5 PORT=19210 TP=8 DP=1 DEVS="8 9 10 11 12 13 14 15" GPU_UTIL=0.92 \
MAX_LEN=1048576 MAX_SEQS=32 BAT_TOKENS=8192 SP_TOKENS=5 ... PREFIX=1 DRAFT_GRAPH=1 \
bash scripts/serve_a2.sh
```

| 项 | 原 run（restore10） | 新 run（restore11） | 一致？ |
|---|---|---|---|
| health | 200 | **200** | ✅ |
| GPU KV cache size | 2,987,836 tokens | **2,987,509 tokens** | ✅（差 0.01%） |
| max_num_batched_tokens | 8192 | 8192 | ✅ |
| max_num_seqs | 32 | 32 | ✅ |
| SP_TOKENS / capture_sizes | 5 / 1..192 | 相同 | ✅ |
| 1M 上下文并发 | — | 2.85× | — |

## 5. 复现

```bash
# 干净的单请求 prefill（nonce + 零命中核验 + 需先预热）
ssh a3-21 'python3 ~/tmp/tp8_prefill_clean.py http://127.0.0.1:19210 4096,16384,65536,131072 8'
# 干净的高并发 prefill
ssh a3-21 'python3 ~/tmp/tp8_prefill_conc_clean.py http://127.0.0.1:19210 8 8192 8'
# 退化检测
ssh a3-21 'python3 ~/tmp/tp8_textcheck.py http://127.0.0.1:19210 1 256'
# tp8k5 占用者排查
ssh a3-21 'npu-smi info -t proc-mem -i 0'
