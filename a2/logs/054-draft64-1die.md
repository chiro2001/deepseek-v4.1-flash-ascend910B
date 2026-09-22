# 054 — ②c（draft block 128→64）在**单 die（A3 c2）**上的数值风险判定：输出逐字节不变、投机解码逐字不变、容量逐字按模型走

> 2026-09-22 11:45–1x:xx CST。执行：子代理 **C2_draft64**。机器：**A3（A3-node1）的 c2 = /dev/davinci7 = NPU 3 chip 1**（**只用 c2**，锁走 `tools/a3_chip.sh`）。
> 模型：`agents/T_draftceiling/models/model-tiny-draft`（= L1_dummy 的 tiny + **只还原 draft 两个键** `num_nextn_predict_layers=3` / `dspark_target_layer_ids=[37,38,39]`）。
> 影子包：`agents/C2_draft64/pkg/dfix2c`（= T_draftceiling 的 `Ci8fix` 基础包 + ②c 生成器 + 探针；**两臂同一个包，只有一个变量 `VLLM_V41_DRAFT_BLOCK`**）。
> 没写 `upstream-v41/`；没用 `/tmp`（`TMPDIR=~/tmp/20260922/C2_draft64`）；没碰 c0/c1/别人的容器；没手设 `ASCEND_RT_VISIBLE_DEVICES`。
> 标记约定：**【实测】**= 有判别力的原始数据；**【算术】**= 由源码几何算出（脚本可复跑）；**【推断】**；**【未确认】**。

---

## 0. 一句话结论

| 问题 | 答案 | 判据 |
|---|---|---|
| **Q1 ②c 改完输出变了吗？** | **没变**：B 档与 D 档各 **7 轮**（4 冷 + 3 连发）输出 `sha256` **逐字节相同**（`0ebccb55…`），**逐 prompt 16/16 相同**，且**跨臂**（128 vs 64）也 16/16 相同。**【实测】** | §3 |
| **Q2 投机解码退化了没有？** | **没有退化，是"逐字不变"**：两臂 `drafts=4239 / drafted=21195 / accepted=2902`、`MeanAccLen=1.685`、`AvgDraftAcc=13.69%`、`per-pos[0]=0.685` **完全一致**；★ 更硬的判据：draft **提案 token 序列 4367/4367 次逐条相同**（探针直读）。**【实测】** | §4 |
| **Q3 容量真的涨了吗？** | **B 档是"降"、D 档是"涨"，两个方向都逐字命中 050 的零参数模型** —— 这就是模型的**符号判别力**证据：B **20,826 → 19,247（×0.9242）**、D **23,651 → 36,825（×1.5570）**。**【实测】** | §5 |
| **tiny 能否替代 8 卡？** | **能替代"几何/寻址/不崩"这三维**（draft 组、真算子、真 KV 路径、逐字节判据都在场），**不能替代**"真权重数值 / TP8 通信 / 图模式 / 卸载池实测 / 绝对 token 数"这五维。详见 §7。**【实测 + 未确认】** | §7 |

---

## 1. 为什么这一条值得用 c2 的排队时间去换

`051` 把 ②c 的**改动面**（2 文件 / 2 处）与**三臂单元自检**做完了，但留下一条它自己标了 `【未确认】` 的风险：

> draft 窗口 `sliding_window=128` 在 `block=64` 下跨 **2–3 块**（带投机 query 共 133 token ⇒ 最坏 4 块）；
> "算子 / 块表 / KV manager 都没有 ≤1 块假设"**只是读源码得出的**，没跑过。

本条实验就是**把这条风险变成实测**，并且**只用单 die**（c0 的 8 卡队列还在等 ⇒ 不占它的时间）。

---

## 2. ★★ 实验设计（两臂只有一个变量 + 三档几何）

| 臂 | tier | `VLLM_V41_DRAFT_BLOCK` | 影子包 | 服务参数差异 | 用途 |
|---|---|---|---|---|---|
| `c2c-b128` | B（纯 BF16） | 128 | 同一个 `pkg/dfix2c` | 无 | B 档基线（= 逐字旧行为） |
| `c2c-b64` | B | **64** | 同上 | 无 | ②c 的**纯净臂**（除 draft 页外无其它变量） |
| `c2c-d128` | D（SWA int8 + ring16 + long-KV int8） | 128 | 同上 | 无 | 交付档基线 |
| `c2c-d64` | D | **64** | 同上 | 无 | ★ 交付档 + ②c（容量应**涨**） |
| `c2c-neg-b64` | B | 64 + **故意缺陷** `[C2-NEGCTRL v1]` | `pkg/neg2c` | 无 | ★★ **阳性对照**：判据若在反例臂上也"全同" ⇒ 判据没判别力 |

**自变量唯一性（可复核）**：两臂的 `PYTHONPATH`/`PGP_*`/`P2_*`/`VLLM_V41_KV8*` 全部逐字相同，唯一差别是 `VLLM_V41_DRAFT_BLOCK`。
影子包两文件 md5（本次四臂**同一对 md5**）：

```
dspark.py          0fd33532abcfcf7c3b43f8ed89d19881   （②c env 开关 + draft-spec 探针）
core/deepseek_v41.py f9b1b560fd54abe69531cf2abd4efed0 （②c：块大小检查改成整除）
```

工作量（每臂）：`16 prompt × 4096 token → max_tokens=64`（**不是 1**，否则 SpecDecoding 只有 ~10 个 drafted token、没有统计功效），
`concurrency=1`、`temperature=0`、`--seed 0`、固定 `prompt_salt=20260922`（**所有臂同一批 prompt**）；
**pass1** = 4 轮**冷**（每轮前 `/reset_prefix_cache`）、**pass2** = 3 轮连发。B 档每轮 ≈ 112 s、D 档 ≈ 123 s。

---

## 3. ★★ Q1：输出 sha256（寻址对不对）

### 3.1 臂内重复一致性（**先证明判据可用**，`037` 的教训）

`037` 在 **8 卡 TP8 + dspark** 上测到 `temperature=0` 下同臂内会抖（首 token 翻转 25–100%），**所以单次比对没有判别力**。
本实验在**单 die** 上先测臂内：**每臂 7 轮全部 `distinct_sha=1`**（B/D 两档、128/64 两臂都是）。

| 臂 | pass1（4 冷轮） | pass2（3 连发轮） | distinct |
|---|---|---|---|
| `c2c-b128` | `0ebccb55…` ×4 | `0ebccb55…` ×3 | **1** |
| `c2c-b64` | `0ebccb55…` ×4 | `0ebccb55…` ×3 | **1** |
| `c2c-d128` | `0ebccb55…` ×4 | `0ebccb55…` ×3 | **1** |
| `c2c-d64` | 见 §5（待填） | 见 §5 | 见 §5 |

⇒ 【实测】**单 die（TP=1、eager）上这个服务是逐字节可复现的** ⇒ `sha256` 判据在这里**有判别力**（与 8 卡相反，原因见 §7）。

### 3.2 跨臂（128 vs 64）

| 档 | 对比 | 结果 |
|---|---|---|
| **B** | `c2c-b128` vs `c2c-b64`，pass1 4 轮 + pass2 3 轮，**逐轮** | `sha_all` **✅逐字相同**（7/7 轮），**逐 prompt 16/16 相同**（每轮） |
| **D** | `c2c-d128` vs `c2c-d64` | 见 §5（待填） |

> ★ **诚实边界**：`out_sha256` 是**生成文本**的 sha，`max_tokens=64` ⇒ 每条 64 token 的文本都参与（不是空文本，`037` 那种 `mt=1` 的空串问题不存在）。

---

## 4. ★★ Q2：投机解码（SpecDecoding 四项 + 一个更硬的判据）

### 4.1 四项（B 档两臂）

| 指标 | `c2c-b128` | `c2c-b64` | 判定 |
|---|---|---|---|
| `Mean acceptance length` | **1.685** | **1.685** | 不回退（**完全一致**） |
| `Avg Draft acceptance rate` | **13.69%** | **13.69%** | 同上 |
| `Per-position acceptance rate`（pos0..4） | `0.685, 0, 0, 0, 0` | `0.685, 0, 0, 0, 0` | 同上 |
| `drafts / drafted / accepted` | 4239 / 21195 / 2902 | 4239 / 21195 / 2902 | 同上 |
| `Accepted | Drafted throughput`（日志行，逐条） | 见 `logs/raw/054-c2-draft64/*.specdecode_lines.txt` | 同量级 |

★ **统计功效（先自证）**：基线不是地板 —— pos0 接受率 **0.685**（不是 0.0x），`MeanAccLen 1.685`
⇒ 若 64 臂把 draft 读坏，**这几项会明显塌**（这正是它在 tiny 上**能**验 Q2 的原因）。

### 4.2 ★★ 更硬的判据：draft **提案 token 序列**逐条对比（探针直读）

`SpecDecoding` 四项是**汇总数**（会互相掩盖：少几步、多几步、接受位置挪动都可能同分）。所以我加了一条**逐条**判据：
把 `NPUModelRunner.propose_draft_token_ids()` 的返回值（每一步 draft 提案的 5 个 token）**全量落盘**，两臂逐条比对。

```
[C2-2C-PROBE v1] hooked vllm_ascend.worker.model_runner_v1.NPUModelRunner.propose_draft_token_ids
                 -> NPUModelRunner.propose_draft_token_ids @0xfffee26114e0 pid=18202
[C2-2C-PROBE v1] DRAFT pid=... call#1 n=5 sha16=... ids=[61498, 126510, 126510, 126510, 126510]
```

| 对比 | 结果 |
|---|---|
| `c2c-b128` vs `c2c-b64`（B 档） | **4367 次调用 / 4367 逐条相同**（不同 **0**；两臂调用次数也相同 ⇒ 步数也相同） |
| `c2c-d128` vs `c2c-d64`（D 档） | 见 §5（待填） |
| **反例臂**（故意缺陷）× | 见 §6 |

> ★ **探针纪律（AGENTS §5b）**：① 先装 hook 再 import（`probe/usercustomize.py` 走 `site` 的 `usercustomize` 机制，
> 放在 `PYTHONPATH` 最前，**不顶掉** `patch/sitecustomize.py` 的 PGP/P2 链）；
> ② 打印**被替换的函数名 + 地址 + pid**（"hooked … @0xfffe…"）—— 第一版探针就是靠这条才发现**钩错了函数**：
> 上游 `SpecDecodeBaseProposer.propose` / `DSparkSpeculator.propose` **一次都没被调用**，本机 dspark 的真入口是
> `NPUModelRunner.propose_draft_token_ids()`；③ 热路径逐次 trace（4367 行落盘）⇒ **可证伪"函数没被调用"**。

---

## 5. Q3：容量（实测 vs 050 §1.6 零参数模型）

### 5.1 模型先自证（脚本 `agents/C2_draft64/scripts/c2_model.py`，可复跑）

```
Σslot_pages = 3 × max(kv+index(ratio2), state, swa, draft) + max(kv+index(ratio1), swa)
P(b)        = cdiv(min(window−1 + max_in_flight, max_len), b) + 1
BPR         = cdiv(max_len, 128) + 1 + 10×P(128) + P(draft_block)
num_blocks  = avail_bytes // Σslot_pages − 1
tokens      = int(num_blocks / BPR × max_len)
```

**9 个已知实测点逐字复现（0 个不符）**：tiny 无 draft B/C/D = 22,719 / 33,295 / 43,469；
tiny +draft128 B/C/D = 20,826 / 20,826 / 23,651；8 卡 +draft128 B/C/D = 427,643 / 427,643 / 485,610。

### 5.2 预测（可判伪）

| 臂 | Σslot_pages | BPR | 预测 token | 相对 128 臂 |
|---|---:|---:|---:|---|
| tiny `draft=128` B | 540,928 | 780 | **20,826** | — |
| tiny `draft=64` B | **540,928（不变）** | **844** | **19,247** | **×0.9242（降！）** |
| tiny `draft=128` D | 476,416 | 780 | **23,651** | — |
| tiny `draft=64` D | **282,880** | **844** | **36,825** | **×1.5570（涨）** |

★ 这两个**相反符号**的预测本身就是判别力：若 64 臂"两边都涨"或"两边都不动"，说明测的不是几何。

### 5.3 实测（待填 D 档）

| 臂 | `GPU KV cache size`（实测） | 预测 | 判定 |
|---|---:|---:|---|
| `c2c-b128` | **20,826** | 20,826 | ✅逐字 |
| `c2c-b64` | **19,247** | 19,247 | ✅逐字 |
| `c2c-d128` | **23,651** | 23,651 | ✅逐字 |
| `c2c-d64` | 待填 | 36,825 | 待填 |

---

## 6. ★★ 阳性对照：判据有没有判别力（AGENTS §5b 第 3 条）

（待填：`c2c-neg-b64` = B 档 + `draft=64` + **故意把 draft 块表行宽按 128 行块算**）

---

## 7. ★★ "tiny 能不能替代 8 卡"

（待填）

---

## 8. 产物与复跑

（待填）
