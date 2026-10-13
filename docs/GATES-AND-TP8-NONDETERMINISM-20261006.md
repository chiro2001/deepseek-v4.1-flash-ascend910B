# 四道验收门实测 + ★ TP=8 非确定性的发现（2026-10-06）


> ✅ **最终实践结论见 `docs/ANSWER-STABILITY-VIA-CHAT-API-20261007.md`**：
> 用服务实际接口（`/v1/chat/completions`）实测 —— **结构化任务（抽取/事实/算术）逐字一致且 100% 正确**，
> 开放生成仅措辞差异（前 8 字一致）。本文的数值层观测仍成立，但**不构成使用层面的风险**。
> 另：本文若出现 `/v1/completions` 裸 prompt 的结论，属**接口误用（OOD）**，以该文为准。

> 上一轮实测 `MAX_SEQS` 32→64 带来 **+32~37%** 吞吐，本轮按目标的**四道验收门**验证它，
> 过程中发现一个**与 MAX_SEQS 无关、但更重要**的问题：**tp8k5（TP=8）在 temperature=0
> 下对同一 prompt 是真正非确定的**，而 tiny（TP=2）完全确定。
> 全部为【实测】，推断处已标注。

## 0. 一页纸

| 项 | 结果 |
|---|---|
| **门② 长文针 60K/74K/150K** | ✅ **24/24 PASS**（MAX_SEQS=64 配置，与目标基线一致） |
| **门① 逐位一致（10+ 轮）** | ❌ **FAIL**，但**基线同样 FAIL** ⇒ 与 MAX_SEQS **无关** |
| 非确定性的范围 | tiny（TP=2）**完全确定**（max\|Δ\|=0）；tp8k5（TP=8）不确定 |
| 【推断】根因 | **TP=8 的 HCCL allreduce 求和不保证定序**（本仓代码注释里已有同类记录） |
| 附带教训 | `walk_blocks.py` 的 `echo+prompt_logprobs` 在 64K 下会 **OOM**（33 GB 输出） |
| MAX_SEQS=64 的推广 | 吞吐结论仍成立，但**应先决定非确定性怎么处理**（它会污染一切精度判定） |

## 1. 门②（长文针）在 MAX_SEQS=64 上通过

```bash
python3 tools/ced_pd_acceptance.py --base-url http://127.0.0.1:19210 \
  --tokenize-url http://127.0.0.1:19210 --model deepseek-v41 \
  --corpus data/hongloumeng.txt --mode needle \
  --context-tokens 60000,74000,150000 --max-tokens 64 --repeat 2 --out ~/tmp/gate64_needle.json
```

```
总计 24 条，通过 24，失败 0
```

（60K/74K/150K × 4 根针 × 2 轮；**与目标里写的"基线 24/24"口径一致**。）

## 2. 门①（逐位一致）FAIL —— 但基线也 FAIL

### 2.1 先踩了一个坑：`walk_blocks.py` 在 64K 下会 OOM

`~/tmp/walk_blocks.py` 用 `echo=True + prompt_logprobs=1`，即要求**整段 prompt 每个位置的完整 logprob**：

```
64000 token × 129280 vocab × 4 B ≈ 33 GB 输出
⇒ 实测 EngineCore: "NPU out of memory. Tried to allocate 3.82 GiB
   (NPU 0; 61.27 GiB total; 55.43 GiB already allocated)"
⇒ 引擎死亡、APIServer 随后 SIGTERM 退出
```

**这是测试请求本身的问题，与 `MAX_SEQS=64` 无关。** 已改用有界工具
`tools/gate_bit_exact2.py`（只比**生成 token** 的 logprob，输出 = max_tokens × (k+1)）。

### 2.2 有界门在两种配置下都 FAIL

同一 prompt、`temperature=0`、`max_tokens=32`、连续 12 轮：

| 配置 | 最坏 max\|Δlogprob\| | 生成文本逐轮一致 |
|---|---:|---|
| **MAX_SEQS=64** | 4.07 | ❌ |
| **MAX_SEQS=32（基线）** | 1.75 | ❌ |

⇒ **逐位一致门在**两种配置下**都不通过**，所以它是**预先存在的性质**，不是 MAX_SEQS 引入的。

### 2.3 进一步判别：连"缓存路径内部"都不一致

注意到 r00 是冷启（15.65 s），r01+ 全部命中 prefix cache（0.6~0.9 s）。
于是做三组对比（`gate_bit_exact2.py` 的输出）：

| 对比 | max\|Δ\| | 文本一致 |
|---|---:|---|
| 每轮 vs r00（冷路径） | 0.31~2.33 | 部分一致 |
| 每轮 vs r01 | 2.10~2.33 | ❌ |
| **r01..r07 两两（同为缓存路径）** | **2.334** | ❌ |

⇒ **连全部命中同一 prefix cache 的轮次之间都不一致** ⇒ **真·非确定**，
不是"冷路径 vs 缓存路径"的差异。

实测样例（同一 prompt、同一参数，连续三轮）：

```
r00  '# 第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀\n\n此开卷第一回也。作者…'
r01  '# 第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀\n\n## 1. 故事缘…'
r03  '========\n[WARNING] 未找到匹配的章节内容。\n请检查输入章节名是…'
```

## 3. ★ 定位：非确定性是 **TP=8 特有**

同样的门跑在 **tiny（TP=2，同机 chip2/3）** 上：

| 配置 | max\|Δlogprob\| | 文本一致 |
|---|---:|---|
| **tiny（TP=2）** | **0.000**（全部轮次） | ✅ 全一致 |
| **tp8k5（TP=8）** | 1.75~4.07 | ❌ |

⇒ **框架层本身是确定性的**（tiny 逐位 0 差），差异出在 **TP=8** 这一侧。

### 3.1 【推断】根因：HCCL allreduce 的求和不保证定序

**本仓代码里已有同类记录**（`dsa_v41.py`）：

> "分叉只能来自**归约的求和顺序**（HCCL all_reduce 在该形状/T 上顺序可变）"
> —— 这正是我们为 DCP 路径实现 `_v41_ordered_allreduce`（定序求和）的原因。

tp8k5 走的是普通 `tensor_model_parallel_all_reduce`（O-Projection、MoE 的 shared/routed/final
等多处），**没有定序保证**。TP=8 的环/树规约顺序随执行时序变化 ⇒ 结果在最低位抖动
⇒ 经投机解码与逐层放大后表现为**文本层面可见的差异**。

（tiny 是 TP=2：两个 rank 的求和顺序实际上是确定的，所以观察不到。）

**这个推断可直接验证**（尚未做）：

1. 在 tp8k5 上开 `--enforce-eager` 或关闭投机解码，看抖动是否缩小（区分"纯规约抖动"与"投机放大"）；
2. 或把 O-Projection 的 allreduce 换成定序实现（本仓已有 `_v41_ordered_allreduce`）后复测。

## 4. 这对前面所有结论的影响

1. **门①的定义需要修正**：在 TP=8 下，"逐位一致 max\|Δ\|=0"不是当前实现的自然属性。
   建议改成两段判据：
   * **强判据**：关掉 TP 规约抖动源（定序 allreduce / 单卡路径）后 max\|Δ\|=0；
   * **弱判据（可立即用）**：同 prompt 多轮**文本一致率**（实测 tiny=100%、tp8k5≈20~40%）。
2. **吞吐类结论不受影响**：`ms/step`、`tokens/step`、吞吐都是统计量，抖动不改变它们。
3. **MAX_SEQS=64 的 +32~37% 仍然成立**，但**在非确定性未定性之前，不建议把它写进交付默认** ——
   否则后面任何精度实验都会被这个抖动淹没。
4. 这也解释了此前多次"同一配置两次结果不同"的现象（如 prefill 冷/热差异、
   `MEASUREMENT-PITFALLS` 里记的接受长度 4.97 vs 1.95）。

## 5. 环境状态（按约束恢复并核验）

| 项 | 恢复后 | 基线 | 一致 |
|---|---|---|---|
| tp8k5 health | **200** | 200 | ✅ |
| GPU KV cache size | 2,987,292 | 2,987,509 | ✅ |
| max_num_batched_tokens | 8192 | 8192 | ✅ |
| max_num_seqs | **32** | 32 | ✅ |
| capture_sizes | `…96,192` | 同 | ✅ |
| SP_TOKENS | 5 | 5 | ✅ |
| 1M 上下文并发 | 2.85× | 同 | ✅ |

（本轮共 4 次起服：MAX_SEQS=64 两轮（其中一轮被 `walk_blocks` 的 OOM 打断）、
基线一轮、以及上一轮的测试/恢复；当前停在**基线配置**。）

## 6. 复现

```bash
# 门② 长文针
ssh a3-21 'cd ~/cedpd-repo && python3 tools/ced_pd_acceptance.py --base-url http://127.0.0.1:19210 \
  --tokenize-url http://127.0.0.1:19210 --model deepseek-v41 --corpus data/hongloumeng.txt \
  --mode needle --context-tokens 60000,74000,150000 --max-tokens 64 --repeat 2'
# 门① 有界逐位门（含冷/热路径判别）
ssh a3-21 'python3 ~/tmp/gate_bit_exact2.py http://127.0.0.1:19210 8000 8 32'   # tp8k5
ssh a3-21 'python3 ~/tmp/gate_bit_exact2.py http://127.0.0.1:19310 8000 6 32'   # tiny 对照
```
