# 按并发切换 decode 路径（低并发走 DSpark、高并发不走）的可行性分析

> 问题：能不能在不同并发下走不同的 decode 路径？低并发开推测解码、高并发不开，
> 两边**都保持图捕获**，只是 dispatch 到不同的图。
>
> **结论：可行，而且这不是新设计 —— vLLM 与 vllm-ascend 都已经为此留好了入口
> （dynamic speculative decoding）。但要真正在 DSpark 上跑起来，Ascend 侧要补两处。**

## 0. 先纠正一个前提：框架给的是"比按并发切"更精细的东西

现有入口有两条，**互不冲突、可以协同**：

| 路线 | 配置 | 决策依据 | K 的取值 |
|---|---|---|---|
| **A. 按 batch size 查表** | `--speculative-config` 里的 `num_speculative_tokens_per_batch_size` | 本步调度的**请求数** | 你显式写死的若干值（**可以是 0**） |
| **B. 按草稿置信度自适应** | `additional-config.dynamic_spec_config.method="dspark"` | **DSpark 自带的 confidence head** 输出的 sigmoid 概率 | 每步、**每请求**独立，范围 `[min_verify_tokens, budget]` |

路线 A 就是你想的那个（按并发开关）；路线 B 更接近"论文级"，它不需要人为设阈值，
而是用草稿自己的置信度决定"这个请求值得验证几个 token"。

**两条路的 K 都可以是 0** ⇒ 都能表达"这一步完全不推测"。

## 1. 现成的部分（已逐行核实到运行时容器）

### 1.1 配置入口存在

```
vllm/config/speculative.py:179
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
vllm/config/speculative.py:1402
    def uses_dynamic_speculative_decoding(self) -> bool:
        return self.num_speculative_tokens_per_batch_size is not None

vllm/v1/spec_decode/dynamic/utils.py:52
    if num_speculative_tokens < 0:
        raise ValueError("...values must be >= 0.")      # ★ K=0 合法
```

### 1.2 调度器每步算出本步的 K（并且**不再做 spec padding**）

```
vllm/v1/core/sched/scheduler.py:309
    if speculative_config.num_speculative_tokens_per_batch_size:
        self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(...)

scheduler.py:1194
    num_spec_tokens_to_schedule = self.num_spec_tokens
    if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
        num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]
    ...
    SchedulerOutput(..., num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, ...)
```

注意 padding 那段的条件（`scheduler.py:885`）：

```python
if (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) and ...:
    num_new_tokens = 1 + self.num_spec_tokens      # 固定 K 时才补
```

⇒ **一旦启用 dynamic SD，调度器就不再强行把每个 decode 请求补到 `1+K`**，
而是按查表得到的 K 走。这正是"同一个引擎、两条形状"的前提。

### 1.3 Ascend 的 model runner 已经消费它

```
vllm_ascend/worker/model_runner_v1.py:1938
    # Dynamic SD: pass the scheduled per-step K explicitly, unified with
    # the other proposers (ngram/suffix/medusa/extract) and matching
    # vLLM's ``propose(num_speculative_tokens=...)``. ``_propose`` sets
    # ``self.num_speculative_tokens`` from it, so the model runner no
    # longer mutates the drafter's state here.
    draft_token_ids = self.drafter._propose(
        num_speculative_tokens=scheduler_output.num_spec_tokens_to_schedule, ...)
```

**而且 per-request 的 K 裁剪也在**（`model_runner_v1.py:2060`）：

```python
dynamic_spec = getattr(self.drafter, "dynamic_spec", None)
per_req_k = dynamic_spec.num_verify_tokens
per_req_k = [max(0, min(int(k), self.num_spec_tokens)) for k in per_req_k]   # ★ 允许 0
cut_tokens = DraftTokenIds(..., draft_token_ids=[tokens[:k] for tokens, k in zip(...)])
```

### 1.4 DSpark **原生带**置信度自适应

```
vllm_ascend/ascend_config.py:906
    # Dynamic speculative-length methods. "dspark" relies on the DSpark
    # confidence head; models without such a head need another method.
    SUPPORTED_METHODS = ("dspark", "dflash")
    method: str | None = None
    # dspark accepts initial_verify_budget_per_req, budget_update_interval
    # and budget_threshold
    method_params: dict[str, Any] = {}

vllm_ascend/spec_decode/dspark_proposer.py:168
    dynamic_spec_config = get_ascend_config().dynamic_spec_config
    if dynamic_spec_config.method == "dspark":
        self.dynamic_spec = DynamicSpecScheduler(
            method="dspark", method_params=..., num_speculative_tokens=...)
```

`DynamicSpecScheduler`（`vllm_ascend/spec_decode/utils.py:208`）的流水线：

```
confidence head 的 sigmoid  →  token_probs [B, D]
  → survival = cumprod(token_probs, dim=1)          # 累积生存概率
  → compute_verify_budget()  每 budget_update_interval 步更新一次共享预算
        mean_k = mean_b( #{i : survival[b,i] >= budget_threshold} )
        budget_k = ceil(mean_k)
  → allocate_verify_budget() 每步按 top-k 生存概率把预算分给各请求
        keep_lens.fill_(min_verify_tokens)          # ★ K 的下界
  → num_verify_tokens [B]
```

默认参数：`initial_verify_budget_per_req=5`、`budget_update_interval=16`、
`budget_threshold=0.3`、`min_verify_tokens=1`。

**⚠️ 默认 `initial_verify_budget_per_req=5` 会把 K 的上限压到 5，而我们已实测
`SP_TOKENS=7` 优于 5**（见 `reports/cannbot-sweep-final-verdict.md` §2 第 1 条）。
所以走路线 B 必须显式把它设成 7，否则会**静默退化**。

### 1.5 框架对"多张图"是**设计好的**

`vllm/v1/worker/gpu/cudagraph_utils.py:265`：

```python
# When using Dynamic SD, num_speculative_tokens is the max number of
# draft tokens. The scheduler might use a smaller number so we need
# to capture graphs for all possible values during decode.
if speculative_config and speculative_config.uses_dynamic_speculative_decoding():
    dense_schedule = build_dynamic_sd_schedule_lookup(...)
    decode_query_lens = sorted({
        num_spec + num_new_sampled_tokens_per_step for num_spec in dense_schedule[1:]
    })
```

⇒ **上游明确按"查表里出现过的每一个 K"去捕获多张图**。
你说的"保持图捕获、只是走不同的图"，就是这个设计意图。

## 2. 真正要补的两处（都在 Ascend 侧）

### 缺口 1：`num_query_per_req` 是构造期派生的，K 变了它不变

```
vllm_ascend/spec_decode/dspark_proposer.py:148
    if self.sample_from_anchor:
        self.num_query_per_req = self.num_speculative_tokens      # ← 只在 __init__ 赋值
    else:
        self.num_query_per_req = 1 + self.num_speculative_tokens
```

全文件 13 处使用 `num_query_per_req`（`set_inputs_first_pass` 里
`cad.query_start_loc = arange * num_query_per_req`、`cad.max_query_len`、
`decode_token_per_req`、`num_query_total`、以及 `dummy_run` 的捕获），
**没有任何一处会在 K 变化时重新派生它**。

而 `llm_base_proposer.py:1299` 每步都会做 `self.num_speculative_tokens = num_speculative_tokens`
⇒ 两者**会脱节**。

修法：把 `num_query_per_req` 改成由当前 K 派生的属性（或在 `_propose` 入口同步），
并让所有下游在每步重算。**这是本方案的核心工作量。**

### 缺口 2：Ascend 的图按**单一** `uniform_decode_query_len` 设计

```
vllm_ascend/utils.py:1071
    uniform_decode_query_len = 1 if not speculative_config else 1 + speculative_config.num_speculative_tokens
vllm_ascend/compilation/compiler_interface.py:73
    uniform_decode_query_len = num_spec_tokens + 1
```

两处都只算**一个值**（= 1 + 最大 K），而 dynamic SD 需要
"查表里出现过的每个 K 各一个 query_len"（上游 GPU 路径的做法，见 §1.5）。

修法：仿照 `cudagraph_utils.py:279`，在 Ascend 的 runner / compiler 里也按
`dense_schedule` 展开成 `decode_query_lens` 集合，逐个捕获。

### 附带的第三处（我们自己的脚本）

`scripts/serve_v2.sh:80` 硬编码了 speculative-config：

```bash
ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SP_TOKENS,\"enforce_eager\":$SE}")
```

要开 dynamic SD 得在这里把 `num_speculative_tokens_per_batch_size` 拼进去
（新增一个 env，例如 `SP_SCHEDULE`）。

## 3. 三个关键的有利事实（会大幅降低工作量）

### 3.1 在 `MAX_SEQS=4` 下，两条路径**天然不撞桶**

我们的口径是 `MAX_SEQS=4`、`SP_TOKENS=7`：

| 路径 | batch=B 时的 `num_tokens` |
|---|---|
| 推测（K=7） | `8B` ∈ {8, 16, 24, 32} |
| 不推测（K=0） | `1B` ∈ {1, 2, 3, 4} |

**两个集合完全不相交**（8B > B 对任意 B≥1）。
⇒ 即使不做 `(num_tokens, query_len)` 联合键控，也不会出现"非推测步误用推测图"。

这是本方案在**当前口径下**能成立的关键；`MAX_SEQS` 一旦 > 8 就会撞
（B=8 非推测 = 8，B=1 推测 = 8），那时必须做联合键控。

### 3.2 现有的 `CAPTURE_SIZES` 已经覆盖了大部分 (B, K) 组合

当前 D 的 `cudagraph_capture_sizes = [1,2,3,4,8,12,16,20,24,32]`。
对 `B ∈ {1..4}`、`K ∈ {0..7}`，`num_tokens = B(1+K)`：

| K | B=1 | B=2 | B=3 | B=4 |
|---:|---:|---:|---:|---:|
| 0 | 1 ✓ | 2 ✓ | 3 ✓ | 4 ✓ |
| 1 | 2 ✓ | 4 ✓ | 6→8 | 8 ✓ |
| 2 | 3 ✓ | 6→8 | 9→12 | 12 ✓ |
| 3 | 4 ✓ | 8 ✓ | 12 ✓ | 16 ✓ |
| 7 | 8 ✓ | 16 ✓ | 24 ✓ | 32 ✓ |

（→ 表示 pad 到该桶。）**绝大多数组合已经有图**，只有少数需要 pad。

### 3.3 draft 图的键**天然包含 K**

`dspark_proposer.py:644` 的注释自己写明：

```
capture key = ``5x1``（num_input_tokens=5 = num_reqs*num_query_per_req）
replay  key = ``6x1``（num_tokens=6 = cudagraph_dispatcher 的 bucket）
```

⇒ draft 图的捕获键是 `num_reqs × num_query_per_req`，**K 一变键就变**，
不会串用。这也意味着：**要让每个 K 都有 draft 图，就必须在捕获阶段遍历这些 K**。

## 4. 收益预估（用上一轮实测的并发曲线）

2K prompt + 128 token 输出、`MAX_SEQS=4`：

| 并发 | DSpark（K=7） | 纯自回归（K=0） | 谁赢 |
|---:|---:|---:|---|
| 1 | **73.1** | 41.4 | DSpark **1.77×** |
| 2 | 57.2 | **70.6** | 自回归 1.23× |
| 4 | 76.7 | **114.5** | 自回归 1.49× |

交叉点在并发 1~2 之间。若用路线 A 的表
`[[1,1,7],[2,4,0]]`（并发 1 用 K=7，≥2 用 K=0），**理论上包络为 73.1 / 70.6 / 114.5**，
相对固定 K=7 分别提升 1.00× / 1.23× / 1.49×。

但要注意两点：

1. **表是按"请求数"查的，不是"用户并发"**。同一时刻 3 个请求里有 2 个长、1 个短，
   表看到的是 3。若要更细，得走路线 B。
2. **高并发下 DSpark 亏在哪**：推测解码把每步行数从 `B×1` 抬到 `B×8`，
   而产出只多 A≈3 倍 ⇒ 每步算子时间涨得比 token 产出快。这条结论与
   profiler 的"设备利用率 97.5%、纯计算受限"完全一致。

## 5. 两条路线的选择建议

| | 路线 A（batch-size 表） | 路线 B（置信度自适应） |
|---|---|---|
| 配置 | `num_speculative_tokens_per_batch_size` | `dynamic_spec_config.method="dspark"` |
| 需要补的缺口 | **缺口 1 + 2**（K 的取值集合有限，可为每个 K 各捕一张图） | **缺口 1 + 2，且要支持 K 的全集**（0..7，图数量更多） |
| 决策粒度 | 每步、全 batch 一个 K | 每步、**每请求**独立 K |
| 是否需要调参 | 需要（试出切换点） | 几乎不需要（`budget_threshold` 等有默认值） |
| 风险 | 低（K 只有两个值，容易验证） | 中（per-request 变长 + 图集合大） |

**建议先走路线 A**，而且**只开两个 K 值（7 和 0）**：

* 只有两个 K ⇒ 只需要 **2 张 target 图 + 2 张 draft 图**，缺口 2 的工作量最小；
* 与 §3.1 的"天然不撞桶"正好契合；
* 单变量、可回退，验证成本低。

跑通之后再考虑路线 B（它才是真正吃掉"长短请求混合"收益的那条）。

## 6. 落地步骤（路线 A）

1. **改 proposer**：让 `num_query_per_req` 随当前 K 派生（缺口 1）。
2. **改图捕获**：Ascend 的 `uniform_decode_query_len` 展开成集合，
   为 `{1, 8}` 两个 query_len 各捕一张 target 图（缺口 2）。
3. **改 draft 捕获**：`dummy_run` 按 K ∈ {0, 7} 各捕一次
   （draft 图键已含 query_len，见 §3.3）。K=0 时**跳过** draft 前向。
4. **改启动脚本**：`serve_v2.sh` 拼入
   `"num_speculative_tokens_per_batch_size":[[1,1,7],[2,4,0]]`（或由 env 传入）。
5. **验证顺序**：
   * 起服后先用 `grep "draft graph" serve.log` 确认**捕获了两个桶**；
   * 单请求 → 确认 K=7 的路径没退化（A ≈ 3.1–3.4）；
   * 并发 4 → 确认 K=0 的路径生效（`SpecDecoding metrics` 的 Drafted 应显著下降）；
   * **144K 四针 + 1M 四针**（正确性回归，参照 `CED-PD-DSPARK-ACCEPTANCE-20260926.md`）；
   * 最后才是吞吐对照。

## 7. 明确的风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **图桶错配** | 历史上踩过 `capture key=5x1 / replay key=6x1`（`dspark_proposer.py:646`），当时是静默错结果 | 保留 `DSPARK_CAPTURE_DISPATCH` 之类的探针，起服后核对 capture/replay 键一致 |
| **K 切换时的 bookkeeping** | `prev_num_spec_tokens`、`num_computed_tokens` 的乐观推进都按"上一步的 K"算（`spec_decode/utils.py:17` 的 `update_num_computed_tokens_for_batch_change`） | 框架已有校正函数；重点是**切换的那一步**要单独验证 |
| **切换抖动** | batch 在 1↔2 之间抖动时，K 会反复切，图来回 dispatch | 表里留滞回（例如 1→7、2→0，实际按 `>=2` 判断），必要时加采样窗口 |
| **起服时间** | 图数量翻倍 ⇒ 捕获时间增加 | 本方案只加 2 个桶，代价可控 |
| **精度** | K=0 等价于纯自回归，**输出会与 K=7 不同**（但都正确） | 正确性验收要覆盖两条路径；注意 `A≈1.0` 的判据不适用于 K=0 的步 |

## 8. 一句话总结

**可行，而且框架层面已经把这套设计好了**（`num_speculative_tokens_per_batch_size`
查表 + 按 K 捕多张图 + Ascend runner 消费 `num_spec_tokens_to_schedule`）。
我们这边缺的是 Ascend 侧的**两处接线**：`num_query_per_req` 随 K 派生、
`uniform_decode_query_len` 从单值展开成集合。
而在 `MAX_SEQS=4` 口径下，两条路径的 token 桶**天然不重叠**，
所以"两张图、两个 K"这条路的风险与工作量都在可控范围内。
