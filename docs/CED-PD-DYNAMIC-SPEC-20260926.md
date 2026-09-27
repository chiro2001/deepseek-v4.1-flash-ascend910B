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

---

## 9. 实现与真机验证（2026-09-28）：**被上游一道降级门挡住**

§1–§7 是分析。这一节记录**实现完成后的真机验证结果**：实现本身跑通了启动前的
所有关口，但第一次真机起服在**模型构造期**失败，根因不在我们的补丁里。

### 9.1 落地清单（已合入 main）

| # | 文件 | 作用 |
|---|---|---|
| ① | `patches/files/patch_cudagraph.py` | 整文件替换 base 镜像的 dispatcher 补丁：认"本步 query_len"、为每个 query_len 各建一组 decode 图、dynamic SD 下跳过桶取整；缺图时**降级而非 raise** |
| ② | `experimental/ced/core_model_runner_dynamic_spec.patch` | 运行期补丁（8 hunk）：`uniform_decode` 改集合判定、`_pad_query_start_loc_for_fia` 用本步 ql（不改会 `assert num_reqs == num_reqs_padded` 打死引擎）、`_dummy_run` 用正在捕的那张图的 ql、dispatch 显式传本步 ql、**显式 opt-in**（防 draft proposer 被连带建图） |
| ③ | `patches/files/draft/dspark_proposer.py` | `num_query_per_req` 随每步 K 派生（原先只在 `__init__` 定型，与每步被覆盖的 `num_speculative_tokens` 会脱节） |
| ④ | `scripts/{serve_v2,serve_a2,serve_a3_ced_pd}.sh` | `SP_SCHEDULE` → vLLM 原生 `num_speculative_tokens_per_batch_size`；`V41_CED_DYNAMIC_SPEC=1` 开关；env 透传 |
| ⑤ | `tools/selftest_dynamic_spec.py` | 23 项离线自检（含负控），已并入 `selfcheck_pkg.sh` 的 9k 节 |

离线自检覆盖的关键不变量：`num_tokens=8` 在 ql=1 与 ql=8 下必须产生**不同**的图键
（否则静默串图）；缺图时降级不抛异常；多 ql 建图后桶列表与 padding 表必须恢复；
**没给 opt-in 的 dispatcher（draft）不得多建图**；不设 `SP_SCHEDULE` 时
`--speculative-config` 与历史**逐字节相同**。

### 9.2 真机第一次起服：补丁全部到位，但引擎在模型构造期失败

2026-09-28 04:25 在 a3-21 用 `V41_CED_DYNAMIC_SPEC=1 MAX_SEQS=8` 起 D
（`main@0247594`）。启动侧全绿：

```
[serve_a2] [DYNAMIC-SPEC] 应用 runner 侧 dynamic-spec 补丁（schedule=1,1,7;2,8,0）
[serve_a2] [DYNAMIC-SPEC] runner 补丁已应用 ✓          ← sha 门 bd250a59… 与真实镜像匹配
[serve_a2] [DYNAMIC-SPEC] patch_cudagraph.py 在位（命中 4 处）✓
```

但 8 个 TP worker 在**构造模型**时全部失败（约 2 分钟后）：

```
vllm_ascend/core/deepseek_v41.py:325, in validate_cache_runtime
    raise NotImplementedError("V4.1 currently supports only eager or
                               FULL_DECODE_ONLY graph mode")
```

### 9.3 根因：上游对 MRV1 上的 dynamic SD **无条件降级图模式**

```
vllm/config/vllm.py:855   _maybe_override_dynamic_sd_cudagraph_mode
    if (speculative_config is None
        or not speculative_config.uses_dynamic_speculative_decoding()
        or not self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        or self.use_v2_model_runner):
        return
    logger.warning_once(
        "Dynamic speculative decoding changes the target verification length at "
        "runtime. Overriding cudagraph_mode from %s to PIECEWISE for reliability. "
        "Use VLLM_USE_V2_MODEL_RUNNER=1 if you want to use full CUDA graphs.", ...)
    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

调用点在 `VllmConfig.__post_init__`（`vllm.py:1308`）——**每个进程都会执行**，
且**没有 env 逃生口**。于是 `compilation_config.cudagraph_mode` 从
`FULL_DECODE_ONLY` 变成 `PIECEWISE`，而 V4.1 的 cache 明确只支持
`NONE` / `FULL_DECODE_ONLY` ⇒ 构造期 raise。

上游降级的理由是"dynamic SD 会在运行时改变 target 的验证长度"——**这正是我们
§1–§7 分析并已修掉的那件事**：MRV1 只有单一 `uniform_decode_query_len`，K 一变
图键就错配。上游给的出路是 MRV2（`vllm/v1/worker/gpu/cudagraph_utils.py` 里
按 `decode_query_lens` 展开），但 **V4.1 的 cache 初始化不支持 MRV2**
（`validate_cache_runtime` 在 `use_v2_model_runner` 时直接 raise）。

### 9.35 第一版"事后改回来"的实现是**死代码**（自查发现，未浪费第二轮重启）

第一版把绕过写成：在 runner 的 `_check_and_update_cudagraph_mode` 里
（`model_runner_v1.py:5603`，由 `initialize_attn_backend` 在 `:5418` 调用）
"看到 mode 不是 FULL_DECODE_ONLY 就改回来"。

**这是无效的**，因为调用顺序是：

```
worker.load_model()                     worker.py:741
  → model_runner.load_model()           model_runner_v1.py:4099
    → get_model() → 模型构造
      → validate_cache_runtime()        deepseek_v41.py:325   ← 在这里 raise
...
worker 后续才 initialize_kv_cache()     worker.py:1031
  → initialize_attn_backend()           model_runner_v1.py:5314
    → _check_and_update_cudagraph_mode()  :5418                 ← 永远到不了
```

即 `validate_cache_runtime` 读 `compilation_config.cudagraph_mode` 的时刻**早于**
任何 runner 代码 ⇒ "事后改回来"永远来不及。本轮已把这处死代码**删除**，
并把绕过改到**它真正的落点**：`vllm/config/vllm.py` 的那道门本身
（`VllmConfig.__post_init__`，早于一切）。

### 9.4 因此只有两条路

| | 做法 | 代价 / 风险 |
|---|---|---|
| **A. 关掉那道降级（已实现，默认关）** | 新增 `experimental/ced/core_config_dynamic_sd_gate.patch`：在 `_maybe_override_dynamic_sd_cudagraph_mode` 的 early-return 条件里加一条 `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1 ⇒ 不降级`。这是降级门的**真正落点**（`VllmConfig.__post_init__`），早于模型构造 | 这是**主动绕过上游的一道可靠性保护**。依据：其余改动已把 MRV1 补成"按本步 query_len 建键/派发"（等价 MRV2 的 `cudagraph_utils.decode_query_lens` 做法），且离线自检覆盖键唯一性与降级路径。但**必须**用 144K/1M 正确性探针验收——主要失效模式是**静默算错**，不是崩溃 |
| **B. 支持 MRV2** | 让 V4.1 的 cache 初始化支持 MRV2（上游建议的出路） | 被 V4.1 明确拒绝：`validate_cache_runtime` 在 `use_v2_model_runner` 时直接 raise。工作量也大得多（MRV2 是另一套 runner），不属于本方案范围 |

已实现的前置检查：`serve_a2.sh` 在 `SP_SCHEDULE` 非空时
**要求** `V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`，否则在起容器之前 `die` 并解释原因；
随后用 sha256 硬门（`66e82e95…`）在容器内 `git apply` 该补丁并做**效果断言**
（grep 开关名 + `py_compile`）。

### 9.5 当前状态（【未确认】的部分要明确）

* 实现链路：**已完成并推送**（`main@0247594`），23 项离线自检 + selfcheck 全绿；
* 启动侧接线：**真机验证通过**（补丁应用、sha 门匹配、两件在位）；
* **动态 K 本身未在真机上跑起来过**：两次尝试之中，第一次就是 9.2 的失败；
  走 A 路之后的图捕获计数、K=7 的 A 值、K=0 的 Drafted 计数、144K/1M 正确性、
  性能三元组，**全部仍是未测**。
* 生产服务已回滚到 `SPEC=0 MAX_SEQS=8` 的高吞吐档（2026-09-27 用户指定口径）。

---

## 10. 真机第二轮（2026-09-28 04:51–05:09）：**起服成功，但 batch 1→2 切换时崩**

### 10.1 这一步的正面结论：gate 补丁打通了

在 a3-21 上**只重启 D**（P 不动，`SPEC=0 MAX_SEQS=8`），命令里带
`V41_CED_DYNAMIC_SPEC=1 V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1`。三条判据同时成立：

| 判据 | 结果 |
|---|---|
| 三个补丁都应用 | `config gate 补丁已应用 ✓` / `runner 补丁已应用 ✓` / `patch_cudagraph.py 在位（命中 4 处）✓` |
| 引擎解析出的图模式 | **`FULL_DECODE_ONLY`**（不再是 PIECEWISE） |
| 上游降级被跳过 | `Skipping the PIECEWISE downgrade` ×3；反向证据 `Overriding cudagraph_mode from` = **0** |
| 上一轮的致命点 | `validate_cache_runtime` 失败 = **0**；全日志 `ERROR` = 0 |

⇒ §9.4 的 A 路**在起服层面是通的**：`V41_CED_DYNAMIC_SPEC_FULL_GRAPHS=1` +
gate 补丁确实能把 dynamic SD 从"起不来"变成"起得来"（D 用时 660 s，比静态档
~300 s 慢，主要是 static kernel 因新形状冷编译 135 个核 + 图数量增加）。

### 10.2 但发现两个实现缺陷，第二个导致**运行期崩溃**

**缺陷 1（静默失效）：`patch_cudagraph.py` 的 `__init__` 补丁晚于 dispatcher 构造。**

容器内实测（`import vllm_ascend` 之后立刻检查）：

```
CudagraphDispatcher.__init__ = vllm.v1.cudagraph_dispatcher.CudagraphDispatcher.__init__
是我们的 _dispatcher_init 吗: False
_create_padded_batch_descriptor 是我们的吗: False
```

即补丁模块的 import 时机**晚于** `GPUModelRunner.__init__` 里
`CudagraphDispatcher(self.vllm_config)`（`gpu_model_runner.py:863`）的构造
⇒ 那个实例上没有 `_dynamic_decode_query_lens` ⇒ runner 侧 `getattr(..., None)`
判成 False ⇒ **整条"多 query_len"路径被静默跳过**（起服日志里看不到任何
`[dynamic-spec] building decode graphs` INFO，也没有 `keeping raw
cudagraph_capture_sizes` 跳过日志）。

连带后果：raw 桶里的非整倍尺寸（12、20）被 `_create_padded_batch_descriptor`
**降级成 non-uniform 键**塞进 FULL 键集，并被打进
`set_draft_graph_params(capture_sizes)` —— 等于给草稿侧喂了非法捕获尺寸。

**已修**：① 改为**懒算**（从 `self.vllm_config` 现算并缓存，完全免疫 import 顺序），
删掉 `__init__` 补丁；② 建图期遇到非整倍桶**直接跳过**（新增
`_v41_building_keys` 标记 + `add_cudagraph_key` 忽略 None），不再产生伪键；
③ runner 侧的判定改为**只读 config**（`speculative_config.num_speculative_tokens_per_batch_size`），
不再依赖 dispatcher 实例属性。三条都补进了离线自检（27 项）。

**缺陷 2（致命，已复现）：batch 1 → 2 切换时 draft 元数据 rows 不匹配。**

探针：先单请求（K=7 路径）跑通，紧接着发 2 并发 ⇒ 502，随后 D 端口拒绝连接。

崩溃点（`patches/files/draft/dsa_v1.py:618`）：

```
build_dspark_swa_indices → block_ids = torch.gather(block_table, 1, safe_nums)
RuntimeError: ... AclNN_Parameter_Error(EZ1001):
  Size does not match at dimension 0, expected index shape 2 smaller than self shape 1
```

调用链是 `sample_tokens → propose_draft_token_ids → drafter._propose
→ build_draft_attn_metadata → build_req_metadata_for_drafting → build_dspark_swa`。

机制（读代码定出）：`build_req_metadata_for_drafting` 里
`num_reqs = common_attn_metadata.num_reqs`，而

```python
dspark_swa_args = (
    self.block_table[:num_reqs],   # 行数 = min(block_table 行数, num_reqs)
    ...
    seq_lens,                       # 行数 = num_reqs（self.seq_lens[:num_reqs]）
)
```

两者都按 `num_reqs` 切片 ⇒ 只有 `self.block_table` **本身不足 `num_reqs` 行**时
才会出现"seq_lens 2 行、block_table 1 行"。即**草稿侧 block table 的尺寸
与实际 batch 不一致**。这与缺陷 1 的连带后果（12/20 这种非法尺寸进了
`set_draft_graph_params`）方向一致，但**尚未定论**——需要在修掉缺陷 1 之后重跑
同一探针确认（如果仍崩，则要在草稿 block table 的分配点继续定位）。

### 10.3 收敛后的判据顺序（下一轮直接用）

1. 起服后先看**是否出现** `[dynamic-spec] building decode graphs for query_lens=(1, 8)`
   —— 这是"多 query_len 路径真的走了"的唯一判据（缺它说明又静默跳过了）；
2. 再看捕获进度条：应为**两组**（ql=8 的 7 个桶 + ql=1 的 5 个桶）；
3. 然后才是 `draft/gen` 的 K 切换探针（单请求 ≈7、并发 2 ≈0）；
4. 最后是 144K/1M 正确性与三元组。

### 10.4 生产服务状态

验证期间 D 崩过一次，**已立即回滚**到你指定的高吞吐档并实测可用：

| 项 | P (18990) | D (18991) |
|---|---|---|
| `--max-model-len` / `--max-num-seqs` | 1048576 / 8 | 1048576 / 8 |
| 推测解码 | 关 | **关**（`--speculative-config` 0 次） |
| 前缀缓存 / 护栏 | ✓ / — | ✓ / loaded ✓ |

经代理单请求 `4*4 → 16`；并发 4 路 4/4 正确（0.53–1.58 s）。
