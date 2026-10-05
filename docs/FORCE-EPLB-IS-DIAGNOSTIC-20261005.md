# ★ `enable_force_eplb` **不是可交付开关**：它绕过门控路由（2026-10-05 二次纠错）

> 这篇是**对上一篇 `FORCE-EPLB-NEGATIVE` 的第二次纠错**，也是本轮最重要的发现。
> 第一次纠错（哑开关）只是"实验无效"；这一次是**该开关本身会让模型算错**。
> 标注：【实测·代码】/【实测·行为】。

## 1. 代码证据：它把 `topk_ids` 整体换掉，而 `topk_weights` 不变

调用点 `ops/fused_moe/routed_experts.py:597-607`：

```python
if get_ascend_config().enable_force_eplb:
    topk_ids = get_force_eplb_topk(topk_ids, num_logical_experts)   # ← 只换 ids
elif enable_force_load_balance:
    ...
return topk_weights, topk_ids                                       # ← weights 仍是门控算出的
```

而被调用的 `_build_round_robin_topk`（`ops/fused_moe/force_eplb.py:108-143`）是这样的：

```python
experts_per_rank = num_logical_experts // ep_size
expanded_tokens  = num_tokens * top_k
expanded_offset  = expanded_tokens * ep_rank + ep_rank
idx    = torch.arange(expanded_tokens)
cursor = idx + expanded_offset
col    = cursor % ep_size
row    = (cursor // ep_size) % experts_per_rank
expert_ids = row + col * experts_per_rank          # ← 只依赖 shape / rank，**完全不看 topk_ids 的值**
```

⇒ 该函数返回的是**形状的纯函数**（`num_tokens × top_k` 的确定性轮转），
结果与"门控选出了哪些专家"**毫无关系**；而权重仍是原门控的权重。

于是这一层实际算的是
`Σ_k w_k · Expert_{roundrobin(shape)_k}(x)` —— **不是模型的 forward**。
（`build_force_eplb_topk()` 的 docstring 也自陈 "Build force-EPLB tables **before ACLGraph capture**"，
即这些表按捕获形状预生成，进一步印证"与本次门控无关"。）

## 2. 行为证据（与代码一致）

`armFE_r2`（已确认真接线生效）**只跑了基准、没跑正确性探针**，观测到：

| 现象 | 值 | 与"路由被换掉"是否自洽 |
|---|---|---|
| conc=1 接受长度 A | 2.89 → **3.11（+7.6%）** | ✅ 退化/重复输出会让 draft 更容易被接受 |
| conc=1 单流 tok/s | 116.4 → **136.9（+17.6%）** | ✅ 同上（不是真加速） |
| conc≥2 | **大幅退化**（N=4 +12.9%、N=8 +27.6%） | ✅ 每 rank 专家数暴增 → 每专家固定开销放大 |
| KV 容量 | 未变 | ✅ 与路由无关 |

## 3. 撤回的两条结论

| 之前写的 | 现在 |
|---|---|
| §1–5 "四个稳定桶全在 ±0.6% 内" | **早已作废**（哑开关，两臂同配置） |
| §6 "单流 +17.6%，值得做两套图的低并发专用开关" | **撤回** —— 那个 +17.6% 是**算错的结果**，不可交付 |

**这个开关的真实身份**：`force_eplb` 是**负载均衡压测工具** ——
它把负载强制做成完美均匀，用来测"如果专家分布完全均衡，MoE 会花多久"，
**前提就是放弃正确性**。它只能出现在诊断/容量规划里，**不能进任何交付配置**。

## 4. ★ 本轮的真正教训（写给后续所有旋钮实验）

我在这一条上连犯两个错，都是**判据选择**问题：

1. **只验证启动器 env，没验证容器内实际配置** ⇒ 第一次 A/B 是同配置对同配置（已由他人复核发现）；
2. **只跑性能基准，没跑正确性探针** ⇒ 第二次 A/B 测的是"算错的模型"，还得出了"单流 +17.6%"这种反向结论。

第二条恰好违反了本项目自己定的纪律（"每项改动单变量 A/B **+ 正确性探针**"）。
**补一条硬规则**：

> 任何**会改变计算结果**的旋钮（路由、量化、融合、精度），
> **必须**先过正确性探针（`ced_pd_acceptance --mode all` 或 144K 四针），**再**看性能；
> 只看时间的"加速"一律按"未验证"处理。

### 4.1 交付默认的正确性证据盘点（本轮顺带核对）

好消息：**当前交付默认里的每一项改动都有正确性证据**，只有 `force_eplb` 缺（而它已排除）：

| 交付默认项 | 正确性证据 |
|---|---|
| 交付基线（armF/IMG_v3） | 144K **11/11 PASS** |
| `ENGRAM_DEVICE_INDEX=1` | 144K 11/11（历史两轮） |
| 6N 捕获桶 / `DRAFT_GRAPH=1` | 144K 11/11 + A 分布监控 |
| `ENGRAM_PAD_SKIP=1` | 144K 11/11（armH） |
| **`ENGRAM_WKV_TP=1`** | 144K 11/11（armH/W 两轮；且机制上每个输出元素仍由一次 matmul 决定） |
| gmm1 armF 内核 | 144K 11/11（历史）+ 内核级指纹 |
| DCP8 形态（含上述默认） | 144K 11/11 |
| **`force_eplb`** | ❌ **无**（本文已判为不可交付） |

## 5. 复现

```bash
# 代码证据
docker run --rm --entrypoint bash local/dsv41-a3-tp8:20261005-1001 -lc \
  "sed -n 108,143p /vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/force_eplb.py; \
   sed -n 595,607p /vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py"
# 行为证据：armFE_r2 的 bench（无正确性探针）+ serve.log 里的 enable_force_eplb:true
```
