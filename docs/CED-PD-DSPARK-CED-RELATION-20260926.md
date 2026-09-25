# DSpark 与 CED 的架构关系（2026-09-26）

> 本文回答两个问题：**DSpark 为什么能进 CED、只能进 D 侧**，以及
> **两者在数据流上到底在哪里相遇**。结论都有代码行号；未实测的标出来。
> 实施路径与风险清单见 [`CED-PD-DSPARK-ANALYSIS-20260926.md`](CED-PD-DSPARK-ANALYSIS-20260926.md)。

## 0. 一句话

**CED 的 128-token 有界重放，恰好就是 DSpark 草稿层需要的那 128 个位置的 target
隐状态来源。** 没有它，PD 分离下的 D 侧根本拿不到 hidden state（P 只传 KV，不传
隐状态），DSpark 在分离部署里无法独立成立。所以这两件事不是"叠加"，而是**互补**：
CED 给 DSpark 提供它唯一缺的输入，DSpark 给 CED 的 D 侧提供它唯一缺的解码加速。

## 1. 先把 DSpark 的数据依赖摊开

### 1.1 草稿层吃什么

| 事实 | 出处 |
|---|---|
| 草稿层是 3 层 `mtp.0/1/2`，各有自己的注意力 | `core/deepseek_v41.py::plan_cache_slots`：`draft` 必须恰好是 `[0,1,2]` |
| 草稿的 KV **不是自己算的**，是 target 隐状态投影出来的 | `models/deepseek_v4/dspark.py::_project_shared_kv`：`kv = attn.kv_norm(attn.wkv(hidden_states))` |
| 投影入口 `main_proj` 的输入宽度 = 3×5120 = **15360** | 权重形状 `mtp.0.main_proj.weight: [5120, 15360]` |
| 这 3 份隐状态来自 target 的 **37/38/39 层**（0-based） | `config.json::text_config.dspark_target_layer_ids=[37,38,39]` → `eagle3_utils.py:48` 转 1-based `[38,39,40]` → `patches/files/model.py` 层循环里 `layer.layer_idx + 1 in aux_hidden_state_layers` 命中 37/38/39 |
| 草稿注意力是 **128 窗口的 SWA**，与 target 同窗口 | `core/deepseek_v41.py`：`draft_spec.sliding_window != swa_spec.sliding_window` 直接报错；`DeepseekV41DraftSWASpec` 强制 BF16 / 单 KV 头 / 无压缩 |

### 1.2 所以：草稿上下文只有 128 个 token

草稿 KV 的可见范围 = 它自己的 SWA 窗口 = **128**。这意味着一件很关键的事：

DSpark **不需要** target 全序列的隐状态，它只需要**最近 128 个位置的**。

### 1.3 它是怎么被写进草稿 cache 的

```
runner._prepare_inputs
  target_hidden_states = cat([h[:num_scheduled_tokens] for h in aux_hidden_states], -1)   # gpu_model_runner.py:5216
    ↓
proposer._propose → set_inputs_first_pass(..., target_hidden_states, cad, ...)
  self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])                    # 本步 query 数
  self._dflash_hidden_states[:num_context] = target_hidden_states[:num_context]
    ↓
llm_base_proposer.py:1945 / :1775 → build_model_inputs_first_pass(num_input_tokens, context_slot_mapping)
  num_context = self._dflash_num_context
  model.precompute_and_store_context_kv(_dflash_hidden_states[:num_context], ...)
    ↓
dspark.py::_store_standard_swa_kv → 逐草稿层写进该组的物理块
```

**`num_context` 就是"本步算出来的 target 隐状态个数"，也就是本步 query 数。**
这一步同时写 target 的 SWA 和草稿的 SWA —— 两者是同一批位置、同一个窗口。

## 2. 为什么 P 上一定不行（回顾，但用架构语言说）

草稿要的是"**第 37/38/39 层的残差流**"。CED 的 P 在层 20 之后 break：
第 37/38/39 层在 P 上**既没有权重也没有计算**。这不是开关问题，是拓扑问题。

顺带一个有意义的观察：**即便 P 跑满 40 层也不行**。P 的角色是生产者
（`model.py:905` 注释：`this role must only serve that internal transfer request`），
它通过 `MooncakeHybridConnector` 只传 KV 块；aux hidden state 是 runner 的
**进程内缓冲**（`gpu_model_runner.aux_hidden_states`），没有任何跨实例传输通道。
→ 要让 P 供草稿上下文，得先把隐状态搬过去，那是新协议，不是新开关。

## 3. CED 的 D 恰好补上这一块

CED 的 D 做的是"对最后 128 个 token 跑完 0..39 层"（`core_scheduler_replay.patch`
把重放长度钉在 128，P 在 transfer params 里传 `ced_replay_tokens=128`，
D 侧 `get_num_new_matched_tokens` 用它做契约校验）。于是：

| CED 的 D 在重放步里发生的事 | 对 DSpark 的意义 |
|---|---|
| 跑完 0..39 层，`num_scheduled_tokens = 128` | `aux_hidden_states` 被填满 128 个位置 |
| 层 37/38/39 的残差流被 `hidden_states.mean(dim=1)` 收走 | 正好是 `main_proj` 需要的 3×5120 |
| target 的 G7..G11 SWA 被这 128 个位置重写 | **同一批位置**的草稿 G12 SWA 也被写入 |
| 重放是 prefill 形状（128 而不是 1） | `num_context = 128` ⇒ 草稿窗口一次填满，不需要预热 128 个 decode 步 |

**这就是整个方案的支点**：CED 的重放长度（128）与 DSpark 草稿窗口（128）
**数值上相等**，这不是巧合 —— 两者都绑在 target 的 SWA 窗口定义上
（`DeepseekV41SWASpec.sliding_window == 128`）。

推论（也是要盯的判据）：**如果重放步没有走 `build_model_inputs_first_pass`，
草稿窗口就会是空的** —— 表现是接受长度 A ≈ 1.0 而 ms/step 看着正常。
A 是唯一的判据，ms/step 会骗人（`reports/draft-graph-negative-control.md` 的负控）。

## 4. 两者相遇的三个接触面

### 4.1 KV 组的拓扑（唯一的硬契约）

| | 组数 | 组 |
|---|---|---|
| P（无 DSpark） | **12** | G0..G6 全量/压缩，G7..G11 上半层 SWA，**P 不传** |
| D（有 DSpark） | **13** | 前 12 组与 P 一一对应，**G12 = 草稿 SWA**（`_uniform(..., "dspark")` 追加在最后） |

契约的锚点是：**P 侧组索引必须是 D 侧的前缀**。
`get_num_new_matched_tokens` 里 D 会校验
`ced_missing_swa_groups == (7,8,9,10,11)`；一旦 DSpark 把某个组插到 12 之前，
这组索引就会指向别的组，D 会在**启动期**报错（已加 `[CED-DSPARK-GUARD]`）。

另一个必然推论：**G12 必须被预清零**。它和 G7..G11 是同一类风险 ——
P 从不发送它们，重放只覆盖 128 token 触及的那 2 页，其余页可能留着上一次请求的残页。
这也是 `[CED-KVRECV]`/G7..G11 预清零那一族 fix 的直接延伸。

### 4.2 注意力执行路径不是同一条

| | target 路径 | 草稿路径 |
|---|---|---|
| 文件 | `attention/dsa_v41.py`（CED 覆盖版） | `attention/dsa_v1.py` + `models/deepseek_v4/dspark.py` |
| CED 改过什么 | `[CED-SWA-CLIP]`：重放首 query 的 128 窗口往回退到重放起点，避免 kernel 读物理块 0 | **没有** |

→ `[CED-SWA-CLIP]` 修的是"重放首 query 的窗口回溯到已回收列"这件事。
草稿的 SWA 走的是另一套 metadata builder（`AscendDSparkProposer` →
`AscendDSAMetadataBuilder`），**是否重现同一个越界读，未验证**。
如果重现，现象会是长上下文静默乱码，而不是崩溃。

### 4.3 图捕获要分别成立

CED 的 D 交付口径已经是**图模式**（`GRAPH=1 EAGER=0` +
`V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` + `V41_CED_SWA_CLIP=1`），
而 DSpark 的草稿图是**另一套**捕获（`DSPARK_DRAFT_USE_CUDAGRAPH=1`，
形状由 `CAPTURE_SIZES` 决定），两者不共享前提：

| | CED 侧前提 | DSpark 侧前提 |
|---|---|---|
| 必须 | `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1`（单 token prompt 尾步强制 eager） | `DSPARK_GRAPH_CAPTURE_METADATA=1`（否则图里没有 attention） |
| 必须 | `V41_CED_SWA_CLIP=1` | `DSPARK_CAPTURE_VALUE_FIX=1`（否则图里没有"写 context KV"） |
| 已知默认 1 | — | `DSPARK_SWA_INDICES_RESIDENT` / `DSPARK_CAPTURE_NCTX_FIX` / `DSPARK_DISPATCH_QUERY_LEN_FIX` |

**"草稿入图"的坑不在 CED 里**：`build_model_inputs_first_pass` 依赖两个
**捕获期会固化**的量（`_context_slot_mapping_buffers` 捕获时是 `None`；
`_dflash_num_context` 是 Python int 会被烘进图）。CED 的 D 是
prefill 形状重放 + decode 形状解码**混在同一个引擎**里，
所以 `num_context` 在同一个实例内会取 **128** 和 **1** 两个值 —— 比单实例
（只有 prefill 阶段和 decode 阶段分开）更容易踩到"切片长度被固化"。
这正是 `DSPARK_HOIST_CONTEXT_KV` 要解决的问题（把 context-KV 写入挪到图外，
每步用真实的 `nctx`/slots）。

## 5. 收益与代价在哪里

| 项 | 影响 |
|---|---|
| **P 侧** | 零。DSpark 不参与 prefill，P 保持 `SPEC=0` |
| **D 侧 prefill（重放）** | 零新增：重放本来就要跑完 40 层，草稿只是顺手多算 3 层投影（`main_proj` 一次 GEMM + 3×`wkv`） |
| **D 侧 decode** | 每步多 3 层草稿前向（eager 时是主要成本），换 ~2.7–3.0 的接受长度 |
| **显存** | G12 多一组 SWA 页；`plan_cache_slots` 说它**与 target SWA 槽位共享**、不新增每块页字节数 ⇒ `num_blocks × 540928 B` 的 4 GiB 寻址上界不变（**待实测复核**） |
| **块表** | 草稿层有自己的块表（`get_draft_kv_cache_layer_names`），D 侧独立分配，与 P 无关 |

## 6. 结论

1. **DSpark 只能进 D**，且原因是拓扑（层 37/38/39）而不是配置。
2. **CED 的重放长度与草稿窗口都是 128**，前者直接产出后者需要的全部输入；
   两者在"最近 128 个位置的 target 隐状态"这一个量上完成对接。
3. 需要动的是**三处接口**（组拓扑契约、G12 预清零、注意力路径的越界读）
   和**一处图前提**（context-KV 写入的捕获期固化）。
4. 判据只有一个：**接受长度 A**。A≈1.0 就是草稿没产出，
   此时 ms/step 反而更好看（每步只出 1 个 token）。
