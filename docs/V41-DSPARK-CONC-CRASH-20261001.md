# ★★★ DSpark × DCP8：并发 2 就崩引擎（2026-10-01 实测）

> 用户要求「DSpark 多流请求时性能掉得很快，测一下」。
> **实测结论：在 DCP8 上不是"性能掉"，而是并发 ≥2 直接把引擎打崩。**
> 单流完全正常（A=3.29、69.3 tok/s），只要第二个流进来，`sample_tokens` 就设备侧崩。

---

## 1. 并发曲线实测（run `dcpcap_1001_1250_s1curve`）

配置：`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 DCP=8 ENGRAM=0 MAX_SEQS=16
BAT_TOKENS=2048 --no-async-scheduling`（**已对齐线 A 的 32.58 基线口径**）

方法：差减法隔离 prefill（同一 prompt 跑 mt=64 与 mt=448 两次，差值 = 纯 decode 时间），
步数用 `/metrics` 的 `spec_decode_num_draft_tokens_total` 增量 / 7 得出。

| 并发 | ms/step | A | ms/token | 聚合 tok/s | 每流 tok/s |
|---:|---:|---:|---:|---:|---:|
| **1** | **40.44** | **3.29** | **14.43** | **69.3** | 69.3 |
| 2 | — | — | — | — | **崩** |
| 4 | — | — | — | — | 未测（服务已崩） |
| 8 / 16 | — | — | — | — | 未测 |

并发 1 两次测量：40.55 / 40.33 ms/step（离散 0.2 ms，很稳）。

---

## 2. 崩溃现场

```
[APIServer] POST /v1/chat/completions  →  HTTP 500
Worker_TP0..7 (8 rank 一致):
  File ".../vllm_ascend/worker/worker.py", line 720, in sample_tokens
  File ".../vllm_ascend/worker/model_runner_v1.py", line 2644, in sample_tokens
  File ".../vllm_ascend/worker/model_runner_v1.py", line 2867, in _bookkeeping_sync
  File ".../vllm/v1/sample/rejection_sampler.py", line 271, in parse_output
  RuntimeError: ACL stream synchronize failed, error code:507011
```

`rejection_sampler.py:271` = `output_token_ids.cpu().numpy()` —— **一次 D2H 拷贝**。

### 2.1 ★ 又一次是"报丧点"而不是凶手

Ascend kernel 是异步执行的，**错误在下一个同步点才暴露**。三次崩在三个不同的地方：

| # | run | 报丧点 | 性质 |
|---|---|---|---|
| 1 | `dcpcap_1001_1145_s1prof` | `dcp_utils.py:331` `valid_sampled_token_count_event.synchronize()` | 同步点 |
| 2 | `dcpcap_1001_1240_s1noasync`（**并发 16 热身**） | `rejection_sampler.py:271` `cpu().numpy()` | D2H 同步 |
| 3 | `dcpcap_1001_1250_s1curve`（**并发 2**） | 同上 | D2H 同步 |

**三次的共同点**：都在 `sample_tokens` 路径、都是 `error code 507011`（AI Core Error）。
⇒ 真正越界的 kernel 在**更早**的 decode 步骤里，只是错误延迟暴露。

### 2.2 `--no-async-scheduling` 没能救

| run | async scheduling | 并发 | 结果 |
|---|---|---|---|
| `..._1145_s1prof` | 开（默认） | 16 | ❌ 崩 |
| `..._1240_s1noasync` | **关** | 16 | ❌ 崩 |
| `..._1250_s1curve` | **关** | **2** | ❌ 崩 |

⇒ 根因**不是** async scheduling，也不是"高并发才有"。**只要 batch_size ≥ 2 就崩。**

---

## 3. 已知事实与未知

| 组合 | 结果 | 出处 |
|---|---|---|
| DSpark × **DCP1** × 并发 4 | ✅ 跑过 | `docs/CED-PD-DYNAMIC-SPEC-20260926.md`（MAX_SEQS=4） |
| DSpark × **DCP8** × 并发 1 | ✅ A=3.29，69.3 tok/s | 本文 |
| **DSpark × DCP8 × 并发 ≥2** | ❌ **崩** | 本文 |

**⇒ 待回答（本轮正在做）**：`SPEC=0 × DCP8 × 并发 ≥2` 是否也崩？

* 若**也崩** ⇒ 是 DCP8 自身在 batch>1 的问题，与 DSpark 无关；
* 若**不崩** ⇒ 是 DSpark × DCP8 的组合问题（batch>1 时 draft 的 slot/verify 路径）。

这个对照是本轮最有价值的一步：它决定后面所有排查往哪边走。

---

## 4. 嫌疑清单（按"batch>1 才触发"这个约束筛选）

1. **`compute_slot_mapping_draft` 在 batch>1 时的分片映射**
   （我加的 `_compute_replicated_slot_mapping` 只在 `effective_dcp_world_size == 1` 命中；
   batch>1 时 `req_indices_mtp` 会跨多个请求，`block_table_indices` 的步长是
   `max_num_blocks_per_req * blocks_per_phys_block` —— 若某个请求的
   `logical_block_idx` 超过 `max_num_blocks_per_req` 就**越界读** device 内存。
   device 越界**不会**像 numpy 那样抛 IndexError，而是静默读到非法地址 ⇒ AI Core Error。
   **这是当前最高嫌疑**，因为 numpy 路径有隐式边界检查、device 路径没有。）
2. `verify` 阶段 `sampled_token_ids` 的 batch 维度（`[batch, max_spec_len+1]`）
3. draft 的 per-group buffer 在 batch>1 时的切片

---

## 5. 下一步（顺序固定）

1. **`SPEC=0 × DCP8 × 并发 1/2/4/8/16`** —— 决定性对照，判断是否 DSpark 专属。
2. 若证明是 DSpark 专属 ⇒ 给 `_compute_replicated_slot_mapping` 加 device 侧边界校验
   （env 门控），用 `V41_DSPARK_DCP_DIAG=1` 复现，打出越界索引的实际数值。
3. 若 SPEC=0 也崩 ⇒ 转向 DCP8 的 batch>1 路径（与 DSpark 无关）。

---

## 6. 对用户问题的直接回答

问：「DSpark 多流请求时性能掉得很快」。

答：**在 DCP8 这个拓扑上，它撑不到"性能掉"那一步 —— 第二个流一进来引擎就崩（HTTP 500，8 worker 全挂）。
单流是健康的（40.44 ms/step、A=3.29、69.3 tok/s），所以这不是性能问题，是可用性问题。**
历史文档里"并发 4 时 DSpark 几乎没收益"那组数据是在 **DCP1（CED-PD）** 上测的，
DCP8 从未跑过多并发 —— 这是本轮的空白，正在补。
