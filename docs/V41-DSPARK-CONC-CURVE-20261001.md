# DSpark × DCP8 并发实测：单流 2.30×，**并发 2 起崩**（2026-10-01）

> 用户：「DSpark 多流请求时性能掉得很快，可以测一下」。
> 本文给出**统一口径的完整并发曲线**，以及一个比"性能掉"更严重的结论。

---

## 0. 统一口径（两边可比）

用**差减法**隔离 prefill：同一 prompt 跑 `max_tokens=64` 与 `448` 两次，
差值 = 纯 decode 时间（prefill 完全相同，被抵消）。

| 指标 | 定义 | 为什么用它 |
|---|---|---|
| `ms/token` | `Δwall / Δout_tokens` | **SPEC=0/1 通用**（SPEC=0 没有 draft 指标，算不出 ms/step） |
| 聚合 tok/s | `1000 / ms_token` | |
| 每流 tok/s | 聚合 / N | 回答"我的单个请求变快了还是变慢了" |
| 每流 token 时延 | `N × ms_token` | 单流视角的端到端时延 |
| `ms/step` / `A` | 仅 SPEC=1（`Δdraft_tokens / 7` 算步数） | |

两个配置**除 SPEC/SP_TOKENS/DRAFT_GRAPH 外逐项相同**：
`DCP=8 ENGRAM=0 MAX_SEQS=16 BAT_TOKENS=2048 PREFIX=1
--no-async-scheduling`（对齐线 A 的 32.58 基线口径）。

---

## 1. ★ SPEC=0（纯自回归）：并发 8 完全正常，线性扩展【实测】

run `dcpcap_1001_1305_spec0`，`SPEC=0 SP_TOKENS=5 DRAFT_GRAPH=0 DCP=8`
容量 8,634,871 tokens，**全程 0 ERROR**。

| 并发 | ms/token | 聚合 tok/s | 每流 tok/s | 每流 token 时延 |
|---:|---:|---:|---:|---:|
| 1 | 33.27 | 30.1 | 30.1 | 33 ms |
| 2 | 16.68 | 60.0 | 30.0 | 33 ms |
| 4 | 8.67 | 115.3 | 28.8 | 35 ms |
| 8 | **4.65** | **215.2** | 26.9 | 37 ms |

* 聚合吞吐 **1→2→4→8 几乎线性**：30.1 / 60.0 / 115.3 / 215.2（理论线性值 30/60/120/240）
* 每流速度只从 30.1 掉到 26.9（**−10.6%**）⇒ 多流几乎没有互相拖累
* 每流 token 时延从 33 ms 只涨到 37 ms

**⇒ 这是 DCP8 的健康基线。多流不是问题，DCP8 也不是问题。**

---

## 2. ★★ SPEC=1（DSpark）：并发 1 健康，**并发 2 崩**【实测】

run `dcpcap_1001_1250_s1curve`，`SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 DCP=8`
容量 8,507,783 tokens。

| 并发 | ms/token | ms/step | A | 聚合 tok/s | 每流 tok/s |
|---:|---:|---:|---:|---:|---:|
| **1** | **14.43** | **40.44** | **3.29** | **69.3** | 69.3 |
| 2 | — | — | — | — | **崩（HTTP 500）** |
| 4 / 8 / 16 | — | — | — | — | 未测（服务已崩） |

并发 1 两次：40.55 / 40.33 ms/step（**离散仅 0.2 ms，非常稳**）。

### 2.1 单流对比：DSpark 赢 2.30×

| | ms/step | ms/token | tok/s |
|---|---:|---:|---:|
| SPEC=0（A≡1） | 33.27 | 33.27 | 30.1 |
| **SPEC=1** | **40.44** | **14.43** | **69.3** |
| 比 | +21.6% | **−56.6%** | **2.30×** |

⇒ **单流下 DSpark 的"ms/step 涨 21.6% 换 token 吞吐 2.30×"成立**，
与历史 CED-PD（DCP=1）的 2.3× 一致。

### 2.2 崩溃

```
POST /v1/chat/completions → HTTP 500
Worker_TP0..7（8 rank 一致）:
  worker.py:720                          sample_tokens
  model_runner_v1.py:2644                sample_tokens
  model_runner_v1.py:2867                _bookkeeping_sync
  vllm/v1/sample/rejection_sampler.py:271 parse_output   ← output_token_ids.cpu().numpy()
  RuntimeError: ACL stream synchronize failed, error code:507011   (AI Core Error)
```

`parse_output:271` 是 **D2H 同步点（报丧点）**，不是凶手 ——
Ascend kernel 异步执行，错误在下一个同步点才暴露。

---

## 3. 定位：DSpark 的哪一段在 batch>1 时出问题

### 3.1 `decode_threshold` 解释了"为什么 SPEC=0 不崩"

`dcp_utils.py::generate_dcp_mtp_input`：

```python
if self.decode_threshold <= 2:
    return                                  # ← SPEC=0 走这里（decode_threshold = 1 + 0 = 1）
extra_tokens = self.decode_threshold - 2    # ← SPEC=1: 1 + 7 - 2 = 6
...
input_batch.block_table.compute_slot_mapping_draft(req_indices_mtp, positions_mtp)
```

`decode_threshold = 1 + num_speculative_tokens`：

| | num_spec_tokens | decode_threshold | 是否走 MTP slot 计算 |
|---|---:|---:|---|
| SPEC=0 | 0 | **1** | ❌ 早退 |
| SPEC=1 | 7 | **8** | ✅ `extra_tokens=6` |

**⇒ `SPEC=0` 根本不执行这段，与"SPEC=0 并发 8 全绿"完全吻合。**

### 3.2 我改的那段在 batch>1 时是否越界（**最高嫌疑，待验**）

`block_table.py::_compute_replicated_slot_mapping`（本轮为修 `Device tensor inputs…`
而新增，见 `docs/V41-DSPARK-X-DCP-FIXED-20261001.md` fix #2）：

```python
block_table_indices = (
    req_indices.to(torch.int64) * self.max_num_blocks_per_req * self.blocks_per_phys_block
    + logical_block_idx
)
block_numbers = self.block_table.gpu.flatten()[block_table_indices].to(torch.int64)
```

**device 侧索引越界是静默的**（numpy 路径会抛 `IndexError`，device 路径只会读到非法地址）
⇒ 正好表现为 **AI Core Error**。batch=1 时索引天然在范围内，batch>1 才可能越界
—— 与"并发 1 正常、并发 2 崩"的现象一致。

⚠️ 但**注意**：`generate_dcp_mtp_input` 传的是 **numpy** 数组（走 numpy 路径，
不经过我这段）。真正会命中我这段的是 `dcp_utils.rebuild_async_spec_decode_inputs`
的 device 重建路径 —— 而它在 `--no-async-scheduling` 下 `should_rebuild=False` **早退**。
⇒ **本假设与本轮的崩溃不完全对得上，标【未确认】，需要实测打点。**

---

## 4. 答案：用户的观察需要修正

| 说法 | 实测 |
|---|---|
| "DSpark 多流性能掉得很快" | 在 DCP8 上**更严重**：并发 ≥2 **直接崩**，撑不到"性能掉" |
| 历史"并发 4 时 DSpark 几乎无收益" | 那是在 **DCP1（CED-PD）** 测的；**DCP8 从未跑过多并发** |

**DCP8 现状**：
* 单流：DSpark 健康且 2.30× 收益
* 多流：**DSpark 不可用**（并发 2 崩）；SPEC=0 可用（并发 8 达 215 tok/s）

---

## 5. 下一步（按信息量排序）

1. **`SPEC=1 × DCP=1 × 并发 2/4`** —— 判断是"DSpark 自身 batch>1"还是
   "DSpark × DCP"组合问题。**正在跑**。
   * DCP1 正常 ⇒ 是 DSpark×DCP 的组合 ⇒ 锁定 DCP 相关的 shape/索引
   * DCP1 也崩 ⇒ 是 DSpark 自身 ⇒ 与我们的 overlay 无关，转上游
2. 按 1 的结果给 `compute_slot_mapping_draft` / verify 路径加 env 门控诊断
   （`V41_DSPARK_DCP_DIAG=1`，默认关以免 `.item()` 污染性能）。
3. 若确认是索引越界 ⇒ 修法是 device 侧先把 `block_table_indices` clamp 并**断言**，
   或改回 CPU 路径（有隐式越界检查）。

---

## 6. 复现命令

```bash
# SPEC=0 曲线
env SPEC=0 SP_TOKENS=5 DRAFT_GRAPH=0 DCP=8 PORT=19210 NAME=dsv41-dspark8 \
    ENGRAM=0 BAT_TOKENS=2048 MAX_SEQS=16 PROFILE=1 \
    EXTRA_KV_ARGS="--no-async-scheduling" STAMP=s0 bash ~/dcp_stage_capacity.sh
python3 ~/tmp/conc_generic.py 19210 1,2,4,8 5

# SPEC=1 曲线（并发 2 会崩）
env SPEC=1 SP_TOKENS=7 DRAFT_GRAPH=1 DCP=8 ... STAMP=s1 bash ~/dcp_stage_capacity.sh
python3 ~/tmp/dspark_conc2.py 19210 1,2,4,8
```

脚本：`~/tmp/{conc_generic,dspark_conc2}.py`（a3-21）
