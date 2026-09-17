# MoE host 同步风暴与 FUSED_MC2：A3-node1 性能线（2026-09-16 00:00–00:45）

> 场地：A3-node1 chips 8-15，容器 `dsv41-a21-perf`，端口 8020
> 基线配置：`static_kernel=1` + `npugraph_ex=1` + `mtpq` + Engram(int8, gate CHUNK=0) + local-owner + hash fast + jemalloc + `MULTISTREAM=1`
> 工具：`torch_npu` profiler（`PROFILE=1` + `/start_profile`）、msprof 导出、`msprof_*.db`（集合通信 payload）、`FRAMEWORK/torch.op_range`（host 侧 op 轨迹，自写解析器）

---

## 0. 一句话结论

decode 每步有 **~23 层 MoE 走 `AlltoAllCommImpl`（`token_dispatcher._preprocess`）**，它每层做 `torch.histc → .cpu().numpy() → all_gather → .cpu().numpy() → repeat_interleave(device repeats) → .item()`：
**host 侧每步约 10.7 ms 的同步等待**（`aten::item` 213 ms + `aten::repeat_interleave` 218 ms / 43 步），
并让每步出现一次 **4 ms 级别的 [8,5120] allReduce 排队**（该 allReduce 62% 的时间在等最慢的 rank）。
切到 `enable_fused_mc2=1`（关闭互斥的 `multistream_overlap_shared_expert`）后：**host op 总时长 −43%、item/repeat_interleave 归零、集合通信 9.19 → 4.55 ms/step、设备 Free 25.1% → 12.8%**。

---

## 1. 证据链（设备 → host 归因）

### 1.1 设备侧：每步一次 3.5–6.4 ms 的 allReduce

`msprof_*.db` 的 `COMMUNICATION_OP` 表带 payload `count`：

| op | count(elem) | n | total | min | max |
|---|---|---|---|---|---|
| allReduce | **40960 = 8×5120** | 6232 | 394.6 ms | 9.1 µs | **6394.9 µs** |
| allReduce | 10362880 | 1296 | 732.6 ms | 293 µs | 4265.6 µs |
| allGather | 1295360 | 656 | 182.4 ms | 136 µs | 4109 µs |

decode 窗口（66 step）里，**每步恰好 1 个 count=40960 的 allReduce，耗时 3.5–6.4 ms**，其余同尺寸 allReduce 只要 9–16 µs
⇒ 不是带宽问题，是 **rank 排队/掉队**（同一 payload 的 P50 只有 14 µs）。

### 1.2 host 侧：把 host op 轨迹和设备时间线对齐后定位调用点

`FRAMEWORK/torch.op_range` 是二进制流，记录布局（实测逆向）：
`[u16 type][u32 payload_size][56B 定长字段(前 16B = t1/t2, ns)][u16][u16][u32 name_len][name]`，下一条 = `off + 6 + payload_size`。
解析后（窗口 1.993 s ≈ 43 个 decode step）：

| host op | n | total | max | 归属（被哪个 `vllm::` 区间包住） |
|---|---|---|---|---|
| `vllm::moe_forward_shared` | 1002 (23.3/step) | **537.3 ms** | 2683 µs | — |
| `aten::repeat_interleave` | 680 | **217.4 ms** | 2286.7 µs | 全部在 `vllm::moe_forward_shared` 内 |
| `aten::item` | 680 | **212.9 ms** | 2279.2 µs | 同上 |
| `aten::histc` | 680 | 1.5 ms | 9.9 µs | 同上 |
| `vllm::dsa_v41_forward` | 720 | 668.2 ms | 372715 µs | — |

最长的一次 `.item()`（2.28 ms）前后 0.6 ms 的 host 序列（节选）：

```
  _C_ascend::moe_gating_top_k_hash → aten::histc → aclnnHistc → aten::sum → aclnnReduceSum
  → aten::to → aten::_to_copy → aclnnInplaceCopy → acl_memcpy_device_to_host
  → c10d::allgather_ → HcclAllgatherBase
  → aten::repeat_interleave (2286.7 µs) → aten::item (2279.2 µs)
  → [同时] 设备侧 hcom_allReduce_ (3912.9 µs, count=40960)
```

调用点：`vllm_ascend/ops/fused_moe/token_dispatcher.py:611 TokenDispatcherWithAll2AllV._preprocess`

```python
input_splits  = num_local_tokens_per_expert.reshape(ep, local).sum(1).to("cpu", non_blocking=True).numpy()
num_global_...= gather_from_sequence_parallel_region(num_local_tokens_per_expert, group=self.ep_group)
output_splits = num_global_tokens_per_local_expert.sum(-1).to("cpu", non_blocking=True).numpy()
global_input_tokens_local_experts_indices = torch.repeat_interleave(
    self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
```

即：**每层 2 次 D2H 强同步 + 1 次 `repeat_interleave(repeats=设备张量)`（内部逐元素 `.item()`）**。

### 1.3 谁走哪条路

| host op | 次/step | 含义 |
|---|---|---|
| `npu::npu_grouped_matmul` | 23.3 | 走 host 侧 MoE 前向的层数 |
| `aten::histc`（`_preprocess`） | 15.8 | 其中走 `AlltoAllV` dispatcher |
| `npu::npu_moe_distribute_dispatch_v2` | 7.5 | 走设备侧融合 dispatcher |
| `c10d::alltoall_base_` | 51.8 | AlltoAllV 路径的 all-to-all（data+scale+combine） |

---

## 2. 改动与实测（同机 A/B）

改动：`enable_fused_mc2=1`（`FUSED_MC2=1`）+ `multistream_overlap_shared_expert=false`
（两者在 `ascend_config` 里互斥，代码会自动关闭后者并告警）。

### 2.1 host / 通信结构（同口径 profile，32K decode）

| 指标 | OLD（AlltoAll + multistream） | NEW（FUSED_MC2） |
|---|---|---|
| host op 总时长（sum of ranges） | 2.931 s | **1.670 s（−43%）** |
| `vllm::moe_forward_shared` | 537.3 ms | **147.1 ms** |
| `aten::item` | 213.6 ms | **0.7 ms** |
| `aten::repeat_interleave` | 218.4 ms | **0.9 ms** |
| `aten::histc` | 680 次 | **0** |
| 集合通信合计（db，去重） | **9.19 ms/step** | **4.55 ms/step** |
| 每步巨型 allReduce（>3 ms） | 1 个/步（3.5–6.4 ms） | 消失（max 0.083 ms 量级，见 §2.2 注） |
| decode 设备 Free | 25.1% | **12.8%** |

### 2.2 NEW 配置的通信账（decode，67 step）

| op | count | 次/step | ms/step | max ms |
|---|---|---|---|---|
| hcom_allReduce | 40960 | 86.8 | 2.878 | 3.977 |
| hcom_allGather | 5120 | 46.5 | 0.692 | 0.023 |
| hcom_alltoallv | 6144 | 2.1 | 0.647 | 1.523 |
| hcom_allReduce | 35840 | 7.4 | 0.147 | 0.051 |
| 其余 | — | — | 0.2 | — |
| **合计** | | | **4.55** | |

（`max 3.977 ms` 是窗口内个别样本；总量已降到 4.55 ms/step。）

### 2.3 mos/step 与 tok/s（单流独占，3 发中位，warmup 后）

| 上下文 | OLD ms/step | NEW ms/step（两次会话） | Δ |
|---|---|---|---|
| 8K | 36.66 | 34.61 / **34.40** | −2.1 ~ −2.3 |
| 32K | 36.48 | 36.11 / **34.35** | −0.4 ~ −2.1 |
| 128K | 38.38 | **38.17** | −0.2（≈0） |

128K：A=2.745，72.0 tok/s（OLD 67.8）。**KV 容量 3,557,104 → 3,388,441 tokens（−4.7%，仍 >3M ✓）。**

---

## 3. 判读

1. **host 侧已经被显著削平**（item/repeat_interleave 归零、MoE host 时长 −73%），
   8K/32K 的改善证实「短上下文时 host 是瓶颈」。
2. **128K 几乎不变** ⇒ 长上下文下 **设备侧是瓶颈**：decode 设备 busy 已占满（NEW 87.2% busy / 44.3 ms 的 profiled step），
   继续在 host 上榨收益已经接近上限。
3. 设备侧当前最大单项（32K decode，每步）：

| 算子 | ms/step | 次/step |
|---|---|---|
| **DispatchFFNCombineW4A8** | **10.89** | 46.0 |
| hcom_allReduce（全部） | 6.20（CSV 含重复，db 去重 2.88） | 190.3 |
| HcPre | 3.20 | 92.0 |
| QuantBatchMatmulV3 | 2.56 | 241.7 |
| QuantLightningIndexerV2 | 2.19 | 8.6 |
| TransposeBatchMatmul | 2.18 | 46.0 |
| MatMulV2 | 1.95 | 75.3 |
| SparseFlashMla | 1.63 | 42.8 |

---

## 4. 下一步（按性价比）

1. ~~**SP_TOKENS 扫描**~~ → **已测，S=5 明显更差（否决）**，见 §6。
2. **DispatchFFNCombineW4A8（10.89 ms/step）**：查它内部是否量化通信 payload；可试 `enable_fused_mc2=2`（MegaMoe，需 `cann_ops_transformer`+sym buffer，风险高）。
3. **HcPre/HcPost 融合**（cannbot 措施 8 类）：`model.py` 每层 2×`npu_hc_pre_v2`+2×`npu_hc_post`，glm5next 已有融合实现可参考；当前合计 ~4 ms/step。
4. **TP allReduce 精简**：40960 尺寸 86.8 次/step、2.88 ms/step，看能否合并（多流/融合）。
5. 若回到 AlltoAll 路径，可只 patch `_preprocess` 去掉 host 同步（保留设备侧更省的 7.7 ms MoE），作为 FUSED_MC2 的替代候选。

---

## 5. 原始证据路径（A3-node1）

| 内容 | 路径 |
|---|---|
| OLD 起服日志 | `logs/perf/a21_mig_2244_serve.log`（prof 会话：`logs/perf/a21_prof_2302_serve.log`） |
| OLD profile 导出 | `logs/prof_a21/`、`logs/prof_a21/extract/{op_summary_r0.csv,op_stat_r0.csv}`、`/tmp/msprof_r0.db`、`/tmp/op_range_r0.bin` |
| NEW 起服日志 | `logs/perf/a21_fmc2_0019_serve.log` |
| NEW profile 导出 | `logs/prof_a21b/`、`/tmp/op_summary_fmc2.csv`、`/tmp/msprof_fmc2.db`、`/tmp/op_range_fmc2.bin` |
| 测量 jsonl | `logs/perf/a21/p42_t4_quote_{8192,32768,131072}_*.jsonl` |
| 分析脚本 | `scripts/{parse_framework,framework_window,attr_item,free_account,free_timeline,gap_anatomy,step_phases,op_window,comm_ops,comm_db,comm_db_window,comm_db_by_op}.py` |

---

## 6. 附：`SP_TOKENS=5` 实测（否决）

同一会话配置（`FUSED_MC2=1`、`static_kernel=1`、`MULTISTREAM=0`）只改 `SP_TOKENS`，3 发中位：

| 上下文 | S=7 ms/step | **S=5 ms/step** | S=7 A | S=5 A | S=7 tok/s | S=5 tok/s |
|---|---|---|---|---|---|---|
| 8K | 34.40 | **41.10** | 1.689 | 1.658 | 49.3 | 40.2 |
| 32K | 34.35 | **42.01** | 2.865 | 2.898 | 83.0 | 69.5 |
| 128K | 38.17 | **45.13** | 2.745 | 2.705 | 72.0 | 59.9 |

接受长度几乎不变（±0.05），**但步时 +6.7 ~ +7.7 ms**，两处都劣于 S=7 ⇒ `SP_TOKENS=7` 保持。
（推测：S=5 使 `max_cudagraph_capture_size = max_seqs*(S+1)` 从 32 变 24，捕获桶与 MC2 容量随之改变，步内固定开销没变但桶变小反而更慢。）

起服日志：`logs/perf/a21_s5_0045_serve.log`；测量：`logs/perf/a21/measure_s5.log`、`logs/perf/a21/p42_t4_quote_*_s5_*.jsonl`。
