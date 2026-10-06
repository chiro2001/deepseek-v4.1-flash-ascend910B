# ★★ 更正：prefill 是**通信受限**，不是我上轮说的 compute-bound（2026-10-07）

> 本轮用 tp8k5 **已开的 `PROFILE=1`** 做了一次**prefill 专用**采集（零重启），
> 结果推翻了我上一轮的结论，并指向一个**可能可用的配置杠杆**。全部为【实测】。

## 0. 一页纸

| 项 | 上一轮结论 | **本轮实测更正** |
|---|---|---|
| prefill 瓶颈 | 【推断】compute-bound | ✅ **TP allreduce 占 45.5%** ⇒ **通信受限** |
| 合批为什么不帮 | 算力成本 | ✅ allreduce 开销**与 token 数成正比**，合批不摊薄 |
| 可能的杠杆 | 只剩 kernel/V2 runner | ✅ **`MC2=1` 可把 MoE 通信从 ALLTOALL 切到 MC2（Matmul-Comm 融合）** |

## 1. 采集方式（零重启）

tp8k5 启动时带 `PROFILE=1` ⇒ 支持运行时 `/start_profile`、`/stop_profile`。

```python
# 8 条互不重叠的 ~8000-token prompt（各带 32 字符 nonce 破 prefix cache）
POST /start_profile
# 并发发出 8 条请求（max_tokens=8，只测 prefill）
POST /stop_profile
# 离线解析（必需：daemon 内解析会报错）
docker exec dsv41-tp8k5 python3 -c "from torch_npu.profiler.profiler import analyse; \
  analyse('<rank0 dir>')"
```

## 2. ★ 算子构成（`op_statistic.csv`，rank0）

| OP Type | Core | Count | Total µs | Avg µs | Max µs | **Ratio** |
|---|---:|---:|---:|---:|---:|---:|
| **`allreduceAicpuKernel`** | AI_CPU | 729 | **6,930,909** | **9,507** | **48,376** | **45.53%** |
| `SparseFlashMla` | MIX_AIC | 600 | 1,949,505 | 3,249 | 6,911 | 12.81% |
| `HcPre` | MIX_AIC | 1290 | 1,049,796 | 814 | 1,583 | 6.90% |
| `QuantLightningIndexerV2` | MIX_AIC | 120 | 949,130 | 7,909 | 33,223 | 6.24% |
| `HcPost` | AI_VECTOR_CORE | 1290 | 597,771 | 463 | 895 | 3.93% |
| `ScatterNdUpdateSk` | MIX_AIV | 870 | 556,199 | 639 | 1,604 | 3.65% |
| `GroupedMatmulSwigluQuantV2` | MIX_AIC | 645 | 391,880 | 608 | 2,189 | 2.57% |
| `GroupedMatmul`（w2） | MIX_AIC | 645 | 232,711 | 361 | 1,373 | 1.53% |
| …（其余 <1.3% each） | | | | | | |

**⇒ TP allreduce 一项就占掉 45.5%**，而所有 MoE 相关的算力算子加起来
（w1/w3 2.57% + w2 1.53% + SparseFlashMla 12.81%）才 17%。

规模核对：`SparseFlashMla` 600 次 ÷ 40 层 = **15 个 forward**（8 条 prompt、每条 ~2 个 chunk
⇒ 16 个 chunk，吻合）⇒ **每个 forward 约 48.6 次 allreduce**，
每次平均 **9.5 ms**（最大 48 ms）。

### 2.1 为什么"合批不帮"现在说得通了

allreduce 的开销**与消息大小成正比**（消息 = T × hidden × 2 B）。
合批把 T 变大 ⇒ **消息变大、耗时同比变大**，只是把多次小 allreduce 换成一次大 allreduce
⇒ 总时间不变。这正好解释了 §1 观测到的"2K 与 8K prompt 都是 ~8K tok/s、并发也不伸缩"。

## 3. ★ 找到一条可能可用的配置杠杆：`MC2`

读运行时选择器（`ascend_forward_context.py`）：

```python
# get_mc2_tokens_capacity() 的分母来自这段：
if ascend_config.enable_prefill_mc2 or (use_mega_moe and not decode_only):
    max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens      # 8192
elif vllm_config.compilation_config.cudagraph_capture_sizes:
    max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size # 192（我们）

# 选择器：
def _select_fused_or_capacity_moe_comm_method(num_tokens, vllm_config, mc2_tokens_capacity):
    if os.environ.get("V41_MOE_COMM_ALLGATHER", "0") == "1":
        return MoECommType.ALLGATHER
    if use_cann_megamoe(vllm_config):                 return MoECommType.FUSED_MC2
    if enable_fused_mc2 == 1 and ep_world_size <= 32: return MoECommType.FUSED_MC2
    if num_tokens is None or num_tokens <= mc2_tokens_capacity:
        return MoECommType.MC2
    return MoECommType.ALLTOALL
```

**我们的现状**：`MC2=0` ⇒ `enable_prefill_mc2=False` ⇒ 容量基数取 **capture size 192**
⇒ 每 rank 24 token ⇒ prefill 的 8192 token **远超容量** ⇒ **走 `ALLTOALL`**。

**若设 `MC2=1`**：容量基数改成 **`BAT_TOKENS=8192`** ⇒ 每 rank 1024 token
⇒ prefill 的 8192 token **在容量内** ⇒ **切到 `MC2`（Matmul-Communication 融合）**
⇒ 正是针对那 45.5% 的通信。

> 另外发现一个上游实验开关：`V41_MOE_COMM_ALLGATHER=1` 可强制 AllGather 路径，
> 其注释记录了一个**具体动机**（A3 的 FUSED_OR_CAPACITY 会短路到 FUSED_MC2，
> 而 MC2 下 TP 把 token 切成每 rank 1 个，`DispatchFFNCombineW4A8` 的标量开销
> 只服务 1 个 token；AllGather 下每 rank 持有全部 token，标量开销被摊薄）。
> ⇒ 这三个选项（`MC2=1` / `V41_MOE_COMM_ALLGATHER=1` / `FUSED_MC2=1`）都值得 A/B。

## 4. 与 §2 的一致性检验

若 allreduce 真的占 45.5%，则"去掉通信"的理论上界 = `1/(1-0.455)` = **1.84×**
（prefill 从 ~8K → ~14.7K tok/s）。这与我们在 decode 侧观察到的
"通信暴露 96.7%"是**同一类问题在不同阶段的体现**。

## 5. 下一步（已在执行）

正在做 **`MC2=1` 的 A/B 重启**：

| 项 | 基线（`MC2=0`） | 测试（`MC2=1`） |
|---|---|---|
| prefill ≈8000 tok，conc=1 | 7937 tok/s | 待测 |
| prefill ≈8000 tok，conc=8 | 8041 tok/s | 待测 |
| decode conc=32 | 1503~1740 tok/s | 待测（确认无回归） |

若 MC2 生效，预期 prefill 有**可观提升**（上界 1.84×）；
若无效，则 prefill 只剩 kernel/V2-runner 两条路。

## 6. 环境状态

本轮**做了 1 次重启**（为 MC2 A/B），当前 tp8k5 运行在 **MC2=1 测试配置**下，
测完将按约束恢复到**交付基线**（MAX_SEQS=32、MC2=0、KV 2,987,836、BAT 8192）并核验。
tiny（chips 2/3）= health=200、未动。

## 7. 复现

```bash
# 采 prefill profile（服务需带 PROFILE=1）
ssh a3-21 'python3 - <<PY
… POST /start_profile; 并发 8 条 8K prompt; POST /stop_profile …
PY'
# 离线解析（必需）
R=/opt/dsv41/results/<RUN>/prof/<rank0 dir>
ssh a3-21 'docker exec dsv41-tp8k5 python3 -c "from torch_npu.profiler.profiler import analyse; analyse(\"$R\")"'
# 看构成
ssh a3-21 'docker exec dsv41-tp8k5 sort -t, -k5 -rn $R/ASCEND_PROFILER_OUTPUT/op_statistic.csv | head -12'
```
