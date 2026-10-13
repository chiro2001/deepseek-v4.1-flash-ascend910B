# prefill 的三条非 kernel 杠杆**全部被封堵**（2026-10-07）


> ✅ 本文的「prefill 是 compute-bound」**最终成立**；但本文 §1 之后我曾一度改口称
> 「通信受限（allreduce 45.5%）」，那是 `op_statistic` 求和假象，**已撤回**。
> 完整的暴露度证据与最终画像见 **`docs/PREFILL-FINAL-PICTURE-20261007.md`**。

> 本轮先**否掉了我自己上一轮提出的 BAT_TOKENS 假设**，随后把 prefill 剩下的三条
> 非 kernel 路径逐条查到底 —— 结论是**三条都不通**。全部为【实测】/【实测·代码】。

## 0. 一页纸

| 项 | 结论 |
|---|---|
| prefill 吞吐 | **~7~8K tok/s**，且**几乎不随并发伸缩**（conc 1→32 仅 1.06~1.22×） |
| 我的 BAT_TOKENS 假设 | ❌ **被证伪**（2000-token prompt 一步可装 4 条，仍只有 1.06~1.16×） |
| `admission_gate` 是不是元凶 | ❌ **不是**。`V41_GATE_MAX_PREFILL` 默认就是 **8**；日志里的 `prefill_reqs=1` 只是因为**一条 8064-token chunk 就吃满了 BAT=8192** |
| prefill 的性质 | **compute-bound**：合批既不帮也不损（2K 与 8K prompt 都约 8K tok/s） |
| **杠杆① PCP**（context parallel） | ❌ 需要 V2 model runner；Ascend 侧 `use_v2_model_runner` 只认 env，我们是 V1 |
| **杠杆② MegaMoE**（官方 prefill 加速） | ❌ **双重封堵**（见 §3） |
| **杠杆③ 加大 BAT_TOKENS/合批** | ❌ 无效（compute-bound） |
| ⇒ 结论 | **prefill 提速只能靠 kernel 级工作或 V2-runner 迁移**，没有配置杠杆 |

## 1. 先否掉我自己的假设（零重启实验）

上一轮我提出：prefill 不伸缩是因为 `BAT_TOKENS=8192` 一步只装得下**一个** ~8.3K prompt 的 chunk。

**检验**：换成 **2000-token** prompt（一步本可装 4 条），同一 BAT_TOKENS：

| prompt | conc=1 | conc=8 | conc=32 |
|---:|---:|---:|---:|
| **≈2000 tok** | 6713 tok/s | 8217（**1.16×**） | 7515（1.06×） |
| ≈8000 tok | 7937 | 8041（1.13×） | （超时未测） |

⇒ **2K prompt 有合批空间，聚合吞吐仍只有 1.16×** ⇒ **BAT_TOKENS 不是瓶颈**，假设**撤回**。

### 1.1 顺带确认 `admission_gate` 也不是元凶

```bash
# scripts/serve_a2.sh
350: # 回退：V41_GATE_MAX_PREFILL=1 …
352: V41_GATE_MAX_PREFILL=${V41_GATE_MAX_PREFILL:-8}
```

脚本默认就是 **8**（不是 1）。日志里的
`[admission_gate] prefill_reqs=1 … total_tokens=8064` 只是因为**单条 chunk 就吃满了
`BAT_TOKENS=8192` 的 token 预算**，与门限值无关。

### 1.2 prefill 的真实性质：compute-bound

| prompt | conc=1 单条墙钟 | 8 条并发总墙钟 | 比值 |
|---:|---:|---:|---:|
| ≈8025 tok | 1.011 s | **7.984 s** | **7.90×**（≈完全串行） |

但**串行不等于浪费**：一条 8025-token prompt 本身就要 1.011s（7.9K tok/s），
8 条首尾相接就是 8.09s —— 实测 7.984s **比串行还略快**。
⇒ **prefill 的瓶颈是"每 token 的算力成本"，不是调度/合批。**

## 2. 杠杆① PCP：被 V1/V2 runner 分裂挡住

已在 `PREFILL-DESIGN-POINT-AND-CORRECTIONS` §8 记录：vllm-ascend 为 v4.1 实现了完整 PCP
（`attention/context_parallel/dsa_cp.py` 的 `AscendDSAPCPMetadataBuilder`），但

```python
# vllm_ascend/platform.py:_validate_parallel_config
if not vllm_config.use_v2_model_runner and parallel_config.prefill_context_parallel_size > 1:
    raise ValueError("PCP (Prefill Context Parallelism) is not supported by vLLM Ascend. "
                     "Please set --prefill-context-parallel-size to 1. …")
```

而 Ascend 把 `use_v2_model_runner` 改成**只认 `VLLM_USE_V2_MODEL_RUNNER` 环境变量**，未设置即 False。
我们整套补丁栈（dspark 入图、DCP、CED、engram）都建在 **V1** 上 ⇒ **迁移成本大**。

## 3. ★ 杠杆② MegaMoE：**双重封堵**（本轮新发现）

官方 v4.1 技术报告的原文写着：

> **MegaMoE** 将量化、token 分发、两次专家 Linear、SwiGLU 和 token 聚合融合，并在算子内部完成
> Shared Expert 计算及结果相加，通过通信与计算的流水重叠减少阶段间等待。
> **当前 DeepSeek-V4.1 在 Prefill 阶段接入 MegaMoE，Decode 保持原有 Dispatch / GMM / Combine 路径。**

我们的启动配置里 `FUSED_MC2=0`，看似"打开即可"。但代码显示**两道门**：

### 3.1 第一道：上游把 MegaMoE **主动回退**了

```python
# vllm_ascend/ascend_config.py:_validate_user_input_ranges
# TODO(zzzzwwjj): Currently, there are many problems with the megamoe op.
# We will first roll back the megamoe internally and keep `enable_fused_mc2=2`
# to enable the megamoe for testing capabilities.
if self.enable_fused_mc2 in (0, 1):
    # When enable_fused_mc2=1, roll back to dispatch_ffn_combine.
    _MEGA_MOE_SUPPORTED = False
elif self.enable_fused_mc2 == 2:
    _MEGA_MOE_SUPPORTED = importlib.util.find_spec("cann_ops_transformer") is not None
    self.enable_fused_mc2 = 1
```

⇒ **`FUSED_MC2=1` 现在等价于 `dispatch_ffn_combine`，不再是 MegaMoE**；
必须传 **`=2`** 才会真正加载 MegaMoE（且注释明说"用于测试能力"、"megamoe op 还有很多问题"）。

### 3.2 第二道：我们的模型配置**不被支持**

```python
# vllm_ascend/ascend_config.py:_is_megamoe_supported_by_config
if hidden_size < 1024 or hidden_size > 8192 or hidden_size % 512 != 0:
    return False
if moe_intermediate_size < 1024 or moe_intermediate_size > 3072 or moe_intermediate_size % 512 != 0:
    return False
```

用**真实 config.json** 逐条核对：

```
hidden_size           = 5120  → in[1024,8192]=True   %512==0 → True   ✅
moe_intermediate_size = 2304  → in[1024,3072]=True   %512==0 → False  ❌  (2304/512 = 4.5)
⇒ _is_megamoe_supported_by_config() = False
```

而 `derive_and_validate` 里有一句**静默回退**：

```python
if self.enable_fused_mc2 == 1 and _MEGA_MOE_SUPPORTED and not self._is_megamoe_supported_by_config(vc):
    self.enable_fused_mc2 = 0
    logger.warning("MegaMoe is not supported for this model config; "
                   "additional_config.enable_fused_mc2 will be set to 0.")
```

⇒ 即使传 `FUSED_MC2=2`，**我们的 `moe_intermediate_size=2304` 也会让它被静默关掉**。
（`2304 = 4.5 × 512`，是 V4.1 的模型既定尺寸，不是可调参数。）

### 3.3 另一条待查的线索（未验证）

同一份 vllm-ascend 代码里还有 `enable_prefill_mc2`（我们的启动配置里是 `false`），
以及 `moe_comm_method` 里非 MegaMoE 的 MC2 路径（`MC2`/`MC2_HIER`/`MC2_ALG` 开关，
我们也是 0）。这些是**与 MegaMoE 不同的机制**（MC2 = Matmul-Communication 融合），
**尚未评估**是否受同样的 shape 约束。

## 4. 于是 prefill 的可行路径只剩两条（都需要大改动）

| 路径 | 代价 | 说明 |
|---|---|---|
| **V2 model runner 迁移** | **大** | 唯一能开 PCP 的路（单条长 prompt 切到多卡并行）；但整套补丁栈要跟着迁 |
| **kernel 级优化** | **大** | 当前 ~8K tok/s；按 compute 估算离理论峰值还有 ~10× 空间，说明是 kernel 效率问题 |

配置层面**没有剩余杠杆**（三条全部封堵）。

## 5. 环境状态

本轮**未重启任何服务**（全部为只读代码核对 + 推理测量）。
tp8k5 = 交付配置（health=200、KV 2,987,836、BAT 8192、MAX_SEQS 32、SP 5、dspark 开启）；
tiny（chips 2/3）= health=200、未动。未提交改动 0。

## 6. 复现

```bash
# 1) 否掉 BAT 假设：2000-token prompt 的并发扫描
ssh a3-21 'for c in 1 8 32; do python3 ~/tmp/tp8_prefill_conc_clean.py \
  http://127.0.0.1:19210 $c 2000 8; done'
# 2) admission gate 默认值
ssh a3-21 'grep -n "V41_GATE_MAX_PREFILL=" ~/cedpd-repo/scripts/serve_a2.sh'
# 3) MegaMoE 两道门
ssh a3-21 'docker exec dsv41-tinyspark bash -lc "sed -n \"470,485p\" \
  /vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py"'
ssh a3-21 'docker exec dsv41-tinyspark bash -lc "sed -n \"763,794p\" \
  /vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py"'
```
