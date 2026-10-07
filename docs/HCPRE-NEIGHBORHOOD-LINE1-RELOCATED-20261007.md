# 线① 重新定位：融合点在 `dsa_v41.py`，但被"bf16 输出仍需消费"阻断（2026-10-07）

> 承接 `HCPRE-NEIGHBORHOOD-THREE-LINES-VERDICT`。该文说"线① 证否"**只对了一半**：
> 原假设（W4A8 分支）确实错，但**真实存在一个融合机会，只是位置不同**。
> 全部【实测】。

---

## 0. 一页纸

| 项 | 结论 |
|---|---|
| 原假设（`_is_w8a8_dynamic` 对 W4A8 判假） | ❌ **错**：`chain wq_b = AscendLinearMethod -> AscendW8A8DynamicLinearMethod`，**判真** |
| `dsa_v1.py` 的融合路径 | ✅ 生效，但**只覆盖 draft 的 3 层**（实测 `fused=341, sep=0`，≈3 次/步） |
| **主模型 43 层的真实位置** | **`dsa_v41.py::multistream_preprocess`**（此前搜错了文件） |
| 该处的 norm+quant 对 | **2 处/层**：`input_layernorm`+`wq_a.quantize`、`q_norm`+`wq_b.quantize` |
| **能否融合** | ❌ **都不能**——两处的 **bf16 输出都被下游消费** |
| 现有算子 | 全是 **2 输出**，**没有 (bf16, int8, scale) 三输出** |
| **残余机会** | **0.66%**（39.4 对/步 × 4.24 µs），**需新 kernel** |

---

## 1. eager vs graph 对照：推翻"图模式分解"假设

上一轮我怀疑"图编译把融合算子分解成 2 个 kernel"。**实测否定**：

| 模式 | `RmsNormDynamicQuant` | `RmsNorm` | `RmsNormCast` | 步数 |
|---|---:|---:|---:|---:|
| **GRAPH** | **3.00/步** | **140.00/步** | 43.00/步 | 112 |
| **EAGER** | **3.00/步** | **140.00/步** | 43.00/步 | 114 |

**两模式完全一致** ⇒ 不是图编译问题。
（eager 下 Python 计数 `fused=341` × 114 步 ≈ 3/步，与 kernel 数吻合 ⇒ **Python 调用与 kernel 一一对应**。）

---

## 2. 那 3 次/步是什么：draft 的 3 层

`dsa_v1.py::_mla_prolog_multistream` 里的融合调用，实测**只被调用 3 次/步**：

```
[QCOUNT] call#1 is_prefill=False is_w8a8=True qa_shape=(5,1280) qa_contig=True | fused=341 sep=0
```

**3 次/步 = draft 模型的 3 个 MTP 层**（`mtp.0/1/2`）。
⇒ `dsa_v1.py` 这条路**不服务主模型**。

---

## 3. 主模型的真实位置：`dsa_v41.py`

**关键**：tp8k5 **没有挂载** `dsa_v41.py`，用的是**镜像内版本**（47 KB / 1092 行）；
宿主 `~/dcpw/` 那份是 264 KB 的另一版本（**不生效**）。

真代码（`dsa_v41.py::multistream_preprocess`，DSA_OVERLAP=1 时走这条）：

```python
# Part 1
q_quant, q_scale = wq_a.quantize(hidden_states)   # ← DynamicQuant [6,5120]
q_a = wq_a.matmul(q_quant, q_scale, bias=...)

# Part 2
qr = attn.q_norm(q_a)                              # ← RmsNorm [6,1280]
q_b_quant, q_b_scale = wq_b.quantize(qr)           # ← DynamicQuant [6,1280]

# Part 3
q = wq_b.matmul(q_b_quant, q_b_scale, bias=...)
return q.to(...), qr                               # ★ qr 返回
```

**⇒ 正是 profile 里那两对 `RmsNorm → DynamicQuant`**（`[6,5120]` 与 `[6,1280]`，各 39.4 对/步 ≈ 43 层）。

---

## 4. ★ 为什么两处都不能用现成的融合算子

### 4.1 `input_layernorm` + `wq_a.quantize`（[6,5120]）

`input_layernorm` 的 bf16 输出 `x` **不只喂 wq_a**，还被 **compressor** 消费：

```python
# _write_compressed_source
latent = compressor(hidden_states)                 # ← 用 bf16
hidden_states_fp32 = hidden_states.float()
kv = compressor.wkv(hidden_states_fp32)
score = compressor.wgate(hidden_states_fp32)
```

**⇒ 必须保留 bf16 输出。**

### 4.2 `q_norm` + `wq_b.quantize`（[6,1280]）

`qr`（bf16）**被 indexer 消费**：

```python
return q.to(hidden_states.dtype), qr   # ← qr 返回给调用方
```

注释也写明：*"V4.1 keeps floating-point qr for its indexer"*。

**⇒ 同样必须保留 bf16 输出。**

### 4.3 现有算子全是 2 输出

| 算子 | 输出 |
|---|---|
| `_C_ascend.npu_rms_norm_dynamic_quant` | `(int8, scale)` |
| `_C_ascend.npu_rms_norm_cast` | `(bf16, fp32)` |
| `torch.ops.npu.npu_rms_norm_quant` | `(int8)` |
| `torch.ops.npu.npu_rms_norm_quant_v2` | 同上族 |

**⇒ 没有任何算子能同时给出 (bf16, int8, scale)。**

这**不是遗漏，而是设计**：`dsa_v41.py` 用分离写法是**被下游需求逼出来的**。

---

## 5. 残余机会与判据

| 项 | 值 |
|---|---:|
| 相邻对（`[6,1280]`） | **39.4 对/步** |
| 分开成本 | **8.4 µs/对**（5.3 RmsNorm + 3.1 DynamicQuant） |
| 三输出融合预估 | 5.0~6.0 µs/对 |
| **可省** | **0.10~0.15 ms/步** |
| **占步长** | **0.4 ~ 0.6%** |
| 判据（本目标） | ≥0.5% |
| **实现代价** | **新 Ascend C kernel**（3 输出 RMSNorm+Quant） |

**⇒ 恰好卡在判据边缘，且需要写新 kernel。**

---

## 6. 三条线的最终状态

| 线 | 状态 | 证据 |
|---|---|---|
| **①（原假设）** W4A8 分支缺融合 | ❌ **证否** | `chain win wq_b = ...W8A8Dynamic`，`is_w8a8=True` |
| **①（重定位后）** dsa_v41 两处 norm+quant | ⚠️ **真实存在但被 bf16 消费者阻断** | `compressor(hidden_states)` / `return q, qr` |
| **②** hc_fn L2 驻留 | ❌ **负结果** | MTE 有效带宽 188 GB/s = 峰值 **11.6%**；hc_fn 仅占每步字节 **0.2%** |
| **③** 展开算子折进 HcPre | ❌ **负结果** | 实测 **0.05 ms/步**（预估 2.5 ms，高估 50×） |

---

## 7. 附带发现：一个必须记住的运维陷阱

重启 tp8k5 时反复失败 `Free memory on device (5.77/61.27 GiB) less than desired`。

**根因**：留了 13 小时的**孤儿 `VLLM::Worker` 进程（ppid=1）仍持有 ~59 GB/芯片**。

**关键**：`kill -9` **杀不掉**（进程 S 态但被设备上下文粘住，kill 后仍有新进程占用）
⇒ **唯一可靠办法是 `docker restart dsv41-tp8k5`**（已验证：重启后 8 die 全回到 ~3 GB）。

---

## 8. 复现

```bash
# eager vs graph 对照
ssh a3-21 'bash /tmp/launch_eager.sh'    # eager 臂
ssh a3-21 'bash /tmp/launch_qc.sh'       # graph 臂 + QCOUNT
# kernel 计数对比
ssh a3-21 'docker cp /tmp/cmp2.py dsv41-tp8k5:/tmp/ && docker exec dsv41-tp8k5 python3 /tmp/cmp2.py'
# 主模型路径（镜像内，非挂载）
ssh a3-21 'docker exec dsv41-tp8k5 sed -n "315,382p" /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py'
```
