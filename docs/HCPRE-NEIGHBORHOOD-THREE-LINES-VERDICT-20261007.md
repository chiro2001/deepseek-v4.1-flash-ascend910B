# HcPre 邻域三条线：全部封闭（含一条证否、两条负结果）（2026-10-07）

> 目标：削减 HcPre 邻域的 decode 固定开销。三条具体线逐条验证。
> 判据：每项端到端 ≥0.5% 才继续。全部【实测】。

---

## 0. 一页纸

| 线 | 结论 | 证据 |
|---|---|---|
| **① 修 W4A8 融合缺失** | ❌ **证否**（融合本来就在生效） | `QCOUNT` 探针：**`fused=N, sep=0`** |
| **② hc_fn 的 L2 驻留** | ❌ **负结果**（延迟受限，且占比仅 0.2%） | MTE 有效带宽 **188 GB/s = 峰值 11.6%** |
| **③ 展开算子折进 HcPre** | ❌ **负结果**（量级只有预估的 2%） | 实测 **≈0.05 ms/步**（预估 2.5 ms） |
| 残余机会（原目标外） | ⚠️ 0.4~0.6%，需新 kernel | `input_layernorm + quantize` 三输出融合 |

---

## 1. 线①：证否 —— 融合路径**本来就是生效的**

### 1.1 我此前的推断错在哪

我从 checkpoint 的 `model_quant_type: W4A8_DYNAMIC` 推断 `_is_w8a8_dynamic(self.wq_b)` 为假。
**错**——那个字段只描述 **MoE** 的 scheme（`@register_scheme("W4A8_DYNAMIC", "moe")`），
而 attention 的 `wq_a/wq_b/wkv` 走的是 **W8A8_DYNAMIC**。

**运行时探针实测**：

```
[QLIN-PROBE] chain wq_b: AscendLinearMethod -> AscendW8A8DynamicLinearMethod
             is_w8a8 wq_a=True wq_b=True wkv=True   share_hs_quant=True

[QBRANCH-PROBE] call#1 is_prefill=False is_w8a8=True
                -> 走 is_w8a8 分支（融合 RmsNormDynamicQuant）

[QCOUNT] call#1 is_prefill=False is_w8a8=True qa_shape=(160,1280) qa_contig=True | fused=38 sep=0
```

**`sep=0` ⇒ 分离分支在整轮运行中一次都没走。**

### 1.2 融合的收益是真的（这部分没错）

单算子实测（chip5，真实形状 [6,W]）：

| 形状 | 方案 | kernel 数 | 设备时长 |
|---|---|---:|---:|
| W=1280 | 分开 `RmsNorm` + `DynamicQuantV2` | 2 | **7.90 µs** |
| W=1280 | **融合 `RmsNormDynamicQuant`** | 1 | **3.66 µs** |
| W=5120 | 融合 | 1 | 4.08 µs |

**省 54%**，且**结果逐位一致**（int8 均值 11.8 vs 11.8、scale 均值 0.0563 vs 0.0563）。
对连续 / 非连续 / NZ 三种布局都成立。

### 1.3 那 profile 里的 140 RmsNorm/步 是什么

**是别的算子**，不是 q_norm：

| kernel | 每步 | 流 |
|---|---:|---|
| `RmsNorm` | **140.0** | s146 88.2 / s144 39.6 / s142 6.9 / s240 3.0 |
| `RmsNormCast` | 43.0 | s146 39.6 / s142 3.0 |
| **`RmsNormDynamicQuant`** | **3.0** | s142 |

（每步 43 层：`input_layernorm` 43 + `q_rms`/`kv_norm` 等 ≈ 97 ⇒ 合计 ≈140，自洽。）

**⇒ 线① 无可修之处。**

---

## 2. 线②：负结果 —— hc_fn 不是 L2 杠杆

| 项 | 实测 |
|---|---:|
| HcPre 每次时长 | 33.3 µs |
| └ **MTE 部分** | **10.4 µs（31%）** |
| `hc_fn` 大小 = 24×20480×4B | **1.97 MB** |
| ⇒ 若 MTE 全用于读 hc_fn，有效带宽 | **188.5 GB/s = 峰值的 11.6%** |
| hc_fn 每步总流量 | **10.2 MB** |
| └ 占每步总搬运（4.38 GB） | **0.2%** |

**两条独立理由都指向"无杠杆"**：
1. **延迟受限**：有效带宽只有峰值 11.6% ⇒ 瓶颈是搬运的启动/寻址延迟，不是容量或带宽；
2. **占比可忽略**：hc_fn 只占每步字节的 0.2% ⇒ 即使做到零成本，端到端也测不出。

**⇒ 线② 关闭。**

---

## 3. 线③：负结果 —— 量级只有预估的 2%

HcPre 前接的"单流→4流展开"算子（`Cast`/`Mul`/`Add`/`Cast`）：

| 项 | 实测 |
|---|---:|
| 有展开算子的 HcPre 占比 | 54%（79851 / 148560） |
| **展开算子总时长** | **≈0.05 ms/步** |
| 我此前的预估 | 2.5 ms/步 |
| **偏差** | **50× 高估** |

**⇒ 线③ 关闭**（0.02% 端到端，远低于 0.5% 判据）。

---

## 4. 残余机会（不在原目标三条线内，供决策）

**序列**（每层每步一次）：

```
x = input_layernorm(hs)          # RmsNorm, 5.3 µs, 输出 bf16
hs_int8, scale = dynamic_quant(x) # DynamicQuantV2, 3.1 µs
q_a = quant_matmul(hs_int8, wq_a.weight, scale)
```

**形状完美衔接**（`[6,5120]`），是标准的"norm + 量化"模式。**但现有算子融不了**：

| 可用算子 | 输出 |
|---|---|
| `npu_rms_norm_dynamic_quant` | `(int8, scale)` —— **没有 bf16** |
| `npu_rms_norm_cast` | `(bf16, fp32)` —— 没有 int8 |
| （无） | `(bf16, int8, scale)` |

而 `input_layernorm` 的 **bf16 输出下游还需要**（attention 本体 + compressor tail）
⇒ **不能用二输出算子替换**。

**残余机会规模**：

| 项 | 值 |
|---|---:|
| 相邻对 | 39.4 对/步 |
| 分开成本 | 8.4 µs/对 |
| 三输出融合预估 | 5.0~6.0 µs/对 |
| **可省** | **0.10~0.15 ms/步 = 0.4~0.6%** |
| 判据 | ≥0.5% |
| **判定** | ⚠️ **恰在边缘**，且需**新 kernel**（现有签名不符） |

---

## 5. 附带发现：一个必须记住的运维陷阱

重启 tp8k5 时反复失败：`Free memory on device (5.77/61.27 GiB) less than desired GPU memory utilization`。

**根因**：**留了 13 小时的孤儿 `VLLM::Worker` 进程（ppid=1）仍持有 ~59 GB/芯片**。

**关键**：`kill -9` **杀不掉它们**（进程显示 S 态，但被设备上下文粘住，kill 后仍有新进程占着）。
⇒ **唯一可靠的办法是 `docker restart dsv41-tp8k5`**（已验证：重启后 8 个 die 全部回到 ~3 GB）。

---

## 6. 环境状态

| 项 | 状态 |
|---|---|
| tp8k5 | ✅ health **200**，KV **2,987,727**（基线区间内），冒烟 `'2'` ✅ |
| tiny | ✅ health 200 |
| 探针 | 3 个（`V41_QLIN_PROBE` / `V41_QBRANCH_PROBE` / `V41_QCOUNT_PROBE`），**全部 env 门控、默认关**，不影响性能 |

---

## 7. 三条线的诚实总结

> **目标假设（"W4A8 分支缺融合导致 ~1.4% 损失"）不成立** —— 融合本来就在生效。
> 另两条线（hc_fn L2、展开算子融合）经实测量级过小，**均不值得做**。
>
> 真正剩下的只有一条：**`input_layernorm + quantize` 的三输出融合（0.4~0.6%，需新 kernel）**，
> 而它恰好卡在 0.5% 判据的边缘。

---

## 8. 复现

```bash
# 线① 证据（运行时探针）
ssh a3-21 'docker exec dsv41-tp8k5 grep -a "QCOUNT\|QLIN-PROBE\|QBRANCH" \
  /opt/dsv41/results/armRESTORE7_1007_090030/serve_qc.log | head -8'
# 融合算子单算子对比
ssh a3-21 'docker cp /tmp/bf2.py dsv41-op-hcfuse:/tmp/ && \
  docker exec dsv41-op-hcfuse bash -lc "cd /tmp && python3 bf2.py"'
# 线②③
ssh a3-21 'python3 /tmp/line23.py ~/tmp/kd'
```
