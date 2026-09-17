# 收尾三补丁的验证结论（含两处**归因更正**）

> 2026-09-16 16:00 CST｜A3-node1 dummy device 账目｜A=tvA（三项关）／B=tvB（三项开）
> 两臂锚点数**完全相同（9460）** ⇒ 直方可比

---

## 0. 结论速览

| 补丁 | 预期 | **实测** | 判定 |
|---|---|---|---|
| `ids64-hoist` | −0.047 ms | **cast 计数完全未变**（21186 → 21186） | ❌ **归因错误**（见 §2） |
| `engram-gate-hoist` | −0.039 ms | `"4,5120"` **dynamic** 的 cast 24 → **0** ✓ | ✅ 部分生效（eager 路径） |
| `pad-skip` | −0.130 ms | `pad` 0.153 → 0.175（**未改善**） | ❌ **撤回**（见 §3） |
| **合计实际可回收** | ~~0.22~~ | **≈ 0.01 ms** | — |

---

## 1. `gate_hoist` 生效（唯一有效的）

| shape | state | tvA | tvB | Δ |
|---|---|---|---|---|
| **`"4,5120"`** | **dynamic** | **24** | **0** | **−24** ✓ |
| `"4,5120"` | static | 856 | 856 | 0 |
| `"6,4,5120"` | static | 1926 | 1926 | 0 |

⇒ eager 路径（dynamic）的那 24 次被消除了；图内（static）的没变
—— 与「`q_weight.float() * k_weight.float()` 在 gate 调用点」的定位一致。
**收益量级：24 × 4.3 µs ≈ 0.1 µs/步**（可忽略，但机制正确）。

---

## 2. ⚠️ `ids64-hoist` 的归因是错的

### 2.1 现象：cast 计数**完全没变**

按 `(输入 dtype, 输出 dtype)` 分组，取 `[6]` 形状：

| (in, out) | tvA | tvB |
|---|---|---|
| **(`INT32`, `INT64`)** | **21186** | **21186** |
| (`INT64`, `INT32`) | 3638 | 3638 |
| (`BOOL`, `INT64`) | 856 | 856 |
| (`INT32`, `BOOL`) | 214 | 214 |

**21186 ÷ 236.5 passes = 89.6/pass —— 逐次完全相同。**

### 2.2 真实来源：**RoPE 的 `pos_tensor.to(torch.long)`**

`89.6/pass` 与 RoPE 的数量吻合：
* `small-op-audit.md` 记载：**RoPE 每层 2 次**（q-RoPE `dsa_v41.py:374` + attention 输出的逆旋转 `:602`）
* 40 层 × 2 = **80**，加 draft 3 层 × 2 = **6** ⇒ **86**，加 compressor 等 ≈ **89.6** ✓

而 `rope_dsv4.py` 里确实有：
```python
gather_idx = pos_tensor.to(torch.long).reshape(-1, 1, 1, 1).expand(...)
```

**⇒ 那 40/pass 从来就不是 router 的 `input_ids` cast，而是 RoPE 的 `positions` cast。**

### 2.3 为什么线 3 的估计错了

线 3 的依据是「真 8 卡 op_summary 里 `Cast "8" INT32→INT64` 42 次/步」+
`fused_topk_router.py:164` 的 `.to(torch.int64)`。

**但 M=8 的 profile 与 M=6 的 profile 里，同一个 shape 的 cast 数量差了一倍多**
（42/pass vs 89.6/pass）—— 这说明 42 那个数**不是**纯 router 的，或者两个 profile 的
cast 构成不同（prefill/decode 混合比例不同）。

**⇒ `ids64` 最多能省 router 的那一份（40/pass 里的一部分），而我们无法从现有 profile 分离它。**

### 2.4 处置

* **不撤回补丁**（它逻辑正确、零风险、且可能省掉 router 那一份）
* **但把预期从 −0.047 ms 下调到「≤0.05 ms，且无法用现有 profile 验证」**
* **不投入更多资源去追**

---

## 3. ⚠️ `pad-skip` 撤回（我自己写的补丁）

我原假设：`pad` 的 0.13 ms = 50 MB memset。

**实测推翻**（`/tmp/pad_bench.py`，NPU 直接测）：

| 操作 | 实测 |
|---|---|
| full `zero_()` **25.2 MB** | **8.48 µs** |
| slice `[:6].zero_()` **74 KB** | **16.72 µs**（**更慢**！） |

⇒ 50 MB memset ≈ 17 µs，**不是 0.13 ms**。
`pad` 的真实构成是 **6 次小设备 kernel 提交**（各 ~16–25 µs）。

⇒ **去掉 full-zero 改用 slice-zero 反而可能更慢**（16.7 vs 8.5 µs）。

**⇒ 撤回。不要写进交付配置。**

**教训**：我当初用「50 MB ÷ 0.13 ms = 387 GB/s，看起来很合理」来"验证"自己的假设
—— **这是循环论证**。正确做法是先单独测 `zero_()` 本体（30 秒的实验）。

---

## 4. ⚠️ 方法论发现：`/proc/PID/environ` **不可信**

我一度用 `cat /proc/<worker_pid>/environ | grep V41` 判断 env 是否传到 worker，
得到「只有 5 个 V41 变量」，据此怀疑 env 传递不完整。

**但这个判断是错的**——同一份 environ 里**缺少 `V41_QLI_NO_CANDIDATE` / `V41_O_PROJ_2D` /
`V41_ROPE_IDXSEL`**，而这三个补丁**都已用 device 账目独立验证生效**（QLI 每 op 减半、
rope 的 `Index` 减少 4104 次、F3 的 `Transpose` 消失）。

⇒ **`/proc/PID/environ` 只反映进程启动时的 env 区域快照，且可能被截断/被 glibc 重写；
不能用它判断"开关是否生效"。**

**正确做法**（本报告用的就是）：**看设备侧行为指纹**
（cast 计数、算子出现/消失、每 op 微秒变化）。

顺带：`set -a; A=1 B=2 C=3` 的 shell 语义**已验证正确**（三个都 export），
所以 inner 脚本的多赋值行没有问题。

---

## 5. 对 110 tok/s 目标的影响

| 项 | 收益 |
|---|---|
| `gate_hoist` | ≈0.0001 ms |
| `ids64` | ≤0.05 ms（无法验证） |
| `pad-skip` | **0（撤回）** |
| **小计** | **≈0.05 ms** |

**⇒ 这三项对目标的贡献可以忽略。真正的杠杆在别处：**

| 项 | 状态 | 收益 |
|---|---|---|
| **numba `hash` + `plan`** | 线 3 已交付（逐位等价） | **−0.57 ms** |
| numba `lookup` | 进行中 | −0.17 ms（预估） |
| **CPython PGO+LTO** | **已构建完成** | 全局 host 开销 **−15~18%** |
| A 的第三源（128K 非确定性） | 线 1 在查 | **决定能否达标** |

---

## 6. 证据

| 内容 | 路径 |
|---|---|
| A/B profile | `logs/prof_tvA/`、`logs/prof_tvB/`；`/tmp/tvA_rank0.csv`、`/tmp/tvB_rank0.csv` |
| 运行日志 | `/tmp/verify_tail.log` |
| pad 微基准 | 容器 `/tmp/pad_bench.py` |
| 脚本 | `exp_tools/verify_tail_patches.sh` |
