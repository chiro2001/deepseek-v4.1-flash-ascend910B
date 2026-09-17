# `rope-idxsel` 验证通过（含 decode 归因）

> 2026-09-16 14:05 CST｜A3-node1 dummy profile（128K）｜A=`V41_ROPE_IDXSEL=0`／B=`=1`
> 两臂锚点数都是 7912 ⇒ 直方可比

---

## 1. 全 profile 事件对照（per-pass 归一化）

| 算子 | A cnt | B cnt | Δ | A 时长 | B 时长 | Δ时长 |
|---|---|---|---|---|---|---|
| `Index` | 6756 | 2652 | **−4104** | 136.5 | 32.1 | **−104.4** |
| `IndexCheck` | 7124 | 3020 | **−4104** | 40.8 | 15.3 | **−25.5** |
| `GatherElementsV2` | 3322 | 802 | **−2520** | 49.1 | 28.1 | −21.0 |
| `BroadcastTo` | 8104 | 5584 | **−2520** | 21.7 | 15.6 | −6.1 |
| **`GatherV3`（新增）** | 0 | 6624 | **+6624** | 0 | 36.6 | +36.6 |
| `Cast` | 60200 | 60992 | +792 | 210.4 | 207.5 | −2.9 |
| **净计** | | | | | | **−123.3 ms** |

**Δ 事件数完全守恒**：`−4104 −4104 −2520 −2520 +6624 = −6624`，
而新增的 `GatherV3` 正好 **+6624** ⇒ **每次旧链（3~4 个 kernel）被替换成 1 个 `GatherV3`**。

---

## 2. **decode 归因**（关键：确认它影响 decode 而不只是 prefill）

取 profile **最后 3 秒**（128K prompt 的 prefill 约 26 s 在前，之后是 96 步 decode，
decode 总时长约 3.2 s ⇒ 最后 3 s ≈ decode 段）：

| 算子 | A（ROPE=0） | B（ROPE=1） | Δ |
|---|---|---|---|
| `Index` | 2620 | 1234 | **−1386** |
| `IndexCheck` | 2764 | 1372 | **−1392** |
| `GatherElementsV2` | 1592 | 276 | **−1316** |
| `BroadcastTo` | 3470 | 2063 | **−1407** |
| **`GatherV3`** | 0 | **2478** | +2478 |
| `InplacePartialRotaryMul` | 10498 | 10005 | −493 |
| `Cast` | 25194 | 23988 | −1206 |

⇒ **decode 段确实在跑这条链，且被替换掉了**。

**decode 收益估算**（按算子平均 µs）：

```
Index          1386 × 20.2 = 28.0 ms
IndexCheck     1392 ×  5.7 =  7.9 ms
GatherElemV2   1316 × 14.8 = 19.5 ms
BroadcastTo    1407 ×  2.7 =  3.8 ms
GatherV3       −2478 × 5.5 = −13.7 ms
                              --------
                              45.5 ms / 约 90 步 ≈ −0.51 ms/step
```

**⇒ decode ≈ −0.45 ~ −0.51 ms/step**，与线 3 的 −0.42 投影一致（略优）。

---

## 3. 结论：**采纳**

| 判据 | 结果 |
|---|---|
| 目标算子归零/变便宜 | ✅ `Index`/`IndexCheck` 各 −4104；`GatherV3` 精确补位 |
| decode 影响 | ✅ 最后 3 s 内 −0.45~−0.51 ms/step |
| 数值等价 | ✅ 线 3 的 21 项测试全 PASS，`max_abs=0.000e+00` |
| 需重捕获 | ❌ 不需要（eager 侧，OP State=dynamic） |

---

## 4. 累积补丁效果（本会话）

| 改动 | decode 收益 | 验证方式 |
|---|---|---|
| MoE AllGather | −4.25 ms | 真权重 A/B（`moe-allgather-breakthrough.md`） |
| `SP_TOKENS=5` + capture 含 6 | −5.9 ms（若漏配） | 真权重 A/B |
| F3（`wo_a` 2D） | −0.31 ~ −0.76 ms | 真权重 A/B/A2 |
| `moe-mask-range` | **−0.51 ms** | device 账目（`moe-mask-range-verified.md`） |
| **`rope-idxsel`** | **−0.45 ~ −0.51 ms** | device 账目（本文） |
| `ids64-hoist` | ≈ −0.047 ms | 待验 |
| `engram-gate-hoist` | ≈ −0.039 ms | 待验 |

**真权重 128K 实测**：AllGather 基线 **35.14** → 当前最好 **32.85（中位）/ 30.655（最好）**

---

## 5. 一个测量陷阱（记下来）

我最初用「40 个连续锚点 + 相邻间隔 ≤1500 µs」找单步窗口，得到 118 个窗口，
里面**只有** `InplacePartialRotaryMul` 和 `Cast`，于是误判「rope 链是 prefill 专属」。

**实际原因**：prefill chunk 也是「40 个锚点」，且层内间隔同样很小 ⇒ 该判据无法区分
prefill chunk 与 decode step。**要区分必须用窗口的 span（decode ≈33 ms，prefill chunk ≈数百 ms）。**

---

## 6. 证据

| 内容 | 路径 |
|---|---|
| A/B profile | `logs/prof_vrA/`、`logs/prof_vrB/`；`/tmp/vrA_rank0.csv`、`/tmp/vrB_rank0.csv` |
| 运行日志 | `/tmp/verify_rope.log`、`/tmp/verify_rope_8k.log` |
| 补丁 | `probe_rope/rope_dsv4.py`（md5 `6a19890850ac7cb41c535b070c2dfbf6`） |
| 启动器 | `serve_a21.sh` 的 `# [ROPE-IDXSEL]` 段（`ROPE_IDXSEL=0/1`） |
| 线 3 交付 | `A3-node2:~/handoff/patches/rope-idxsel/` |
