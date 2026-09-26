# 118 — ★★★ `r8-tbl` 臂判决：**列并行改写在生产几何下收益为 0**（`width=1040` 让 NC≡1）+ **臂间 `hp` 噪声 ±2 ms ⇒ 小格改动不能用 `hp` 判**

> 2026-09-23 05:50–06:1x CST。执行：**主代理**（起臂 + 判据 + 离线核对）。臂 `r8-tbl`：`dsa_dir_D=MERGED_TBL`、8051 独占、8 卡 Phy-ID 8–15。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

**上游预测的 0.40–0.53 ms/step 不成立**，且不是"没生效"，而是**建模的几何错了**：
`swa_table` 的 `width` 不是 `max_num_batched_tokens`（8192），而是 **`block_table.shape[1]` = 1040**（`133120 / 128`，`dsa_final_merged.py:570`）。
⇒ 改写用的 `BLOCK_W=2048` 在 `width=1040` 下 `range(0, 1040, 2048)` **只迭代 1 次** ⇒ `grid=(N,1)` ⇒ **第二维恒为 1，没有"列并行"这回事**（trace 逐字：`[kv8fuse] swa table grid=(1,1) width=1040`）。
★ 更彻底的判据：这一格的**天花板本来就只有 0.129 ms/step**（prof 实测 `_kv8_swa_table_kernel` = 40.00 次/步 × 3.23 µs = 0.1291 ms/步），因为生产宽度只有 1040 而不是 8192。
★★ **2026-09-23 06:2x 按子代理 `GEOM_MICRO` 的单卡实测修正本句措辞**（**本卷首发时写错了**）：
改写**不是**"退化为原样"——它把"**一个 program 循环 9 次**（`ceil(1040/128)`）"换成了"**一次 2048 宽的掩码写**"，
单卡实测 **2.83 → 1.37 µs/层（NC=1 最快；NC=9 反而 4.59 µs）** ⇒ **已落袋 0.053–0.058 ms/step**，且 48 例 × 9 变体 `torch.equal` 全过。
⇒ 准确的说法是：**"没有列并行"（`NC≡1`）+ "但顺手省掉了一个 9 次循环"，两者都成立**；原估的 0.40–0.53 仍**不成立**。
★★ 同时暴露一个**测量学问题**：**臂间 `hp` 噪声达 ±2 ms**（同配置的 `r8-merged` 29.93/29.95 vs `r8-merged-suite` 32.09）⇒ 0.1–0.5 ms 的小格**不能用 `hp` 判**，必须用 **op 级计数**（profiler）或**生产几何的单卡微基准**。

---

## 1. 【实测】臂事实（起服四条判据 + 指纹）

| 项 | 值 |
|---|---|
| 臂 ID | `r8_r8-tbl_20260923_055010`（05:50:10 起，05:59:49 结束） |
| 前置门 | ✓ 8051 空闲 ｜ ✓ c0 锁可抢 ｜ ✓ `MemAvailable=1308 GiB ≥ 1200` ｜ ✓ 三个件 md5 全对 |
| 容器内 `dsa_v41.py` | **`30ecf49b3fe11fb704fe505425b2dfa5`**（= 融合件接线版）✅ |
| 容器内 `kv8_fuse_triton.py` | **`9bdcdaf54fd8b128540009e7137fb3dc`**（= **列并行**版）✅ ⇒ J1/J2 **全过** |
| G2b | ✓ `dsa_dir_D=~/projects/dsv41-upstream-pr/agents/MERGED_TBL` |
| `GPU KV cache size` | **427,643**（与预注册逐字相同 ⇒ 两个修复都不动容量）✅ |
| draft 入图 | `Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL` ×8 rank ✅ |
| 起服期 `EE1016 / capture failed` | **0** ✅ |

---

## 2. 【实测】★ 列并行**没有并行机会**（这是本卷的核心）

```
(Worker_TP5_EP5) [kv8fuse] swa table grid=(1,1) width=1040 pp=3 use_qlen=True
```
全部出现过的栅格形态（`sort -u`）：
```
grid=(1,1) grid=(2,1) grid=(3,1) grid=(4,1) grid=(6,1) grid=(7,1) grid=(8,1) grid=(16,1) grid=(32,1)
```
**第二维永远是 1**。而**对照臂 `r8-merged` 的 trace 是 `grid=(1,)`（一维）** —— 说明：
1. 改写**确实接线了**（栅格从 1 维变 2 维）；
2. 但 **`width=1040 < BLOCK_W=2048` ⇒ `NC = ceil(1040/2048) = 1`** ⇒ **列并行退化为原样**。

### 2.1 `width` 到底是什么（源码级）

`a2/agents/FUSE_MULTIROW/out/dsa_final_merged.py:570` 逐字：
```python
width = block_table.shape[1]
```
而 `:626` 用它构造表：
```python
columns = torch.arange(width, device=device, dtype=torch.int64).view(1, width)
```
⇒ `width` = **每个请求的 block table 列数** = `ceil(max_model_len / block_size)` = `ceil(133120 / 128)` = **1040**。
★ **`MAX_LEN=133120` 与 `BLOCK=128` 都在臂的 `serve_cmd.txt` 里逐字可查** ⇒ 1040 不是巧合。

### 2.2 上游建模错在哪

`FUSE_TUNE` 用 **`WIDTH=8192`** 做标定，得到"原样 13.17 µs/层 → `BLOCK_W=2048` 1.39 µs/层 ⇒ 可回收 0.40–0.53 ms/step"。
但 `WIDTH=8192` 对应的是 **`max_num_batched_tokens`**（= `bat=8192`），**不是 `block_table.shape[1]`**。
两边**同名不同物**（`width`）—— 与本仓已记录的"`089` 差第三个变量 / `097` 两个同名不同义的计数器"**同族**。

---

## 3. 【实测】这一格的真实天花板 = **0.129 ms/step**

在 **prof 臂**（`dsa_dir_D=MERGED_FULL`，即**未改列并行**的原版）的 decode 稳态窗上：

| OP | 次/步 | **ms/步** | 单次 µs | `aiv_vec` | `aiv_mte3` | `aiv_scalar` |
|---|---:|---:|---:|---:|---:|---:|
| `_kv8_swa_rows_kernel` | 40.00 | 0.4226 | 10.6 | 0.059 | 0.207 | 0.123 |
| **`_kv8_swa_table_kernel`** | **40.00** | **0.1291** | **3.23** | **0.524** | **0.430** | 0.427 |
| `ViewCopy` | 30.73 | 0.2503 | 8.1 | 0.002 | 0.001 | **0.499** |
| `TensorMove` | **1.00** | **0.0099** | 9.9 | — | — | — |

⇒ **`swa_table` 全窗只有 12.4 ms（3840 次 × 3.23 µs）⇒ 即便把整个 kernel 降到 0，也只省 0.129 ms/step。**
⇒ **对比 FUSE_TUNE 的单卡读数 13.17 µs/次（width=8192）⇒ 生产是 3.23 µs/次，差 4.1×，正是 1040 vs 8192 的循环次数之比（9 次 vs 64 次）。**
★ **结论：这一格关闭。**（不是"改动无效"，而是"对象本来就小"+"改写在生产几何下无并行机会"。）

---

## 4. 【实测】★★ 臂间 `hp` 噪声 ±2 ms —— 比任何小格都大

| 臂 | 配置 | **`[bneck] hp`（末行，rank0/rank7）** | `d2h` |
|---|---|---:|---:|
| `r8-merged`（04:03:53） | chunk + 融合（`8057b3eb`） | **29.932 / 29.950** | 12.38 / 12.87 |
| `r8-merged-suite`（04:23:30） | **同一配置**（bneck 消融臂） | **32.094 / 32.092** | 15.19 / 15.27 |
| **`r8-tbl`（05:50:10）** | chunk + 融合 + 列并行（`9bdcdaf5`） | **28.015 … 28.064**（8 rank） | **10.58 … 11.30** |

**读法（★ 必须同时读这两条）**：
1. **`hp` 的臂间差异（29.93 → 28.04，−1.89）不能归因给列并行** —— 因为 §2/§3 已证列并行**无并行机会**、天花板 **0.129 ms**。
2. **同配置的两条臂自己就差了 2.16 ms**（29.93 vs 32.09），且 **`d2h` 与之同幅变化**（12.4 → 15.2）⇒ **`hp` 主要由 host 侧负载/噪声驱动**（与 `FUSE_TUNE §③` 独立发现的"`hp` 与 `d2h` 同幅变动"**互相印证**）。
3. ⇒ ★★ **0.1–0.5 ms 量级的改动不要用 `hp` 判**：判据必须换成 **op 级计数**（`_kv8_swa_table_kernel` / `ViewCopy 16384` / `ScatterNdUpdateSk` 的 Count × µs）或**生产几何的单卡微基准**。本仓已多次栽在"判据绑错对象/口径"，这是同族的第 N 次。

---

## 5. 【未确认】本臂的 quote 被污染，**不采用**

`TBL_POST/quote_{1024,8192}.jsonl` 的两组读数（1K: 87.08 ms/step、8K: 23.68 ms/step）**全部作废**，原因是：
* 我的判据脚本在臂的**自身压测（`PROMPTS=6 ROUNDS=3`）仍在进行时**跑了 quote（时间戳 05:59，臂 05:59:49 才结束）⇒ **两个负载叠加**；
* 逐发离散度自证污染：1K 三发 = **89.9 / 87.1 / 39.0 ms**（相差 2.3×），而干净臂的同类读数是**极稳**的（ARM_1M 的 1K 三发 `178.127/178.162/178.126`）；
* 且 `accept_length` 异常（1.47–2.46，干净臂应 ~4.0）。
⇒ ★ **教训**：**判据脚本必须自带"臂已空闲"的前置断言**（`running=0 waiting=0` + `[bneck]` 稳定），而不是假设"健康就能测"。已写进 `a2/scripts/tbl_arm_postcheck.sh` 的后续修订项。

---

## 6. 对 goal 的净影响（**可动清单收缩**）

| # | 项 | 原估 ms/step | **本轮后** | 依据 |
|---|---|---:|---|---|
| 1 | `swa_table` 列并行 | 0.40–0.53 | ❌ **关闭**（天花板 **0.129**，且改写无并行机会） | 本卷 §2/§3 |
| 2 | `_slot_mapping_2d` 视图写（`ViewCopy "16384"`） | — | **0.250**（★新，最小改法 + 4 条预注册判据已给） | 子代理 `ARM_1M` 的 `reinplace_audit.md` |
| 3 | fp16 scale 写并入 int8 K 核 | 0.263 | 0.263（未动） | `PROF_MINE §4.1` |
| 4 | MoE gating 的 3×int32→int64 + 1×float→bf16 | 0.147 | 0.147（未动） | `PROF_MINE §4.1` |
| 5 | `HcPost`+`HcPre` 跨支融合 | 0.30–0.64 | 0.30–0.64（未动） | `PROF_MINE §4.1` |
| | **合计（我们可改的代码）** | | **≈0.96–1.30** | ⇒ 27.905 − 1.3 ≈ **26.6** |

⇒ **【推断·强】服务端 8K ≤24 ms 在本轮可动范围内不可能达成**；缺口仍需 A2 侧 MC2（≤1.153，仅 A2、不支持入图/quant）或 **C 层供应商算子/再量化**。
★ 同时：**已达成且无争议**的部分不变（四轴同开、保精度含 1M 冷算==取回逐字节、8K/32K 超越加 int8 前基线）。

---

## 7. 下一步（**已按本卷的测量学结论重新排序**）

| # | 动作 | 成本 | **判据（不用 `hp`）** |
|---|---|---|---|
| 1 | **生产几何单卡微基准**：`swa_table`(width=1040, BLOCK_W=128/208/…)、`_slot_mapping_2d` 改法、scale 融合 —— 三格一次测准 | 单卡 c1/c2，约 40 min，**不占 8 卡** | 每格 µs/层 × 40 ⇒ ms/step；与 prof 的 `3.23 / 8.1 / 46.7` µs **对账** |
| 2 | `_slot_mapping_2d` 改法上臂（若 1 证实 ≥0.15） | 8 卡 ×25 min | `ViewCopy "16384"` 计数 **2280 → 0**（op 级，不受 `hp` 噪声影响） |
| 3 | 8K vs 128K **两档 profile 对差**（`117 §4.3` 的 3.6 ms） | 8 卡 ×2 | 两档 `op_statistic` 之差逐项归因 |
| 4 | 臂间噪声本身要不要压（例如固定 `CPU_BIND`/预热轮数） | —— | 同配置两条臂 `hp` 之差 ≤0.3 才算压住 |

---

## 8. 复现配方（只读）

```bash
# ① 臂事实与 trace
d=~/projects/dsv41-upstream-pr/shadow-pkg/results/r8_r8-tbl_20260923_055010
grep -a "\[bneck\] mode=stock" $d/serve.log | tail -8
grep -ao "\[kv8fuse\] swa table grid=([0-9, ]*) width=[0-9]*" $d/serve.log | sort -u   # ★ 第二维恒为 1
grep -a "GPU KV cache size" $d/serve.log | tail -1
# ② 对照臂（同配置、无列并行）的 hp —— 证明噪声
grep -a "\[bneck\] mode=stock" ~/projects/dsv41-upstream-pr/shadow-pkg/results/r8_r8-merged-suite_20260923_042330/serve.log | tail -2
# ③ 这一格的天花板（prof 臂，原版 kernel）
grep -E "^_kv8_swa_table_kernel|^ViewCopy|^TensorMove" ~/projects/dsv41/a2/agents/HC_PROBE/out_bottleneck.csv
# ④ width 的来源
grep -n "width = block_table.shape\[1\]" ~/projects/dsv41/a2/agents/FUSE_MULTIROW/out/dsa_final_merged.py
```

**产物**：本卷 ｜ `TBL_POST/`（**污染的 quote，勿引用**）｜ `a2/agents/HC_PROBE/out_bottleneck.csv` ｜ `a2/scripts/tbl_arm_postcheck.sh`（判据包）。
