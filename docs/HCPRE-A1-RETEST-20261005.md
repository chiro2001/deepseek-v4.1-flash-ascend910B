# HcPre A1 复测：**机制已生效，但服务内效应低于采纳门槛**（2026-10-05）

> 背景：R5 曾把 A1（HcPre 自适应 `K_L0`）判为"**在服务里从未执行**"——
> 根因是 static kernel 缓存（`cache/skcache/compile_outputs`）的 key **不含 `.o` 内容**，
> 换 vendor 内核后运行时仍复用旧内核（`KERNEL-CACHE-STALE-20261004`）。
> 修复手段是 `V41_OPP_CLEAR_SKCACHE=1`。本轮用**规范的单变量 A/B**重新验证 A1。

## 1. 被测改动（trackB 交付，非本仓发明）

`hc_pre_cube_compute.h` 里 `K_L0_SIZE` 原本硬编码 32；A1 改为按 M 自适应：
`mAligned ≤ 64 → kL0Size = 128`（L0 主循环轮数 32→8），M>64 时退化为 64/32。

| 证据（trackB 提供，本仓复核过包指纹） | 结果 |
|---|---|
| 数值等价 | eager **514/514**、服务 16 个捕获桶 **16/16** 全零差异 |
| 设备时长 | M=6 图模式 **33.33 → 30.72 µs（−2.61）** ⇒ 预期 **−0.21 ms/步**（80 次） |
| 包 | `~/tmp/trackB/pkg_hcpre_A1`，HcPre `.o` md5 `de790a50…` |

## 2. 单变量条件（先证明，再测）

| 检查 | 结果 |
|---|---|
| 两臂的 **gmm1 内核**是否相同 | **相同**（都是 `5dab79be…`）⇒ 只有 HcPre 这一个自变量 |
| 交付实例当前 HcPre | stock `5aa7b15c…`（复测后已确认回到 stock） |
| 两臂是否都清 skcache | **是**（日志：`compile_outputs → .stale_20261005_053642` / `_054924`） |
| 运行日志 | `[V41-HC-FUSE] opp pkg: …/pkg_hcpre_A1`、`[OPP-OVERRIDE] 已注入 inner.sh 覆盖块` |
| 冷启 | A1 臂 437 s、对照臂 1035 s+（两者都重编译，时间差来自编译规模） |

## 3. 结果（各 1 run；判据 = `[bneck] hp` 按 batch 归一 + 聚合吞吐）

### 3.1 `hp` 配对差（`tools/ab_gate.py`，负数 = A1 更快）

| 对照 | n=6 | n=12 | n=18 | n=24 | n=36 | n=48 | **中位** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `armF_r6_ctl`（同期、也清了缓存） | −0.28 | −0.10 | −2.39 | −0.02 | nan | −0.11 | **−0.104 ms** |
| `armF_r6_restore`（旧交付基线） | −0.23 | −0.06 | +1.46 | −0.02 | — | +0.72 | **≈ 0** |

> 对照臂的 n=18 与 n=36 两个档异常（35.14 / nan，而同期另一臂是 31.29 / 38.85）
> ⇒ 同期对照本身有噪声，**不能把 −0.104 当作净收益**。

### 3.2 聚合吞吐（A1 − 对照）

N=1 **+4.7%**、N=2 **+6.0%**、N=4 +1.0%、N=8 **−2.8%**
⇒ **方向不一致**，在 ±5% 的 run-to-run 漂移内。

### 3.3 正确性

`ced_pd_acceptance --mode all`（144K）：**11/11 通过，失败 0**。

## 4. ★ 执行指纹：A1 **确实被编译并执行了**（本轮新增证据）

清缓存时被移走的两个目录正好是"stock 派生"与"A1 派生"的编译产物：

| 目录 | 内容 |
|---|---|
| `compile_outputs.stale_20261005_053642` | A1 臂启动前移走的**累计 stock 缓存**（含 2384 个 HcPre 文件、**790 个唯一静态内核 sha**） |
| `compile_outputs.stale_20261005_054924` | A1 臂**新编译**的产物（40 个文件、**12 个唯一 sha**） |

**两个 sha 集合的交集 = 0**（`comm -12` 输出为空）。

* 该 sha 不是 pid 相关：stock 累计 2384 个文件才 790 个唯一 sha ⇒ 同一 (op, shape) 稳定复现同一个 sha；
* tiling 描述符（`*_opcompile/HcPre_*.json`）两边**完全相同**（8/8 same）——符合预期，
  因为 A1 只改内核、不改 tiling ⇒ **描述符相同 + 内核 sha 全不同** 正是"A1 生效"的签名。

⇒ **R5 的"从未执行"问题已被 `V41_OPP_CLEAR_SKCACHE=1` 彻底解决**；
这一条对所有后续 kernel A/B 都是前提，值得单独立账。

## 5. 判决

**不采纳为交付默认。** 理由：

1. **机制已验证**（数值逐位等价 + 内核确实换上了），但**服务内效应只有 −0.10 ms/步（0.4%）**，
   低于本机单跑噪声底（±0.16 ms，`LEVERS-R5` 实测）；
2. 换一个对照（旧基线）该效应变成 ≈0 ⇒ **未达到"可采纳"的证据强度**；
3. 处置与 `armH`（`IDS64_HOIST`+`ENGRAM_PAD_SKIP`）一致：**保留为可选包，不计入已兑现收益**。

**要采纳它需要什么**：`ab_gate` 建议的"≥2/3 轮为负"，即再做 **2–3 轮交替 A/B**
（每轮 ≈35 min：起服 8.5 min + 基准 12 min + 验收 6 min，两臂都需清 skcache）。
预期收益仅 0.8%，**建议与其它"逐位等价但收益 <1%"的候选打包成一次多轮验证**，
而不是单独为它再烧 2–3 h。

## 6. 复现

```bash
# A1 臂（含清缓存）
bash tools/run_arm_suite.sh ~/tmp/launch_armA1.sh armF_r6_A1 1 0 1
# 对照臂（同样清缓存，避免互相污染）
bash tools/run_arm_suite.sh ~/tmp/launch_armFctl.sh armF_r6_ctl 1 0 1
# 配对判据
python3 tools/ab_gate.py armF_r6_ctl,armF_r6_restore -- armF_r6_A1
# 执行指纹（sha 集合交集）
#   见 §4：比对 cache/skcache/compile_outputs.stale_* 下 static_kernel_HcPre_* 的 sha 集合
```
