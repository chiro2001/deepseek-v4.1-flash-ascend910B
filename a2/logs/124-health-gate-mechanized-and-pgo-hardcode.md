# 124 — ★★ **把"行为健康闸"机械化**（双向自测过：事故臂 FAIL rc=9 / 健康臂 PASS rc=0）+ **PGO 的真缺陷是"字面量硬编码"**

> 2026-09-23 07:1x–07:2x CST。执行：**主代理**（写闸 + 真实数据双向自测 + 定位 `PYTHON_PGO` 硬编码点）。**零占卡**。
> 标记：**【实测】/【推断】/【未确认】**。

---

## 0. 一句话

`logs/123` 那条"`ms/step` 没变但行为已回退"的教训，**不能只写在文档里**（本仓反复栽在"写了但没人执行"）⇒ 做成**可执行的门**：
`a2/scripts/arm_health_gate.py`（已部署 A3 `~/tmp/arm_health_gate.py`，md5 `d0560ed9…`），并在**真实数据上双向自测**过。
★ 同时把 **PGO 为什么一直没生效**追到**具体一行**：`run_arm_r8.sh:226` 的 `PYTHON_PGO=0` 是**字面量**，不是 `${PYTHON_PGO:-0}`。

---

## 1. 【实测】健康闸：判据、自测、以及它抓到了什么

### 1.1 判据（全部来自 vLLM 自己的 `metrics.py` 行 —— **零成本、已在 `serve.log` 里**）

| # | 判据 | 健康区间 | 事故值（`r8-g2g3`） |
|---|---|---|---|
| H0 | 日志里**必须存在** `Per-position acceptance rate` 行 | 存在 | （不存在 ⇒ 按 FAIL） |
| H1 | `Mean acceptance length` | **2.5–3.0**（阈值 ≥1.8） | **1.02 / 1.04** ❌ |
| H2 | **`Per-position acceptance rate` 的 position-0** | **0.62–0.93**（阈值 ≥0.40） | **0.017 / 0.036** ❌ |
| H3 | `Accepted > 0` 且 `Drafted > 0` | 在跑 | ✓（2/575） |
| H4 | （给了 `--ref` 时）position-0 比参考臂低 ≤0.25 | —— | ❌ |

### 1.2 ★★ 双向自测（**没做过双向自测的门不许当门用**，同 `AGENTS.md §5b` 第 3 条）

```
$ python3 ~/tmp/arm_health_gate.py .../r8_r8-g2g3_20260923_064643/serve.log        # 事故臂
  ⛔ H1 FAIL: Mean acceptance length=1.02 < 1.8
  ⛔ H2 FAIL: position-0=0.017 < 0.40
  ⛔⛔ 健康闸 FAIL ⇒ ★ 本臂性能读数作废
  rc=9   ✅ 正确报警

$ python3 ~/tmp/arm_health_gate.py .../r8_r8-merged-suite_20260923_042330/serve.log  # 健康臂
  ✓ H1 Mean acceptance length=2.83 ｜ ✓ H2 position-0=0.937 ｜ ✓ H3 spec 在跑（349/955）
  ✅ 健康闸全过 ⇒ 本臂的性能读数可用于对照
  rc=0   ✅ 正确放行
```

⇒ **两臂给出不同的数 ⇒ 它有判别力**（不是"全 0 假阳性"）。

### 1.3 部署

| 位置 | 内容 |
|---|---|
| `a2/scripts/arm_health_gate.py` | 实现（纯流式，O(1) 内存；接受 `serve.log` 或 run 目录） |
| A3 `~/tmp/arm_health_gate.py` | 同 md5 `d0560ed999818f31367d9136dde25389` |
| **`a2/AGENTS.md` §5b.0** | ★★★ **新增硬规则**：任何 8 卡性能臂，把 `ms/step` 记进对照表**之前**必须先跑这道闸，`rc=9` ⇒ **读数作废**。写进 `AGENTS.md` 是为了让**以后每个子代理都自动拿到**（本目录的 agent 指南）。 |

★ 特别条款：**改到 attention / KV / 图输入**的改动（本轮正是这一类）⇒ **健康闸与性能判据必须同时报**。
`_slot_mapping_2d` 是**图输入**（`npugraph_ex` 捕获时是 Placeholder）⇒ **"换成新 tensor"有失配风险；"就地写回同一缓冲"是安全形态**。

---

## 2. 【实测】PGO 为什么一直没生效：**字面量硬编码**

`agents/R_8card_int8/scripts/run_arm_r8.sh:226`（逐字）：
```
  CPU_BIND=0 DROPCACHE=0 PATCH_MODE=mount PYTHON_PGO=0 \
```
★ 它是**字面量 `0`**，**不是** `${PYTHON_PGO:-0}` ⇒
① 外部 `export PYTHON_PGO=1` 被**静默丢掉**；② `serve_a2.sh:194` 的默认值 **1** 永远用不上。
⇒ **这就是"我们全部 8 卡臂都是 `pgo=0`"的机制**（`serve_cmd.txt` 逐字可查；`faB` 基线那种 `pgo=1` 我们从没复现过）。

**修法（必须改成私有副本，共用那份现在有臂在跑）**：
```bash
cp .../agents/R_8card_int8/scripts/run_arm_r8.sh <你的目录>/run_arm_r8.pgo.sh
sed -i 's/PYTHON_PGO=0 \\/PYTHON_PGO="${PYTHON_PGO:-0}" \\/' <你的目录>/run_arm_r8.pgo.sh
```
**判据**：起服日志须出现 `pgo=1` 且 `[serve_a2] PYTHON_PGO=1 pgo_target=/usr/local/python3.12.13/lib/libpython3.12.so.1.0`；
若打 `pgo=0` 或 `pgo_target=none` ⇒ **改动没生效，本臂作废**。
★（目标落点与产物我已独立验证：见 `logs/121 §3`。）

---

## 3. 对"改动生效性"这一类坑的收口（本仓第 N 次）

| 坑 | 机制 | 通用做法 |
|---|---|---|
| `081` 合并件过期 | 改了源、没重新合并 | 合并件 md5 进门 |
| `093` dsa 挂错份 | runner 默认指向未修目录 | 起服后**容器内** md5 反查 |
| `099` 别人的 health | 端口门被别的容器满足 | 门绑**容器内** `health` |
| `112` 注释截断参数表 | `\`+`#` 把 `env` 参数表切断 | 长参数表里**不插注释** |
| **`124`（本卷）`PYTHON_PGO=0` 字面量** | 调用方的 env **被 runner 覆盖** | ★ **凡 runner 里写的 env，都要检查是不是 `${X:-default}`；字面量 = 调用方无法覆盖** |
| **`123` 行为回退** | `ms/step` 看不出、读数是"有效但无意义" | ★ **健康闸**（本卷机械化） |

⇒ ★ 归纳一条：**"我传了这个变量"与"这个变量真的生效了"是两件事** ——
**判据必须落在"实际生效后的可观测痕迹"上**（容器内 md5 / 起服日志的 `pgo_target` / `SpecDecoding metrics` 的接受率），
**不能落在"我传了"这个动作上**。

---

## 4. 复现配方

```bash
# 健康闸自测（在 A3 上）
R=~/projects/dsv41-upstream-pr/shadow-pkg/results
python3 ~/tmp/arm_health_gate.py $R/r8_r8-g2g3_20260923_064643/serve.log;      echo "rc=$?  # 期望 9"
python3 ~/tmp/arm_health_gate.py $R/r8_r8-merged-suite_20260923_042330/serve.log; echo "rc=$?  # 期望 0"

# PGO 硬编码点
grep -n "PYTHON_PGO" ~/projects/dsv41-upstream-pr/agents/R_8card_int8/scripts/run_arm_r8.sh   # 期望看到字面量 0
grep -n "PYTHON_PGO" ~/projects/dsv41-upstream-pr/shadow-pkg/scripts/serve_a2.sh             # 期望看到默认 1
```
