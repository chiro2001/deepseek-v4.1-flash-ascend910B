# 082 — ★★★ 四轴同开的**第二个**前置：档 C + 图模式必须显式把 `dsa_v41.py` 指向 graphsafe 包（否则捕获期 `EE1016`）

> 2026-09-22 22:4x CST。执行：**主代理**（A3-node1 只读核查；**不占卡、不起容器**）。
> 目的：把「`ENGRAM=1` × 卸载 × int8 档 C × `DRAFT_GRAPH=1`」这条臂的**起服前置**全部钉死。
> 标记：**【实测】/【推断】**。

---

## 0. 一句话

8 卡 runner（`run_arm_r8.sh`）在**档 C + 图模式**下，默认会把 `attention/dsa_v41.py`
挂成 **`X_integrate/pkg-ring` 里那份**（md5 `9db97849…`）—— 而那份**既没有 graphsafe 修复、
也没有 scratch 的 role 分键** ⇒ **捕获期 `EE1016` 必炸**（`logs/048` §2 实测过；
`s_graphfix` 的反例臂 `EE1016=7 / stream_id=34`）。

带修复的那份只有 **`S_graphfix/pkgs/pkg-kv8pf/.../dsa_v41.py`（md5 `94aeebb7…`）**。
⇒ 四轴臂起服时**必须**显式给这两个 env（`run_arm_r8.sh` 都支持透传）：

```bash
DSA_SRC=D                                   # ⇒ R8_DSA_SRC=D ⇒ dsa 取自"D 那个目录"
R8_KV8_DIR_D=$S/pkgs/pkg-kv8pf              # ⇒ 而 D 目录指向 graphsafe 包
```

---

## 1. 【实测】四份候选 `dsa_v41.py` 的关键标志位

| 路径 | md5 | `rows_bound` | `VLLM_V41_KV8_GRAPH_SAFE` | role 分键 | 结论 |
|---|---|---:|---:|---:|---|
| `X_integrate/pkg-ring/…/dsa_v41.py` | `9db97849…` | **0** | **0** | — | ⛔ **既无 graphsafe 也无 role 键** |
| `X_integrate/pkg-kv8pf/…/dsa_v41.py` | `75f4e565…` | **0** | **0** | — | ⛔ 无 graphsafe |
| `S_graphfix/pkgs/pkg-ring/…/dsa_v41.py` | `9db97849…` | **0** | **0** | — | ⛔ 与 X 的 pkg-ring **逐字节相同** |
| ★ `S_graphfix/pkgs/pkg-kv8pf/…/dsa_v41.py` | ★ **`94aeebb7…`** | ★ **14** | ★ **1** | ★ 21 处 | ✅ **唯一带 graphsafe 的** |

判据来源：
```
$ grep -c rows_bound                     <file>      # graphsafe 的 rows_bound 上界逻辑
$ grep -c VLLM_V41_KV8_GRAPH_SAFE        <file>      # 运行期开关（387 行）
$ grep -n "_kv8_graph_rows_bound"        <file>      # 471 行，函数定义
```
★ 顺带一条**判据口径**教训：`grep -c "rows_bound\|max_query_len"` 会**误报**
（`max_query_len` 在非 graphsafe 文件里本来就有 4 处，是别的用途）
⇒ 必须用**精确的模式**（`rows_bound` 单独数），否则会得出"已经修了"的假结论。

---

## 2. 【实测】默认路径为什么会挂错

```
run_arm_r8.sh:97    R8_KV8_DIR_D=${R8_KV8_DIR_D:-$X/pkg-kv8pf}      ← ★ 默认指向"没修"的那份
run_arm_r8.sh:213   R8_INT8_TIER="$TIER" R8_DSA_SRC="${DSA_SRC:-auto}" \
shadow  serve_a2.sh:  case "$_R8DSA" in
                        auto) if [ "$_R8TIER" = "D" ]; then _R8DSA=D; else _R8DSA=${R8_DSA_DEFAULT:-C}; fi ;;
                      esac
                      C) _R8DSAFILE="$_R8C/attention/dsa_v41.py" ;;
                      D) _R8DSAFILE="$_R8D/attention/dsa_v41.py" ;;
```
⇒ 档 C + 无人指定 `DSA_SRC` ⇒ `_R8DSA=auto→C` ⇒ 取 `_R8C`（`X_integrate/pkg-ring`）
⇒ **`9db97849…`（无 graphsafe）** ⇒ `EE1016`。

★ **那 `logs/048` 的档 C 图模式是怎么过的？** 用的是 `S_graphfix/scripts/run_arm_sg.sh`，
它比 `run_arm_r8.sh` 多两个可覆盖项：
```
run_arm_sg.sh:173   echo "R8_GRAPH_SAFE=${R8_GRAPH_SAFE:-0} SG_PKG_D=${SG_PKG_D:-（未设：用 X_integrate/pkg-kv8pf）}"
run_arm_sg.sh:213   R8_KV8_DIR_C="${SG_PKG_RING:-$X/pkg-ring}" R8_KV8_DIR_D="${SG_PKG_D:-$X/pkg-kv8pf}" \
```
⇒ 通过臂（`sg-a-c-graph`）是 **`SG_PKG_D=$S/pkgs/pkg-kv8pf` + `R8_GRAPH_SAFE=1`** 起的；
它的 `serve_a2.log` 原文也印证了这一点：
```
[serve_a2] [R8-INT8] 已挂载 int8 影子文件：tier=C src=…/X_integrate/pkg-ring/shadow/vllm_ascend dsa=D=带 role 分键 core=draft-aware…
```
（`dsa=D=带 role 分键` ⇒ 那次确实是走 D 的那份。）

★ 而 `run_arm_r8.sh` **没有** `SG_PKG_*` 这两个名字，但它照样能覆盖 ——
直接给 `DSA_SRC=D` 与 `R8_KV8_DIR_D=<S 的 pkg-kv8pf>`（两处都在 `run_arm_r8.sh` 里透传）。

---

## 3. 四轴臂的**完整**起服配方（把三次踩到的前置都写进来）

```bash
R8=$HOME/projects/dsv41-upstream-pr/agents/R_8card_int8
S=$HOME/projects/dsv41-upstream-pr/agents/S_graphfix
X=$HOME/projects/dsv41-upstream-pr/agents/X_integrate

# ① 合并件新鲜度门（logs/081；派生件过期 = 功能静默消失）
bash a2/scripts/check_merged_fresh.sh \
  --img ~/tmp/20260922/merge4axis/model.img.py \
  --kv8 $X/pkg-kv8pf/shadow/vllm_ascend/models/deepseek_v41/model.py \
  --prod $HOME/projects/dsv41-upstream-pr/shadow-pkg/patches/files/model.py \
  --installed $R8/patched/model_merged.py --merger $R8/patch/merge_model.py

# ② 先看 dmesg（logs/080：OOM 会伪装成"代码挂了"）
sudo -n dmesg -T | grep -iE "killed process" | tail -3

# ③ 起臂
TAG=r8-4axis TIER=C GRAPH=1 EAGER=0 \
ENGRAM=1 DRAFT_GRAPH=1 \
OFFLOAD_BYTES=23068672000 \
DSA_SRC=D R8_KV8_DIR_D=$S/pkgs/pkg-kv8pf \
PROMPTS=16 PROMPT_TOKENS=131072 REPLAY_PROMPT_TOKENS=65536 MAX_TOKENS=128 \
bash $R8/scripts/run_arm_r8.sh
```

★ 起服期**必须**看到（否则立刻停，别等压测）：
1. `[serve_a2] [R8-INT8] … dsa=D=带 role 分键`（不是 `dsa=C=原样`）
2. `KV8_GRAPH_SAFE=1`
3. `[serve_a2] [R8-INT8] model.py 用**合并版**`（且 md5 = `5c990b04…`）
4. 捕获期 `EE1016` = 0

---

## 4. 这一格对目标的含义

| 目标轴 | 现状 |
|---|---|
| `ENGRAM=1` | ✅ 已有（`logs/077` 的修复） |
| 卸载 | ✅ 已有（`logs/077` + `logs/078` 逐字节保真） |
| int8 档 C | ✅ **单轴**验过（`logs/048`），但需要**本节这两个 env** 才能在图模式下成立 |
| `DRAFT_GRAPH=1` | ✅ 单轴验过（`logs/069`） |
| **四轴同开** | ⏳ 本条把它最后一个**起服前置**补齐（前两个：`logs/081` 的合并件、`logs/080` 的 OOM 门） |
