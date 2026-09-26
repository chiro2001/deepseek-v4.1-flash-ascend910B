# 074 — ★★★ 交付脚本的 P0 静默降级：**档 B 起服时「完全没有 DRAM 卸载」**（已修 + 已加回归门）

> 2026-09-22 20:4x–20:5x CST。执行：**主代理**（本机；`DRY=1` 对照 + `bash -x` 逐行跟踪）。
> 标记：【实测】/【推断】。红线遵守：只读发布仓 + 改 `a2/scripts/` 一处；未碰 `upstream-v41/`；未占卡。

---

## 0. 一句话

`a2/scripts/serve_a2_offload.sh` 里那个 `if [ 开了 int8 ] ... fi` 的 **`fi` 落错了位置**，
把「shadow 存在性检查 + 拷 4 个补丁 + `export OFFLOAD_*_PATCH=1` + `export P2_*`」**整段吞进了 int8 分支**
⇒ **不开 int8 时（= 档 B，也就是 A2 最稳的那条路）**：

1. 4 个卸载补丁**一个都不挂** ⇒ 服务照常 READY，但**完全没有 DRAM 卸载**（零报错）；
2. `P2_POOL_PATCH` / `P2_COMP_JSON` 不导出 ⇒ **L1 失效** ⇒ 宿主内存从 **≈3.49 × 池** 变 **≈7 × 池**
   （85 GiB 池：≈297 GiB → **≈590 GiB**，A2 根本装不下）。

⇒ 这正是本项目反复在防的失败模式（`065` §3 / `067` §7 记的那一类"静默 no-op"），
而且**是我自己这一晚引进的**：这条路线在 06:3x 的 `cecc056` 补 dry-run 挂载断言时就已经埋下。

---

## 1. 【实测】证据（唯一变量 = 有没有 `KV8_*`）

| 配置 | `[a2-dry] MOUNTS(...)` | 4 个卸载补丁 |
|---|---|---|
| 档 C（`KV8_SWA=1 KV8_RING_FP16=1`） | `MOUNTS(24)` | **全在** |
| 档 B（不带 `KV8_*`） | `MOUNTS(10)` | ★ **一个都没有** |

★ **`bash -x` 逐行跟踪（档 B）**：`mkdir -p "$PATCHDIR"` / `cp ...` / `export OFFLOAD_SCHED_PATCH=1` /
`export A2_KV8_SWA=...` **一条都没执行**（146 行 trace 里 0 命中）⇒ 不只是"没挂上"，是**整段没跑**。

★ **追溯**：该结构至少从 `32ddbee` 起就存在（A2 那份 clone 的 `8a90062` 同样）。
**此前所有 8 卡臂都没暴露**，因为 runner 是**显式**传 `OFFLOAD_SCHED_PATCH=1 OFFLOAD_NPU_WORKER_PATCH=1`
的（`027` §6 / `042` §6 的臂参数表可以逐字看到）⇒ 只有"A2 用交付脚本起服"这一路会踩。

---

## 2. 修法（已落 `a2/scripts/serve_a2_offload.sh`，三处）

1. **移出分支**：shadow 存在性检查 + 拷 4 个补丁 + `export OFFLOAD_*_PATCH/NPU_OFFLOAD_HOST_MEM/
   PREFIX_MATCH_UNIT/ENGRAM/P2_*/PGP_MGR_*` 移到 int8 判断**之外**（`A2_*` 那套名字仍在里面，它们只对 int8 有意义）；
2. **起服前 fail-closed**：断言 `OFFLOAD_SCHED_PATCH` 与 `OFFLOAD_NPU_WORKER_PATCH` 必须都是 `1`，否则 `exit 2`；
3. **dry-run 回归门**：把子进程输出收进变量后，**断言真实 MOUNTS 里必须出现那 4 个补丁文件名**，缺任一 ⇒ `exit 2`。

---

## 3. 【实测】修后复跑（三条路 dry-run 全过；本机假模型 + 真 shadow）

| 路 | 配置 | 档位自报 | MOUNTS | 新断言行 |
|---|---|---|---|---|
| 路 1 | 档 B + `ENGRAM=0` + `DRAFT_GRAPH=1` | **B** | `MOUNTS(10)`（含 4 个卸载补丁） | ✓ |
| 路 2 | 档 C（`KV8_SWA=1 KV8_RING_FP16=1`）+ `ENGRAM=0` | **C** | `MOUNTS(24)` | ✓ |
| 路 3 | 档 B + `ENGRAM=1` | **B** | `MOUNTS(10)` | ✓ |

★ **反向对照**：把 `A2_OFFLOAD_FILES` 指到一个空目录 ⇒ shadow 立刻 `die`（`rc=2`）
⇒ 这道门**有判别力**（不是"看不见所以全绿"）。

---

## 4. 影响面

| 对象 | 影响 |
|---|---|
| **A2 上线（档 B 路线）** | 若带这个 bug：服务 READY、**零卸载**；且 85 GiB 池要 ≈590 GiB 宿主 ⇒ 大概率起不来/被 OOM killer 干掉 ⇒ **必须 pull 后再起服** |
| A3 全部结论 | **不受影响**（runner 显式传了那两个 env） |
| 文档 | `docs/A2-DEPLOY-NOW.md` §1 的命令里本来就显式写了那两个 env ⇒ 照它抄是对的；**本次修好之后单跑 `serve_a2_offload.sh` 也对** |
