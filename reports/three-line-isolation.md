# 三条线的文件隔离（git worktree + 跨机 handoff）

> 2026-09-16 11:45 CST｜起因：线1 与线3 原本共用 `A3-node2:$W/wt-graph` 同一个 worktree ⇒ 会互相覆盖

---

## 0. 隔离后的布局

| 线 | 负责人 | 机器 / 卡 | **工作目录** | git 分支 |
|---|---|---|---|---|
| **线 1 正确性** | 子 Agent `line_correctness` | A3-node2 chips 8-15 | `A3-node2:~/projects/dsv41-workspaces/**wt-graph**` | `draft-graph` |
| **线 2 dummy 性能** | **主 Agent（我）** | A3-node1 chips 8-15 | `A3-node1:~/projects/dsv41` | `main` |
| **线 3 单算子/单层** | 子 Agent `line_single_op` | A3-node2 chip7 | `A3-node2:~/projects/dsv41-workspaces/**wt-op**` ← **新建** | `single-op` |

**关键**：线1 与线3 现在在**两个独立的 worktree**，共享同一个 `.git` 但工作树完全分离
（`git worktree` 保证同一分支不会在两个 worktree 同时 checkout）。
线2 在**另一台机器**的独立仓库上，天然隔离。

`git worktree list`（A3-node2 实测）：

```
/home/user/projects/dsv41                        fa96208 [main]
/home/user/projects/dsv41-workspaces/wt-graph    18a21a4 [draft-graph]   ← 线1
/home/user/projects/dsv41-workspaces/wt-op       fa96208 [single-op]     ← 线3（新）
... (wt-hotspike / wt-mtpq / wt-stage1 / wt-vision 为历史保留)
```

---

## 1. 文件边界（谁碰什么）

| 文件/目录 | 线1 | 线2 | 线3 | 说明 |
|---|---|---|---|---|
| `probe_dsa/dsa_v1.py` | 只读 | **写** | 只读（取基线） | F3 的 patch 与 mount 在 A3-node1 |
| `probe_idx/indexer.py` | **写**（非确定性诊断） | 只读 | 只读 | A3-node1 有 CAND_MODE 版本可参考 |
| `probe_bneck/model.py.probe` | 只读 | **写** | 只读 | IDS64 / ENGRAM-DUMMY |
| `scripts/serve_a21.sh`（A3-node1） | — | **写** | — | 线2 独占 |
| `scripts/serve_a22_v2.sh`（A3-node2, wt-graph） | **写** | — | 只读（拷走自用） | 线1 独占 |
| `layer_bench/`（wt-op） | — | — | **写** | 线3 独占（已复制到 wt-op） |
| `probe_f24/` | — | 只读（取 patch） | 只读 | 已复制到 wt-op |
| `reports/*.md` | `correctness-line.md` | `three-line-plan.md` 等 | `single-op-line.md` | **文件名不重叠即可** |

**规则**：任何线要改不属于自己的文件，**先通过 handoff 通知**。

---

## 2. 跨机 handoff（线3 → 线2 的 patch 通道）

线3 在 A3-node2，线2 在 A3-node1，**不能直接共享工作树**。用共享目录：

```
A3-node2:~/handoff/{patches,reports,logs}/     ← 线1/线3 写
A3-node1:~/handoff/{patches,reports,logs}/     ← 线2 读（我自己 scp 同步）
```

**交付协议**（线3 每完成一个改动就做）：

1. 把 patch 脚本 + 说明放到 `A3-node2:~/handoff/patches/<改动名>/`
   - `<改动名>.py`（幂等 patch 脚本，锚点唯一自检 + `py_compile` 通过）
   - `README.md`（改哪一行、减少多少算子、单卡 µs 改前→改后、数值等价证据、是否需要重捕获）
2. 把单卡原始证据放 `A3-node2:~/handoff/logs/<改动名>/`
3. **发消息给主 Agent**（不要自己 scp 到 A3-node1，我来同步，避免并发写）

---

## 3. 为什么不用"每线一个独立 clone"

`git worktree` 优于独立 clone 的地方：
* 共享同一份 `.git` 对象库 ⇒ **不占额外磁盘**（这个仓库带 600GB 级的模型/产物不在 git 里，但工作树本身也有几十 MB）
* 分支互锁：`git worktree` 禁止同一分支被两个 worktree 同时 checkout ⇒ **从机制上防止两个 Agent 在"同一分支"上互相覆盖**
* 合并容易：线3 的 `single-op` 分支可以直接 `git merge` 到主线，不用跨库 `format-patch`

---

## 4. 仍未隔离的共享资源（有意保留，风险已评估）

| 资源 | 冲突风险 | 处理 |
|---|---|---|
| **模型目录**（`/home/user/models/out/...`） | 只读，无冲突 | 保持共享 |
| **静态内核缓存**（`p36_static_kernel_a21` / `p36_static_kernel`） | 各线独立路径 | 保持 |
| **vllm/npugraph 缓存**（`.cache/vllm-a21perf` 等） | 各线独立路径 | 保持 |
| **`/tmp` 下的 switch 文件**（`/tmp/v41_*`） | ⚠️ 容器内独立，不共享 | 无需处理 |
| **NPU 卡** | 已按线分配（见上表） | 保持 |
| **A3-node1 的容器名** `dsv41-a21-perf` | 线1 只读 `docker exec`，线2 独占起停 | 线1 遇到 `docker exec` 失败即重试 |
