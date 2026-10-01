# V4.1 DCP 运维坑位（2026-10-01）

> 记录三种会**静默毁掉现场**的情况。每条都写了"症状 → 根因 → 判据 → 处置"。
> 判据一律落在**可观测的最终事实**上，不看"命令是否成功"。

---

## 坑 1：`git reset --hard` 冲掉 DCP 挂载机制（本轮踩到，代价一轮 10 分钟起服）

### 症状

```
NotImplementedError: V4.1 initial runtime requires PP=DCP=PCP=1
  at /vllm-workspace/vllm-ascend/vllm_ascend/core/deepseek_v41.py:348
```

起服"成功提交"、容器启动、然后秒崩。**看起来像 overlay 没生效**，
实际是 **`serve_a2.sh` 里的 DCP 挂载块整体消失** ⇒ 容器里根本没有 overlay。

### 根因（已用 reflog 钉死，不是推断）

DCP 挂载机制（`V41_DCP_MOUNT` 整文件覆盖 + `DCP_EXTRA_ENV` 透传 +
`DCP-MOUNT-GUARD` 逐个 md5 校验 + `*.so` 白名单，共 ~100 行）**从未提交进 git**。

```
$ git reflog show --date=iso
9a2fa2c HEAD@{2026-10-01 11:17:39}: reset: moving to origin/main   ← 凶手
b337e35 HEAD@{2026-10-01 11:16:58}: commit: fix(dspark-dcp): ...
```

`reset --hard` 把**已跟踪但未提交**的 `scripts/serve_a2.sh` 还原成了 HEAD 版本，
DCP 挂载块随之蒸发。文件 mtime `11:17:39` 与 reflog 时间戳**逐秒吻合**。

> ⚠️ 我一度把原因写成 `tools/selfcheck_pkg.sh`。**那是错的**，已更正。
> selfcheck 虽然会实际执行 `bash scripts/serve_a2.sh`（gate 用例，
> `selfcheck_pkg.sh:415-421`），但都带 `V41_GATE_GUARD_CHECK_ONLY=1`，
> 而该 guard 在 serve_a2.sh 第 280 行、**早于任何写文件动作**；全仓也搜不到
> `git checkout/restore/stash/reset`。归因必须靠 reflog，不能靠"谁看起来像"。

### ★ 教训

**在 a3-21 上跑任何 `git reset --hard` / `git checkout -- <file>` 之前，
先 `git status` 看有没有未提交的现场改动。** 这台机器上的实验现场
（`~/cedpd-repo` 的 `scripts/`）长期带着未提交改动。

### 判据（起服后查，**不要**看"起服成功"）

```bash
# ① 容器里到底挂了几个文件（应 >= 15）
docker inspect <name> --format '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' \
  | grep -c vllm_ascend

# ② 关键文件是不是我们的版本（md5 要比对 overlay 源，逐字节）
docker exec <name> md5sum \
  /vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py

# ③ overlay 生效的铁证（这一行由我们的 patch_v41_dcp.py 打印）
grep -a "PP/DCP/PCP guard bypassed" <run>/serve.log
```

> ⚠️ **不要**用 `grep [DCP-MOUNT-GUARD] <run>/serve.log` 判断 —— 那是宿主侧输出，
> 会被 `serve_a2.sh` 的 `: > "$LOG"` 截断（见坑 3）。grep 不到 **不代表** 没生效。

### 处置：永久存档（已做）

三件套，都落在 `~/dcp_durable/`：

| 文件 | 内容 |
|---|---|
| `serve_a2.sh.dcp-mount-20261001` | 完整文件（HEAD + DCP 挂载 + `.so` 白名单，**含 KV32 守卫**） |
| `serve_a2.sh.dcp-mount-20261001.sha256` | 校验和（`431c2e54e186f34b…`） |
| `dcp-mount.patch` | 相对 `origin/main` 的增量补丁（已用 `git apply --check` + 逐字节比对验证） |

一键恢复（幂等，已校验 sha256 后才覆盖）：

```bash
bash ~/restore_serve_a2_dcp.sh
```

**恢复时不要用 `serve_a2.sh.bak_dcpson`** —— 它是 09:44 的旧备份，
**不含** `f95acf3` 引入的 KV32 池守卫（会把 4 GiB 寻址回绕的防护一起丢掉）。
本轮踩过一次：用旧备份恢复后 `diff` 发现少了 100 行 KV32 守卫。

### 真正的修法（未做，需决策）

把 DCP 挂载机制提交进 git。它是**默认关闭**的（不设 `V41_DCP_MOUNT` 完全不生效），
进 main 是安全的；但 main 是交付分支，要单独决策。

---

## 坑 2：宿主侧诊断不能写进 `serve.log`

`serve_a2.sh` 里有 `: > "$LOG"`，会把宿主侧的早期输出整段截掉。所以：

* 宿主侧 → `$OUT/harness.log`（`dcp_stage_capacity.sh` 已经这么做）
* 引擎原始日志 → `$OUT/serve.log`
* 两个日志分开，不要混

**直接后果**：`[DCP-MOUNT-GUARD]` 不在 `serve.log` 里 —— 见坑 1 判据的警告。

---

## 坑 3：`msprof --export=on` 会清掉同目录已有的 `mindstudio_profiler_output/*.csv`

线 A 提交的踩坑记录。抓新 profile 必须**换新 run_id**，
否则会把上一份已导出的 CSV 覆盖掉（原始 16G 数据不受影响）。

---

## 附：本次 8 卡启动的完整时间线（供追溯）

| 时刻 | 事件 |
|---|---|
| 11:16:58 | a3-21 提交 `b337e35`（DSpark×DCP 修复） |
| **11:17:39** | **`git reset --hard origin/main` ⇒ DCP 挂载块被冲掉** |
| 11:18:26 | 起服尝试 #1 → 容器起来但秒崩（`NotImplementedError`） |
| 11:28 | 从 `bak_dcpson` 恢复（**但丢了 KV32 守卫**） |
| 11:31:47 | 起服尝试 #2 → 容器 `dsv41-dspark8`，挂载 16 个 ✅，md5 逐字节匹配 ✅ |
| 11:45 | 发现恢复版本缺 KV32 守卫 ⇒ 用 `merge_serve_a2_dcp.py` 正确重建 |
| 11:50 | 补回 `*.so` 白名单 ⇒ 落 `~/dcp_durable/` 存档 + `dcp-mount.patch` |
