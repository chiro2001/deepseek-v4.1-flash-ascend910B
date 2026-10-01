# V4.1 DCP 运维坑位（2026-10-01）

> 本文记录**在跑 DCP 实验时会静默毁掉你现场**的两件事。两件都是本轮踩到、
> 都花了 10 分钟以上的起服代价才发现。

---

## 坑 1：`tools/selfcheck_pkg.sh` 会抹掉**未提交**的工作区改动

### 症状

```
NotImplementedError: V4.1 initial runtime requires PP=DCP=PCP=1
  at /vllm-workspace/vllm-ascend/vllm_ascend/core/deepseek_v41.py:348
```

起服"成功提交"、容器启动、然后秒崩。**看起来像 overlay 没生效**，
但真正的原因是：**`serve_a2.sh` 里的 DCP 挂载块没了** ⇒ 容器里根本没有 overlay。

### 根因

`serve_a2.sh` 的 DCP 挂载机制（`V41_DCP_MOUNT` 整文件覆盖 + `DCP_EXTRA_ENV`
透传，约 106 行）**从未提交进 git**，一直只存在于 a3-21 的工作区。

而 `tools/selfcheck_pkg.sh` 会**实际执行** `bash scripts/serve_a2.sh`
（gate 用例，见 `selfcheck_pkg.sh:415-421`，用 `GATE_MAX_TOKENS=2048/8192` 等
共 4 次），其中某一步把工作区文件还原成了 HEAD 版本。

复现证据（时间戳）：
```
serve_a2.sh        mtime 11:17:39   ← selfcheck 期间被改
起服               11:18:26
容器报错           11:20:45
```
而 `serve_a2.sh` 的行数从 1842（无 DCP 块）变回 1827（有 DCP 块）即可判别。

### 判据（不要看"起服成功"，要看挂载）

```bash
# ① 容器里到底挂了几个文件（应 >= 15）
docker inspect <name> --format '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' \
  | grep -c vllm_ascend

# ② 关键文件是不是我们的版本（md5 要比对 overlay 源）
docker exec <name> md5sum \
  /vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py

# ③ 起服日志里应出现（overlay 生效的铁证）
grep -a "PP/DCP/PCP guard bypassed" <run>/serve.log
```

### 处置

`~/restore_serve_a2_dcp.sh`（已落盘）—— 幂等，已含 DCP 挂载时不动作：

```bash
bash ~/restore_serve_a2_dcp.sh
```

**纪律**：跑 `selfcheck_pkg.sh` **之前**先把工作区改动 commit 或
`cp serve_a2.sh serve_a2.sh.bak_$(date +%H%M)`。

### 真正的修法（未做）

把 DCP 挂载机制提交到 git。它是**默认关闭**的（不设 `V41_DCP_MOUNT` 完全不生效），
所以进 main 是安全的；但 main 是交付分支，需要单独决策。

---

## 坑 2：`msprof --export=on` 会清掉同目录已有的 `mindstudio_profiler_output/*.csv`

线 A 提交的踩坑记录。抓新 profile 必须**换新 run_id**，
否则会把上一份已导出的 CSV 覆盖掉（原始数据不受影响）。

---

## 坑 3（历史，重复踩过）：宿主侧诊断不能写进 `serve.log`

`serve_a2.sh` 里有 `: > "$LOG"`，会把宿主侧的早期输出整段截掉。
所以：
* 宿主侧 → `$OUT/harness.log`（`dcp_stage_capacity.sh` 已经这么做）
* 引擎原始日志 → `$OUT/serve.log`
* 两个日志分开，不要混。

**直接后果**：`[DCP-MOUNT-GUARD]` 那行**不在** `serve.log` 里，
所以"grep 不到守门行"**不能**推断"挂载没生效" —— 必须用坑 1 的三个判据。
