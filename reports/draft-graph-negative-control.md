# DSpark draft 入图：负控确认（缺 `DSPARK_GRAPH_CAPTURE_METADATA=1`）

> 2026-09-16 18:55 CST｜A3-node1 chips 8-15｜真权重｜`DRAFT_GRAPH=1`，`SP_TOKENS=5`，全补丁

---

## 结果

**臂 `rgnif`（`DRAFT_GRAPH=1`，但启动器当时**没有**注入 `DSPARK_GRAPH_CAPTURE_METADATA=1`）**

| # | ms/step | A | tok/s |
|---|---|---|---|
| r1–r2 | — | **1.0** | — |
| r3 | — | 1.0 | — |
| **r4** | **30.527** | **1.008** | 33.1 |
| r5 | 30.764 | 1.000 | 32.6 |
| r6 | 30.421 | 1.000 | 33.0 |
| r7 | 30.215 | 1.000 | 33.2 |
| r8 | 30.520 | 1.000 | 32.9 |

* 起服 **READY=401 s**，`Wrapping draft model with ACLGraphWrapper` = **8**（8 rank 各一次），`Target sizes` = **0**（修复后不再崩）
* `dspark-graph-capture` 打印次数 = **0** ⇒ 捕获期确实没建 draft attention metadata
* `static_kernel.py:650` 降级 = 0；KV cache = 4,142,652 tokens
* 文本探针（2 发，128K）：见 `/tmp/a21_seq1.log` 的 `rg_text`

## 判读

1. **`num_query_tokens` 崩溃已修复**（旧版在第一个请求就 `RuntimeError: Target sizes: [6,2] vs [5,2]`，本臂 8 发全跑完）。
2. **但 A 恒为 1.0** ⇒ draft 的 token **全部被拒绝**，与历史上"draft 图模式 ms 达标但 tok/round 崩到 1.0"完全一致。
3. 这就是**缺 `DSPARK_GRAPH_CAPTURE_METADATA=1` 时的负控指纹**：图捕获走了
   `AscendDSAImpl.forward` 的 `attn_metadata is None` 兜底分支（`output.fill_(0)`），
   replay 里没有 draft attention ⇒ draft 输出全错。
4. **ms/step 仍然是 30.2–30.8**（与 eager 同量级）⇒ 单看时延**无法**发现这个错误，
   必须同时看 `A` / `pos` 形态 / 文本。

## 已落地的启动器修复

`scripts/serve_a21.sh` 的 `DRAFT_GRAPH=1` 分支现在无条件注入：
```bash
MOUNTS="$MOUNTS -e DSPARK_DRAFT_METADATA_MODE=sync -e DSPARK_GRAPH_CAPTURE_METADATA=1"
```
md5（改后）：`9d5dd8bf8dadf570f36e6b2391d94058`

## 待办

- **正控**：`exp_tools/a21_seq2.sh` 用修复后的启动器重跑（判据：`graphcap>0` 且 `A>1.5`）
- 真权重判据④（A 不劣于 eager）需要与 `faB`（eager，A 中位 2.748）对照
