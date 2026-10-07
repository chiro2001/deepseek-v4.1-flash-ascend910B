# 第2步实测：DBO 双流在 conc=2 已逼近盈亏平衡（落后仅 7%）（2026-10-07）


> ⚠️ **本文的核心结论（conc=2「只落后 7%」）已被撤回**：DBO 在 conc=2 时**根本没触发**
> （16 token < 阈值 32）⇒ 那 −7% 不是 DBO 的效果。强制触发后 **−42.5%**（与历史 −42% 一致）。
> 详见 `S14-CORRECTION-DBO-NOT-TRIGGERED-20261007.md`。**§3.3「第 2 步已实现」的代码观察仍然成立。**

> 承接 `S12-BM-CURVE-AND-CBP-CRITERIA`。本文用 **DCP=1 基线** + **DBO 双流臂**做直接对照。
> 环境：tiny（`dsv41-tinyspark`），dummy 权重，SPEC=1 SP_TOKENS=7，graph 模式。全部【实测】。

---

## 0. 一页纸

| 项 | 结果 |
|---|---|
| 第 2 步（每 ubatch 独立 `compute_stream`）是否已实现？ | ✅ **overlay 里已经实现了**（`_run_ubatches_graph` 把 ubatch1 放 side 流、ubatch0 放 root 流） |
| DBO 双流 @ conc=2 | **62.1 tok/s vs 基线 67.0** ⇒ **0.93×（落后 7%）** |
| 需要的门槛 S(2) | **1.137** |
| 实测争用税 tax(2) | **1.227** |
| **差距** | **8%** —— 这是所有尝试里最接近盈亏平衡的一次 |
| 对比历史 | DBO 此前在 conc=4 报告 **−42%**；本轮 conc=2 只落后 **7%** |

---

## 1. DCP=1 基线（与 DBO 同并行度，可比）

| conc | 聚合吞吐 R_N | 单流 tok/s | 单流效率 | **S(N)=N×R₁/R_N** |
|---:|---:|---:|---:|---:|
| 1 | **38.1** | 37.7 | 100% | — |
| 2 | **67.0** | 34.4 | 91.2% | **1.137** |
| 4 | **107.5** | 28.6 | 76.0% | **1.418** |

（DCP=1 比 DCP=2 快很多：R₁ 38.1 vs 25.8、R₄ 107.5 vs 71.6。所以后续对照都用 DCP=1。）

---

## 2. DBO 双流臂

| conc | 聚合吞吐 | 单流 tok/s | **vs 同 conc 基线** | 需要的 tax | **实测 tax** |
|---:|---:|---:|---:|---:|---:|
| **2** | **62.1**（61.9/62.7/62.1） | 33.0 | **0.927×** | **<1.137** | **1.227** |
| 4 | 64.4（63.7/64.4/64.7） | 17.4 | **0.599×** | <1.418 | **2.084** |

**tax 推算**：`tax = N × R₁ / 实测吞吐`
* conc=2：`2 × 38.1 / 62.1 = 1.227`
* conc=4：`4 × 38.1 / 64.4 = 2.366`（若按"两个 2-请求批次的 solo 速率"算则是 2.084）

---

## 3. 三个结论

### 3.1 conc=2 只差 8%

```
需要 tax < 1.137，实测 1.227  ⇒  差 7.9%
```

**这是整条线最接近的一次。** 作为对比：
* DBO 历史报告：conc=4 **−42%**、conc=8 **−40%**
* 本轮 conc=2：**−7.3%**

### 3.2 conc=4 反而大幅恶化（0.599×），原因待查

按 conc=2 的 tax（1.227）外推，conc=4 应该拿到 `2 × 67.0 / 1.227 = 109` tok/s，
实测只有 **64.4** —— 说明 conc=4 时出现了**额外的惩罚**（不是单纯的争用税）。

**候选原因**（未验证）：
1. `enable_dbo` 固定 `num_ubatches=2`，conc=4 时每个 ubatch = **2 个请求（16 token）**，
   而 conc=2 时 = 1 个请求（8 token）——**半批变大后，ubatch 之间的独立性/收益窗口变了**；
2. 日志显示 `[DBO-NODM] enable_device_metadata: use_ubatching=True -> enabled=False`
   —— **device-metadata 被关掉了**（DBO 线已知的负结果路径），可能引入串行化；
3. conc=4 时 4 个请求 → 2 个 ubatch 的切分可能触发了额外同步。

### 3.3 第 2 步的"实现"其实早就有了

```python
# npu_ubatch_wrapper.py :: _run_ubatches_graph
e_fork.record(root_stream)
with torch.npu.stream(side):
    side.wait_event(e_fork)
    outputs[1] = _submit(ubatch_metadata[1], model, side)     # ← side 流
    e_join.record(side)
outputs[0] = _submit(ubatch_metadata[0], model, root_stream)  # ← root 流
root_stream.wait_event(e_join)
```

**⇒ 两个 ubatch 确实在两条不同的流上。** 之前 `ALIGNMENT` 里说的
"只有一条 `compute_stream`"指的是**上游 PR 的参考实现**，不是我们的 overlay。

---

## 4. 下一步（按性价比）

| # | 动作 | 目标 | 成本 |
|---:|---|---|---|
| 1 | **查清 conc=4 恶化 0.599× 的原因**（看 ubatch 切分与 device-metadata 关闭的影响） | 若是可修的串行化 ⇒ 直接收益 | 0.5 天 |
| 2 | **相位配平**：让 ubatch1 相对 ubatch0 偏移半个"层内相位" | tax 从 1.227 → <1.137 | 1~2 天 |
| 3 | 若 1+2 后仍不过门槛 ⇒ **CBP 关闭归档** | — | — |

**判据不变**：tax < S(N) 才继续。conc=2 现在是 1.227 vs 需要 1.137。

---

## 5. 复现

```bash
# DCP=1 基线（无 DBO）
ssh a3-21 'bash /tmp/launch_armMcp1.sh'
# DBO 双流臂
ssh a3-21 'bash ~/tmp/launch_tiny_dbo.sh'
# 测量
ssh a3-21 'cd ~/cedpd-repo && python3 ~/tmp/bench_conc.py --base-url http://127.0.0.1:19310 \
  --model deepseek-v41 --concurrency 2,4 --prompt-tokens 1024 --output-tokens 96 --repeats 3'
```
