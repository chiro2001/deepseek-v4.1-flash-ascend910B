# ★★ 突破：NPU **一张图可以捕获双流并行分支** —— ubatching 的图模式前提成立
+
+> 起因：线 B（ubatching）被上游 PR #11273 的 "eager only, no ACL Graph" 卡住。
+> 而上游 **GPU 版**把两个 ubatch 线程的 join 整个包进一次 `torch.cuda.graph`。
+> 本文是 NPU 侧等价能力的**实测验证**。全部为【实测】。
+> 工具：`tools/tiny_graph_ms.py`（最终可用版）。
+
+## 0. 结论
+
+| 项 | 结果 |
+|---|---|
+| **一张 NPUGraph 能否捕获"两条独立流的并行分支"** | **✅ 可以** |
+| 关键规则 | **fork event 必须在「捕获根流」上 `record`**（不是默认流） |
+| 图重放 | **0.989 ms** |
+| 同工作量 eager | 1.611 ms（图 **1.63×** —— 消除了 host 开销） |
+| **同工作量串行单流** | 1.293 ms ⇒ **多流图 = 1.31×** |
+
+⇒ **ubatching 的图模式前提成立**；PR #11273 的 eager-only 限制**不是平台限制，
+而是那个 PR 的实现范围限制**，可以自己补。
+
+---
+
+## 1. 失败 → 成功的两步
+
+### 1.1 第一次尝试（失败）
+
+```python
main = torch.npu.current_stream()      # 默认流
s1, s2 = torch.npu.Stream(), torch.npu.Stream()
e_fork = torch.npu.Event()
def body():
    e_fork.record(main)                # ← 在**默认流**上 record
    with torch.npu.stream(s1):
        s1.wait_event(e_fork)
        ... working ...
    with torch.npu.stream(s2):
        s2.wait_event(e_fork)
        ...
g = torch.npu.NPUGraph()
with torch.npu.graph(g, stream=s1):    # 捕获根流 = s1
    body()
```
+
+报错（决定性）：
+
+```
+rtStreamWaitEvent execution failed, reason=**in the model capture scenario,
+the event wait task has no corresponding event record task**
+```
+
+**原因**：捕获只发生在 `s1` 这条子图上；`e_fork.record(main)` 在默认流上，
+**没有被捕获** ⇒ 捕获区里的 `s1.wait_event(e_fork)` 找不到对应的 record。
+
+### 1.2 第二次尝试（成功）
+
+```python
+root = torch.npu.Stream()              # 捕获根流（非默认）
+s2 = torch.npu.Stream()
+e_fork, e_join = torch.npu.Event(), torch.npu.Event()
+
+def body():
+    e_fork.record(root)                # ★ 在**捕获根流**上 record
+    with torch.npu.stream(s2):
+        s2.wait_event(e_fork)          # 侧流 fork
+        ... 侧分支工作 ...
+        e_join.record(s2)
+    ... 根流上跑另一条分支 ...
+    root.wait_event(e_join)            # 汇合
+
+with torch.npu.graph(g, stream=root):  # 捕获根流 = root
+    body()
+```
+
+⇒ **捕获成功**，重放正确。
+
+---
+
+## 2. 这条规则为什么重要（对 ubatching 的实现含义）
+
+上游 `ubatch_context` 的设计是**每条 ubatch 一条 compute_stream**，
+两条线程通过 CPU 事件（`cpu_wait_event` / `cpu_signal_event`）保证
+**同一时刻只有一条线程在提交**（`_cpu_yield` 的注释：
+*"It is critical for correctness that only one thread is running at a time"*）。
+
+把这件事映射到 NPU 图捕获，需要：
+
+1. **选一条流作为捕获根**（= ubatch 0 的 compute_stream）；
+2. 在根流上 `record` 一个 fork event；
+3. ubatch 1 的流 `wait_event(fork)`；
+4. ubatch 1 结束时在自己的流上 `record` join event；
+5. 根流 `wait_event(join)`。
+
+**⇒ 上游 GPU 版把"两个线程的 join"包进 `torch.cuda.graph` 之所以能工作，
+是因为 CUDA 的捕获对侧流有隐式 fork/join；NPU 需要显式 event，
+而只要 record 落在捕获根流上就能工作。**
+
+---
+
+## 3. 对线 B 的影响
+
+| 项 | 之前 | 现在 |
+|---|---|---|
+| 图模式多流 | 【未确认】 | **✅ 已验证可行** |
+| PR #11273 的 eager 限制 | 阻塞 | **不再是阻塞**（自己补图支持即可） |
+| B2 的工作量 | 不明 | **明确**：把 `NPUUBatchWrapper._run_ubatches` 的线程 join 改成"根流 record fork → 侧流 wait → 侧流 record join → 根流 wait"，并放进 `torch.npu.graph(g, stream=root)` |
+
+**收益参考**：本次微基准（40 层 × 2 条链）**1.31× vs 串行**；
+与之前 `pingpong_v4` 的 1.37×（多张图并发重放）量级一致。
+
+---
+
+## 4. 复现
+
+```bash
+ssh a3-21 'docker cp ~/tmp/graph_ms2.py dsv41-tinyspark:/tmp/ && \
+  docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 graph_ms2.py"'
+# 期望：
+#   ✓ eager OK / ✓ 多流捕获成功
+#   图重放 ~0.99 ms | eager ~1.61 ms | 串行 ~1.29 ms
+#   ⇒ 多流图 1.31× vs 串行
+```
+
+工具已入仓：`tools/tiny_graph_ms.py`（成功版）、`tools/tiny_graph_multistream.py`（失败版，保留作反例）。
