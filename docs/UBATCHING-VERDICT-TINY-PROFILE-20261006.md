# ★★ 决定性数据：并发越高、串行度越高 ⇒ **ubatching 的收益随并发上升**

> 起因：执行线 B 前，先花一次 profiling 回答"conc=8 时 AIC 是否已被填满"。
> 答案：**没有，而且串行度更高**。这直接确认了用户"大吞吐场景收益更大"的判断。
> 平台：tiny（a3-21 chips 2/3，TP2+DCP2，BAT=2048，重排几何）。
> 全部为【实测】。工具：`tools/prof_ab_overlap.py`。
+
+## 0. 一句话
+
+| | conc=1 | **conc=8** |
+|---|---|---|
+| 步长 | 34.95 ms | 23.88 ms |
+| **AIC busy** | 13.77（**39%**） | 8.43（**35%**） |
+| **AIV busy** | 10.55（30%） | 8.45（35%） |
+| COMM | 2.40（7%） | **3.16（13%）** |
+| CPU | 1.02（3%） | **2.09（9%）** |
+| **四者之和 / 步长** | **79%** | **93%** ← **几乎完全串行** |
+| AIC∩AIV | 1.124 | 0.912 |
+| **AIC∩COMM** | **0.000** | **0.000** |
+| **完美重叠理论上限** | 2.54× | **2.83×** |
+
+**结论三条**：
+1. **并发提高到 8，AIC 利用率反而从 39% 掉到 35%** ⇒ 批处理**不会**填满 AIC；
+2. **串行度从 79% 升到 93%** ⇒ 步长几乎等于"各资源 busy 之和"，没有任何重叠；
+3. **通信与 AIC 的重叠在两个并发档都是 0.000** ⇒ 集合通信**从未**与计算并行。
+
+---
+
+## 1. 为什么并发提高反而"更串行"
+
+vLLM 把 8 个请求塞进**一个 batch、一条图、一次前向**：
+
+```
+conc=8： step = [10 个 M=48 的算子] → [5 个 M=48 的算子] → … 一条流
+  每个算子内部：AIC 相位 → AIV 相位（严格先后）
+  ⇒ 8 个请求的相位**完全重合**（不是错开）⇒ 空档依旧
+```
+
+**请求数增加只让 M 变大（算子变宽、数量变少：3349→1603 个/步），
+不产生相位多样性。** 所以：
+
+* 算子数减少 ⇒ 固定开销摊薄 ⇒ 步长从 34.95 降到 23.88 ms（这是批处理的收益）；
+* 但**每个算子内部 AIC→AIV 仍是串行** ⇒ 串行度反而上升到 93%。
+
+⇒ **批处理优化的是"每 token 的固定开销"，而 ubatching 优化的是"资源重叠"。
+两者正交，后者在高并发下空间更大。**
+
+---
+
+## 2. 与 tp8 交付实例的对照（口径一致）
+
+| | tp8 conc=1（profile，换算后） | tiny conc=1 | **tiny conc=8** |
+|---|---|---|---|
+| 步长 | 24.59 ms | 34.95 | 23.88 |
+| AIC | 55% | 39% | **35%** |
+| AIV | 37% | 30% | 35% |
+| 通信 | 11%（∩AIC=0） | 7%（∩AIC=0） | **13%（∩AIC=0）** |
+| AICPU | 5% | 3% | 9% |
+
+两者结构一致（AIC 空闲、通信零重叠），tiny 的 AIC 占比更低（因为 spec 接受长度
+只有 1.00，draft 工作量占比更高 —— tiny 是 dummy 权重，推测解码接受率天然低）。
+
+**⇒ tiny 的"高并发更串行"结论可以外推到 tp8。**
+
+---
+
+## 3. 上游 PR #11273 的可用性评估【实测·代码】
+
+拉取了 PR diff（35 KB / 6 文件 / +813 行），抽出核心新文件
+`vllm_ascend/worker/npu_ubatch_wrapper.py`（258 行）逐段读过。
+
+### 3.1 它做了什么（可复用的部分）
+
+```python
+class NPUUBatchWrapper:
+    def __init__(self, runnable, vllm_config, device):
+        self.comm_stream = torch.npu.Stream(device=device)
+        self.ready_barrier = threading.Barrier(num_ubatches + 1)
+
+    def __call__(self, *args, **kwargs):
+        ubatch_slices = getattr(get_forward_context(), "ubatch_slices", None)
+        if ubatch_slices is None:
+            return self.runnable(*args, **kwargs)          # 不切分时透传
+        # 按 tokens_slice 切 input_ids / positions / inputs_embeds / intermediate_tensors
+        return self._run_ubatches(ubatch_metadata, self.runnable)
+
+    def _run_ubatches(self, ubatch_metadata, model):
+        # 每个 ubatch 一条线程，with ubatch_metadata.context 进入 UBatchContext
+        # 该 context 自带 compute_stream + comm_stream
+        # 最后 torch.cat(sorted_results, dim=0)
+```
+
+**它的价值**：把上游 `ubatching.py`（纯 Python，与 CUDA/NPU 无关）与 NPU 结合的最小骨架，
+以及 `_slice_model_inputs` 的切分规则。**这部分可以直接借鉴。**
+
+### 3.2 ★ 它的致命限制（对我们不适用）
+
+```python
+class NPUUBatchWrapper:
+    """NPU version of UBatchWrapper for Ascend DBO support.
+    Supports eager mode only (no ACL Graph capture in phase 1)."""
+```
+
+**Phase 1 明确只支持 eager、不支持 ACL Graph。** 而：
+
+* 我们的 decode 走 **NPUGraph**（`FULL_DECODE_ONLY`）；
+* 我们**已实测**：无图时 host 提交开销吃掉一切（Python 交错双流 **0.24×**）。
+
+⇒ **PR 原样不能用于我们的交付形态。** 它的实测收益（+3.5~9.9%）是在
+`max_model_len=2048` 的**小模型**上、算子数少、host 开销占比低的情况下取得的。
+
+### 3.3 但上游 GPU 版**支持图**，这是我们要补的那一环
+
+`vllm/v1/worker/gpu_ubatch_wrapper.py:284-294`：
+```python
+with torch.cuda.graph(cudagraph, stream=compute_stream, pool=self.graph_pool):
+    ubatch_metadata[0].context.cpu_wait_event.set()
+    for thread in ubatch_threads:
+        thread.join()
+    ...
+```
+
+⇒ **上游把"所有 ubatch 线程的 join"整个包进一次 CUDA Graph 捕获**。
+NPU 侧需要等价的 `torch.npu.graph(...)` 版本 —— 这是我们线 B 的核心工作量。
+
+---
+
+## 4. 修正后的线 B 计划
+
+| 步 | 内容 | 判据 |
+|---|---|---|
+| **B1** | ✅ **已完成**：确认 ubatching 骨架可复用；PR #11273 的 eager 限制已查明 | 本文 §3 |
+| **B2** | 借鉴 `NPUUBatchWrapper` 的切分逻辑 + 上游 GPU 版的**图内 join**，在 tiny 上实现**图模式 2-ubatch** | `[bneck] hp` 下降 + 聚合 tok/s 上升 |
+| **B3** | 扫 k=2/4，量 conc=1/4/8 的 hp 与总吞吐 | 复现 §0 的曲面 |
+| **B4** | 四道门（逐位一致 / 长文针 / hp / 带宽） | 全绿才上 tp8 |
+
+**预期收益（修正）**：
+* 之前估 +25~35%（基于"填 AIC 空转"）；
+* **现在按"串行度 93% → 重叠"重估，理论上限 2.83×**；
+* 扣掉：（a）MoE GEMM 带宽受限部分、（b）拆 batch 后 M 变小的固定开销、
+  （c）fork/join 开销 ⇒ **现实预期 1.3~1.6×（高并发）**，仍需实测。
+
+---
+
+## 5. 当前环境状态
+
+| 项 | 状态 |
+|---|---|
+| tiny | 运行中，**带 profiler**（`PROFILE_DIR=/opt/dsv41/results/ab_mkc_1001_215615/prof_cap8`），重排几何，BAT=2048 |
+| tiny 上跑的是什么 | `~/tmp/deepseek_v41.tiny_full.py`（**DCP 旁路 + repack + cap** 三要素齐全） |
+| tp8k5 | **未动**（本次全部在 tiny） |
+| 两次 profile | `prof_cap8/...073805994`（conc=1）、`prof_cap8/...074459342`（conc=8），已离线解析 |
+
+> ⚠️ 记录一个操作失误：`launch_tiny_prof.sh` 里的 `RUN=cap8` 没改，导致 conc=8 的
+> profile 也写进了 `prof_cap8`（两个 session 目录区分）。**数据无污染**，但下次应改 RUN。
+
+---
+
+## 6. 复现
+
+```bash
+# 采集（需要 tiny 以 PROFILE=1 启动）
+bash ~/tmp/prof_ab.sh 1 c1
+bash ~/tmp/prof_ab.sh 8 c8
+# 离线解析（必须是 sync 模式，daemon 内解析会失败）
+docker exec dsv41-tinyspark python3 -c "
+from torch_npu.profiler.profiler import analyse
+analyse('/opt/dsv41/results/ab_mkc_1001_215615/prof_cap8/<rank0 dir>')"
+# 资源重叠分析
+python3 tools/prof_ab_overlap.py <...>/ASCEND_PROFILER_OUTPUT <tag>
+```
