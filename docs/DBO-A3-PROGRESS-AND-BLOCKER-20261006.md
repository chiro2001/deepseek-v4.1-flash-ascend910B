# DBO（ubatching）在 Ascend 上的推进与**结构性阻塞**（2026-10-06，本轮）
+
+> 线索：线 B（pingpong / ubatching）。上一轮已证"并发越高串行度越高"（conc=8 达 93%），
+> 且微基准证明多流有效（1.37×）。本轮把它往真机推。全部为【实测】。
+
+## 0. 本轮四件事
+
+| # | 结果 | 性质 |
+|---|---|---|
+| **1** | **NPU 的一张图可以捕获双流并行分支**（1.31× vs 串行） | ✅ **突破**（扫清最大技术未知） |
+| **2** | 打通 DBO 的**配置校验**（修掉 platform.py 两处 `all2all_backend` 覆盖） | ✅ 进展 |
+| **3** | 发现 **PR #11273 与 vllm-ascend main 都缺关键接线**，自己补齐后继续推进 | ✅ 进展 + **上游发现** |
+| **4** | 撞上 **V4.1 专有的结构性阻塞**：`_publish_task` 的组级共享缓存与 ubatch 冲突 | 🔴 **阻塞点** |
+
+---
+
+## 1. 突破：一张 NPUGraph 能捕获双流并行分支
+
+上游 GPU 版 DBO 把"两个 ubatch 线程的 join"整个包进 `torch.cuda.graph`。
+NPU 侧此前【未确认】。实测：
+
+| 项 | 结果 |
+|---|---|
+| 两流并行分支能否被一张图捕获 | **✅ 可以** |
+| 关键规则 | **fork event 必须 record 在「捕获根流」上**（不是默认流） |
+| 图重放 | 0.989 ms |
+| 同工作量 eager | 1.611 ms（图 **1.63×**） |
+| **同工作量串行单流** | 1.293 ms ⇒ **多流图 = 1.31×** |
+
+**失败→成功的关键**（第一次的报错是决定性的）：
+
+```
+rtStreamWaitEvent failed, reason=in the model capture scenario,
+the event wait task has no corresponding event record task
+```
+
+⇒ 捕获只覆盖根流子图；在**默认流**上 `record` 的 fork event 不在捕获区内，
+侧流的 `wait_event` 就找不到对应的 record。把 `record` 挪到**捕获根流**即成功。
+
+工具：`tools/tiny_graph_ms.py`（成功版）、`tools/tiny_graph_multistream.py`（失败版反例）。
+
+---
+
+## 2. 打通 DBO 的配置校验（platform.py 两处覆盖）
+
+`--enable-dbo --all2all-backend=deepep_low_latency` 启动后报：
+
+```
+Assertion failed, Microbatching currently only supports the deepep_low_latency,
+deepep_high_throughput, and nixl_ep all2all backends. flashinfer_all2allv is not supported
+```
+
+vllm-ascend 在 `platform.py` **两处**强制 `all2all_backend = "flashinfer_all2allv"`
+（`:1161` 与 `:1254`，注释 "a tricky way to disable SP moe"）。
+PR #11273 只提到一处（"after worker_cls resolution"）—— 实测**两处都要放开**才过校验。
+
+> 附带教训：`VLLM_ALL2ALL_BACKEND` 不是有效的 vLLM env（会打 "Unknown environment variable"），
+> 必须用 CLI 参数 `--all2all-backend=`。
+
+---
+
+## 3. ★ 上游发现：PR #11273 与 main 都缺关键接线
+
+### 3.1 PR #11273 的 diff 里**没有**它自己声称的那个函数
+
+PR 描述明确写：
+
+> "New module-level `ascend_split_attn_metadata()`: preserves `AscendCommonAttentionMetadata`
+> subclass fields when splitting per-ubatch attention metadata. Upstream
+> `_make_metadata_with_slice` hardcodes `return CommonAttentionMetadata(...)`, which drops
+> Ascend-specific fields (`attn_state`, `prefill_context_parallel_metadata`, `kvcomp_metadata`,
+> etc.) and causes AttributeError on long prompts that trigger ubatch splitting."
+
+**但实测 diff（两种下载途径都试过，35,289 B / 3 commits / 6 files / +807−15）里
+`grep -c "ascend_split_attn_metadata" = 0`、`grep -n "split" = 0`。**
+
+⇒ **PR 声称的关键修复并不在代码里**，它无法端到端工作。
+
+### 3.2 vllm-ascend main（`cd96b9d`，2026-10-06）**也缺** ubatch 循环
+
+`_build_attention_metadata` 里 GPU 版有：
+
+```python
+for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
+    if ubatch_slices is not None:
+        for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
+            _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)
+    else:
+        _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
+```
+
+而 vllm-ascend（**main 与我们的镜像都一样**）只有 else 分支：
+
+```python
+for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
+    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm, ...)   # 没传 ubid
+```
+
+⇒ 一旦 `create_attn_groups` 建了 `num_ubatches` 个 builder，
+`attn_metadata` 变成 list 而调用方按 dict 用 ⇒ **`assert isinstance(attn_metadata, dict)` 失败**。
+
+**⇒ 结论：DBO 在 vllm-ascend 上从来没有真正跑通过**（这也解释了 PR 为何挂了 3 个月无人合并）。
+
+### 3.3 我自己补齐的部分
+
+**① `ascend_split_attn_metadata()`**（低风险实现）：
+父类字段**直接复用上游 `split_attn_metadata`**（已验证），只回填 ~10 个 Ascend 专有字段：
+
+| 类别 | 字段 | 切法 |
+|---|---|---|
+| per-request | `seq_lens_cpu`、`num_computed_tokens_cpu`、`context_parallel_metadata.*` | `request_slice` |
+| per-token | `positions`、`positions_cpu`、`actual_seq_lengths_q` | `token_slice` |
+| **按 ubatch token 数** | **`num_input_tokens`** | `len(base.slot_mapping)` |
+| scalar/共享 | `decode_token_per_req`、`graph_pad_size`、`attn_state` | 原样 |
+
+长度不匹配时**保守保留原值**（宁可多带，不静默切错）。
+
+**② ubatch 循环**（`_build_attention_metadata`）：加 `ubatch_slices is not None` 分支并传 `ubid`。
+
+**③ `num_input_tokens` 必须切**（第一版保守保留 → 报错，见下）。
+
+---
+
+## 4. 🔴 结构性阻塞：V4.1 的 `_publish_task` 与 ubatch 冲突
+
+补齐 ①②③ 后，错误推进到：
+
+```
+File ".../attention/dsa_v41.py", line 4521, in build
+RuntimeError: V4.1 compressor metadata must have one owner
+```
+
+### 4.1 根因
+
+```python
+# dsa_v41.py:3975  （V4.1 的 device-metadata 共享机制）
+def _publish_task(self, shared, key, buffer, stage, run) -> torch.Tensor:
+    existing = shared.get(key)
+    if existing is not None:
+        return existing            # ← 命中缓存就复用
+    shared[key] = buffer
+    ...
+    return buffer
+
+# dsa_v41.py:4517
+compressor_group = self._publish_task(shared, "c2:compressor", self._c2_complete_mask, ...)
+if compressor_group is not self._c2_complete_mask:
+    raise RuntimeError("V4.1 compressor metadata must have one owner")
+```
+
+而 `shared`（= `common_v41_batch_metadata`）是**在 `_build_attention_metadata` 里创建一次、
+跨所有 cache group 复用**的组级 dict。
+
+⇒ ubatching 时：
+* ubatch 0 的 builder 把 `self._c2_complete_mask`（**对象 A**）发布进 `shared`
+* ubatch 1 的 builder（**自己的 builder 实例、自己的对象 B**）命中缓存 ⇒ 拿到对象 A
+* `A is not B` ⇒ **断言失败**
+
+### 4.2 为什么这是"结构性"的
+
+这不是漏了一个参数，而是 V4.1 的 device-metadata 设计**假设"每个 cache group 每步只有一个
+metadata 消费者"**（这样它可以只建一份 mask、只发一次 device task）。
+ubatching 天生要**两个消费者**。
+
+### 4.3 两条候选修复（未验证）
+
+| 方案 | 做法 | 风险 |
+|---|---|---|
+| **A（小改）** | 把 `shared` 的 key 变成 per-ubatch：`shared[f"{key}:ub{ubid}"]`；`_build_attn_group_metadata` 已有 `ubid` 形参，透传即可 | 中：`common_v41_metadata` 在循环**之后**还要被用来组装最终 metadata，per-ubatch 版本需要一并收集 |
+| **B（改缓存作用域）** | 在 ubatch 循环里给每个 ubatch 传**独立的** `common_v41_batch_metadata` / `common_v41_metadata` | 中：要确认这两者在下游是否按 ubatch 分开消费 |
+
+**关键判断**：无论 A 还是 B，都要动 `dsa_v41.py` 的共享语义 ——
+而那是**当前单批路径正常工作的基础**。改动必须保证单批路径逐位不变
+（用 `walk_blocks` 十轮 `max|Δ|=0` 守住）。
+
+---
+
+## 5. 当前状态与产物
+
+| 项 | 状态 |
+|---|---|
+| 多流图捕获 | ✅ 已验证（微基准 1.31×） |
+| DBO overlay | ✅ 已建（5 文件），**本轮结束时已从容器回退**，tiny 恢复常规配置 |
+| 上游发现 | ✅ PR #11273 不完整；main 缺 ubatch 循环 |
+| V4.1 阻塞 | 🔴 已定位到具体行与机制，两条修复方案待验 |
+| tp8k5 | **未动** |
+
+**产物**：
+* `tools/tiny_graph_ms.py` / `tools/tiny_graph_multistream.py`（多流图捕获）
+* `tools/ref_pr11273.diff` / `tools/ref_pr11273_npu_ubatch_wrapper.py`（上游 PR 存档）
+* `~/tmp/apply_dbo_overlay.py`、`~/tmp/patch_mr_full.py`（一键重建 overlay）
+* `~/tmp/launch_tiny_dbo.sh`、`~/tmp/launch_tiny_prof.sh`（tiny 起服）
+
+### 5.1 下一步（下次接手直接做）
+
+1. 按 **§4.3 方案 A** 改 `dsa_v41._publish_task` 的 key（透传 ubid），
+   同时把 `common_v41_metadata` 的收集改成 per-ubatch；
+2. 起服（eager 先验证正确性）→ `walk_blocks` 十轮逐位比对；
+3. 通过后切图模式，测 `[bneck] hp` 与聚合 tok/s；扫 k=2/4。
+
+## 6. 复现
+
+```bash
+# 1) 多流图捕获（独立验证，与 DBO 无关）
+ssh a3-21 'docker exec dsv41-tinyspark bash -lc "cd /tmp && python3 graph_ms2.py"'
+# 2) 重建 DBO overlay 并起服
+ssh a3-21 'python3 ~/tmp/apply_dbo_overlay.py && python3 ~/tmp/patch_mr_full.py \
+  && python3 ~/tmp/fix_numin.py && bash ~/tmp/launch_tiny_dbo.sh'
+# 3) 观察阻塞点
+ssh a3-21 'grep -aE "must have one owner|must match the size" \
+  ~/tmp/ab_mkc/run_ab_mkc_1001_215615/serve_dbo.log | tail -3'
+# 4) 回退到常规 tiny
+ssh a3-21 'python3 ~/tmp/revert_pf.py && bash ~/tmp/launch_tiny_prof.sh'
+```
