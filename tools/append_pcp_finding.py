#!/usr/bin/env python3
"""把「PCP 被 V1/V2 runner 分裂挡住」的调查结论追加到 PREFILL 文档。"""
from pathlib import Path

DOC = Path("/home/chiro/projects/dsv41/main-merge/docs/PREFILL-DESIGN-POINT-AND-CORRECTIONS-20261006.md")

ADD = '''
## 8. ★ §6 假设①的调查结果：**PCP 被 V1/V2 runner 分裂挡住**（决定性）

结论：**在当前配置下 PCP 不是"可开的开关"，而是需要换整个 model runner。**

### 8.1 证据链

1. **vllm-ascend 确实为 v4.1 实现了 PCP**（不是缺功能）：
   * `vllm_ascend/attention/context_parallel/dsa_cp.py` 里有
     `AscendDSAPCPMetadataBuilder`（2200+ 行）；
   * `vllm_ascend/core/deepseek_v41.py`、`attention/sfa_v1.py`、
     `attention/attention_v1.py` 都有 `prefill_context_parallel_size > 1` 的分支；
   * 官方技术报告也写明「V4.1 在 Prefill 阶段采用 Context Parallel 把单条请求切到各卡并行计算」。

2. **但它硬性要求 V2 model runner**：
   ```python
   # vllm_ascend/platform.py:_validate_parallel_config
   if not vllm_config.use_v2_model_runner and parallel_config.prefill_context_parallel_size > 1:
       raise ValueError(
           "PCP (Prefill Context Parallelism) is not supported by vLLM Ascend. "
           "Please set --prefill-context-parallel-size to 1. ...")
   ```

3. **而 Ascend 上的 `use_v2_model_runner` 被 vllm-ascend 改成了"只认环境变量"**：
   ```python
   # vllm_ascend/patch/platform/patch_use_v2_model_runner.py
   def _patched_use_v2_model_runner(self) -> bool:
       """On Ascend the v2 runner is controlled purely by the
          VLLM_USE_V2_MODEL_RUNNER environment variable"""
       use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
       if use_v2 is not None:
           return use_v2
       return False          # ← 未设置 ⇒ 一律 V1
   VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
   ```
   （即：上游 vLLM 里 `dspark ⇒ V2` 的那条规则**在 Ascend 上被绕开了**。）

4. **我们的部署没有设这个变量**：实测 `inner.sh` 与 API server / EngineCore
   进程环境里都**没有** `VLLM_USE_V2_MODEL_RUNNER`（`grep` 为空）。
   ⇒ tp8k5 跑的是 **V1 runner** ⇒ `--prefill-context-parallel-size > 1` 会**直接启动失败**。

### 8.2 这意味着什么

| 项 | 状态 |
|---|---|
| PCP 功能本身 | **已实现**（vllm-ascend 为 v4.1 写了完整路径 + 官方报告背书） |
| 我们能否开 | **不能**，除非迁移到 V2 model runner |
| 迁移代价 | **大**：V1↔V2 是两套 runner（`worker/model_runner_v1.py` vs `worker/v2/model_runner.py`），
而我们这一整套补丁栈（dspark 入图、DCP、CED、engram、bneck 探针…）都是围绕 **V1** 建的 |

⇒ **"开 PCP 提 prefill" 不是低成本选项**，应记为**独立的大工程**（相当于 runner 迁移），
而不是本轮能做的调参。

### 8.3 于是 prefill 侧剩下的可行手段（按成本排序）

| 手段 | 成本 | 说明 |
|---|---|---|
| 调整 **chunked-prefill 的 chunk 大小**（`BAT_TOKENS`） | 低（改参数） | prefill 是延迟型，chunk 越小则交错更细、首 token 更快，但总吞吐可能下降。**未测** |
| 把长 prompt 的 **TTFT 目标与吞吐目标分开**（准入/调度策略） | 中 | 服务层策略，不改 kernel |
| 迁移到 **V2 runner 以启用 PCP** | **大** | 唯一能真正突破"单条 prefill 只有 ~7K tok/s"的手段 |
| 重写 prefill 的 MoE/attention kernel | 大 | compute-bound 路径的 kernel 级工作 |
'''

s = DOC.read_text()
if "PCP 被 V1/V2 runner 分裂挡住" in s:
    print("已存在，跳过")
else:
    DOC.write_text(s.rstrip("\n") + "\n" + ADD)
    print("appended", len(ADD), "chars")
