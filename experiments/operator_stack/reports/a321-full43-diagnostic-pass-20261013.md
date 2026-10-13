# a321 full43 真实权重诊断首次通过（2026-10-13）

job `formal_dspark_full43_diag_v17`（v26 包，快照 `src_dspark_full43_diag_v17`，physical 8–15）**passed=True，exit=0**：

- `diagnostic_request_completed=true`：真实 32 代诊断请求完成，输出 192 token（greedy/min=max=192/ignore_eos）。
- **8/8 rank 各 37 个完整 generation**（`complete_43_decoder_and_graph_output_generations=37`）：
  每轮 40 main + 3 draft 的 `decoder.output` 全部 `copy_count=1, rows>0, any(valid)`，且 main/draft 的
  `actual_graph_output_exports` 完整（1684/202 tensors，`all_explicit_outputs_observed`、`all_values_equal_owned_after_fence`、
  `all_outputs_alias_original_storage` 全 true）。
- 收敛链（v11→v26 全部真实权重、HCCL strict、native DSpark ON）：
  1. v11/v12：`seal_native_parameters` 对 host-resident CPU Engram 参数调 `get_npu_format` → 跳过 CPU 张量，记 `host_resident=true`；
  2. v13：把 4 个审计入口导出为原生模块属性，满足 dynamo `LOAD_GLOBAL` 守卫；
  3. v14：外层 envelope 接受编译容器 list，并写结构诊断；
  4. v15：post-load 身份/扩展 tag 审计 + envelope 幂等重挂（bound method 经 JSON 安全投影）；
  5. v16：把 envelope 挂到 `@support_torch_compile` 真正调用的 `_compiled_callable`/`aot_compiled_fn`；
  6. v17：adapter 用**动态子类**实现（`isinstance(model,_HIDDEN_STATE_DRAFTER_TYPES)` 通过、共享 `__dict__`）；
  7. v18：draft 强制 eager（unwrap `_runnable`）+ SDK shutdown 150s 窗口（`owned_worker_cleanup.json` 8×exit0、grace=45s）；
  8. v19/v21：draft root `do_not_compile=True`；调用账本用 `torch.compiler.is_compiling()` 保护；
  9. v22/v23/v24：draft 观测窗放宽到 6 行 + positions/valid 独立补位（修掉一处 `torch.bool` 笔误，并加 CPU 回归）；
  10. v25/v26：观察者 Python 计数定位 `last_skip=dtype` → tap 内 positions/mask dtype 归一（int64/bool），43/43 全写入。

关键证据（本地冻结于 `evidence/formal_a321/full43_diag_v17/`）：

- `gate/result.json`：sha256 `92c0facdc05602bd194ad5eeb89ae30023202569723637f75e8a011a844861bf`
- `gate/owned_worker_cleanup.json`：sha256 `a152cbd0a37edb4cbb21ee21155acede4558ac0379491c75f74072e779b5b803`
- `run.log`（119,187B）sha256 `8158d4decc86e8969d97cb8e93a0e327a0e1352ea2dc41b9ea41570f6e7b4ede`；`run.exit`=0
- `gate/rank_exports/rank0/epoch_0004.json` sha256 `5422a8ca7d19369873f1f951fd57b87d4c3e113aa4cbd1e90699d3850e80223e`
  （43/43 decoder 完整；draft.40 observer calls=writes=7）

范围声明：这是**私有诊断**通过，不是正式准入。剩余 UNCOVERED：前端 scheduler/rollback 对应、prepack 前后顺序、
17 消费者设备覆盖、原生 A/A 数学与 logprob、独立通用质量（GSM8K100/100、Vision23/23）与包含 DSpark 的 ≤20ms/step 性能。
全部 formal/performance 标志仍为 false。
