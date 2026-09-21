# vllm-ascend CI 剖析：有没有自动的性能与功能验证

> 2026-09-21，基于 `upstream/main`（`c173a64a`）实测。**结论：功能验证是自动的，
> 性能验证存在但基本不在 PR 上跑。**

---

## 0. 一句话结论

| 问题 | 答案 |
|---|---|
| 有自动**功能**验证吗？ | **有，而且是重头戏**。PR CI 会按改动文件**精准推荐**并跑到 NPU 上 |
| 有自动**性能**验证吗？ | **有机制**（`run_vllm_bench_case` + 硬编码基线 + 0.97 阈值），但**PR CI 里几乎不跑**，主要在 nightly/weekly |
| 我们能自己触发 NPU 测试吗？ | **不能**。要 maintainer 打 `ready-precise` / `ready-all` label（或 `/e2e` 斜杠命令，但**需要 triage+ 权限**） |

---

## 1. PR CI 结构（`.github/workflows/pr_test.yaml`，名字叫 "E2E"）

触发：`pull_request` 的 `opened / synchronize / reopened / labeled`，目标 `main` / `*-dev` / `releases/v*`。

### Job 链

```
pre-commit ──► cpu-ut ──► recommend-tests ──► select-tests ──► NPU runs
   │                            │                  │
   │                            │                  └─ 路由到 one/two/four card、A2/A3/A5/310p
   │                            └─ 覆盖率/AST 的「精准选测」（test_selector.py）
   └─ 标题前缀校验 + ruff/codespell/typos/clang-format/markdownlint + mypy + Gitleaks
```

### 1.1 `pre-commit`（第一个 job，**硬门槛**）

**★ PR 标题前缀被 CI 强制校验**（原文）：

```
VALID_PREFIXES='\[(BugFix|Performance|Test|CI|Feature|Doc|Misc|Community|Refactor)\]'
```

⇒ 我们用的 `[Performance]` **正确**；不符合会**直接 fail**，连测试都不跑。

其余：pre-commit 全家桶、mypy、Gitleaks 密钥扫描、paths-filter（判断 `src` / `ci_pipeline` 是否变化）。

### 1.2 `recommend-tests` —— 精准选测（很有意思）

不是"改哪个文件跑哪个测试"的静态映射，而是：

1. 下载 **test case map + coverage package**
2. 用 `.github/workflows/scripts/test_selector.py`（**覆盖率 + AST**）推荐测试
3. `test_config.yaml` 只放路由元数据（**不再做路径→测试的映射**）

### 1.3 跑 NPU 测试需要 label

| label | 行为 |
|---|---|
| `ready-precise` | 跑**推荐**的测试（走 coverage 结果） |
| `ready-all` | 跑**全量**测试 |
| `main2main` | 强制全量 + 双 vLLM 版本 |

**没有 label ⇒ `select-tests` 直接跳过**（原文："all-tests and unlabeled runs select nothing"）。

⇒ **外部贡献者加不了这些 label**（GitHub 上打 label 需要 triage+ 权限）。
**我们的 PR 提上去后会停在 CPU 阶段，等 maintainer 打 label 才会有 NPU 验证。**

### 1.4 斜杠命令（也需要权限）

| 命令 | 作用 | 权限 |
|---|---|---|
| `/e2e` | 触发 e2e | "Check user authorization"（triage+） |
| `/nightly` | 触发 nightly 配置 | 同上 |
| `/rerun` | 重跑失败 job | **PR 作者**或 triage+ |

---

## 2. 性能验证：有机制，但 PR 上基本不跑

### 2.1 工具：`tools/vllm_bench.py`

```python
def run_vllm_bench_case(model_name, port, config, baseline, threshold=0.97, ...):
```

⇒ **吞吐低于基线的 97% 即失败**。这是真正的性能回归门禁。

### 2.2 两个性能测试（**一个被跳过，一个活着**）

| 测试 | 基线 | 状态 |
|---|---|---|
| `tests/e2e/pull_request/two_card/test_qwen3_performance.py` | **1514.0 tok/s**（Qwen3-8B，500 prompts） | ❌ **在 `skip_tests` 里**（"Temporarily skipped due to flaky failures"） |
| `tests/e2e/pull_request/four_card/test_profiling_chunk_performance.py` | **`BASELINE_TTFT_S = 5.45`**，断言 `median_ttft <= 5.45` | ✅ **活跃**（列在 `accuracy_tests`，会改派到专用 560T 机器） |

**注意 `test_qwen3_performance.py` 里的原话**：

```python
# NOTE: Any changes for the baseline throughput should be approved by team members.
# The origin baseline: 1600.0. For some uncertain reasons, the throughput is decreased to 1514.0
```

⇒ **基线是团队共识产物，不能自己改。**

### 2.3 nightly / weekly 里的性能

| 配置 | 内容 |
|---|---|
| `nightly_config.yaml` | **`qwen3-30b-a3b-bf16-a2-performance`**、**`qwen3-30b-a3b-w8a8-a2-performance`**、accuracy 组 |
| `weekly_config.yaml` | `Qwen3.5-397B-Memcache-perf`、`Qwen3.6-35B-A3B-w8a8-A3-accuracy`、**`Qwen3.5-122B-A10B-w4a8-A3-accuracy`** |

⇒ 有意思的是：**nightly 的性能用例在 A2 上，而且已经有 w8a8/perf 的组合**；
weekly 里**已经有 W4A8 的 A3 accuracy**（和我们做的事相关）。

### 2.4 ⇒ 结论

| 场景 | 会自动出性能吗 |
|---|---|
| 普通 PR | ❌ **不会**（PR CI 的唯一 perf-guard 是那个 TTFT 测试，且只在被推荐时才跑） |
| 带 `ready-all` / `/nightly` | ⚠️ 可能（取决于配置，且多是 accuracy 而非 perf） |
| nightly 定时 | ✅ 会（`*-a2-performance`） |
| weekly 定时 | ✅ 会（`*-perf`） |

**⇒ 对性能类 PR，我们必须自带证据**（不能指望 CI 出）。

---

## 3. 对我们的具体影响（行动项）

### 3.1 改我们的文件会触发哪些测试（实测对应关系）

| 我们改的文件 | CI 会推荐的测试 |
|---|---|
| `vllm_ascend/ops/rope_dsv4.py` | `tests/ut/ops/test_rope_proxy.py`、`tests/ut/attention/test_dsa_v1.py` |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | `tests/ut/ops/test_token_dispatcher.py`、`test_moe_comm_method.py`、`test_moe_runtime_args.py`、`tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_moe.py` |
| `vllm_ascend/attention/dsa_v1.py` | `tests/ut/attention/test_dsa_v1.py` + 若干 |

⇒ **rope 分支已经在 `test_rope_proxy.py` 里加了 12 个测试**，正好落在会被推荐的那个文件里 ✅
⇒ **MoE mask 分支必须补 `tests/ut/ops/test_token_dispatcher.py` 的测试**（否则改动没有对应的 UT 覆盖）

### 3.2 三条硬约束（写进执行纪律）

1. **标题前缀**：`[Performance][MoE] …` —— CI 第一关就查，错一个字就 fail
2. **必须有对应 UT**：改哪个文件，就把测试加在**会被推荐到的那个测试文件**里
3. **性能证据自带**：CI 不会给我们的 PR 出性能数字

### 3.3 流程上的一个现实

```
我们提 PR
   → pre-commit 跑（CPU，自动）           ✅ 我们能自己看到
   → cpu-ut 跑（CPU，自动）               ✅ 我们能自己看到
   → select-tests **跳过**（没有 label）   ⚠️ 卡住
   → 等 maintainer 打 ready-precise       ← 需要主动请（或在 PR 里说明）
   → NPU 测试跑
```

⇒ **PR 描述里应该主动写一句**"please add `ready-precise` to run the relevant NPU tests"，
并且**把我们自己的证据附上**（因为 NPU 测试要等，而证据能立刻给 reviewer 判断）。

---

## 4. 对今晚计划的影响

| 原计划 | 调整 |
|---|---|
| MoE mask 的测试"参照上游风格写" | **明确写进 `tests/ut/ops/test_token_dispatcher.py`**（这是会被推荐到的文件） |
| rope 分支的测试已就绪 | ✅ 无需调整 |
| 性能证据 | **必须自带**（CI 不出）；这也正是今晚跑单卡 A/B 的理由 —— 更有必要 |
| PR 描述 | 加"请加 `ready-precise`"的说明段 |

---

## 附：原始出处

| 内容 | 文件 |
|---|---|
| PR CI 主流程 | `.github/workflows/pr_test.yaml` |
| 选测/路由配置 | `.github/workflows/scripts/test_config.yaml` |
| 精准选测实现 | `.github/workflows/scripts/test_selector.py` |
| 性能断言工具 | `tools/vllm_bench.py::run_vllm_bench_case`（`threshold=0.97`） |
| 性能测试 | `tests/e2e/pull_request/two_card/test_qwen3_performance.py`（被 skip）、`four_card/test_profiling_chunk_performance.py`（活跃） |
| 定时性能 | `.github/workflows/configs/{nightly,weekly}_config.yaml` |
