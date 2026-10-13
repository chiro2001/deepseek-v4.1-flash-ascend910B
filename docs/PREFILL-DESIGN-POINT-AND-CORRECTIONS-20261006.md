# prefill 设计点：几乎不随并发伸缩；并更正两处我自己的错误（2026-10-06）

> 本轮把 prefill 从「疑似崩塌」追到「稳定且平坦」，过程中**更正了上一轮的两个结论**。
> 全部为【实测】。

## 0. 更正清单（都是上一轮的结论）

| 上一轮结论 | 实测更正 |
|---|---|
| 「prefill conc=8 崩到 **0.59×**（4,173 tok/s）」 | **冷启动 artifact**。稳定值是 **7,965 tok/s = 1.11×**（三次复现：7973 / 7977 / 7945） |
| 【推断】「崩塌源于 BAT=8192 的 **chunk 边界效应**」 | **被证伪**：带块拆分的 8299 tok（7,994）反而**快于**不拆分的 7963 tok（5,565） |
| `docs/TP8-CURVE-CORRECTED-20261006.md` 的 decode 曲线 | 仍然成立 |
| `docs/MEASUREMENT-PITFALLS-AND-PREFILL-20261006.md` §2 的单请求 prefill（7.0~7.5K） | 成立 |

## 1. 稳定性验证：同一配置连跑三次

conc=8、每条 8,300 token、nonce 破缓存、零缓存命中：

| run | 墙钟 | 聚合 prefill | 首 token（min / 中位 / max） |
|---:|---:|---:|---|
| 1 | 8.328 s | **7,973 tok/s** | 1.179 / 6.098 / 8.324 s |
| 2 | 8.330 s | **7,977 tok/s** | 1.182 / 6.103 / 8.325 s |
| 3 | 8.362 s | **7,945 tok/s** | 1.179 / 6.126 / 8.357 s |

⇒ **离散度 0.4%**，测量本身完全可靠。所以之前那次 4,173 tok/s 不是随机噪声，
而是**该次运行的顺序位置**导致的状态差异（它是服务重启后的首个 conc=8）。

**结论**：在固定 `BAT_TOKENS=8192` / `MAX_SEQS=32` 的交付配置下，可以把 §3 的曲线当作设计点；
报告数字时必须声明是**冷**运行还是**热**运行。

（测量后 `num_requests_running == 0`、`num_requests_waiting == 0`，无遗留请求。）

## 2. chunk 边界假设被证伪

| conc=8 配置 | 是否跨 8192 块 | 聚合 prefill |
|---|---|---:|
| 7963 token | **否**（单 chunk） | **5,565 tok/s** |
| 8299 token | 是（8192 + 107） | **7,994 tok/s** |

若块边界是主因，不拆分那条应当更快；实际相反，且两条都落在 §1 的冷/热包络内。
⇒ **块边界不是原因，撤回该推断。**

## 3. 真正的结论：prefill 几乎不随并发伸缩

同一会话、同一 prompt 长度（约 8,300 token）、零缓存命中：

| 并发 | 墙钟 | 聚合 prefill | 相对单请求 | 首 token（min / 中位 / max） |
|---:|---:|---:|---:|---|
| 1 | 1.155 s | **7,195 tok/s** | 1.00× | 1.153 / — / — |
| 8 | 8.330 s | **7,965** | **1.11×** | 1.179 / 6.098 / 8.324 |
| 16 | 16.078 s | **8,262** | 1.15× | 1.191 / 10.045 / 16.069 |
| 32 | 30.302 s | **8,771** | **1.22×** | 0.963 / 16.532 / 30.288 |

三条读法：

1. 并发 1 到 32（32 倍），聚合 prefill 只涨 **1.22 倍**；
2. 墙钟几乎严格线性（8 条 8.33 s、16 条 16.08 s、32 条 30.30 s）
   ⇒ 引擎**近似串行处理 prefill**，只在边缘有约 15% 的重叠；
3. 首 token 的中位/最大时间随并发线性增长（8 条中位 6.1 s、32 条中位 16.5 s）
   ⇒ **多用户同时提交长 prompt 时，后到的请求要排很久**。

与 decode 形成鲜明对照（decode 在 conc 1 到 32 涨 **14.8 倍**）：

| 阶段 | conc 1→32 的总吞吐倍数 | 性质 |
|---|---:|---|
| **prefill** | **1.22×** | 近似串行；单请求只有约 7K tok/s，硬件未打满 |
| **decode** | **14.8×** | 每步固定成本被摊薄，随并发近似线性获益 |

⇒ prefill 是**延迟型**，decode 是**吞吐型**。

## 4. 对用户体验与容量的含义

以 8.3K prompt + 500 token 输出、单请求为例：

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| prefill | 1.155 s | 18% |
| decode（500 token ÷ 2.36 token/步 × 24.58 ms） | 5.21 s | 82% |
| **合计** | **6.37 s** | — |

而长上下文场景（131K token）prefill 单独就占 **19.8 s**（见 §5），完全主导。

⇒ **短 prompt 场景瓶颈在 decode（每步固定成本），长 prompt 场景瓶颈在 prefill（约 7K tok/s）。**
两者需要不同手段，而 prefill 侧目前**看不到合批红利**。

## 5. 单请求 prefill 参考曲线（上一轮已验证干净）

| prompt tokens | TTFT | prefill tok/s |
|---:|---:|---:|
| 4,153 | 0.553 s | 7,504 |
| 16,414 | 2.228 s | 7,368 |
| 65,820 | 9.300 s | 7,077 |
| 131,317 | 19.805 s | 6,631 |

## 6. 下一步该测什么（仍未验证的假设）

prefill 只有约 7K tok/s 且不随并发提升，说明**单条 prefill 本身就没打满硬件**。
【未确认】候选原因（按可测性排序）：

1. **CP（Context Parallel）没有开**：官方文档明确写「长序列下单卡 prefill 成为首 token 时延瓶颈，
   V4.1 在 Prefill 阶段采用 Context Parallel 把单条请求切分到各卡并行计算，Decode 仍按 DP 执行」。
   我们 tp8k5 是 `TP=8 / DP=1`，prefill 走 TP 而非 CP ⇒ 单条长 prompt 没有序列维并行。
2. **chunk 之间串行**：8192-token chunk 的依赖（KV 写入完成才能算下一个 chunk）导致无法流水。
3. **MoE 在 prefill 的算力利用率**：我们的 kernel 迁移自 decode，可能未针对 compute-bound 调优。

建议下一步先查 ①——它在**配置层面**（官方已有实现），不需要改 kernel，
且直接对应「长文 TTFT 19.8 s」这个用户可感知的痛点。

## 7. 复现

```bash
# 稳定性验证（连跑三次）
ssh a3-21 'for i in 1 2 3; do python3 ~/tmp/tp8_prefill_conc_clean.py \
  http://127.0.0.1:19210 8 8252 8; done'
# 并发扫描
ssh a3-21 'for c in 1 8 16 32; do python3 ~/tmp/tp8_prefill_conc_clean.py \
  http://127.0.0.1:19210 $c 8252 8; done'
# 服务端空闲核验
ssh a3-21 'curl -s --noproxy "*" http://127.0.0.1:19210/metrics | grep num_requests_running'
```

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
