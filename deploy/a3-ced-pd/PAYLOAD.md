# 部署形态 payload 清单（唯一事实源）

本文件是 **GitHub 侧启动器** 与 **镜像层 patch 包** 共用的 payload 定义。
两边都由它派生 ⇒ **可保证一致**。

* 生成镜像：`bash deploy/a3-ced-pd/build_image.sh`
* 打包成镜像层 patch：`sudo python3 <main-wt>/tools/make_image_patch_kit.py --job 'dsv41-a3-ced-pd|<image>|v41' ...`
* 一致性校验：`bash deploy/a3-ced-pd/verify_consistency.sh`（比对镜像内文件 md5 与本仓文件 md5）

---

## A. vllm-ascend 补丁件（整文件覆盖；目标根 = `/vllm-workspace/vllm-ascend/vllm_ascend`）

| 仓库源（相对仓库根） | 容器目标（相对 ASCEND_PKG） | 类型 |
|---|---|---|
| `patches/files/engram_hbm.py` | `models/deepseek_v41/engram_hbm.py` | 覆盖 |
| `patches/files/engram_hash.py` | `models/deepseek_v41/engram_hash.py` | 覆盖 |
| `patches/files/engram_gate.py` | `models/deepseek_v41/engram_gate.py` | 覆盖 |
| `patches/files/engram_jit_kernel.py` | `models/deepseek_v41/engram_jit_kernel.py` | 新增 |
| `patches/files/engram_plan_kernel.py` | `models/deepseek_v41/engram_plan_kernel.py` | 新增 |
| `patches/files/engram_device_index.py` | `models/deepseek_v41/engram_device_index.py` | 新增 |
| `patches/files/engram_graph.py` | `models/deepseek_v41/engram_graph.py` | 新增 |
| `patches/files/model.py` | `models/deepseek_v41/model.py` | 覆盖 |
| `patches/files/indexer.py` | `models/deepseek_v41/indexer.py` | 覆盖 |
| `patches/files/ascend_forward_context.py` | `ascend_forward_context.py` | 覆盖 |
| `patches/files/rope_dsv4.py` | `ops/rope_dsv4.py` | 覆盖 |
| `patches/files/block_table.py` | `worker/block_table.py` | 覆盖 |
| `patches/files/token_dispatcher_moemask.py` | `ops/fused_moe/token_dispatcher.py` | 覆盖 |

## B. CED 件（本部署形态的核心；挂载模式下由 CED 角色分支挂入）

| 仓库源 | 容器目标 | 说明 |
|---|---|---|
| `experimental/ced/mooncake_hybrid_connector.py` | `distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py` | P→D 的 12 组契约、D 侧预清零、`[CED-32BIT-GUARD]` |
| `experimental/ced/dsa_v41.py` | `attention/dsa_v41.py` | `[CED-SWA-CLIP]` 等；**仅在 `V41_CED_ROLE=decode` 时改变行为**（见 §E） |

> ⚠️ 只烘一份 `dsa_v41.py` 给 **P/D 共用**是有前提的，见 §E。已按该前提验证。

## C. vLLM core 补丁（**运行期**在容器内 `git apply`，不烘进文件）

| 仓库源 | 容器目标 | 触发条件 |
|---|---|---|
| `patches/admission_gate.patch` | `/opt/dsv41/admission_gate.patch` | **镜像构建期就打好**（baked 模式不再现场打） |
| `experimental/ced/core_scheduler_replay.patch` | `/opt/dsv41/ced_scheduler_replay.patch` | `V41_CED_ROLE=decode` |
| `experimental/ced/core_scheduler_prefill_hit.patch` | `/opt/dsv41/ced_scheduler_prefill_hit.patch` | `V41_CED_ROLE=prefill` |
| `experimental/ced/core_model_runner_prompt_tail.patch` | `/opt/dsv41/ced_runner_prompt_tail.patch` | `V41_CED_GRAPH_PROMPT_TAIL_EAGER=1` |

> 这三个补丁**故意留到运行期**打：每个都带**基线 sha256 硬门**
> （scheduler.py / model_runner_v1.py 的固定哈希），
> 一旦基底 vLLM 版本不同就会**拒绝起服**，而不是静默跑错版本。
> 烘进镜像反而会绕过这层保护。

## D. 起服脚本

| 仓库源 | 容器目标 |
|---|---|
| `scripts/serve_a2.sh` | `/opt/dsv41/scripts/serve_a2.sh` |
| `scripts/serve_v2.sh` | `/opt/dsv41/scripts/serve_v2.sh` |
| `scripts/serve_a3.sh` | `/opt/dsv41/scripts/serve_a3.sh` |
| `scripts/serve_a3_pd.sh` | `/opt/dsv41/scripts/serve_a3_pd.sh` |
| `scripts/serve_a3_pd_proxy.sh` | `/opt/dsv41/scripts/serve_a3_pd_proxy.sh` |
| `scripts/serve_a3_ced_pd.sh` | `/opt/dsv41/scripts/serve_a3_ced_pd.sh` |
| `scripts/serve_a3_ced_single.sh` | `/opt/dsv41/scripts/serve_a3_ced_single.sh` |
| `scripts/run_test.sh` | `/opt/dsv41/scripts/run_test.sh` |

## D2. decode 侧请求边界护栏

| 仓库源 | 容器目标 | 触发条件 |
|---|---|---|
| `patches/files/v41_decode_guard.py` | `/opt/dsv41/guards/v41_decode_guard.py` | `V41_CED_ROLE=decode` 时由 `serve_v2.sh` 注册为 `--middleware` |

作用：decode 半边拒绝**没有 `kv_transfer_params`** 的生成请求（400，不进引擎）。
背景是 2026-09-27 00:01 的事故：一条直连 18991 的普通请求让 D 自己去 prefill，
撞上固定 128-token replay 守卫，worker `raise` ⇒ EngineCore 退出 ⇒ 整个 D 实例死掉、
要重载 20 分钟权重。详见 `docs/CED-DECODE-API-GUARD-20260927.md`。

判据（起服日志）：`[V41-DECODE-GUARD] middleware loaded` 与
`[serve-v2] decode API guard: ON`。缺文件时 `serve_v2.sh` 会打 WARNING，
`serve_a2.sh` 在 decode 角色下会直接 `die`（不带着未加固的 D 起服）。

## E. ★ 为什么一份 `dsa_v41.py` 能同时供 P 和 D

挂载模式下，P **不挂** `experimental/ced/dsa_v41.py`，D 挂 ⇒ 两实例可以用不同文件。
烘成一份镜像后两边只能共用一份，所以必须先证明"共用也安全"。

**已逐行核实**：`experimental/ced/dsa_v41.py` 相对**基础镜像原版**只有 371 行差异，
其中对原有代码的**修改只有 4 处**，且全部满足"角色为 P 时行为与原版逐位一致"：

| # | 改动 | P 角色下的取值 |
|---|---|---|
| 1 | `_attention(...)` 增加默认参数 `replay_chunk=False` | 默认值 ⇒ 等价 |
| 2 | `seqused_ori` 在裁剪分支里被换成 `local_lens` | 该分支要求 `replay_chunk=True` ⇒ 不进入 |
| 3 | `if self.role.is_kv_source:` → `and not replay_chunk` | `replay_chunk=False` ⇒ 等价 |
| 4 | 调用处多传 `replay_chunk=replay_chunk` | 见 1 |

而 `replay_chunk` 的定义（`dsa_v41.py:927`）：

```python
replay_chunk = (
    _CED_DECODE_ROLE                      # ← os.environ["V41_CED_ROLE"] == "decode"
    and not getattr(forward_context, "in_profile_run", False)
    and metadata.swa.num_prefills > 0
    and metadata.swa.max_query_len > 1
)
```

⇒ `V41_CED_ROLE != "decode"` 时 **`replay_chunk` 恒为 False**，四处改动全部短路。

**真机验证**（见 `docs/CED-PD-ACCEPTANCE.md`）：P 与 D 用同一份 CED 代码，
144K/1M 四针 **21/21** 通过。本部署形态在此基础上再叠一层：
`deploy/a3-ced-pd/verify_consistency.sh` 会逐文件比对镜像内 md5 与本仓 md5。

## F. 不随包携带的（体积与版权）

| 项 | 原因 |
|---|---|
| 模型权重（273 GB） | 现场准备；`MODEL=<路径>` 传入 |
| `optim/pgo/`（PGO 产物） | 与镜像内 libpython md5 绑定，跨镜像会**静默降级**；本形态不依赖 PGO（`PYTHON_PGO=0`） |
| 语料（`data/hongloumeng.txt`、`data/dihuo.txt`） | 有版权；用 `tools/fetch_corpus.sh` 自行下载（含 sha256 校验） |
