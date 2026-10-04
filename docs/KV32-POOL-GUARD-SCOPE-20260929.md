# KV32 池守卫的作用域修正（2026-09-29）

**改动**：`scripts/serve_a2.sh` 的 `[CED-POOL-GUARD]` → `[KV32-POOL-GUARD]`，
并把它从"所有形态默认 pin"改成"只在会撞 32 位页偏移回绕的形态默认使能"。

**入口**：本文件是 `docs/CED-PD-BLOCK-BOUND-20260925.md`（根因与上界）的配套，
讲的是"**谁**该被钳、**谁**不该被钳"。

---

## 0. 一句话

守卫原先挂在**所有形态共用的底层**（`scripts/serve_a2.sh`）上，且
"未设置 `KV_CACHE_MEMORY_BYTES` ⇒ 直接 pin 到 29076 块"
（`9706454`，2026-09-25 17:31）。于是**非 CED 部署也被强制 pin**，vLLM 因此走

```
v1/worker/gpu_worker.py:474
  "reserved X GiB for KV Cache as specified by kv_cache_memory_bytes,
   skipping memory profiling. This does not respect the gpu_memory_utilization config."
```

即**跳过自动显存 profiling**、`GPU_UTIL` 对 KV 池不再生效。现在只有
CED 角色 / Mooncake PD 默认 pin，其它形态交回自动 profiling，并新增**起服后复核**。

---

## 1. 三种形态与三态开关

判定发生在 `scripts/serve_a2.sh` 的 `[KV32-POOL-GUARD]` 块（`KV_ARGS_EXTRA`
与 `V41_CED_ROLE` 都已在作用域内）：

| `V41_KV32_POOL_GUARD` | 行为 |
| --- | --- |
| `auto`（**默认**） | CED 角色（`V41_CED_ROLE` 非空）**或** `KV_ARGS_EXTRA` 含 `MooncakeHybridConnector` ⇒ pin；其它 ⇒ 不 pin |
| `on` | 无条件 pin（旧行为，逃生用） |
| `off` | 完全不干预：不 pin / 不 clamp / 起服后不复核 |
| 非法值 | 打 WARNING，按 `auto` 处理 |

另有两个**与钳位无关**的规则：

* **显式给值且超过上界 ⇒ 一律钳位并 WARNING**（含非 CED）。
  理由：回绕是**模型级**风险（槽位 3 的页步长 147712 B 是打包布局的性质），
  显式设置不改变物理事实。要完全绕过用 `off`。
* 旧逃生口 `V41_CED_ALLOW_32BIT_OVERFLOW=1` 等价于 `off`
  （但它**只作用于宿主侧** —— 见 §4）。

各形态实际效果：

| 入口 | 改动前 | 改动后 |
| --- | --- | --- |
| `scripts/serve_a3_ced_pd.sh`（CED PD） | pin（role 层有第二份守卫） | pin（收敛到共享层） |
| `scripts/serve_a3_pd.sh`（普通 PD，Mooncake） | pin | pin |
| `scripts/serve_a3_ced_single.sh` | 自设 1 GiB（不受影响） | 同左 |
| A2 单实例 / `a2/scripts/serve_a2_offload.sh` / `run_test.sh` | **pin** | **不 pin**（自动 profiling，`GPU_UTIL` 生效） |

---

## 2. 起服后复核（新增）

放开 pin 之后，非 CED 的池大小由 vLLM 自动 profiling 决定 —— 若它恰好算出
> 29076 块，长上下文会**静默**变成"HTTP 200 + 1 token（EOS）"。
所以 `serve_a2.sh` 在**就绪之后**加了一道复核（`起服必查 ②`）：

* 只在"既没 pin、也没被 clamp、调用方也没显式给值"时执行；
* 数据源：profiling 路径会打印
  `Available KV cache memory: X GiB`（`format_gib = round(b/GiB, 2)`）；
* 取**所有 rank 的最小值** —— vLLM 最终 `num_blocks` 正是各 rank 取 min
  （`v1/core/kv_cache_utils.py`："Change the num_blocks of each rank to the smallest"）；
* 换算：`blocks = ⌊X·2³⁰ / 540928⌋`；`> 29076` ⇒ **拒绝起服**并给出三条处置。

**判据校准**（用仓库内证据回代）：

| 样本 | 日志值 | 换算块数 | 判定 |
| --- | --- | --- | --- |
| A2 历史 profiling | 14.40 GiB | 28,583 | ≤ 29076 ✅（文档记载 28,577，差 7 是 2 位小数的舍入） |
| CED P 侧越界现场 | 15.16 GiB | 30,092 | > 29076 ⛔（文档记载该形态默认 29721 块、profiling 曾算 30080） |

日志只保留 2 位小数 ⇒ 估算误差 ≤ ±10 块；上界两侧的余量分别是 492 与 1004 块，
判据可用。

---

## 3. 离线自检

```bash
bash tools/selftest_kv32_scope.sh
```

覆盖：作用域 10 例（正控）+ 起服后复核 5 例 + **内置负控**
（把 `auto` 默认改回 `on` 复现"所有形态都 pin"，case1 必须不通过）。

* 不占卡、不起容器、不需要镜像（用 `PATH` 前置的 docker 桩走到守卫尾的钩子）；
* 判据绑在脚本自己打印的 `KV32_RESOLVED scope=… pin=… enforce=… pinned=… bytes=…` 上，
  不绑"我传了哪个 env"；
* 手工负控：`SERVE_A2_UNDER_TEST=<别的副本> bash tools/selftest_kv32_scope.sh`。

---

## 4. 一个口径纠正：`V41_CED_ALLOW_32BIT_OVERFLOW` 绕不过连接器硬门

文档与连接器报错文案原先都写"可设 `V41_CED_ALLOW_32BIT_OVERFLOW=1` 绕过"。
**实测·代码**：该变量**没有**进 `scripts/serve_a2.sh` 的 `docker run -e` 白名单
（1375–1397）⇒ 容器内的 `experimental/ced/mooncake_hybrid_connector.py`
读不到它 ⇒ `[CED-32BIT-GUARD]` 那道硬门**不会**被它关掉。

本次按"**文档向**"修正（改文案，不加透传）：一个环境变量同时关掉宿主钳位与连接器
硬门，会让实验性起服静默跑到错误数据上。若将来确实需要关硬门，应另设一个独立变量
并显式透传。

---

## 5. 判据（真机可观测痕迹）

```bash
# 非 CED：不应再有 pin 行，且能看到自动 profiling 的池大小
grep -a '\[KV32\]'            <run>/driver.log        # 期望 …pin=0 ⇒ 不设置 KV_CACHE_MEMORY_BYTES
grep -a 'KV_CACHE_MEMORY_BYTES=' <run>/inner.sh        # 期望 export … KV_CACHE_MEMORY_BYTES= SEED=
grep -aE 'Available KV cache memory|GPU KV cache size' <run>/serve.log
# CED：仍是 pin
grep -a '\[KV32\]'            <run>/driver.log        # 期望 …pin=1 ⇒ 池按 4 GiB 上界 pin：15728022528 B
```

（历史对照样本：非 CED 空值 —— `evidence/ced_metadata_inline_single_variable_20260924/inner.sh:6`；
CED pin 值 —— `logs/ht-20260927-115218/d-decode-18991.inner.sh:6` = `15728022528`。）
