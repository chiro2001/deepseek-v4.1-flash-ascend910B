# 42 — RoPE 的 int32 回退被消除：index build 提到每次调用一次

**日期**：2026-09-21 14:2x–14:4x CST
**触发**：`logs/41` 的任务二把 int32 两格的回退**归因到 host 侧**，并给出"每方向多 build 一次 index"的
判断；本轮**把这个判断落成代码**，然后重新验证。
**结论**：**PR 里最后一处回退消失**。改后 **32/32 + 5/5 checks 全过**，两格由 **+20.7 / +28.9 µs**
变成 **−17.9 / −17.7 µs**，device kernel 数由 4.00 降到 **3.00 /次调用**；PR 分支 HEAD
**`ed5b928c`**（fork 已更新）。

---

## 1. 改了什么（两处，逻辑等价）

`pr/PR-rope-index-select.md` 的 §5a 原来有两格 ⚠️：int32 的 eager 小尺寸格比 `main` **慢**
（n=192 连续 +20.7 µs、非连续 +28.9 µs，`logs/35`）。`logs/41` §6.2/§6.3 的逐 op 时序把它归到
**host 侧**：device 上 PR 一直**赢 22–28 µs**，但 `_rope_index_1d()` 被 cos / sin **各调一次**，
int32 源因此**每个方向都付一次 `.to(torch.int64)`** 的下发代价。

| 位置 | 改前 | 改后 |
|---|---|---|
| `_rope_gather_rows()` 签名 | `(src, pos_tensor, gather_idx, out)` —— 内部自己调 `_rope_index_1d` | `(src, select_idx, gather_idx, out)` —— **接收已经展平好的 1-D index** |
| 调用点（`use_cache=True`） | cos / sin 各自触发一次 flatten+cast | `select_idx = _rope_index_1d(pos_tensor)` **算一次**，两次 lookup 共用 |
| `use_cache=False` 分支 | 两个 `index_select` 各自 `_rope_index_1d(pos_tensor)` | 同样只算一次 |

两条 1-D / 4-D 路径仍然互斥（`gather_idx is None` 时才用 `select_idx`），
**2-D fallback 分支的行为一字未变**。

---

## 2. 验证（A3 槽位 c1 = die6，同一张 910C 类卡）

被测代码**按路径 import**：`main` = `upstream/main`（`0f9177f5…`）、PR = 改后的分支 head
（`5f1c92d6…`）。

### 2.1 边界矩阵：32/32 通过（与改前同数）

```
SUMMARY  checks=32  pass=32  fail=0
```

含 `n=0` 空批、6 个尺寸、非连续 stride / 倒序 / 重复位置 / int32 / 2-D fallback、
`draft_index=1..5`（10 组只写目标 `spec_row[K-1]`）。原始 JSON：
`logs/raw/42-rope-hoist-edge-a3.json`。

### 2.2 ★ 两格回退变成收益（端到端，中位数，µs）

| 用例 | 改前（`logs/35`） | **改后** |
|---|---:|---:|
| n=192 int32 连续（dflash buffer dtype） | **+20.7** ⚠️ | **−17.9** |
| n=192 int32 非连续 strided | **+28.9** ⚠️ | **−17.7** |
| n=192 int32 连续（n 取自 decode 批） | −144.2 | −190.4 |
| n=4096 int64（prefill 尺寸） | −376.3 | **−390.8** |
| n=0 空批 | −4.0 | −11.1 |

**全部 27 个计时格现在都是负的（PR 更快）**，无回退。

### 2.3 device kernel 数：int32 由 4.00 降到 3.00（每调用）

profiler 相位跑了 **3 次**（1 次全量 + 2 次专项），两次对齐的样本给出同样的数：

| 配置 | `main` | PR（改前） | **PR（改后）** |
|---|---:|---:|---:|
| int64, `use_cache=True` | 6.00 | 2.00 | **2.00** |
| int32, `use_cache=True` | 7.00 | 4.00 | **3.00** |
| int64, `draft_index=3` | 6.00 | 2.00 | **2.00** |

构成上，int32 的 `InplaceCopy_Cast` 由 **6 个/3 步（=2/调用）降到 3 个/3 步（=1/调用）** ——
正是被 hoist 掉的那一次 cast。

> ⚠️ **测量坑（已写进 PR 正文）**：本 harness 的 profiler 窗口**不总是按调用对齐**。
> 三次跑里有一次 `stock|int32` 抓到了"第 4 次调用的 gather 但没有它开头那次 cast"
> （原始 27 行而不是 21 行），于是那一次的 `stock` 被算成 9.00/lookup。
> 本文只引用**两次对齐**的样本；专写的 probe 按构造就是调用对齐的，结论一致。

### 2.4 独立探针（调用对齐，5 组配置）

`probe_gather_kernels_both.py`（`logs/41` 的产物，本机无 torch 故在容器里跑）：

```
  stock  T=1M n=192 1 call/step                    6.00/lookup
  stock  T=8K n=192 1 call/step                   12.00/lookup
  pr     T=1M n=192 1 call/step                    2.00/lookup
  pr     T=8K n=192 1 call/step                    2.00/lookup
  pr     T=1M n=192 5 calls/step, fresh positions  2.00/lookup
```

⇒ 生产表与 8K 表、单发与连续发，**改后一律 2.00 kernels/lookup**。
原始：`logs/raw/42-rope-hoist-probe-a3.json`。

### 2.5 图口径（ACLGraph）也跟着小幅变好

| n | 改前 Δ（µs/call） | **改后 Δ（µs/call）** |
|---:|---:|---:|
| 1 | −12.07 | −12.06 |
| 8 | −17.29 | **−16.68** |
| 192 | −29.70 | −28.28 |
| 2048 | −200.13 | −199.39 |
| **4096** | −383.95 | **−389.29** |

图口径基本不变（host 下发在 capture 里本来就被摊掉），与"这次改的是 host 侧"的判断一致。
原始：`logs/raw/42-rope-hoist-graph-a3.json`。

---

## 3. 分支与文档的同步

| 对象 | 处理 |
|---|---|
| `perf/rope-fused-index-select` | **amend 成单提交**，HEAD `ed5b928c`（基于 `upstream/main` = `5fbcfaa9`），已 force-push（`--force-with-lease`） |
| `perf/rope-index-select-on-16285` | 用同一改动**重建**组合分支 = `4abfa85e`，已 push（旧的 `0121f203` 作废） |
| `pr/PR-rope-index-select.md` | §3b/§3d 换成新数字；§5a 的 ⚠️ 两格改为收益并保留"更早一版曾回退"的记录；§5c/§5d/§5e 同步；正文 HEAD 改为 `ed5b928c` |
| `pr/PR16285-rope-merged-dsv4.py` | 更新为新组合版快照 |
| `pr/PR16285-rope-composition-check.py` | 允许 `BASE` / `MERGED` 用环境变量覆盖（本机无 torch，原来只能在特定机器上跑） |

---

## 4. 【实测】/【推断】/【未确认】

| 项 | 状态 |
|---|---|
| hoist 后 32/32 + 5/5、逐位一致 | 【实测】 |
| 两格 int32 由回退变收益、全部 27 格为负 | 【实测】（单次全量；改前的对照是 `logs/35`，同机同脚本） |
| kernel 数 int32 4.00 → 3.00 | 【实测】（两次对齐样本 + 原始 CSV 计数） |
| 图口径基本不变 | 【实测】 |
| 组合分支与 `#16285` 合并后**行为不变** | **【实测】已补跑**：`BASE`=本分支 head、`MERGED`=重建的组合版，6 组配置**全部 `base==merged` 且都等于表查表结果**，另外组合版独有的 `cached_output_len=12 / 6 positions` 六项语义（返回长度、token 行、pad 行 cos=1/sin=0、buffer 地址保持、未触及行保持填充）全 True ⇒ `ALL CHECKS PASSED`。<br>证据：`logs/raw/42-rope-composition-check-a3.txt`（**原始输出**，未手抄）。<br>⚠️ 第一次尝试确实因三卡被 `T1_realtable` 占满而 500 s 超时；槽位一空即补跑成功 |
| 单元测试 13 个在新 head 上重跑 | **【实测】13 passed / 0 failed**。<br>绕开 conftest 的办法：容器里的 vllm 比分支旧，`tests/ut/conftest.py` 的 `adapt_patch()` 在 `EagleModelMixin.AUX_HIDDEN_STATE_KEY` 上就死了（**环境不匹配，不是测试失败**），所以写了 `pr/run_rope_ut_standalone.py` —— 直接按路径 import 测试模块、自带 `rope_state` / `monkeypatch` 两个 fixture，逐条跑 `Test*` 类。**它不是 CI 的替代**（跳过了 collection、参数化与 conftest 的平台桩），只是让"这个 revision 的测试通过"在 CI 之外也能被检查。<br>证据：`logs/raw/42-rope-ut-standalone-a3.log`（13 条 ok + 汇总行） |
| hoist 对 int64 的影响 | 【推断】在噪声内（改后 int64 各格与改前差异 ≤ 2 µs，正负都有）——与"flatten 是 view"的判断一致 |

---

## 5. 还缺什么

1. **CI 的正式单测**仍要等上游跑（本轮的 13 passed 是独立驱动器跑出来的，见 §4 —— 它不覆盖 collection / 参数化 / conftest 桩）；
2. 组合校验只覆盖**行为等价**，没有重跑性能（组合版的性能不是本 PR 的主张）；
3. `logs/35` 里其余【未确认】项（n>4096、跨 CANN 版本）不受本次改动影响，仍挂着。
