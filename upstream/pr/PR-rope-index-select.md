# PR 草稿（待评审后再发）

- **目标仓**：`vllm-project/vllm-ascend`
- **源分支**：`chiro2001:perf/rope-fused-index-select`，HEAD **`ed5b928c`**（已 push，基于 `upstream/main` = `5fbcfaa9`）
- **标题**：`[Performance][Attention] Fuse DSA RoPE index selection into a single index_select`
  （CI 强制 `\[(BugFix|Performance|Test|CI|Feature|Doc|Misc|Community|Refactor)\]`，本标题命中 `[Performance]`）
- **前置**：follow-up to **#14428**（已合入；把 index+copy 融成 `torch.gather(..., out=)`）
- **RFC 归属**：[87] "Fuse eligible operations **before the indexer**, including query
  normalization/**RoPE**/layout conversion/quantization and index-key normalization/quantization/cache
  writes …" —— 本 PR 改的 `get_cos_and_sin_dsa()` 正是该条目点名的 **RoPE 前置融合**：
  把 cos/sin 查表从「4-D 广播索引 + gather」换成单次 1-D `index_select`。
  并**部分**命中 [91] "Validate numerical accuracy, **non-contiguous cache strides, empty/padded
  batches**, and prefill/decode shapes for each fusion. Benchmark both individual kernels and the
  full pipeline"：数值逐位一致、单 kernel 计数、eager/ACLGraph 双口径已有；
  **non-contiguous / empty / padded 的形状矩阵**已于 2026-09-21 在 A3 单 die 上补测完毕
  （`pr/rope_edge_cases.py`：**32/32 checks 全过**，另有 ACLGraph 尺寸扫描 5/5），
  **已回填到正文 §5**（原始 JSON 见 `logs/raw/35-rope-edge-cases-*.json`）。
  （条目号 = RFC #16375 正文行号；快照 `pr/refs/RFC-16375-body.md`，sha256 `459c6328…`）
- **状态**：**草稿，数字已填实**。§3 与 §5 是**本分支 head 的单卡实测**（§3：op 计数 45→15、eager 与
  ACLGraph 双口径；§5：RFC [91] 的形状矩阵 / draft / 序列 / 图 / 计数），§2 是部署上的 profile 账目
  （已标注口径差异）。**待你审完再决定是否发。**

---

## 正文（英文，直接可贴）

### What this PR does / why we need it?

Follow-up to #14428, which fused the "index-then-copy" pattern in
`get_cos_and_sin_dsa()` into a single `torch.gather(..., out=...)`. That change
combined the indexing and the write, but it still materialises a 4-D index along
the rotary dim first:

```python
gather_idx = pos_tensor.to(torch.long).reshape(-1, 1, 1, 1).expand(
    num_tokens, 1, 1, full_rope_cos.size(-1)
)
torch.gather(full_rope_cos, 0, gather_idx, out=buf_cos[:num_tokens])
```

It is the `expand()` that makes the index 4-D: every token index is repeated
along the rotary dim so that `torch.gather(..., dim=0)` accepts it.
`torch.index_select(src, 0, index)` takes a plain 1-D index and produces the same
`[num_tokens, 1, 1, rotary_dim]` result, so the broadcast is avoidable.

This PR uses `index_select` in both branches of `get_cos_and_sin_dsa()`:

| Branch | Before | After |
|---|---|---|
| `use_cache=True` | `expand()` + `gather(..., out=)` | one `index_select(..., out=)` |
| `use_cache=False` | `full_rope_cos[pos_tensor]` (advanced indexing) | one `index_select` |

The 4-D `gather_idx` path is **kept as a fallback** for position tensors that are
not 1-D, because flattening those would change the result shape.

A note on the cast, to keep the claim precise: the `.to(torch.long)` inside the
old expression is a no-op for the positions that every in-tree caller passes
(the attention call sites slice `common_attn_metadata.positions[...]` and call
`.long()`; the model runner's positions buffer is created with
`dtype=torch.int64`). This PR therefore does **not** claim any cast removal — the
saving is the broadcast, plus the `Index`/`IndexCheck` pair that advanced
indexing costs in the uncached branch.

### Does this PR introduce _any_ user-facing change?

No. Values, dtypes and shapes are unchanged. There is no new configuration, no
new dependency, and no behavior change for non-1-D positions. The returned
tensors still come from the same pre-allocated runtime buffers, so the stable
addresses that graph capture/replay depends on are preserved.

### How was this patch tested?

**1. Unit tests (new, `tests/ut/ops/test_rope_proxy.py`)**

```bash
python -m pytest tests/ut/ops/test_rope_proxy.py -v
```

These are pure tensor-math tests and need no accelerator. They compare the new
lookup against *both* earlier formulations bit-for-bit (`torch.equal`), and cover
the four paths this patch touches: `use_cache=True`, `use_cache=False`,
`draft_index` (speculative buffers) and the non-1-D fallback. They assert
behaviour only — no torch op is patched or mocked — and check that the lookup
still writes into the pre-allocated buffers without disturbing the rows beyond
`num_tokens`.

Local result: `13 passed` (12 new tests plus the pre-existing equivalence test).
These were re-run on the current head with a small standalone driver, because
the NPU container we use for the benchmarks ships a vLLM older than this branch
and `tests/ut/conftest.py` cannot complete its platform stubs there; the driver
imports the test module directly and supplies the two fixtures it needs. It
does not replace CI (no collection, no parametrisation, no conftest stubs).

**2. Op-level A/B profile (existing evidence, collected on our deployment)**

> Provenance: these numbers come from our own DeepSeek-V4.1 W4A8 deployment
> (8×910C) with **additional non-upstream optimizations enabled**. The absolute
> wall-clock is therefore environment-specific; the environment-independent part
> is the operator-level event accounting below. A single-card re-measurement on
> this branch head is in **3**, and the shape/stride/draft matrix is in **5**.

A/B on the same session, 128K prompt, `V41_ROPE_IDXSEL=0` (A) vs `=1` (B), both
arms at 7912 anchors:

| op | A count | B count | Δ | A dur | B dur | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `Index` | 6756 | 2652 | **−4104** | 136.5 | 32.1 | −104.4 |
| `IndexCheck` | 7124 | 3020 | **−4104** | 40.8 | 15.3 | −25.5 |
| `GatherElementsV2` | 3322 | 802 | **−2520** | 49.1 | 28.1 | −21.0 |
| `BroadcastTo` | 8104 | 5584 | **−2520** | 21.7 | 15.6 | −6.1 |
| **`GatherV3`** (new) | 0 | 6624 | **+6624** | 0 | 36.6 | +36.6 |
| `Cast` | 60200 | 60992 | +792 | 210.4 | 207.5 | −2.9 |
| **net** | | | | | | **−123.3 ms / pass** |

The two removed chains account for exactly the new `GatherV3` count
(4104 + 2520 = 6624), i.e. each replaced chain — `Index` + `IndexCheck` on the
uncached path, `BroadcastTo` + `GatherElementsV2` on the cached one — became one
`index_select`. The `Cast` row is listed for completeness and is flat, which is
consistent with `.to(torch.long)` being a no-op on int64 positions.

**3. Measurement on this branch head: a single card, both frames.**

The numbers above come from a deployment that also carries other non-upstream
optimizations, so they are context rather than proof. To check the branch head
itself, the same lookup was measured on a single Ascend 910 (PCI `19e5:d803`)
with production shapes — `full_rope_*` `[1048576, 1, 1, 64]` fp32 and a
`[4096, 1, 1, 64]` runtime buffer — comparing `main`'s form against this PR's
form back to back in one process.

**3a. Device op counts** (torch_npu profiler, 15 lookups per arm) — the mechanism:

| kernel | `main` | this PR |
|---|---:|---:|
| `aclnnGather_BroadcastToAiCore_BroadcastTo` | 15 | — |
| `aclnnGather_CastAiCore_Cast` | 15 | — |
| `aclnnGather_GatherElementsV2_GatherElementsV2` | 15 | — |
| `aclnnIndexSelect_GatherV3AiCore_GatherV3` | — | 15 |
| **total** | **45** | **15** |

⇒ **three kernels per lookup become one**, i.e. `BroadcastTo` and `Cast` are
eliminated and the gather itself becomes a single `GatherV3`.

**3b. Wall clock, in both frames.** The eager and captured numbers are reported
together on purpose: for some changes the eager frame is misleading, because it
charges per-op host dispatch while the deployed code runs inside an ACLGraph.
Here both frames agree in sign.

| tokens | eager Δ (µs) | **ACLGraph replay Δ (µs)** |
|---:|---:|---:|
| 1 | −8.9 | **−12.1** |
| 8 | −8.0 | **−16.7** |
| 192 | −28.3 | **−28.3** |
| 2048 | −200.9 | **−199.4** |
| 4096 | −390.8 | **−389.3** |

(negative = this PR is faster; median of 50 (eager) / 60 (graph) iterations, 40
back-to-back calls inside one capture for the graph rows. The saving survives
capture because it removes two *device* kernels, not host dispatch.)

**3d. Kernel count per lookup, and the int32 variant** (same single card, both
revisions driven in one process; 3 profiler steps × N calls, so the count is
call-aligned):

| positions | `main` | this PR |
|---|---:|---:|
| int64 (what every in-tree caller passes) | 6.00 | **2.00** |
| int32 (the DFlash positions buffer dtype) | 7.00 | **3.00** |

The int32 row is 7 → 3 rather than 6 → 2 because an int32 source needs one
`.to(torch.int64)` per call, and this PR builds the flattened index **once per
call** and hands the same tensor to the cos and the sin lookup, so that cast is
paid once instead of once per direction.

**4. Bit-exactness** — on this branch head, five token counts (1, 8, 32, 192,
2048) × both branches (`use_cache=True` with `out=`, and `use_cache=False`)
report `torch.equal == True` and `max_abs_diff = 0.000e+00`. The 21 equivalence
checks from the deployment are consistent with this.

**5. The RFC's shape matrix: stride, empty/padded, draft, sequence, and the captured frame.**

RFC [91] asks for "non-contiguous cache strides, empty/padded batches, and
prefill/decode shapes ... both individual kernels and the full pipeline". The
measurements above are single call sites; the matrix below is the systematic
sweep. `pr/rope_edge_cases.py` drives the **real `get_cos_and_sin_dsa()` from both
revisions** — the module files exported from git at the merge base (`c173a64a`,
`sha256 0f9177f5…`) and at this branch head (`d4167f52`, `sha256 982bb28d…`) —
loaded by path and handed identical module state (one shared
`[1048576, 1, 1, 64]` fp32 table, separate output buffers). So this compares
code against code, including the `pos_tensor.dim() == 1` branches, not a
paraphrase of them. Eager timings are the median of 50 reps; every "identical"
below is `torch.equal == True` with `max_abs_diff = 0.000e+00`, never a
tolerance. **32/32 checks pass, 0 fail** (a second harness adds **5/5** for the
graph sweep in 5d), and the raw JSON is attached as
`logs/raw/42-rope-hoist-edge-a3.json` + `42-rope-hoist-graph-a3.json`.

**5a. Shapes, strides, row order, dtype** (eager). "≡ `main`" = the two revisions
agree; "≡ `full[pos]`" = the output also equals the pre-#14428 semantics
(`full_rope[pos]`, advanced indexing); "tail" = rows `≥ n` in the pre-allocated
buffer were not disturbed, i.e. the padded-batch property; "buf" = the write
still lands in the same pre-allocated buffer. Both branches (`use_cache=True`
and the allocating `use_cache=False`) are checked per row.

| positions | ≡ `main` | ≡ `full[pos]` | shape | tail | buf | eager Δ (µs) |
|---|:--:|:--:|:--:|:--:|:--:|---:|
| **n = 0 (empty batch)** | ✅ | ✅ | `(0,1,1,64)` | ✅ | ✅ | **−11.1** |
| n = 1 | ✅ | ✅ | `(1,1,1,64)` | ✅ | ✅ | −8.9 |
| n = 8 | ✅ | ✅ | `(8,1,1,64)` | ✅ | ✅ | −8.0 |
| n = 192 | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −28.3 |
| n = 2048 | ✅ | ✅ | `(2048,1,1,64)` | ✅ | ✅ | −200.9 |
| **n = 4096 (prefill = `max_num_batched_tokens`)** | ✅ | ✅ | `(4096,1,1,64)` | ✅ | ✅ | **−390.8** |
| int32, n = 192, contiguous (dflash buffer dtype) | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −17.9 |
| int32, n = 2048, contiguous (dflash dtype, n from the decode batch) | ✅ | ✅ | `(2048,1,1,64)` | ✅ | ✅ | −190.4 |
| **int64, n = 192, non-contiguous strided view (stride 2)** | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −37.1 |
| int64, n = 192, descending row order (`flip`) | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −32.5 |
| int64, n = 192, repeated/duplicate positions (4 distinct rows) | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −31.8 |
| int32, n = 192, non-contiguous strided (sliced after the cast) | ✅ | ✅ | `(192,1,1,64)` | ✅ | ✅ | −17.7 |
| int64, n = 192, 2-D `[n,1]` | ✅ | ✅ (flattened) | `(192,1,1,64)` | — | — | fallback branch |
| int64, n = 192, 2-D `[n,1]` strided (`as_strided`) | ✅ | ✅ (flattened) | `(192,1,1,64)` | — | — | fallback branch |
| int64, n = 192, 2-D `[n,1]` transposed view | ✅ | ✅ (flattened) | `(192,1,1,64)` | — | — | fallback branch |
| int64, n = 32, 2-D `[2,16]` (expand-incompatible) | ✅ both raise `RuntimeError` | — | — | — | — | — |

Two notes on 5a. The three 2-D rows are the cases this patch deliberately leaves
on the 4-D `gather` fallback, so "identical" there means *behaviour unchanged*,
not "the new path is faster" — and the last row confirms the PR does not change
the pre-existing error for a shape the old formulation also rejected. Second, an
earlier revision of this patch made the two int32 rows **slower** (+20.7 / +28.9
µs) for a reason worth recording: the flattened index was built inside the cos
and the sin lookup separately, so an int32 source paid two `.to(torch.int64)`
casts per call where the old 4-D formulation paid one. Two independent
single-card runs reproduced the sign, and a per-op timing decomposition put the
cost on the host side (the device-side lookup was already 22–28 µs *faster*, with
2–3 fewer kernels). Building the flattened index once per call removes it: the
rows are now −17.9 / −17.7 µs and the kernel count is 3.00 per call instead of
4.00. No row in this table is a regression.

**5b. `draft_index = 1..5`** (speculative decode). Each check also asserts that
only `spec_runtime_buffer[cfg][group][K-1]` was written, that the *other* spec
rows were left untouched, and that the plain runtime buffer was not aliased.

| K | n = 8 Δ (µs) | n = 192 Δ (µs) | bit-exact | correct spec row | other rows untouched |
|---:|---:|---:|:--:|:--:|:--:|
| 1 | −12.6 | −31.1 | ✅ | ✅ | ✅ |
| 2 | −11.4 | −29.2 | ✅ | ✅ | ✅ |
| 3 | −12.5 | −28.7 | ✅ | ✅ | ✅ |
| 4 | −13.0 | −30.1 | ✅ | ✅ | ✅ |
| 5 | −11.4 | −28.7 | ✅ | ✅ | ✅ |

**5c. Real call sequence.** Back-to-back calls on one positions tensor (the
40-layer burst a production step looks like), then a true spec step (1 main + 4
draft lookups):

| case | `main` (µs) | this PR (µs) | Δ |
|---|---:|---:|---:|
| warm single call, n = 192 | 104.18 | 76.10 | −28.1 |
| **40-layer burst, n = 192 (per call)** | 59.98 | 53.81 | **−6.2** |
| **one spec step (1 main + 4 draft), total** | 340.49 | 288.58 | **−51.9** |
| same, per lookup | 68.10 | 57.72 | −10.4 |

The spec step also has a structural check: the main buffer and spec rows 0..3
all equal `full_rope[pos]`, and the unused row 4 stays untouched — in both
revisions.

**5d. ACLGraph replay, swept over the token count.** Captured 40-layer burst,
replay time divided by the 40 lookups, median of 30 replays:

| n | `main` (µs/call) | this PR (µs/call) | **Δ (µs/call)** | bit-exact after replay |
|---:|---:|---:|---:|:--:|
| 1 | 19.93 | 7.86 | **−12.1** | ✅ |
| 8 | 24.57 | 7.89 | **−16.7** | ✅ |
| 192 | 44.93 | 16.64 | **−28.3** | ✅ |
| 2048 | 218.63 | 19.25 | **−199.4** | ✅ |
| **4096** | **411.61** | **22.32** | **−389.3** | ✅ |

The edge-case run's own graph phase measures n = 192 independently as
48.69 → 18.80 (−29.9 µs/call), which agrees with the swept value above to within
run-to-run noise. Bit-exactness after capture+replay is asserted against
`full_rope[pos]`, so the captured frame is checked to produce the same values,
not just the same timing.

**5e. Kernel counts per `get_cos_and_sin_dsa()` call** (torch_npu profiler,
production `[1048576, …]` table, 3 active steps). One call = one cos lookup plus
one sin lookup:

| configuration | `main` | this PR | composition |
|---|---:|---:|---|
| int64, `use_cache=True` | **6.00** | **2.00** | `main`: `BroadcastTo` 6 + `Cast` 6 + `GatherElementsV2` 6 → PR: `IndexSelect_GatherV3` 6 |
| int32, `use_cache=True` | 7.00 | **3.00** | `main`: the same three + `InplaceCopy_Cast` 3 → PR: `IndexSelect_GatherV3` 6 + `InplaceCopy_Cast` 3 |
| int64, `draft_index=3` | 6.00 | 2.00 | identical to the int64 row — the draft path adds no kernel |

i.e. **3 kernels per direction become 1**, and 6 per call become 2. The int32
row is where the cast shows up: an int32 source needs one `.to(torch.int64)`,
and because the flattened index is built once per call that cast is paid once
(+1), exactly as the old code did — the extra direction-local cast that an
earlier revision paid is what produced the two ⚠️ rows in 5a. Why the old path costs *three* kernels
per direction is confirmed separately: at the production table size it is
always `BroadcastTo` + `Cast` + `GatherElementsV2`, independent of `n` and of
whether positions are rebuilt per call (five probe configurations, see the
attached probe log); at a small (8K) table `main` is *worse* (12 per call, because 3 extra
`Transpose` per direction appear), which is another reason the smaller-table
numbers in §3a and here are quoted for the production table only.

> Measurement note, because it bit us once: the harness derives "kernels per
> call" by dividing an active-step window by the number of steps, and that window
> is not always call-aligned. Three repeated runs of the profiler phase gave
> `main`/PR = 6.00/2.00 for int64 and 7.00/3.00 for int32 twice, and once a
> drifted window that caught the gathers of a fourth call without its initial
> cast (27 raw rows instead of 21). The table above quotes the two aligned runs;
> the dedicated probe is call-aligned by construction and agrees.

### Overlap with #16285

`#16285 [Feature] Support DSpark ACLGraph and DSA_CP` edits the same function: it
adds a `cached_output_len` argument and reworks how the two cache branches decide
their output length. It does **not** change how rows are selected, so the two
changes are logically independent — but they do share the four
`torch.gather(..., out=...)` call sites, so they will conflict textually and
reviewers will probably want to read them together.

Whichever lands second only needs a rebase. I checked that direction locally: with
#16285's hunks applied first, re-applying this commit conflicts in a single place,
and the resolution is to keep their `output_len` / padding / `[:output_len]`
logic while routing the four gathers through `_rope_gather_rows(...)`. The
composed file is a pure superset of this branch (diffing it against this branch
yields exactly #16285's own patch), it passes the new unit tests, and it is
`torch.equal` to this branch on every path across six configurations — so if this
PR lands first, that is all the follow-up work #16285 needs.

### A note on CI

This PR touches `vllm_ascend/ops/rope_dsv4.py`, which the coverage-based test
selector maps to `tests/ut/ops/test_rope_proxy.py` and
`tests/ut/attention/test_dsa_v1.py`. As an outside contributor I cannot add
labels, so **could a maintainer add `ready-precise` once the CPU jobs are green**,
so that the recommended NPU tests actually run? Thanks!

---

## 发送前的 checklist

- [x] 单测已并入，并在当前 head 上重跑 **13 passed / 0 failed**（用独立驱动器绕开容器里
      损坏的 conftest，见正文 §1 末段；CI 仍是正式口径）
- [x] `ruff check` / `ruff format --check` / `codespell` 全过（ruff 0.14.0）
- [x] commit 带 `Signed-off-by:`（DCO），标题用上游格式 `[Performance] ...`
- [x] 分支基于最新 `upstream/main`（`c173a64a`），已 push 到我们的 fork（`d4167f52`）
- [x] 正文里没有我们自己的发布包 / 项目名 / 广告
- [x] **单卡实测已填进 §3**：op 计数 **45 → 15**（3→1 kernel/次）、
      eager **−25 ~ −139 µs**、**ACLGraph −6 ~ −128 µs**（双口径同号）、
      逐位一致 `max_abs_diff = 0.000e+00`（5 尺寸 × 2 路径）
- [x] **形状矩阵已填进 §5**（RFC [91]）：空批 n=0、n=1/8/192/2048/4096、非连续 stride、
      倒序、重复位置、int32、2-D fallback、draft_index=1..5、40 层 burst、spec step、
      ACLGraph 五尺寸（含 **n=4096 −384 µs**）、三种 profile 计数；
      **32/32 + 5/5 checks 全过**。⚠️ 更早一版这里有两格 int32 eager **回退**（+20.7 / +28.9 µs），
      已在 `ed5b928c` 里**修掉**（index 每次调用只 build 一次）⇒ 现在是 **−17.9 / −17.7 µs**，
      **27 个计时格全为负**。§5a 保留了这个"曾经回退"的记录（见该节末尾的说明）
- [ ] 提 PR 后请 maintainer 打 `ready-precise`（正文末尾已写）

## ⚠️ 已知边界（必须在 PR 里说明，不能藏）

1. **§2 的 profile 数字来自我们的部署**（含其他非上游优化）—— 已在正文显式标注为"context rather
   than proof"；**§3 与 §5 才是本分支 head 的单卡实测**（均已补齐）。
   ⚠️ §3 的单卡是 **910C 类（PCI `19e5:d803`）**，驱动 25.5.5 / CANN 9.1.0；
   与 A2（`d802`）不是同一机型。§5 的矩阵跑在 **A3 的 `Ascend910_9382`** 上
   （torch_npu 2.10.0.post4），与 §3 又不是同一台机器。
2. **单测在 CPU 与真机两个环境都跑过**：CPU harness 上是 13 passed（`torch 2.14.0+cpu` +
   `torch_npu` 桩）；当前 head 又在 **A3 的 NPU 容器里**用 `pr/run_rope_ut_standalone.py`
   重跑了一次，同样 **13 passed / 0 failed**。⚠️ 该驱动器**跳过了 conftest 的平台桩、
   collection 与参数化**（容器里的 vLLM 比本分支旧，conftest 会在 `adapt_patch()` 处失败），
   所以**正式口径仍是上游 CI**。
3. 只覆盖 `rope_dsv4.py` 的 `get_cos_and_sin_dsa()`；
   `full_rope_*[pos]` 这类模式如果还在别处出现，**本 PR 不涉**（留给后续）。
4. 与在途 **#16285** 改同一个函数（见附 A）：文本上会撞、逻辑上不撞。
   若它先合入，按附 A 的做法 rebase 即可；若**我们先合入**，正文
   "Overlap with #16285" 一节已经写清它要做的四行改动。
5. #16285 的兄弟 PR **#16925**（V4.1 + Engram host offload）用的是
   `aclrtHostRegisterV2(MAPPED|PINNED)` 路径，与我们记录的 pinned 反例同形，
   但那**只是一条可检验假设**（我们没在他们的实现上复现过）——
   详见 `PR16285-ROPE-OVERLAP-ANALYSIS.md` 的附录。**与本 PR 无关，别写进正文**，
   留给 Engram 那条线去沟通。
6. ~~§5a 里有两格 PR 更慢~~ —— **已不再是问题**。
   更早一版确实有两格 int32 eager 回退（n=192 连续 +20.7 µs、非连续 +28.9 µs），
   根因是 `_rope_index_1d()` 被 cos 与 sin **各调一次**（int32 源要各做一次 cast）。
   `ed5b928c` 把**展平后的 index 提到每次调用只 build 一次、两个方向共用**，
   于是这两格变成 **−17.9 / −17.7 µs**；`use_cache=False` 那一路也一并只算一次。
   **证据**：`logs/42-20260921-rope-index-hoist.md`（改前 32/32 → 改后 32/32 + 5/5，
   27 个计时格全为负，device kernel 数 int32 由 4.00 降到 3.00）。
   §5a 仍保留"曾经回退"的段落（并写明原因），因为那是这个 PR 的设计理由之一 ——
   **不要删，但也不要再把它当成现存缺陷。**

---

## 附 A：#16285 的交互（评审备注，不必贴进 PR）

`#16285 [Feature] Support DSpark ACLGraph and DSA_CP`（open，`mergeable_state=dirty`）
也改了 `get_cos_and_sin_dsa()`：

* 它加的是 **输出长度语义**：新参数 `cached_output_len`、`output_len` 校验、
  超出 `num_tokens` 的行填 `cos=1 / sin=0`、返回 `buf[:output_len]`，
  并把 draft 分支的 `buf_cos[draft_index-1]` 提成局部变量。
* 它**没有动索引方式** —— 4-D `gather_idx` 与两处 `torch.gather` 只是被
  重新排版（多行参数）并换成局部变量。

⇒ **逻辑不冲突（已独立验证），但 patch 会撞**（同一片 4-D 索引构造 + 四处
`torch.gather` 调用点）。本地实测：把它的 rope hunk 单独 apply 到最新 main，
再 cherry-pick 我们的 commit，冲突**只有 `rope_dsv4.py` 的
`if draft_index is None:` 这一段**（即那四处 gather 调用点）；
解法 = 保留它的 `output_len` / `fill_` / 返回切片 + 用我们的
`_rope_gather_rows(...)` 替换四处 `torch.gather`。

合并后的文件与我们的文件逐行对比，差异**恰好等于它自己的 patch**（纯增量；
见 `pr/PR16285-rope-resolution.patch`）。★ **而且这不是纸上推演**：重建组合分支后
用 `pr/PR16285-rope-composition-check.py` 在同一进程里驱动两个模块，结论是
`ALL CHECKS PASSED` ——

* 6 组配置（`use_cache` 两种 × `draft_index` × int32 × 2-D fallback）下，
  base 与合并版输出 `torch.equal` 全等，且都等于表查表结果；
* 合并版新增的 `cached_output_len` 语义（返回长度、token 行、pad 行 `cos=1/sin=0`、
  缓冲区地址稳定、未触及行保持填充）六项全部成立。

组合分支已经推到 fork（`4abfa85e`），与 `perf/rope-fused-index-select`
（`ed5b928c`）配对，随时可重跑。

## 附 B：#16285 的另一层背景（给协调者）

`#16285` 的 base 是 `fe167d93`，**落后最新 main 53 个提交**，而 main 在
`9dc67045` 里 revert 了 V4.1 框架支持（#16544 —— 正是要重新合入的 #16925）。
所以它的 `mergeable_state=dirty` 主要来自这里：把它 merge 进最新 main，
冲突文件只有
`tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py`
与 `vllm_ascend/spec_decode/dspark_proposer.py`（`rope_dsv4.py` 能自动合并；
那个 revert 不会因为合并它而复活）。⇒ 它合入前必然要自己 rebase 一次，
届时我们按附 A 跟一下即可，成本约四行冲突解决。
