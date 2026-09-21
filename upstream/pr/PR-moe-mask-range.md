# PR 草稿（待评审后再发）

- **目标仓**：`vllm-project/vllm-ascend`
- **源分支**：`chiro2001:perf/moe-contiguous-expert-map`，HEAD **`206d39c9`**
  （已 push，基于 `upstream/main` = `5fbcfaa9`）
- **标题**：`[Performance][MoE] Compare the routed-expert mask against the local expert range`
- **RFC 归属**：[63] "Enable and tune the V4.1 routed-expert W8A8 paths on A2/A3 …
  including checkpoint packing, scale layouts, **dispatch/combine**, and expert GEMMs" —— **主归属**
  （本 PR 改的正是 dispatch 里的掩码计算，而且**量化无关**：只改比较方式 ⇒ W8A8 同样受益。
  **不要**挂在 W4A8 名下）。
  [65] 只**部分**命中其 "…and **document selection rules**" 子句：本 PR 给出一条可文档化的规则
  （连续线性 expert map ⇒ 范围比较；其余布局 ⇒ 回退查表），但**没有**做 fullmesh_v2 与
  collective 的对比 —— 那句话属于 MoE AllGather 那条线，本 PR 不声称。
  （条目号 = RFC #16375 正文行号；快照 `pr/refs/RFC-16375-body.md`，sha256 `459c6328…`）

---

## 正文（英文，直接可贴）

### What this PR does / why we need it?

`TokenDispatcherWithAllGather` masks the router weights of the tokens that were
not routed to this rank:

```python
mask = expert_map[topk_ids] != -1
topk_weights = topk_weights * mask
```

With the default ("linear") EP placement, `determine_expert_map()` gives each
rank one **contiguous** slice of the global expert space
(`expert_map[e] == e - first_expert_idx` for `e in [first, last)`, `-1`
elsewhere). For that layout the mask is exactly a range comparison, so the
gather — an aclnnIndex `Index` plus the `IndexCheck` that validates the indices
— is unnecessary work on every MoE layer of every step.

This PR adds that range comparison behind
`VLLM_ASCEND_MOE_MASK_RANGE=1` (default `0`):

```python
topk_weights = topk_weights.masked_fill(
    (topk_ids < first_expert_idx) | (topk_ids >= last_expert_idx), 0.0
)
```

The mask itself is **kept**. The `-1` entries in `expanded_row_idx` make
`npu_moe_token_unpermute` read rows that were never written, and only the zeroed
weights suppress them; dropping the mask is not an option.

The fast path is only taken after the map layout has been validated once on the
host (a single `torch.equal` against the canonical linear map, cached per
`(first_expert_idx, last_expert_idx, numel)` and re-validated when the map
storage changes). Everything else keeps the original lookup:

* EPLB — a static `expert_map_path` or dynamic EPLB may rewrite the map while
  the engine runs, so the cached layout must not be trusted;
* any non-contiguous placement (`round_robin`, redundant experts, …).

### Does this PR introduce _any_ user-facing change?

Only an opt-in environment variable, `VLLM_ASCEND_MOE_MASK_RANGE` (registered in
`vllm_ascend/envs.py`, default `0`). Behavior, dtypes and shapes are unchanged:
the range comparison is numerically equivalent to `expert_map[topk_ids] != -1`
for the layout it is enabled for.

### How was this patch tested?

**1. Unit tests** — `tests/ut/ops/test_token_dispatcher.py`:

* the range comparison reproduces `expert_map[topk_ids] != -1` element by element
  for the production shape (384 experts, EP8, 48 local experts, `topk_ids[8, 6]`);
* a non-contiguous (EPLB / round-robin) map falls back to the lookup, and the
  test asserts the two forms really disagree for that map;
* dynamic EPLB and a static `expert_map_path` both fall back;
* the default (variable unset) never evaluates the layout gate;
* the layout validation runs once and is reused across steps, and a re-laid-out
  map invalidates it.

We also verified the tests fail when the implementation is mutated (dropping the
EPLB guard, always taking the range path, or dropping the cache's storage guard).

**2. Operator-level A/B — and an important frame distinction.**

A single Ascend 910 (PCI `19e5:d803`) was used to compare the two forms back to
back in one process, at the production shape (`expert_map[384]`, EP8 ⇒ 48 local
experts, `topk_ids[8, 6]`, ranks 0/3/7). The two frames disagree, so both are
reported.

Because a single process is not enough to quote a magnitude, each row below is
**n = 7 independent processes** (7 separate `python bench_mask_graph.py` runs, each
re-initialising ACL, the NPUGraph and the profiler), 200 replays per arm per rank,
3 ranks per process. `Δ = median(range) − median(lookup)`, negative = the range
comparison is faster:

| tokens | eager Δ (µs), n=7 | **inside an ACLGraph, replay Δ (µs), n=7** |
|---:|---:|---:|
| 8 | **+34.2** median, IQR [+33.4, +35.0], 7+/0− | **−10.7** median, IQR [−12.4, −10.5], 0+/7− |
| 192 | **+35.7** median, IQR [+33.8, +37.7], 7+/0− | **−12.3** median, IQR [−14.9, −11.8], 0+/7− |
| 2048 | **+26.0** median, IQR [+23.3, +27.8], 7+/0− | **−12.9** median, IQR [−14.2, −10.3], 0+/7− |

**The sign is stable, the magnitude is not.** Across the 21 pairings (3 token
counts × 7 processes) the range comparison is slower in eager **21/21** times and
faster inside an ACLGraph **21/21** times; individual runs spread from −8.5 to
−19.5 µs on the graph rows, i.e. up to ~2.3×. Only the sign and the median are
quoted for that reason — a single run should not be used to claim a magnitude.
An EP-range sweep (`--ranks 0` and `--ranks 0..7`, one process each) keeps the same
signs at all six (frame, size) combinations, so the conclusion does not depend on
which ranks are compared.

**Why the sign flips:** the two forms issue a similar number of device kernels —
7 per call for the lookup path (`Index`, `IndexCheck`, `NotEqual`, `Mul`, `Cast`,
`Fill`, `Arange`) versus 6 for the range path (`Less`, `GreaterEqual`, `Cast`×2,
`LogicalOr`, `MaskedFill`) — measured with the torch_npu profiler on the same
card. In **eager** mode the extra host dispatch of the comparison kernels
dominates, so the range form loses. In the deployed configuration this mask runs
**inside an ACLGraph**, where the dispatch is amortised away, and the device-side
kernel count is what remains.

⇒ **Please evaluate this change in the graph frame.** We report the eager number
too because it is the one a naive microbenchmark produces, and quoting it alone
would argue *against* the change.

The deployment-level profile (A3, 8 chips, DeepSeek-V4.1 decode) showed the same
direction: `Index` + `IndexCheck` cost 0.927 ms/step and the replacement 0.414,
≈ **−0.51 ms/step**; GSM8K **100/100** and Vision **23/23** unchanged. Those are
our deployment's numbers (it carries other non-upstream changes) and are given
as context; the single-card numbers above are measured on this branch head.

**3. Reproducers** (single card, `npu:0`), both attached:

* `bench_moe_mask.py` — production shapes, `torch.equal` between both forms for
  several EP ranks, plus the EPLB fallback case.
* `bench_mask_graph.py` — the same A/B **captured in an NPUGraph and replayed**,
  plus the device op counts behind the frame distinction. This is the one whose
  numbers should be compared against a deployment.

> Please add `ready-precise` to run the recommended NPU tests
> (`tests/ut/ops/test_token_dispatcher.py`, `test_moe_comm_method.py`,
> `test_moe_runtime_args.py`, `tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_moe.py`).

### Notes for reviewers

* No overlap with #14933: it touches `TokenDispatcherWithMC2` and the imports,
  this PR touches the `expert_map` mask inside `TokenDispatcherWithAllGather`.
* The change is quantization-agnostic; it is listed under the A2/A3 W8A8 items of
  the V4.1 roadmap rather than under W4A8.
