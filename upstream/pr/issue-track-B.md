`Tracks RFC #16375 items [63][65]`

> **Draft implementation-issue body — not posted.** Written in the form RFC #16375
> line 3 invites: *"Release targets and owners can be attached to individual
> implementation issues as they are agreed."*
>
> **Proposed issue title:** `[Performance][MoE] AllGather dispatch when TP=EP + range-compare expert mask`

| Field | Value |
|---|---|
| Target branch | `main` (RFC line 18) |
| Suggested labels | `performance`, `moe` |
| Proposed owner | @chiro2001 — offered, not asserted; maintainers may attach whoever they prefer |
| Evidence root | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B |
| Hardware behind every number below | 8 × Ascend 910C (A3), **TP8 / EP8, single node**, W4A8 |
| Scope warning | Every number here was measured at **EP=8 on one node** and on a **W4A8** checkpoint. The RFC's A2/A3 target is **W8A8**. We do not extrapolate |
| What this issue is **not** | a claim that AllGather is generally the better collective, and not a proposal to change the RFC's quantization matrix |

---

## 1. What "done" looks like for these items

| Item | Completion criteria |
|---|---|
| **[63]** | The routed-expert path for each supported hardware × quantization combination is enabled and tuned end to end — checkpoint packing, scale layouts, dispatch/combine selection, expert GEMMs — with routing weights, output reduction and padded/empty expert assignments validated, and a documented backend fallback. For A2/A3 that means a **W8A8** result; for A5 a W4A8 result |
| **[65]** | `fullmesh_v2` is enabled on the topologies that support it and **compared against the existing collective strategies by prefill/decode batch shape and EP scale**, with the selection rules written down — including the cases where the newer collective is *not* chosen |

An acceptable answer to [65] can be "we tested it at EP=8 and EP=32 on these four batch
shapes and here is the rule". An unacceptable answer is a single number without a rule,
which is exactly the shape of evidence we can currently offer.

## 2. Existing evidence

Every row is a same-session paired measurement with a gate that can be turned off, so a
reviewer can reproduce the A/B on their own node.

| Item | Numbers we hold | Scope | One command | Source / repo link |
|---|---|---|---|---|
| **[63]** dispatch/combine | 128K single-stream **−4.25 ms/step**; 32K **−1.35**; 8K **−1.23**; KV capacity **3,390,000 → 4,160,000 tokens**; output **byte-identical** between arms | A3, 8×910C, TP8/EP8, W4A8, single-stream | `V41_MOE_COMM_ALLGATHER=1 bash scripts/serve_a3.sh` (vs unset) then `MODEL="$MODEL" MODE=full bash scripts/run_test.sh` | [`patches/README.md` §3 (0001)](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/README.md), [`reports/moe-allgather-breakthrough.md`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/moe-allgather-breakthrough.md) |
| **[63]** expert mask | **−0.51 ms/step**; GSM8K 100/100, Vision 23/23; output-equivalent | A3, TP8/EP8 | gate `V41_MOE_MASK_RANGE=1`, then `MODEL="$MODEL" MODE=full bash scripts/run_test.sh` | [`reports/moe-mask-range-verified.md`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/moe-mask-range-verified.md), [`patches/files/token_dispatcher_moemask.py`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/files/token_dispatcher_moemask.py) |
| **[65]** collective comparison | AllGather vs MC2 compared at **one EP scale (EP=8, single node)** over three context lengths; the mechanism is that with TP=EP the MC2 path splits a small token count across ranks, so per-rank work collapses and the scalar overhead is not amortised | A3, decode | `V41_MOE_COMM_ALLGATHER=1` on/off, then `MODEL="$MODEL" MODE=full bash scripts/run_test.sh`; and `bash tools/bench_concurrency.py --base-url http://127.0.0.1:8020 --model deepseek-v41 --concurrency 1,4,8,32,64 --prompt-tokens 1024 --output-tokens 256` | [`patches/files/ascend_forward_context.py`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/files/ascend_forward_context.py) |

Candidate selection rule (offered as a hypothesis to be tested, not as a conclusion):

> When TP = EP and the per-rank token count under the MC2 dispatch path collapses (small
> batch or short decode step), AllGather measured faster on 8×910C. The gate is a single
> environment variable and defaults **off**, so the comparison is a two-line experiment on
> any node.

Related work that must be read before this rule is believed (all upstream, all citeable):

| Reference | Why it matters |
|---|---|
| **#15043** *Support W4A8 fused MoE operator on Ascend 910B* | argues the **opposite** direction on 910B (W4A8 should take the fused MC2 path). Conditions differ from ours; the branch is ~579 commits behind `main` and the function it edits no longer exists on `main`, so it cannot merge as-is — but the reasoning stands and may well be right on that machine class |
| **#14933** *Reuse MC2 expert scales for combine* | same file (`token_dispatcher.py`), different class and region; no code overlap with the mask change |

## 3. ★ What is still missing

1. **No W8A8 measurement at all.** The RFC's matrix (line 11) sets A2/A3 = W8A8. Our entire
   MoE result set is W4A8. The dispatch/combine change is quantization-independent in
   mechanism, but "quantization-independent in mechanism" is an argument, not a measurement,
   and it should not be accepted as one.
2. **EP=8 on a single node is the only EP scale tested.** No EP=16/32, no multi-node, no DP
   composition. The MC2-vs-AllGather crossover is very likely an EP-scale-dependent curve,
   and a crossover curve is exactly what [65] asks for.
3. **No `fullmesh_v2` result.** We have not run it.
4. **No prefill/decode batch-shape sweep.** We have context lengths (8K/32K/128K) at
   single-stream and a concurrency sweep (C1…C64) — neither is the same as the item's
   *"compare by prefill/decode batch shape"*.
5. **Nothing on packing, scale layouts or expert GEMMs.** [63] names those explicitly; we
   touched none of them.
6. **The mask change keeps the invalid-row mask on purpose** (removing it makes `unpermute`
   read unwritten rows). An optional "zero invalid rows" variant exists but is **not
   validated end to end**; it must not be presented as part of this work.
7. **No upstream-style unit tests yet.** The port to current `main` (English commit body,
   `VLLM_ASCEND_*`-style gate naming, `tests/ut/ops/...` coverage) is in progress on a
   separate branch and is not finished.
8. **Empty/padded expert assignments and backend fallback** are covered by our end-to-end
   accuracy gates, not by focused tests.

## 4. How we would like to use this issue

* The useful split is: (a) the mask range-compare (small, self-contained, no collective
  change), (b) the dispatch/combine selection rule (needs the EP-scale sweep before it can
  be stated as a rule at all), (c) a W8A8 rerun of both on an A2/A3 node.
* (c) is the one we cannot do on our own hardware/checkpoint mix. If a W8A8 checkpoint on
  A2/A3 is available to the team already running #16828-style experiments, the comparison
  is a two-arm run of `scripts/run_test.sh` with and without one environment variable.
* If the maintainers already have an EP-scale rule in mind for [65], we would rather
  measure against it than publish a rule derived from a single EP scale.

**Claim discipline for this issue.** No comparison of our implementation against any other
implementation appears here. The AllGather result is stated with the exact conditions it was
measured under, next to a note that an open upstream PR reasons the other way on a different
machine class. Where the RFC names a quantity we have not measured (W8A8, EP scale,
`fullmesh_v2`, batch shapes), it is written as "not measured".
