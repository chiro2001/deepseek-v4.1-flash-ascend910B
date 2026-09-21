`Tracks RFC #16375 items [73][75][77]`

> **Draft implementation-issue body — not posted.** Written in the form RFC #16375
> line 3 invites: *"Release targets and owners can be attached to individual
> implementation issues as they are agreed."*
>
> **Proposed issue title:** `[Performance][Graph] DSpark ACLGraph with explicit eager fallback and host-sync boundary`

| Field | Value |
|---|---|
| Target branch | `main` (RFC line 18) |
| Suggested labels | `performance`, `graph` |
| Proposed owner | @chiro2001 — offered, not asserted; maintainers may attach whoever they prefer |
| Evidence root | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B |
| Hardware behind every number below | 8 × Ascend 910C (A3) **and** 8 × Ascend 910B3 (A2), TP8/EP8, single node, W4A8, DSpark S=5 |
| What this issue is **not** | a claim that a graph-captured stack is better than an eager one, and not a claim about anyone else's graph support |
| What it is **no longer** (v2) | a proposal to land DSpark graph capture in `main` — see the re-aim below |

---

## ⚠️ v2 re-aim (2026-09-21) — read this before §2

**A maintainer has publicly ruled out the centrepiece of v1 of this draft.**
On PR **#16285** (2026-09-20T11:38:27Z), reviewing
`vllm_ascend/spec_decode/dspark_proposer.py:76`, **drslark** wrote (verbatim):

> We don't plan to support DSpark graph mode in v1.
>
> Supporting it in v1 would require extensive changes due to the current graph_pad_size
> design, which would add unnecessary implementation and maintenance complexity.
>
> Also, based on the profiling results, DSpark is consistently sync-bound rather than
> graph-bound, so graph support is unlikely to bring meaningful performance benefits here.
>
> If DSpark graph support is needed, please use v2 instead.

and four minutes later (11:41:58Z) added:

> Given the extensive changes required to support DSpark graph mode in v1, as well as the
> significant long-term maintenance burden they would introduce, we cannot accept these
> changes in their current form.

Separately, **pisceskkk** (2026-09-17T01:54:02Z) on the same PR:

> The modifications in `vllm_ascend/attention` is okay for me and I leave some nits. But for
> speculative decoding part, could you adapt these in model runner v2 directly? MRv1 will be
> deprecated in soon.

### What this means for us

1. **Our DSpark-v1 graph-capture work is not a contribution to `main`.** It lives on MRv1,
   which is being deprecated, and the reviewer who would see it has already said no — with a
   reason that matches our own measurements (*"DSpark is consistently sync-bound rather than
   graph-bound"*). Submitting it as a PR or an issue would spend credibility, not buy any.

| Our asset | v1 draft's claim | **v2 (this re-aim)** |
|---|---|---|
| DSpark draft graph gains (A3 36.9→23.9 ms/step, A2 64.8→34.3) | "evidence for [73]" | **Supporting data for their v2 direction only** — offered as a measurement, explicitly *not* as a `main` PR. Also independently consistent with the reviewer's sync-bound diagnosis. |
| Draft-path host-sync accounting (metadata build 20000/20000 samples through a blocking D2H; **118.5 s blocked of 787.6 s**) | not in v1 draft | **The useful part.** It is the number behind "sync-bound", it corroborates our own [77] claim, and it is directly relevant to open PR **#16465** (which removes a blocking D2H on the draft path). Offer as evidence, let them own the fix. |
| Engram lookup host-sync 3.379 → 0.058 ms/step + pointer-stability contract | "evidence for [77]" | **Unchanged — still our strongest [77] evidence.** It sits on our Engram path, not on DSpark. |
| static-kernel silent-disable trap (missing `LOCAL_WORLD_SIZE` ⇒ 4–5 ms/step, ~12%) + compile-cache false positive | "evidence for [75]" | **Unchanged, and the best-value item in this track**: documentation-shaped, independent of any draft-model design, and [75] explicitly asks for "warmup and compilation-cache behavior". |
| comm ∩ compute overlap = 0.000 ms at 32K with the dependency identified | "[73]/[77]" | **Unchanged** (measurement, not implementation). RFC line 104 wants *trace evidence* for overlap items — still missing on our side. |

2. **Revised ask.** Drop item (d) ("DSpark draft capture") from §4's split. The re-aimed
   track is **(a) fallback rules + log observability, (b) static-kernel cache/warmup
   documentation, (c) Engram lookup capture with the pointer contract ([77]), plus (e) sync
   accounting offered to whoever owns #16465.** All four are independent of MRv1 vs v2.

3. **One thing we should NOT do**: re-submit the same design "adapted to v2" without asking
   first. The objection was about `graph_pad_size` maintenance burden and the sync-bound
   profile, not about v1-vs-v2 plumbing.

See `logs/18-20260921-track-c-reaim.md` for the full evidence and the review URLs.

---

## 1. What "done" looks like for these items

| Item | Completion criteria |
|---|---|
| **[73]** | ACLGraph decode support exists with **explicit eager fallback rules** — every fallback trigger is enumerated, tested, and observable in a log — and the graph paths that are supported (decode, then prefill/mixed-batch, piecewise or full) are stated per backend capability |
| **[75]** | `npugraph_ex` compilation and the supported static-kernel optimizations are integrated for attention, mHC, Engram and MoE, with **warmup and compilation-cache behaviour documented**, including what happens when the cache is cold, where the cache lives, and how a user can tell that the optimization silently did not apply |
| **[77]** | The eager/graph boundary for CPU-side Engram lookup and dynamic communication metadata is defined; persistent graph inputs are refreshed before replay; host synchronisation stays **off** the captured computation path — and there is a measurable number attached to the last clause |

A reviewer should be able to answer "which of these triggers a fallback, and how do I see
it in the log?" without reading the source.

## 2. Existing evidence

| Item | Numbers we hold | Scope | One command | Source / repo link |
|---|---|---|---|---|
| **[73]** | DSpark draft forward: A3 same-process paired arms **36.9 → 23.9 / 24.9 ms/step**, single-stream **66.5 → 100.7 / 109.8 tok/s**, acceptance length flat (2.455 → 2.403–2.738); **A2 64.8 → 34.3 ms/step**, **54.7 → 88.7 tok/s** | A3 and A2, 8 dies each, decode | `bash tools/draft_ab_launch.sh <chip> && bash tools/draft_ab_run.sh` | [`CHANGELOG.md §6.2`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/CHANGELOG.md), [`reports/a2-draft-graph-20260920.md`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/a2-draft-graph-20260920.md) |
| **[73]** | Runtime eager fallback without restarting: write `DRAFT_FORCE_EAGER=1` into `/tmp/v41_dspark_flags` and the same session switches arms — this is how the A3 paired numbers above were produced | A3 | `echo DRAFT_FORCE_EAGER=1 > /tmp/v41_dspark_flags` (then compare `ms/step` and spec-decoding metrics in the same session) | [`tools/draft_ab_run.sh`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/tools/draft_ab_run.sh) |
| **[75]** | `npugraph_ex=1` + `STATIC_KERNEL=1` are part of the shipped default configuration. One quantified trap: if `LOCAL_WORLD_SIZE` is missing from `os.environ`, torch_npu **silently** disables static kernels and the identical config costs **4–5 ms/step (~12%)** | A3, same config, different sessions | `grep -ac "static_kernel.py:650" "$LOG"` must print `0` | [`reports/static-kernel-silent-disable-fix.md`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/static-kernel-silent-disable-fix.md) |
| **[75]** | Compile-cache behaviour: the static-kernel output directory is derived from `Path.cwd()`, so a wrong cache mount means **every restart recompiles** (A3 ≈5 min extra, A2 first run 15–20 min) *and* the "cache hit" check reports a false positive | A2 + A3 | `V41`-agnostic: mount the real output dir and check that `static_kernel_cache/*.json` is ≥512 B instead of "directory exists" | [`CHANGELOG.md §4.2`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/CHANGELOG.md) |
| **[77]** | Host-synchronous time on the Engram path **3.379 → 0.058 ms/step** once the lookup is captured; step time **29.5 → 28.4** (C1) and **35.3 → 32.1** (C4); the residual 0.695 ms of device work overlaps because `Graph.replay()` enqueues asynchronously | A3, decode, real weights | `V41_ENGRAM_DEVICE_INDEX=0` vs `auto`, then `grep -h "\[bneck\]" "$LOG" \| tail -n 20` | [`CHANGELOG.md` v8 §0–§1](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/CHANGELOG.md) |
| **[77]** | Boundary rule in use: persistent input buffers, pointer-stable index buffers and per-`(n, n_reqs, block_width)` graph keys may be captured; dynamic collective metadata, changing slot mappings and first-touch allocations must stay outside; every replay validates `data_ptr`+shape of the captured inputs first | A3, decode | `grep -a "\[DEVICE-INDEX\]" "$LOG"` and the pointer-check path in the graph module | [`patches/files/engram_graph.py`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/files/engram_graph.py) |
| **[73]/[77]** | Related measurement: communication ∩ compute overlap is **0.000 ms** at 32K decode, with the cause identified as a data dependency (allReduce consumes the matmul output that the next matmul consumes), not as core selection | A3, 32K decode, 70 steps | method in the report (interval-union of comm and compute ranges from a `PROFILE=1` run) | [`reports/comm-compute-overlap-cannbot.md`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/comm-compute-overlap-cannbot.md) |

## 3. ★ What is still missing

1. **Decode only.** [73] explicitly asks for prefill/mixed-batch and piecewise/full-graph
   modes to follow "according to backend capabilities". We have decode. Prefill and
   mixed-batch graph capture is unbuilt.
2. **Nothing under SP / DCP / PD.** The boundary rules we wrote are validated for one
   single-node TP8/EP8 decode configuration and nowhere else.
3. **A rare, unreproduced failure mode exists in our graph path.** Acceptance length stuck
   at `1.00` with accepted throughput `0.00` (empty output) was observed **once**; six
   follow-up reproduction attempts (~36 measurement points, 10.4 min of continuous load)
   failed, and five mechanistic hypotheses were falsified. This is why the graph path
   ships **off by default** even though the measured gain is large, and why we report the
   failure mode next to the gain rather than after it.
4. **No per-component `npugraph_ex` attribution.** We know it is enabled and we know the
   static-kernel-disable trap costs 4–5 ms/step. We do **not** have separate A/B numbers for
   `npugraph_ex` on attention, mHC, Engram and MoE individually, which is what [75] names.
5. **Warmup and cache behaviour are documented only as "cold start costs more".** No
   compile-time statistics per stage in a form that could go into a support matrix.
6. **No trace evidence for the overlap claim.** RFC line 104 requires *"trace evidence"* for
   overlap items. We have interval-union measurements and a report; we do not have a
   published trace artifact attached to this issue.
7. **Changing request lengths and slot mappings are handled by falling back to eager**,
   not by a graph-safe mechanism. That is a legitimate design choice but it must not be
   presented as "the boundary is solved".
8. **No upstream-style tests for fallback triggers**, which is the one thing [73] makes
   testable by construction.

## 4. How we would like to use this issue

* Reviewable split (v2): (a) explicit fallback rules + log observability as a standalone PR
  (useful even without the captured graph), (b) the static-kernel cache/warmup
  documentation, (c) the Engram lookup capture with its pointer-validation contract,
  (e) the draft-path **sync accounting** offered to whoever owns #16465.
* **(d) the DSpark draft capture is withdrawn** — see the v2 re-aim section above; a
  maintainer has ruled it out for v1 with a reason that matches our own profile.
* (a) and (b) are close to ready and low risk.
* On [75]: if someone already owns `npugraph_ex` enablement for attention/mHC, we would
  rather contribute the static-kernel trap measurement to that item than fork the work.

**Claim discipline for this issue.** No statement here compares our stack to another
implementation. Large gains are reported together with the failure mode and the reason the
feature is default-off; unmeasured sub-items (`npugraph_ex` per-component, prefill graph,
SP/DCP/PD, trace artifact) are listed as missing rather than approximated.
