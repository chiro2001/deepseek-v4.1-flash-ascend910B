# Draft comment for RFC #16375

> **Draft — not posted.** One copy-pasteable comment for
> https://github.com/vllm-project/vllm-ascend/issues/16375
>
> Form: item reference + numbers + one reproducible command, per item, then one closing
> sentence. Nothing here claims an item is complete; RFC line 104 defines completion as
> merged + NPU-tested + documented. Sources are in
> [`deepseek-v4.1-flash-ascend910B`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B);
> all measurements are on 8 × 910C (A3) or 8 × 910B3 (A2), TP8/EP8, single node, W4A8.

---

Thanks for writing this roadmap — the item structure makes it easy to attach evidence to
specific lines instead of arguing about whole implementations. Below is what we can
contribute per item: a measurement, its scope, and a command that reproduces it. Where an
item asks for something we have not measured, it says "not measured" rather than offering a
nearby number.

## Engram memory management — [46] [47] [48] [49] [50]

**[46]** *"Support Engram CPU offload with CPU-resident embedding tables, bounded
pinned-memory staging, batched lookups, and asynchronous H2D prefetch overlapped with model
computation."*

* We have CPU-resident INT8 tables (206.0 GiB, 4 shards) with batched lookups, and we take a
  different branch: the table stays in host DRAM and **device operators index it directly**
  (`aclrtHostRegister(..., MAPPED)`), so the H2D leg is removed rather than overlapped.
* Removing that path moved decode from **35.3 → 32.1 ms/step** at concurrency 4 and
  **29.5 → 28.4 ms/step** at concurrency 1 (A3, decode, same session).
* **Not done on our side:** bounded pinned staging + async prefetch. On `host_mem_pool=0`
  machines (910B3-class, PCI `19e5:d802`) device-side lookup is unavailable, and that is
  precisely where the mechanism in this item is needed.

```
du -sBG --apparent-size "$MODEL"/engram_int8/* | sort -n      # 11.4 / 91.6 / 11.4 / 91.6 GiB
```

**[47]** *"…measure table footprint, lookup latency, transfer volume, and NUMA/bandwidth
sensitivity under realistic concurrency."*

We measured **4 of the 5** named quantities (A3, 8 ranks):

* footprint: **206.0 GiB** (INT8 weights + FP32 group-32 scales, 2 layers);
* lookup latency: hash **0.427 → 0.076 ms**, plan **0.261 → 0.068 ms** (numba JIT);
* transfer volume, host-lookup path, per rank: `d2h` **0.19–3.41 ms**, `route`
  **1.34–2.89 ms** (TP0 largest — rank-0 role cost, not a fault);
* realistic concurrency: C1…C64 sweep, 7 levels × 2 repeats, `ok=64/64`, service alive.
* machine sensitivity we do have: A3 `host_mem_pool=1` (HCCS) vs A2 `host_mem_pool=0`
  (PCIe); **NUMA/bandwidth sensitivity as such is not measured**.
* **registration cost, measured on a single card** (four sizes spanning 512×, three
  repetitions, anonymous *and* file-backed mappings):

  | block | 8 MiB | 128 MiB | 1 GiB | 4 GiB |
  |---|---:|---:|---:|---:|
  | ms/MiB (anonymous) | 11.08 | 11.38 | 11.46 | 11.45 |
  | ms/MiB (file-backed) | 11.27 | 11.63 | 11.49 | 11.84 |

  A 512× size step costs 529× the time, i.e. **linear** on that stack: **≈11.4 ms/MiB
  ⇒ ≈40 minutes for a 206 GiB table per rank**.

  **We then re-measured this on A3 hardware and the 40-minute figure did not survive.**
  With a *materialised* file (100% blocks allocated) and an idle chip on driver 26.1.1:
  1024 MiB → 799.7 ms, 4096 MiB → 3119.6 ms, 8192 MiB → 4839.0 ms, i.e.
  **0.59–0.78 ms/MiB ⇒ 206 GiB ≈ 2.0–2.5 minutes per rank** — which reconciles with our own
  A3 (133 s) and A2 (149.9 s) bring-up records. **The 18× outlier is the software stack of
  that one container (driver 25.5.5), not the hardware.**

  Two traps worth passing on, because we fell into both:
  * **A "file-backed" number is only as real as the file.** Our first cheap file result
    (0.009 ms/MiB) came from a file made with `ftruncate` — **0 blocks allocated**, i.e. no
    physical pages to register. Writing real bytes moved it to 0.591 ms/MiB, a 65×
    difference. Check `st_blocks`. (This also withdraws our earlier "the memory kind does
    not matter" claim, which was drawn from that artefact.)
  * **the `host_mem_pool` procfs flag reads `1` on that container *and* on A3** despite the
    18× cost difference, and A2 records 0.71 ms/MiB on the `host_mem_pool=0` slow path ⇒
    the flag does not predict cost. Measure it.
  If a bring-up ever looks like it is hanging, a one-minute 1 GiB registration predicts
  the full-table time well enough to tell "slow but bounded" from "stuck".

```
V41_ENGRAM_JIT=1 NUMBA_CACHE_DIR=./cache/numba bash scripts/serve_a3.sh
grep -h "\[bneck\]" "$LOG" | tail -n 20        # d2h / hash / route / plan per step, per rank
```

```
python bench_host_register_scale.py --sizes-mib 8,128,256 --reps 1     # <10 s, predicts the curve
python bench_host_register_scale.py --sizes-mib 8,128,1024,4096 --memory both
```

**[48]** *"…Distinguish node-level table sharding from model TP, and define lookup routing,
result redistribution, and projection reduction."*

* Our measured design keeps **one full table per rank** and indexes it on device; node-level
  sharding is deliberately not used, which is itself a datapoint for this item.
* The sharded alternative was measured and rejected: `all_to_all` + `broadcast` ≈
  **0.5 ms/step**, net gain ceiling ≈ **6%**.
* **Not done:** a row/feature ownership split with projection reduction. We have no result
  for a deployment that cannot afford 206 GiB of resident host DRAM per node.

```
V41_ENGRAM_LOCAL_OWNER=fast bash scripts/serve_a3.sh    # vs the sharded arm in reports/engram-host-breakdown.md §2
```

**[49]** *"Eliminate unnecessary duplicate queries across TP ranks and batch queries across
Engram layers…"*

* Metadata `all_gather` and ids `all_to_all` are removed from the decode path; with the
  lookup inside the captured graph, per-step host-synchronous time is **3.379 → 0.058 ms**
  and `route` goes **2.462 → 0.058 ms** (the residual 0.695 ms of device work overlaps).
* **Not measured / not done:** empty-query ranks, collective ordering, and batching queries
  across layers (layers are still called per step).

```
# same session, two arms
V41_ENGRAM_DEVICE_INDEX=0 … bash scripts/serve_a3.sh   # then: grep -h "\[bneck\]" "$LOG" | tail
V41_ENGRAM_DEVICE_INDEX=auto … bash scripts/serve_a3.sh
```

**[50]** *"Validate Engram CPU offload and TP with SP, DCP, PD, and graph replay…"*

* graph replay: **yes** — one ACLGraph per batch shape, zero-copy capture on the model's own
  buffers, `data_ptr`+shape validated before every replay; shipped default on A3-class.
* **SP, DCP, PD: not covered.** Token history, padding masks and transfer-event lifetimes
  under those configurations are untested on our side.

```
V41_ENGRAM_DEVICE_INDEX=auto bash scripts/serve_a3.sh && grep -a "\[DEVICE-INDEX\]" "$LOG"
```

## MoE — [63] [65]

**[63]** *"Enable and tune the V4.1 routed-expert W8A8 paths on A2/A3 and W4A8 path on A5,
including checkpoint packing, scale layouts, dispatch/combine, and expert GEMMs."*

* At TP=EP on one node, changing only the dispatch/combine collective selection measured
  **−4.25 ms/step at 128K**, −1.35 at 32K, −1.23 at 8K, KV capacity **3.39M → 4.16M tokens**,
  with **byte-identical output** between arms. A range-compare expert mask measured
  **−0.51 ms/step** with GSM8K 100/100 and Vision 23/23.
* These two changes are quantization-independent in mechanism — they change which
  collective is used and how a mask is computed, not the numeric path.
* **One methodological note that changes how to read the mask number**, because we got it
  wrong ourselves first. On a single card, comparing the two mask forms **eager** makes the
  range form look **28–40 µs slower per call**; captured in an ACLGraph, the same comparison
  makes it **14–23 µs faster**. The kernel counts explain it — the lookup path issues
  7 kernels per call (`Index`, `IndexCheck`, `NotEqual`, `Mul`, `Cast`, `Fill`, `Arange`)
  and the range path 6 (`Less`, `GreaterEqual`, `Cast`×2, `LogicalOr`, `MaskedFill`) — so
  the change trades one device kernel for several host dispatches. Eager charges the
  dispatches; the graph, which is where this code runs in the deployment, does not.
  In the same way a rope lookup change (3 kernels → 1) measured **−6 … −128 µs** in the
  graph and **−25 … −139 µs** eager, i.e. the same sign, because there the saving is
  *device* kernels. The general rule we took away: **if the saving is host dispatch it will
  not show up in a graph; if it is device kernels it will.** Suggest benchmarking any
  change of this kind in the frame it will run in.
* **Not measured:** we have **no W8A8 run**. Our numbers are W4A8 (A2/A3), so they do not
  validate this item's A2/A3 target. No work on packing, scale layouts or expert GEMMs.
* Note for whoever owns this: #15043 argues the opposite direction for W4A8 on 910B (fused
  MC2). Conditions differ from ours and we are not claiming generality; the gate here
  defaults off, so the comparison is two runs of one script on any node.

```
python bench_mask_graph.py --tokens 8,192,2048 --reps 100   # eager vs ACLGraph + op counts
```

```
V41_MOE_COMM_ALLGATHER=1 bash scripts/serve_a3.sh     # vs unset
MODEL="$MODEL" MODE=full bash scripts/run_test.sh     # 8K/32K/128K + Vision + GSM8K
```

**[65]** *"Enable `fullmesh_v2` … Compare against existing collective strategies by
prefill/decode batch shape and EP scale, and document selection rules."*

* We have one point on that curve: **EP=8, single node**, three context lengths, AllGather vs
  MC2, with the mechanism identified (with TP=EP the MC2 path spreads a small token count
  across ranks, so per-rank work collapses and scalar overhead is not amortised).
* **Not measured:** `fullmesh_v2` at all; EP=16/32; multi-node; DP composition; and a
  systematic prefill-vs-decode **batch-shape** sweep. One EP scale is not a selection rule.

```
python3 tools/bench_concurrency.py --base-url http://127.0.0.1:8020 --model deepseek-v41 \
  --concurrency 1,4,8,32,64 --prompt-tokens 1024 --output-tokens 256 --repeats 2
```

## Graph execution — [73] [75] [77]

> **⚠️ Do not send this paragraph as-is.** On PR #16285 a maintainer ruled out DSpark graph
> mode for v1 — *"we don't plan to support DSpark graph mode in v1 … please use v2 instead"*
> (drslark, 2026-09-20T11:38:27Z), followed by *"we cannot accept these changes in their
> current form"* (11:41:58Z). Our draft-graph work is on MRv1. See `pr/issue-track-C.md`
> §v2 re-aim before quoting any of it. The paragraph that survives is the host-sync
> accounting, which corroborates their own diagnosis instead of competing with it.

**[73]** *"Establish V4.1 ACLGraph decode support … with explicit eager fallback rules."*

* **What we are NOT claiming:** a graph-captured DSpark draft. We built and measured one on
  MRv1 (A3 **36.9 → 23.9 / 24.9 ms/step**, A2 **64.8 → 34.3 ms/step**, acceptance length
  flat), but we accept the maintainer's ruling that v1 should not carry that complexity, and
  we are not proposing it for `main`.
* **What we can contribute instead:** an independent **measurement** of why DSpark is
  sync-bound rather than graph-bound — a draft-path host-sync accounting showing the
  metadata build going through a blocking D2H on 20000/20000 samples, **118.5 s blocked out
  of 787.6 s**. That is evidence for the v2 direction, and it is directly relevant to open PR
  #16465, which is already removing a blocking D2H on this path. We would rather hand over
  the number than fork the fix.
* **Not done:** prefill and mixed-batch graphs; piecewise/full-graph modes.

```
bash tools/draft_ab_launch.sh <chip> && bash tools/draft_ab_run.sh   # eager vs graph, numerical comparison
echo DRAFT_FORCE_EAGER=1 > /tmp/v41_dspark_flags                     # in-process arm switch
```

**[75]** *"Integrate `npugraph_ex` compilation and supported static-kernel optimizations for
attention, mHC, Engram integration, and MoE, including warmup and compilation-cache
behavior."*

* `npugraph_ex=1` and `STATIC_KERNEL=1` are in our shipped default configuration. One
  quantified trap worth adding to this item's documentation: if `LOCAL_WORLD_SIZE` is absent
  from `os.environ`, torch_npu **silently** disables static kernels and the identical config
  costs **4–5 ms/step (~12%)**.
* Compile-cache behaviour: the static-kernel output directory is derived from `Path.cwd()`,
  so a wrong cache mount recompiles at every restart (A3 ≈5 min extra, A2 first run 15–20 min)
  *and* the "cache hit" check reports a false positive; we now check that
  `static_kernel_cache/*.json` is ≥512 B rather than that a directory exists.
* **Not measured:** per-component `npugraph_ex` attribution for attention / mHC / Engram /
  MoE. We know the switch is on; we do not have separate A/B numbers per component.

```
grep -ac "static_kernel.py:650" "$LOG"     # must print 0; anything else = static kernels silently disabled
```

**[77]** *"Define the eager/graph boundary for CPU Engram lookup and dynamic communication
metadata; refresh persistent graph inputs before replay and keep host synchronization off
the captured computation path."*

* This is the item we match most closely. Host-synchronous time per decode step
  **3.379 → 0.058 ms**; step time **29.5 → 28.4** (C1) and **35.3 → 32.1** (C4).
* Boundary rule in use: persistent input buffers, pointer-stable index buffers and
  per-`(n, n_reqs, block_width)` graph keys may be captured; dynamic collective metadata,
  changing slot mappings and first-touch allocations stay outside; every replay validates
  `data_ptr`+shape of the captured inputs before running.
* **Not covered:** SP/DCP/PD; changing request lengths and slot mappings fall back to eager
  rather than being graph-safe.

```
V41_ENGRAM_DEVICE_INDEX=0 vs auto, same session, then: grep -h "\[bneck\]" "$LOG" | tail -n 20
```

## Fusion and validation — [90] [91] [97]

**[90]** *"Optimize compressor, mHC, and Engram gather/dequantization/gating fusion where
profiling shows launch or memory-traffic overhead."*

* Engram gate without the fixed 2048-row padding: **−1.56 ms/step at 8K** (KV slightly
  lower, not higher). Host kernels: hash **0.427 → 0.076 ms**, plan **0.261 → 0.068 ms**.
* Known counter-example: the fused INT8 Triton kernel `gather_dequantize_engram_int8`
  rejects host-mapped pointers (the pointer location is neither DEVICE nor HOST_NUMA), so
  the device-lookup path falls back to aclnn. At decode scale that is ~0.03 ms — but it means
  the fusion this item names is not applied on that path.
* **Not done:** compressor and mHC fusion.

```
V41_ENGRAM_GATE_CHUNK=512 vs 0, same session; and `grep -h "\[bneck\]" "$LOG" | tail` for hash/plan
```

**[91]** *"Validate numerical accuracy, non-contiguous cache strides, empty/padded batches,
and prefill/decode shapes for each fusion. Benchmark both individual kernels and the full
pipeline…"*

* We report op-level counters **and** end-to-end step time for each change (the RFC asks for
  both), plus `torch.equal` checks where outputs are directly comparable; accuracy gates are
  GSM8K-200 198–199/200, Vision 23/23, long-context retrieval 10/10, real agent traces 10/10.
* **Partly measured now — for the RoPE fusion ([87], the one in PR above).** A dedicated
  edge-case harness drives the *real* production function from two git revisions (merge base
  and PR head, loaded by path, sha256 printed) in one process, and reports bit-exactness,
  per-call wall clock and device kernel counts: **32/32 checks pass** (plus 5/5 in a second
  ACLGraph size sweep). Covered: empty batch `n = 0` (bit-exact, shape `(0,1,1,64)`), the
  prefill size `n = 4096` (**eager −376 µs, inside an ACLGraph −384 µs per call**),
  **non-contiguous** strided positions, descending order, duplicate positions, **int32**
  positions (the DFlash buffer dtype), 2-D fallback, and `draft_index = 1..5` (10/10
  bit-exact, only the targeted `spec_row[K-1]` written). Kernel count per lookup drops
  **6 → 2** (three per cos/sin direction to one), which is why the saving survives capture.
  **Two cells regress and are reported as such:** int32 *eager* at small n (+20.7 µs
  contiguous, +28.9 µs strided, attributed to one extra cast per cos/sin direction —
  count-level attribution only, no per-op timing yet).
* **Still not measured:** the same matrix for the *other* fusions (Engram gather/dequant/gate,
  QLI, `wo_a`); prefill/mixed-batch **graph** modes; and a true negative-stride tensor
  (PyTorch refuses to construct one, so a `flip` copy was used instead). We would treat these
  as gaps, not formalities.

```
MODEL="$MODEL" MODE=full bash scripts/run_test.sh
```

**[97]** *"Publish reproducible TTFT, inter-token latency, throughput, HBM usage, Engram
transfer cost, and communication/compute-overlap measurements against the corresponding
baseline."*

All six categories exist with a command in our repository:

* concurrency sweep C1…C64 with per-stream and total throughput and TTFT
  (`tools/bench_concurrency.py`), `ok=64/64` per level;
* HBM/KV accounting (`EXPECTED_PERF.md §6`, `docs/prefill-memory-headroom.md`);
* Engram transfer cost (`d2h` **0.19–3.41 ms**, `route` **1.34–2.89 ms** per rank);
* overlap: measured communication ∩ compute = **0.000 ms** at 32K decode, with the cause
  identified as a data dependency (`reports/comm-compute-overlap-cannbot.md`).
* Caveat: the baseline for most of these is **our own stock configuration**, not upstream
  `main`. A same-card, same-session comparison against upstream code is in progress for the
  Engram gate function and will be posted when it is finished.

```
python3 tools/bench_concurrency.py --base-url http://127.0.0.1:8020 --model deepseek-v41 \
  --concurrency 1,2,4,8,16,32,64 --prompt-tokens 1024 --output-tokens 256 --repeats 2
```

If useful, these can be attached as completion evidence for this item.
