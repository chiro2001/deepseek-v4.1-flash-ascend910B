# RFC #16375 Contribution Report — DeepSeek-V4.1 on Ascend (910B / 910C, W4A8)

> **Draft — not posted anywhere.** Prepared 2026-09-21 as material for
> [RFC #16375 *[RFC]: DeepSeek V4.1 Roadmap*](https://github.com/vllm-project/vllm-ascend/issues/16375).
>
> Evidence root (public, Apache-2.0):
> https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B
> Code fork: https://github.com/chiro2001/vllm-ascend
> Patch base of the measured stack: `GDzhu01/vllm-ascend-v41-private@46856f89e`
> (public, read-only; **not** on `vllm-project/vllm-ascend` main).

---

## 0. How to read this document

**Question this document answers:** *for which items of RFC #16375 do we hold
reproducible evidence, and what is still missing?*

**Question this document explicitly does not answer:** *is our implementation better
than anything else?* That question is not asked here and no section answers it.

### 0.1 Status vocabulary (read this before any table)

| Status | Meaning |
|---|---|
| **Have** | We measured the quantity or ran the configuration the item names, on real NPU hardware, and can give a command that reproduces it. |
| **Partial** | We measured part of what the item names. The missing part is listed explicitly in §2 and in the matching issue draft. |
| **None** | We have nothing. Stated plainly so the roadmap is not accidentally over-credited. |

**No item in this document is "complete".** RFC #16375 line 104 defines completion as
*"merged into its stated target branch, passes the required NPU correctness tests, and is
documented in the support matrix or deployment guide. Performance items additionally
require reproducible comparisons."* Nothing below is merged into
`vllm-project/vllm-ascend` main. Every status here is **evidence availability only**.

### 0.2 Item numbering

Item numbers are **body line numbers of the RFC as fetched on 2026-09-21**
(`gh api repos/vllm-project/vllm-ascend/issues/16375 --jq .body`).
Example: `[46]` = line 46 = the first bullet of *Engram memory management and tensor
parallelism*; `[97]` = line 97 = the *Publish reproducible TTFT …* bullet.
If the body is edited, numbers shift — always quote the item text next to the number.

### 0.3 Number provenance convention

Every number carries a scope and a source. Sources are repository-relative paths under
the evidence root above, e.g. `patches/README.md §3`. Numbers measured on the
**host-lookup path (v7 and earlier)** and on the **device-index path (v8)** are kept
separate; they are different mechanisms and must not be added together.

### 0.4 Hardware and configuration under test (the only configuration we claim)

| Dimension | Value |
|---|---|
| Chips | 1 node × 8 × Ascend 910C (A3) **and** 1 node × 8 × Ascend 910B3 (A2) |
| Parallelism | TP8 / EP8, single node, no DP, **no SP, no DCP, no PD** |
| Quantization | **W4A8** (weights INT4, activations INT8, dynamic) |
| Model | DeepSeek-V4.1-Flash, 128K context, DSpark speculative decoding S=5 |
| Serving config | `MAX_SEQS=64 PREFIX=1 GPU_UTIL=0.92 STATIC_KERNEL=1` (release default) |
| Source of config | `README.md §3`, `CHANGELOG.md §9`, `EXPECTED_PERF.md §6` |

The RFC's hardware matrix (line 11) sets **A2/A3 = W8A8** and **A5 = W4A8**.
Our measured stack is **W4A8 on A2/A3**, i.e. off-matrix in the *good* direction but still
off-matrix: it is a claim about *what W4A8 also runs on*, **not** a validation of the
RFC's W8A8 path. Where an item names a quantization backend, that is stated in the row.

### 0.5 ★ Which frame a number was measured in — read this before quoting any of them

The same optimisation can measure as a **win** or a **loss** depending on whether the
measurement runs **eager** or **inside an ACLGraph**, because the deployed code runs in
the graph while a naive microbenchmark runs eager. We hit this twice on one card, in
opposite directions, so it is worth stating as a rule rather than a footnote:

| Optimisation | Device kernels / call | Eager Δ | **In-graph Δ** | Verdict |
|---|---|---:|---:|---|
| **rope lookup** → `index_select` | **3 → 1** (removes `BroadcastTo` + `Cast` + `GatherElementsV2`) | −25 … −139 µs | **−6 … −128 µs** | **same sign** |
| **expert-mask** → range compare | **7 → 6** (adds `Less`/`GreaterEqual`/`LogicalOr`, drops `Index`/`IndexCheck`) | **+28 … +40 µs** | **−14 … −23 µs** | **sign flips** |

**The rule that came out of it:** what matters is *what the saving consists of*.

* A saving made of **device kernels** survives capture — the graph replays the same
  kernels, just without the host dispatch in the loop.
* A saving that is mostly **host dispatch** does **not** survive capture — the graph
  amortises the dispatch away whether or not the change is present.

Applied to our own stack, this predicts which of our numbers stay honest inside a graph:

| Patch | What it removes | Frame-sensitive? |
|---|---|---|
| 0001 AllGather dispatch | device-side collective work (fewer/cheaper kernels) | no — device work |
| 0002 expert-mask range | 1 device kernel per call | **yes — measured both ways above** |
| 0003 rope `index_select` | 2 device kernels per call | **yes — measured both ways above** |
| 0004 QLI no-candidate | an entire dedup/sort chain (device kernels) | no |
| 0005 `wo_a` 2-D matmul | a degenerate batched matmul | no |
| 0006 chunked gate | device activation size (and kernel count) | no |
| 0007 host-resident tables | **host** collective work | **not applicable** — it is host time, already outside the graph |
| 0008 numba JIT (hash/plan) | **host** compute | **not applicable** — host time |
| 0009 device-index lookup | moves the lookup **into** the graph | reverse direction: it *creates* the captured work |

We have measured the frame for 0002 and 0003 only. **The other rows are reasoning from
the mechanism, not measurements** — flagged so a reader does not assume otherwise. The
practical request to reviewers is therefore narrow: **judge a change in the frame it
will run in**, and if the change is host-side, the graph is irrelevant to it either way.

---

## 1. TL;DR — which RFC items have reproducible evidence behind them

Item ↔ evidence map. Details, caveats and commands are in §2 and §8.

| RFC item | Short form of the item (see RFC for the full text) | What we hold | Status |
|---|---|---|---|
| **[46]** | Engram CPU offload: CPU-resident tables, bounded pinned staging, batched lookups, async H2D prefetch | CPU-resident INT8 tables (206.0 GiB) + batched lookups + overlap; the pinned-staging/H2D leg was **measured and rejected** in favour of device-side lookup over host-mapped DRAM | **Partial** (mechanism differs) |
| **[47]** | Residency policy + measure footprint, lookup latency, transfer volume, NUMA/bandwidth sensitivity under concurrency | Footprint, lookup latency, transfer volume, per-rank spread, machine split, concurrent profile — **5 of 5 named quantities**. The NUMA/bandwidth one now has **both frames**: on a *synthetic* 2 GiB table (single die **107 GB/s** contiguous / 95 GB/s gather; the cap is **per CPU socket ≈115 GB/s**, so 3 dies on 3 sockets scale to 321 GB/s while 3 on 1 socket fall to 39 GB/s each — `logs/38`) **and on the production 206.0 GiB table with 256 B rows** (contiguous 107.0 GB/s, but uniform random gather only **7.55 GB/s** — rows 80× narrower; hot-row skew **pays** here, 2.6–4.3×; 3 dies registering it concurrently all `ret=0`, no 207001/507011 — `logs/40`) | **Partial** |
| **[48]** | Engram TP: explicit ownership; distinguish node-level table sharding from model TP | Full-table-per-rank + device index (`LOCAL_OWNER=fast`); cost of sharded registration measured (a2a+bcast ≈ 0.5 ms/step, net ceiling ≈ 6%) | **Partial** |
| **[49]** | Eliminate duplicate queries across TP ranks; batch across Engram layers; empty-query ranks, collective ordering | Metadata `all_gather` and ids `all_to_all` removed from the decode path; `route` 2.462 → 0.058 ms/step (pre-graph device path → captured graph) | **Partial** |
| **[50]** | Validate Engram offload/TP with SP, DCP, PD **and** graph replay | Graph replay: yes (decode, A3). SP / DCP / PD: **not covered** | **Partial** |
| **[63]** | Routed-expert W8A8 on A2/A3, W4A8 on A5: packing, scale layouts, dispatch/combine, expert GEMMs | Dispatch/combine selection rule measured at TP=EP=8 (quantization-independent); expert-mask path −0.51 ms. **No W8A8 run, no packing/scale/GEMM work** | **Partial** |
| **[65]** | `fullmesh_v2` MoE communication; compare collectives by prefill/decode batch shape and EP scale; document selection rules | AllGather vs MC2 comparison at EP=8, one node, 8K/32K/128K single-stream; candidate selection rule. **No `fullmesh_v2`, no EP sweep** | **Partial** |
| **[73]** | ACLGraph decode support + explicit eager fallback rules | Decode graph capture for DSpark draft and for Engram lookup; fallback rules enumerated and switchable at runtime. Decode-only. **⚠️ v2 re-aim: upstream has ruled out DSpark graph mode for v1** (*"we don't plan to support DSpark graph mode in v1 … please use v2 instead"*, drslark on #16285, 2026-09-20T11:38:27Z) — the DSpark half of this row is **withdrawn as an upstream contribution**; see `logs/18-20260921-track-c-reaim.md`. The Engram half, and the "explicit eager fallback rules" deliverable, stand | **Partial** |
| **[75]** | `npugraph_ex` + static-kernel optimizations for attention, mHC, Engram, MoE, incl. warmup and compile-cache behaviour | `npugraph_ex=1` + `STATIC_KERNEL=1` in the production config; silent-disable trap quantified at **4–5 ms/step** and fixed. No per-component attribution | **Partial** |
| **[77]** | Eager/graph boundary for CPU Engram lookup and dynamic metadata; keep host synchronisation off the captured path | Boundary rules written down; host-synchronous time **3.379 → 0.058 ms/step**; persistent-input refresh + `data_ptr`/shape validation before replay | **Have** (within our stack) |
| **[90]** | Fuse compressor / mHC / Engram gather-dequant-gating where profiling shows overhead | Chunked gate without 2048-row padding (**−1.56 ms @ 8K**); numba JIT for hash/plan kernels (0.427→0.076 / 0.261→0.068 ms). Known counter-example: the fused INT8 Triton kernel rejects host-mapped pointers | **Partial** |
| **[91]** | Validate each fusion numerically, incl. non-contiguous strides, empty/padded batches; benchmark kernels **and** full pipeline | Bit-exact (`torch.equal`) checks per patch, op-level *and* end-to-end numbers. No systematic stride/empty-batch matrix | **Partial** |
| **[97]** | Publish reproducible TTFT, ITL, throughput, HBM, Engram transfer cost, overlap measurements vs baseline | All six measurement categories exist with commands. Baseline = our own stock config, **not** upstream main | **Have** (with that caveat) |

**Coverage summary** (counted over all 50 RFC items in the ledger in §2.5, not just the 13
above): **2 Have, 20 Partial, 28 None, 0 complete.** The 28 items we hold nothing for are
listed in the ledger rather than hidden. Several of the 20 Partial rows are marked
*adjacent* in the note column — a nearby measurement, not the item's own subject. Read the
note, not only the status.

### 1.1 What we are *not* claiming

* Not that any checkbox can be ticked. See the completion rule in §0.1.
* Not that W4A8 on A2/A3 replaces the RFC's W8A8 target. It is an additional data point.
* Not that our 8-NPU numbers are comparable to a 32-NPU instance. See §4.
* Not that a difference in behaviour on concurrent workloads is a defect in anyone's code.
  §6 is written as a debugging aid for open issue #16828 and states a falsifiable hypothesis.

---

## 2. Item-by-item mapping

### 2.1 Requested quantities that we measured: [47] as the worked example

RFC [47] asks for five named quantities. This is the clearest case of "we have most of it",
so it is shown in full; the other items follow in the same form.

| # | Quantity named by [47] | Scope | Value | Source |
|---|---|---|---|---|
| 1 | table footprint | A3, INT8 table, 2 Engram layers | **206.0 GiB** in 4 shard files (11.4 / 91.6 / 11.4 / 91.6 GiB); INT8 weights + FP32 group-32 scales | `quant/README.md` L2; `CHANGELOG.md` v8 §0 |
| 2 | lookup latency (host kernels, per step) | A3, 8 ranks, numba JIT on/off | hash **0.427 → 0.076 ms**, plan **0.261 → 0.068 ms** | `patches/README.md` §3 (0008); `reports/engram-jit-verified.md` |
| 3 | transfer volume (host-lookup path, per rank) | A2 (910B3), decode, `ENGRAM_DEVICE_INDEX=0`, real weights, 8 ranks | `d2h` **0.19–3.41 ms**, `route` **1.34–2.89 ms** (TP0 largest; rank-0 role cost) | `reports/a2-draft-graph-20260920.md` §3.1 |
| 4 | machine / bandwidth sensitivity — **incl. the NUMA + bandwidth sweep** | A2 (910B3) vs A3 (910C); then A3 host-DRAM read by a device operator through the legacy `aclrtHostRegister(…, MAPPED)` path, 1/2/3 dies concurrently — first synthetic 2 GiB / 20480 B rows, then **the production 206.0 GiB table, 256 B rows, 3 dies (3/8-rank proxy)** (2026-09-21) | A2: PCI `19e5:d802`, PCIe, `host_mem_pool=0`, full-table registration fails `ret=207001`; A3: PCI `19e5:d803`, HCCS, `host_mem_pool=1`, 8 ranks × 206 GiB bring-up in ~133 s. **New — one die alone, synthetic:** **107 GB/s** contiguous (1 GiB; 104 at 256 MiB, 66 at 16 MiB), **95 GB/s** random-row gather (4096 × 20480 B), **57 GB/s** at 512 rows; HBM anchor 601 / 724 GB/s ⇒ host DRAM is 5.6–7.6× slower. **Single-die bandwidth is flat across NUMA nodes 0–5 (≤0.7 %, two passes), but the cap is per CPU *socket*, not per die:** two dies reading from the same socket share ≈115 GB/s (57 each) while two dies on different sockets both keep 107 (214 total); three dies on three sockets reach **321 GB/s**, three dies on one socket collapse to 39 GB/s each (117 total). The kernel's own first-touch placement put 2 of the 3 dies on one socket unaided — under an unbound allocator, placement was worth 2.8× per die. **New ★ — the same quantities on the real 206.0 GiB table (rows are 256 B, not 20480 B; 3 free dies = 3/8 proxy):** registration one die **122.2 s / 0.580 ms/MiB** (reproduces the 119.4 s of `logs/29`; 99.9 s hot; **83.9 s / 0.398 ms/MiB** through upstream's `aclrtHostRegisterV2(MAPPED\|PINNED)`); **three dies registering it concurrently 240.9 / 241.3 / 238.9 s, all `ret=0`, no 207001 / 507011** (second run 227.0 / 247.5 / 247.6 s); contiguous read **107.0 GB/s** single-die and **38.5 GB/s per die (115.6 total)** with three dies; uniform random gather only **7.55 GB/s** on the real 256 B rows, and **7.5–7.7 GB/s** for the deployed weight+scale lookup; **hot-row skew pays here** (80 % of queries into 0.1 % of rows: 27.9–32.2 GB/s, 2.6–4.3× uniform) — the synthetic "no gain" was a small-table artefact; and **registration does not perturb peers** (one die re-registering all 206.0 GiB in 108.9 s moved the other two dies' eight read arms by ≤0.8 %). Caveats measured the same day: the writable mapping the API requires is **dirtied** by registration, so a bring-up writes the whole 206 GiB back (`Dirty` ≈108–126 GiB, mtimes move, content byte-identical), and `fadvise(DONTNEED)` silently drops 0 pages while ranks still map it | `CHANGELOG.md` v8 §3.2, §3.3; `README.md` §1; **`logs/38-20260921-host-dram-bandwidth.md`** (synthetic topology evidence, method, caveats) and **`logs/40-20260921-real-table-concurrency.md`** (real table, 3/8 proxy, side effects); raw JSON per run in **`logs/raw/38-host-dram-bw-*.json`** and **`logs/raw/40-real-table-*.json`** |
| 5 | under realistic concurrency | A3, 1K prompt / 256 output, C1…C64 | 7 levels × 2 repeats, `ok=64/64`, service alive throughout | `CHANGELOG.md` §9; `results/bench/conc_dihuo_v8.json` |

**Missing from [47]:** the sweep is now **measured on the production table itself**
(`logs/40`, 2026-09-21) rather than only on the synthetic 512 MiB / 2 GiB one — but at
**3 dies, i.e. a 3/8-rank proxy** (only three dies are free on this box; the production
service owns the other eight). On the real 206.0 GiB / four-shard table, whose rows are
**256 B wide (not the synthetic 20480 B)**:

* registration, one die: **122.2 s / 0.580 ms/MiB** as found (the `logs/29` 119.4 s figure,
  reproduced to 2.3 %) and 99.9 s hot; upstream's exact
  `aclrtHostRegisterV2(MAPPED|PINNED)` + `HostGetDevicePointer` call **83.9 s / 0.398 ms/MiB**
  (all `ret=0`);
* **three dies registering the same 206.0 GiB concurrently: 240.9 / 241.3 / 238.9 s (a second
  run 227.0 / 247.5 / 247.6 s), every shard `ret=0`, no `207001` and no `507011`** — 24
  registrations across two barrier-aligned rounds. The physical pages are shared
  (`MAP_SHARED` page cache), so three dies do **not** need 3 × 206 GiB;
* device reads with the registration held: **107.0 GB/s** contiguous (1 GiB) but only
  **7.55 GB/s** on a uniform random gather of the real 256 B rows, and **7.5–7.7 GB/s** for
  the deployed two-gather lookup shape (weight row + its scale row);
* three dies reading concurrently: **38.5 GB/s per die = 115.6 GB/s total** contiguous
  (the per-socket cap of `logs/38` reappears at the real size) and 7.56 GB/s per die uniform
  gather;
* **hot-row skew does pay at this size**, opposite to the synthetic arm: 80 % of queries into
  0.1 % of rows (≈94 MiB of hot rows) gives **27.9–32.2 GB/s vs 7.5–10.8 GB/s uniform
  (2.6–4.3×)**, 2.8× per die with three dies running. That is a *locality* effect on the
  real 92 GiB shard, not an explicit hot-row cache (we implemented none);
* **registration does not perturb the peers' steady state**: while one die unregistered and
  re-registered all 206.0 GiB (108.9 s), the other two dies' per-arm medians moved by
  **0.992–1.002 across all 8 read arms** (≤0.8 %).

Two side effects found on the way, both in `logs/40`: registering the **writable** mapping
the API requires marks the table's pages dirty, so a bring-up triggers a full **206 GiB
write-back** (`Dirty` ≈ 108–126 GiB observed; the shards' mtimes move; a private-file control
shows the content is byte-identical before/after). And `posix_fadvise(DONTNEED)` **silently
does nothing** while any rank still maps the table (`rc=0`, 306 s spent, 0 of 24 M pages
dropped — only `mincore` tells the truth).

Still unmeasured: **8 ranks** (the 3-die figure above is a proxy, and the 8-rank mixed
registration load cannot be extrapolated linearly), the **production access distribution**
(the skew above is our model of a hot-row workload, not a measured trace), and end-to-end
lookup at realistic batch shapes. The machine-class split and the per-rank spread stand as
before.

Row 4's new bandwidth numbers were measured through the **legacy `aclrtHostRegister`**
with `MAPPED`; one control run repeated the same measurements through upstream's exact
**`aclrtHostRegisterV2(MAPPED|PINNED)`** call (+ `aclrtHostGetDevicePointer`) and reproduced
them within 0.9 % (`logs/38` §2.5), so the numbers are not an artefact of the entry point.
On the **real table** (`logs/40`) the read arms were likewise taken through the legacy path
with the registration held; the V2(`MAPPED|PINNED`) call was exercised there as a **separate
full-table registration pass** (83.9 s, every shard `ret=0`, device pointer via
`aclrtHostGetDevicePointer`), but the read arms were **not** repeated through it — that is
still open.

**Do these numbers apply to the upstream design?** No — they were measured on our
host-mapped path. On the upstream `aclrtHostRegisterV2(MAPPED|PINNED)` path the analogous
quantities have not been measured by us — **except** the row 4 bandwidth numbers, which
`logs/38` §2.5 repeats through that exact V2 call (same numbers within 0.9 %). That is
stated again in §3.

### 2.2 Engram: [46] [47] [48] [49] [50]

| Item | What we have | What is missing |
|---|---|---|
| **[46]** CPU-resident tables, bounded pinned staging, batched lookups, async H2D prefetch | CPU-resident INT8 tables (206.0 GiB) with batched lookups; the H2D leg is *removed* rather than overlapped — device operators index host-mapped DRAM directly. Removing the H2D/pinned leg moved decode concurrency-4 from **35.3 → 32.1 ms/step** and concurrency-1 from **29.5 → 28.4 ms/step** | We did not build the bounded-pinned-buffer + async-prefetch design the item names. If that design is required for A2 (where device-side lookup is unavailable), it is unbuilt on our side. No `hot-row caching` policy either |
| **[47]** | See §2.1 — the sweep is now measured **on the real 206.0 GiB table with 3 dies (3/8 proxy)**: 122.2 s single-die registration (83.9 s through upstream's `V2(MAPPED\|PINNED)`), **three concurrent full-table registrations 240.9 / 241.3 / 238.9 s all `ret=0` with no 207001 / 507011**, 107 GB/s single-die contiguous vs **7.55 GB/s** uniform gather on the real 256 B rows, **27.9–32.2 GB/s with hot-row skew (2.6–4.3× uniform)**, 115.6 GB/s total for three dies, and **≤0.8 % peer impact while one die re-registers the whole table** (`logs/40`); synthetic NUMA/socket sweep in `logs/38` | **8 ranks** (3/8 proxy here) and a **real production access trace** to replace our modelled skew; also the registration side effects in `logs/40` §6 (full-table write-back; `fadvise` silently ineffective) are measured but not yet worked around |
| **[48]** | Explicit rule: one full table per rank, indexed on device (`LOCAL_OWNER=fast`), i.e. node-level sharding deliberately **not** used. Cost of the sharded alternative measured: `all_to_all` + `broadcast` ≈ **0.5 ms/step**, net gain ceiling ≈ 6% | Row/feature ownership split across ranks and projection reduction are not implemented in the measured path. "result redistribution" exists only as the plan we measured and rejected |
| **[49]** | Removed metadata `all_gather` and ids `all_to_all` from the decode path. Pre-graph device path stepped `route` at **2.462 ms**; with the lookup inside the captured graph the host enqueues in **0.058 ms** (the remaining 0.695 ms of device work overlaps) | Empty-query-rank behaviour and collective ordering are not separately tested. "Batch queries across Engram layers" is not done (each layer is called per step) |
| **[50]** | Graph replay: yes — per-batch-shape ACLGraph, zero-copy capture on the model's own buffers, `data_ptr`+shape validation before every replay | SP, DCP, PD: **not covered at all**. Token history, padding masks, persistent input-buffer refresh and transfer-event lifetimes under those configurations are untested |

#### 2.2.1 The host-mapping API matrix (relevant to the open bug in §6)

Both host-registration APIs were exercised on one card, on three kinds of host memory,
with a **real device read** as the verdict (a 4096×256 int8 table, `torch.index_select`
on the device, `torch.equal` against the bytes read on the host). Full output:
`results/probe/probe_hostmap_v2_<date>.txt`.

| # | API | Host memory | Result | Note |
|---:|---|---|---|---|
| 1 | `aclrtHostRegister` | writable file mmap | **read back byte-for-byte** | the production shape |
| 2 | `aclrtHostRegisterV2` (`MAPPED`) | writable file mmap | **read back byte-for-byte** | |
| 3 | `aclrtHostRegisterV2` (`MAPPED\|PINNED`) | writable file mmap | **read back byte-for-byte** | ← the call PR #16925 uses |
| 4–5 | `aclrtHostRegister` / `V2` | anonymous mmap | read back byte-for-byte | |
| 6–7 | `aclrtHostRegister` / `V2` | `aclrtMallocHost` (pinned) | read back byte-for-byte | |

Four findings worth carrying into the `#16828` discussion. **Each is a hypothesis
generated by a probe, not a diagnosis of the upstream implementation** — the upstream
failure was never reproduced here:

1. **`aclrtHostMemMapCapabilities` is not a usable gate.** It returned
   `AIC rc=207000, AIV rc=207000` (feature-not-support) on a machine where the
   registered table **was** readable by a device operator. Gating on that query would
   refuse to run a configuration that works. Gate on `host_mem_pool` plus **one actual
   device read** of a registered table instead.
2. **`aclrtHostRegisterV2` returns only a status code — never a device pointer.**
   `aclrtHostGetDevicePointer(ptr, flag=0)` must be called afterwards. Code that
   migrates from the legacy API and keeps reading its second return value as a pointer
   silently gets **0**.
3. **A read-only VMA is rejected with `107017`** (`ACL_ERROR_RT_INVALID_HANDLE`) here,
   not with the `507899` recorded in our own notes — the same code on a different
   driver. Worth knowing before trusting either number.
4. **Registration cost: ≈0.6–0.8 ms/MiB on real A2/A3 hardware — not the ≈11.4 ms/MiB
   this test container charged.** We first measured 11.4 ms/MiB on a single-card test
   container (driver 25.5.5) and extrapolated "a 206 GiB table costs ≈40 minutes per
   rank". **That extrapolation was wrong** — re-running the same script on A3 hardware
   (driver 26.1.1, chip idle, `host_mem_pool=1`) and, critically, **with a materialised
   file** instead of a sparse one:

   | size | `aclrtHostRegister`, real file (100% blocks allocated) | ms/MiB |
   |---:|---:|---:|
   | 1024 MiB | 799.7 ms | 0.781 |
   | 4096 MiB | 3119.6 ms | 0.762 |
   | 8192 MiB | 4839.0 ms | 0.591 |

   ⇒ **206 GiB ≈ 2.0–2.5 minutes per rank**, which reconciles with both of our own
   production records: A3's 133 s for 8 ranks × 206 GiB (0.63 ms/MiB) and A2's 149.9 s
   for one rank (0.71 ms/MiB). The three independent numbers now agree.

   **And then we stopped extrapolating.** Recognising that no synthetic size can fully stand
   in for the real thing, we registered the **production tables themselves** on an idle A3
   chip, in a throwaway container, one file at a time:

   | shard | size | cost |
   |---|---:|---:|
   | `layers_14_..._embed.scale.safetensors` | 11719.3 MiB | 4711.7 ms (0.402 ms/MiB) |
   | `layers_14_..._embed.weight.safetensors` | 93754.1 MiB | 53406.3 ms (0.570 ms/MiB) |
   | `layers_1_..._embed.scale.safetensors` | 11718.9 MiB | 72.3 ms (0.006 ms/MiB) |
   | `layers_1_..._embed.weight.safetensors` | 93751.5 MiB | 61247.5 ms (0.653 ms/MiB) |
   | **total** | **206.0 GiB** | **119.4 s ⇒ 0.566 ms/MiB** |

   **206 GiB = 119 seconds, measured on the real table, no extrapolation.** That is also
   within 11% of the "133 s for 8 ranks × 206 GiB" in our own release notes — so that
   figure, which we had flagged as unverifiable, now has an independent confirmation.
   (The 65× spread between the first and third shard is page-cache residency, not size.)

   Three things follow that are worth passing on:

   * **The 40-minute figure is a property of that test container, not of Ascend hardware.**
     It is the only outlier (15–19×) in the set; two different real machines on two
     different driver versions land at 0.6–0.8 ms/MiB. Do not size timeouts from it.
   * **A "file-backed" measurement is only as real as the file.** Our first cheap-looking
     file number (0.009 ms/MiB) came from a file created with `ftruncate`, i.e. **0 blocks
     allocated** — there were no physical pages to register. Manufacturing a file with
     real bytes moved the same measurement to 0.591 ms/MiB, a 65× difference. **Always
     check `st_blocks` before believing a file-backed registration number.**
     (This also **withdraws** an earlier claim of ours that "the memory kind is not the
     variable" — that conclusion was drawn from the sparse-file artefact.)
   * **The `host_mem_pool` flag does not predict the cost.** It reads `1` on the test
     container *and* on A3, which differ by 18×; and A2 records 0.71 ms/MiB on the
     `host_mem_pool=0` per-page slow path. Gate on an actual measurement, not on the flag.

   Practical consequence for the timeout family of failures: **one minute spent
   registering 1 GiB predicts the full-table time closely enough to catch a timeout in
   advance** (`bench/bench_host_register_scale.py --sizes-mib 8,128,256 --reps 1`).

### 2.3 MoE: [63] [65]

| Item | What we have | What is missing |
|---|---|---|
| **[63]** routed-expert paths incl. dispatch/combine and expert GEMMs | A dispatch/combine selection change that is **quantization-independent** (it changes which collective is used when TP=EP, not the numeric path): 128K single-stream **−4.25 ms/step**, 32K −1.35, 8K −1.23, and KV capacity **3.39M → 4.16M tokens**; output byte-identical. Plus an expert-mask range comparison: **−0.51 ms**, GSM8K 100/100, Vision 23/23 | **No W8A8 run at all** (the RFC's A2/A3 target). No checkpoint packing, no scale-layout work, no expert-GEMM work. Mask change is adjacent to the item, not one of its named sub-items |
| **[65]** collectives by batch shape and EP scale, selection rules | Contrast between AllGather and MC2 dispatch at **EP=8, single node**, three context lengths (8K/32K/128K), with the numeric equivalence checked. A candidate selection rule: use AllGather when the per-rank token count in MC2 collapses | **No `fullmesh_v2`.** No EP=16/32, no multi-node, no DP, no systematic prefill-vs-decode batch-shape sweep. Related upstream work must be read first: #15043 argues the opposite direction for W4A8 on 910B; #14933 touches the same file (different class) |

**Wording discipline for this row.** Our measurements were taken at TP=EP=8 on 8×910C.
We are not able to say AllGather is generally better; we can only say that under those
conditions it measured faster, and that the gate is an env var that can be turned off to
compare. That limitation is repeated in the issue draft for Track B.

### 2.4 Graph execution, fusion, validation: [73] [75] [77] [90] [91] [97]

| Item | What we have | What is missing |
|---|---|---|
| **[73]** | ACLGraph decode capture for the DSpark draft head and for the Engram lookup; capture is gated on the graph-mode configuration and the proposer's eager setting, and a runtime switch (`DRAFT_FORCE_EAGER=1`) returns to eager without restarting the server. A3 same-process paired arms: **36.9 → 23.9 / 24.9 ms/step**, single-stream **66.5 → 100.7 / 109.8 tok/s**; A2: **64.8 → 34.3 ms/step**, **54.7 → 88.7 tok/s** | Prefill, mixed-batch, piecewise/full-graph modes are **not** covered, and the fallback triggers are not yet written as a tested enumeration. One rare failure mode remains in the DSpark graph path (see §6.5) and is why it ships default-off |
| **[75]** | `npugraph_ex=1` and `STATIC_KERNEL=1` in the production configuration, including warmup/compile-cache handling. We quantified one failure mode worth naming: when `LOCAL_WORLD_SIZE` is absent from `os.environ`, torch_npu **silently** disables static kernels and the same config costs **4–5 ms/step (~12%)**. Startup log check: `grep -ac "static_kernel.py:650" <log>` must be 0 | No per-component A/B isolating `npugraph_ex` for attention, mHC, Engram or MoE. No compilation-cache statistics beyond "warm vs cold start" timing |
| **[77]** | The item we would call our strongest single match. Boundary rule: what may stay inside the captured graph (persistent input buffers, pointer-stable indices, per-shape graph keys) and what must stay outside (dynamic collectives metadata, changing slot mappings, first-touch allocation). Host-synchronous time per decode step **3.379 → 0.058 ms** on the Engram path; the residual device work overlaps with the next step | The boundary is validated for our decode path only. Changing request lengths / slot mappings are handled by falling back to eager, not by a graph-safe mechanism |
| **[90]** | Chunked gate: removes a fixed 2048-row padding (**−1.56 ms @ 8K**, and KV usage slightly lower, not higher). Host kernels: hash **0.427 → 0.076 ms**, plan **0.261 → 0.068 ms** after JIT. Counter-example recorded: the fused Triton kernel `gather_dequantize_engram_int8` rejects host-mapped pointers (location is neither DEVICE nor HOST_NUMA), so the device path falls back to aclnn (~0.03 ms at decode scale) | No compressor or mHC fusion work. The fusion that the item names most directly (gather+dequant+gating in one kernel) is exactly the one we could **not** use on host-mapped memory |
| **[91]** | Per-patch numerical equivalence (`torch.equal` on outputs where applicable), op-level counters **and** end-to-end step time reported separately (the RFC asks for both). Accuracy gates: GSM8K-200 198–199/200, Vision 23/23, long-context retrieval 10/10, real agent traces 10/10 | No systematic matrix over non-contiguous cache strides, empty/padded batches, and prefill/decode shapes per fusion. This is a real gap, not a formality |
| **[97]** | TTFT, inter-token latency, throughput, HBM usage, Engram transfer cost and communication/compute overlap all exist with a command each (§8). Concurrency table C1…C64 with `ok=64/64` | The baseline for most numbers is **our own stock configuration**, not upstream main. A same-card, same-session comparison against upstream code exists only for the Engram gate function (§3) and is still being completed |

### 2.5 Full 50-item ledger (including everything we do **not** have)

`Have` / `Partial` / `None` as defined in §0.1. Items are listed so the roadmap is not
over-credited: **28 of the 50 items carry no evidence from us at all**, and several of the
20 `Partial` rows are flagged *adjacent* — a nearby measurement rather than the item's own
subject.

| Item | Topic (abbreviated) | Status | Note |
|---|---|---|---|
| [22] | Land V4.1 model integration | None | The model code lives in a personal dev repo, not in main; we consume it, we do not land it |
| [23] | Correctness baselines incl. compression ratio 1/2, cross-layer sharing | Partial | Prefill/decode/long-context/prefix-reuse covered by our acceptance runs; ratio-2 and cross-layer sharing not separately validated |
| [24] | Publish support matrix + reproducible serving examples | Partial | We publish one for our own stack (`README.md`, `EXPECTED_PERF.md`), not in upstream documentation form |
| [28] | DCP for sliding-window and compressed sparse attention | None | — |
| [29] | DCP-aware compressor state and cross-layer sharing | None | — |
| [30] | Complete SP across mHC/norm/MoE/Engram | None | — |
| [31] | Token-aligned inputs consistent under SP | None | — |
| [32] | Validate TP/EP/DCP/SP combinations, uneven tokens, empty ranks | None | We validate TP8/EP8 only |
| [36] | PD disaggregation incl. Mooncake/AscendStore | None | — |
| [37] | Complete V4.1 handoff state for PD | None | — |
| [38] | Preserve/reconstruct Engram N-gram history at the P→D boundary | None | — |
| [39] | Heterogeneous P/D parallel configurations | None | — |
| [40] | PD pooling through a shared external KV pool | None | — |
| [41] | Layerwise pool transfers, HBM staging reuse | None | — |
| [42] | Validate transfer completion, cancellation, pool metrics | None | — |
| [46] | Engram CPU offload (pinned staging + async H2D) | Partial | Different mechanism; see §2.2 |
| [47] | Residency policy + 5 measurements | Partial | 4 of 5; no NUMA/bandwidth sweep |
| [48] | Engram TP ownership; node sharding vs model TP | Partial | Full-table-per-rank; no row/feature split |
| [49] | Remove duplicate cross-rank queries | Partial | `route` 2.462 → 0.058 ms; empty ranks untested |
| [50] | Validate with SP/DCP/PD/graph replay | Partial | Graph replay only |
| [54] | Extend multistream Cube/Vector schedule to V4.1 QKV/compressor/indexer | None | We measured overlap, we did not build a schedule |
| [55] | Pipeline Q/KV preprocessing | None | — |
| [56] | Compressor projections in the schedule, ratio-1/2 paths | None | — |
| [57] | Schedule indexer projections/quantisation/selection | None | — |
| [58] | Preserve source-to-consumer dependencies for shared KV/index | None | — |
| [59] | Validate stream/event ownership; publish overlap traces | Partial | We have a measured finding that comm∩compute overlap is **0.000 ms** at 32K decode, with the dependency identified as the cause (`reports/comm-compute-overlap-cannbot.md`) |
| [63] | Routed-expert W8A8(A2/A3) / W4A8(A5) paths | Partial | Quantization-independent dispatch evidence only; no W8A8 |
| [64] | Integrate MegaMoE, validate routing weights/fallbacks | None | — |
| [65] | `fullmesh_v2` + collective comparison + selection rules | Partial | EP=8 only; no `fullmesh_v2` |
| [66] | Dynamic EPLB | None | — |
| [67] | Shared-expert DP | None | — |
| [68] | Overlap shared-expert with routed-expert dispatch/compute/combine | Partial | We measured that overlap is zero and why; we did not implement the overlap |
| [69] | Validate EPLB + overlap under graph replay | None | — |
| [73] | ACLGraph decode + eager fallback | Partial | Decode only |
| [74] | Keep attention outputs/TopK/compressor state stable across capture/replay | Partial | Done for our buffers (persistent inputs, per-shape keys, pointer checks); not for attention outputs or compressor state |
| [75] | `npugraph_ex` + static kernels for attention/mHC/Engram/MoE | Partial | Enabled in production config; no per-component A/B |
| [76] | Make multistream schedules graph-compatible | Partial | Only the pieces we capture (draft dispatch, Engram lookup) are covered |
| [77] | Eager/graph boundary; keep host sync off the captured path | Have | 3.379 → 0.058 ms/step |
| [81] | Selective quantisation exclusions; keep Wqkv unquantised | None | Adjacent only: our recipe leaves 670 tensors FLOAT, but we never analysed them as a Wqkv policy |
| [82] | Map Wqkv policy to `wq_a`/`wq_b`/`wkv` | None | — |
| [83] | Compare unquantised vs quantised projections | None | — |
| [87] | Fuse pre-indexer ops (norm/RoPE/layout/quant) | Partial | Adjacent: our RoPE cos/sin table-index fusion (**−0.45…0.62 ms/pass**, 6 kernels → 2) is a fusion in this area but not one of the named fusions |
| [88] | Fuse post-index-selection ops | None | Adjacent: our QLI no-candidate fast path (99.3 → 50.3 µs ⇒ −0.49 ms) is a short-circuit, not a fusion |
| [89] | Use `quantize_indexer_query` / QLI / `prepare_indexer_indices` boundaries | Partial | We have per-op data at the QLI boundary |
| [90] | Fuse compressor/mHC/Engram gather-dequant-gating | Partial | Gate chunking + JIT; the fused int8 kernel cannot read host-mapped pointers |
| [91] | Validate fusions numerically; benchmark kernels and pipeline | Partial | Numerical checks yes; stride/empty-batch matrix no |
| [95] | Focused regression tests for ownership/alignment/layout/config/transfer | None | Our assets are end-to-end acceptance gates, not upstream-style unit tests |
| [96] | NPU end-to-end coverage A2/A3 W8A8 + A5 W4A8 across TP/EP/SP/DCP/PD/pooling/Engram/graph | Partial | A2/A3 W4A8, TP8/EP8, Engram + graph; no A5, no W8A8, no SP/DCP/PD |
| [97] | Publish reproducible TTFT/ITL/throughput/HBM/transfer/overlap vs baseline | Have | Baseline is our stock config, not upstream main |
| [98] | Attach implementation PRs, owners, target releases, config ranges | None | This document plus the three issue drafts are the input to that item |

---

## 3. Reproducible comparison (RFC [97] / [91])

> **STATUS: RUN COMPLETE, RE-RUN AND CROSS-CHECKED (§3.2 filled).** The main table below is
> from a **2026-09-21 run on an A3 die** (Ascend 910C-class `Ascend910_9382`, **die 3** = host
> `/dev/davinci3`, CANN 9.1.0 `V100R001C11SPC001B243`, driver 26.1.1, torch 2.10.0+cpu,
> torch_npu 2.10.0.post4, python 3.12.13), one process, arms interleaved, 30 repetitions,
> slot held by `tools/a3_chip.sh c0` and released on exit. A repeat run on the same die and an
> earlier single-card run (`Ascend910_9362`, CANN 9.1.0, torch_npu 2.10.0) are carried in §3.2
> as the cross-run check — **including the rows where the two runs disagree.** The §3.3
> comparisons were run afterwards and are **no longer open** (see §3.3).

The RFC asks for *"reproducible comparisons"* ([97]) and for benchmarking *"both
individual kernels and the full pipeline"* ([91]). The strongest form of that is a single
script that runs **upstream code and our code back to back on the same card in the same
session** and reports time, peak HBM and a numerical equality check — so the reader can
draw the conclusion rather than being handed one.

### 3.1 Harness

Draft file: `upstream-v41/pr/bench_engram_gate_head2head.py`; to be published as
`bench/bench_engram_gate_head2head.py` in the evidence repository together with the raw JSON.

| Item | Value |
|---|---|
| Script | `bench/bench_engram_gate_head2head.py` (**draft ready, to be published**; working copy sha256 `6b9455634c433a9f…`) |
| Upstream arm | `engram_gate()` copied **verbatim** from `refs/remotes/pr/16925` = `382dc9289d6a7dec37203c3a886f3c35bd7c0966` (committed 2026-09-20), file `vllm_ascend/models/deepseek_v41/engram/common.py`, function at line 164 to EOF — no edits |
| Our arm | `engram_gate()` copied verbatim from `patches/files/engram_gate.py` (sha256 `955e40b9…`), shipped gate values `V41_ENGRAM_GATE_CHUNK=512`, `V41_ENGRAM_GATE_MAX_TOKENS=2048` |
| Arms per run | `arms_for()` yields **five**: (1) upstream verbatim, (2) **control** = our embedded file with `V41_ENGRAM_GATE_CHUNK` unset, (3) ours shipped `CHUNK=512 MAX=2048`, (4) ours `CHUNK=512 MAX=4096`, (5) ablation `CHUNK=512 MAX=ceil(n/512)·512` (**not a shipped configuration**) |
| Provenance check | `--verify-verbatim` re-diffs both embedded copies against their sources; needs no NPU. **PASS on 2026-09-21**: upstream block 25 lines ≡ `refs/remotes/pr/16925:…/common.py` line 164; our block 184 lines ≡ `patches/files/engram_gate.py` line 42 |
| Interleaving | all live arms run inside one loop with the arm order alternated every repetition, so no arm pays warm-up |
| Metrics | median / min wall time per call, peak HBM (`torch.npu.max_memory_allocated()` delta), `torch.equal(upstream_out, ours_out)` |
| Machine (main run) | A3 die 3 (`Ascend910_9382`, 910C class), CANN 9.1.0, driver 26.1.1, torch 2.10.0+cpu, torch_npu 2.10.0.post4, python 3.12.13 |
| Raw output | `logs/raw/36-engram-gate-h2-a3-a3c0-20260921.json` (sha256 `0228423a…`) and `…-run2.json` (sha256 `7cf8941…`) in the working repo; to be published as `results/head2head/engram_gate_20260921.json` |
| Control arm | **in the script, and measured** (the earlier "not in the script yet" note was stale — the 02:17 raw JSON already contains this arm). `ours default (CHUNK unset → reference)` runs our embedded file with the chunking gate off, i.e. the stock algorithm from our file. **Time: 1.00–1.06× the upstream median at every size** in both A3 runs *and* in the 02:17 run (worst cell 1.059× at n = 8 in the 02:17 run; the A3 runs stay within 1.05×), so the file swap is invisible to the clock. **Peak HBM: 1.20× upstream at every size in all three runs** — the extra is one FP32 `[n, hc_mult, 5120]` buffer (`n × 4 × 5120 × 4 B`, up to allocator rounding: +0.08 / +0.63 / +2.50 / +16.0 / +160.0 / +320.0 MB at n = 1 / 8 / 32 / 192 / 2048 / 4096), i.e. the second `hidden.float()` cast our reference path keeps live (the `[ENGRAM-GATE-HOIST]` block); at n = 2048 the shipped arm's 420 MB is 0.44× this control. `torch.equal = True`, `max\|d\| = 0` everywhere. **Both halves matter**: the control proves the file swap is not what the other rows measure, and it also shows that a 1.2× HBM difference can be real and reproducible without costing measurable time |

**Honest boundaries stated by the harness itself** (kept because they bound what the numbers
may be used for): function level, not end-to-end; both arms run eager, so graph-replay
overhead is not represented; inputs are synthetic fixed-seed tensors, not captured from a
live request; one card, no concurrency, no TP; upstream's branch documents W8A8 while our
deployment is W4A8, so no claim may be extended beyond "this function".

### 3.2 Results

Production shapes: `hidden`/`key` `[tokens, hc_mult=4, hidden=5120]` bf16,
`rotation_block` 32×32, `token_mask` with 1/8 rows masked. Median of 30, warmup 5.
Main table = A3 die 3, 2026-09-21 12:50 CST (`36-…-a3c0-20260921.json`).

**★ The headline is not the speed — it is the memory. On time the two forms are at parity
for n ≥ 2048 (0.94–1.00× across the two A3 runs); the earlier single-card run put the same
rows at 1.12–1.15×. A ≤15% delta whose sign does not survive a machine change is not evidence
of a speed-up, and we do not quote it as one.**

| tokens | upstream median (ms) | ours, `CHUNK=512` (ms) | time ratio | **upstream peak HBM (MB)** | **ours peak HBM (MB)** | **HBM ratio** | `torch.equal` |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.587 | 2.681 | 4.57× | 0.4 | 420.0 | 1063× | ✅ `max\|d\| = 0` |
| 8 | 0.581 | 2.852 | 4.91× | 3.1 | 420.0 | 134× | ✅ |
| 32 | 0.581 | 2.761 | 4.75× | 12.5 | 420.0 | 33.6× | ✅ |
| 192 | 0.578 | 2.835 | 4.90× | 80.0 | 420.0 | 5.25× | ✅ |
| 2048 | 3.334 | 3.275 | 0.98× | 800.1 | 420.0 | **0.52×** | ✅ |
| 4096 | 6.750 | *rejected by contract* | — | 1600.3 | — | — | — |

**Machine and raw data for the table above**: A3-21 slot `c0` = **die 3**
(`Ascend910_9382`, 910C class), CANN 9.1.0, driver 26.1.1, torch 2.10.0+cpu,
torch_npu 2.10.0.post4, python 3.12.13; raw JSON
`logs/raw/36-engram-gate-h2-a3-a3c0-20260921.json` (sha256 `0228423a…`), repeat run
`logs/raw/36-engram-gate-h2-a3-a3c0-20260921-run2.json` (sha256 `7cf8941…`).
§3.2.1 and §3.2.2 are the same run; their `02:17` columns are the earlier single-card run.

**Two independent runs, plus an earlier one — does the direction survive?** Time ratio =
ours ÷ upstream. The HBM ratio is *identical* in all three runs (1 / 8 / 32 / 192 / 2048:
1063× / 134× / 33.6× / 5.25× / 0.52×) because it is set by the allocation sizes, not by
timing; the time ratio is not.

| tokens | 02:17 single card (`Ascend910_9362`) | 12:50 A3 die 3 | 12:51 A3 die 3 (repeat) | same sign? |
|---:|---:|---:|---:|---|
| 1 | 7.20× | 4.57× | 4.72× | yes — ours slower, both runs |
| 8 | 6.51× | 4.91× | 4.96× | yes |
| 32 | 5.47× | 4.75× | 4.81× | yes |
| 192 | 5.56× | 4.90× | 4.41× | yes |
| 2048 | 1.12× | **0.98×** | **1.00×** | **no — ±15% band, sign flips** |
| 4096 (`MAX=4096`) | 1.15× | 0.94× | 0.96× | **no — ±15% band, sign flips** |

Raw rows: `logs/raw/03-engram-gate-h2-20260921.json` (sha256 `2e5ce488…`, single card),
`logs/raw/36-engram-gate-h2-a3-a3c0-20260921.json` (sha256 `0228423a…`, A3 die 3),
`logs/raw/36-engram-gate-h2-a3-a3c0-20260921-run2.json` (sha256 `7cf8941…`, A3 repeat).

`torch.equal` is `True` for **every arm, every size and all three runs** (29 of the 30 cells
per run; the 30th is the contract rejection at n = 4096), including the padded chunked path
and the masked rows: `max|d| = 0.00e+00` throughout.

At the model's real `max_num_batched_tokens` (2048) **our peak HBM is 0.52× of upstream's in
all three runs** (420.0 MB vs 800.1 MB), and the time ratio is inside ±15% in all three runs
— the entire point of the chunking gate is device activation, not latency.

### 3.2.1 What the small-token rows actually show: the padding ceiling, measured

The 4.4–5.0× penalty at n ≤ 192 (5.5–7.2× on the earlier single-card run — same sign) is
**not** caused by chunking. It is caused by `V41_ENGRAM_GATE_MAX_TOKENS` padding every call up
to a static token ceiling (a deliberate graph-capture choice). That ceiling is the knob, so it
was swept **twice: in eager mode, and inside a captured + replayed ACLGraph** — the second frame
being the one that ships. Both sweeps fix `CHUNK` at 512, sweep `MAX_TOKENS` = 256 / 512 / 1024
/ 2048 / 4096 over n = 1 … 4096, run one card / one process with every arm interleaved inside
the same run, `reps=30`, and re-measure the §3.1 arms unchanged (eager raw rows
`logs/raw/41-engram-gate-ceiling-{small,large}-a3c0.json`, write-up
`logs/41-20260921-engram-gate-ceiling-sweep.md`, harness
`agents/T2_ceilings/bench/bench_engram_gate_head2head_sweep.py`, sha256 `66470619…`);
the graph harness imports *that* module's arms by path, so the two tables cannot diverge. It
captures one graph per (arm, n) — the ceiling is a compile-time constant, so n *is* the captured
batch size — with L = 1 / 4 / 16 / 40 back-to-back gate calls inside a single graph, replays it
30 times, and validates every graph by `torch.equal` on the replayed output against the same
arm's eager output (**126/126 cells exact, `max|d| = 0.00e+00`**; graph raw rows
`logs/raw/43-gate-ceiling-graph-{small,large}-a3c{0,1}.json`, write-up
`logs/43-20260921-gate-ceiling-in-graph.md`, harness
`agents/T3_graphceil/bench/bench_gate_ceiling_graph.py`, sha256 `8761d4cf…`). **Eager** (median
ms per call, die 3):

| `MAX_TOKENS` | n=1 | n=8 | n=32 | n=192 | n=512 | n=1024 | n=2048 | n=4096 | peak HBM (small n) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.774 | 0.820 | 0.824 | 0.859 | 0.871 | — | — | — | 285 MB |
| 1024 | 1.417 | 1.455 | 1.499 | 1.558 | 1.636 | 1.697 | — | — | 330 MB |
| **2048 (shipped)** | 2.811 | 2.834 | 2.896 | 2.911 | 3.006 | 3.140 | 3.346 | — | 420 MB |
| 4096 | 5.675 | 5.591 | 5.683 | 5.709 | 5.864 | 5.712 | 5.993 | 6.357 | 600 MB |

**In-graph**, same arms, L = 16 (median ms per call; parentheses = the capture's own pool peak
MB). This is the production frame: the deployed model captures 40 layers in one graph, and an
L = 40 control on a third die reproduces the table within 3.7% (pool peaks identical, i.e. the
pool saturates):

| `MAX_TOKENS` | n=1 | n=8 | n=32 | n=192 | n=512 | n=1024 | n=2048 | n=4096 | pool peak (small n) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.679 | 0.688 | 0.690 | 0.742 | 0.787 | — | — | — | 285 MB |
| 1024 | 1.303 | 1.322 | 1.329 | 1.399 | 1.529 | 1.548 | — | — | 330 MB |
| **2048 (shipped)** | 2.597 | 2.629 | 2.662 | 2.752 | 2.853 | 2.944 | 3.185 | — | 420 MB |
| 4096 | 5.224 | 5.298 | 5.206 | 5.358 | 5.611 | 5.581 | 5.899 | 6.365 | 600 MB |

A n = 8192 tail (over the shipped contract) was measured in both frames: `MAX_TOKENS=8192` runs
zero-padded at **12.859 ms / 1120 MB** eager and **12.771 ms** in-graph (pool peak 1120 MB at
L = 1, 1440 MB at L = 4), versus upstream's **13.295 / 3200.5 MB** and
**13.255 / 3520.5 MB** (0.96–0.97× time, **0.35–0.41× HBM**).

Median ms per call; **“—” = the arm raises** (n > MAX is a contract violation, not a crash —
the padder cannot shrink a dimension). **Neither frame has a knee above `CHUNK`**, and the slope
is the same in both: **in-graph t ≈ 0.011 + 0.651 × (MAX/512) ms** (n = 1 row; per-row slopes
0.639–0.691 ms per 512 rows across two dies, max residual 0.05 ms) against eager's
**t ≈ 0.122 + 0.648 × (MAX/512)** (n = 1 row, same run; per-row 0.648–0.693, and 0.660–0.710 in
`logs/41`). Graph capture removes the
intercept — mostly host dispatch, worth 0.11 ms at n = 1 — but *not* the slope: **a padded row
costs what a real row costs, and it costs that inside a graph too.** The padded buffer costs
≈ 45 MB per 512 rows in both frames (the graph pool peaks 0–160 MB above the eager single-call
peak, saturating by L = 4). **The honest statement is therefore unchanged: the padding constant,
not the chunking, is the cost — and here is its price list, now in the frame that ships.**

**What the in-graph frame does change is the comparison with upstream — in our disfavour.** The
verbatim upstream gate has no padding, so its eager number is mostly host dispatch at small n,
and a captured graph deletes exactly that: upstream drops to **0.099 ms at n = 1** (5.5× below
its 0.546 ms eager), 0.180 at n = 32, 0.270 at n = 192, and only reaches its eager value at
n ≥ 512. Our arms, being pad/device-bound, gain 1–11% from capture. The in-graph ratios against
upstream are therefore:

| n | ours at the minimal ceiling `ceil(n/512)·512` | shipped `MAX=2048` | (eager, for contrast — minimal / shipped) |
|---:|---:|---:|---:|
| 1 | **6.84×** | **26.1×** | 1.32× / 4.76× |
| 8 | 4.54× | 17.3× | 1.19× / 4.12× |
| 32 | 3.83× | 14.8× | 1.20× / 4.24× |
| 192 | 2.75× | 10.2× | 1.26× / 4.28× |
| 512 | 1.41× | 5.11× | 1.38× / 4.81× |
| 2048 | 0.97× | 0.97× | 1.00× / 1.00× |
| 4096 | 0.95× (zero padding) | — (illegal) | 0.94× / — |

The previous revision of this section quoted "**1.19–1.38× upstream instead of 4.12–4.78×**" for
the minimal ceiling — that was an eager-only number and understates the ratio by 3–5×
(die 6 reproduces every cell within 2%: 6.88× / 4.48× / 3.93× / 2.77× / 1.43× at 512, and
5.11–26.4× for the shipped constant). The mechanism is not subtle: upstream's device work is ∝ n,
ours is ∝ MAX.

**Recommended value** — the smallest `CHUNK` multiple that covers the largest token count the
captured graph can see, i.e. `MAX_TOKENS = max(CHUNK, ceil(B_max / CHUNK) * CHUNK)`. This does
**not** change with the frame; the in-graph data makes it sharper:

* small-batch / decode-only graphs (`B_max ≤ 512`): **512** — **0.679–0.816 ms** and 285–305 MB
  per call in-graph (1.41× upstream at n = 512 … 6.84× at n = 1);
* the shipped prefill contract (`max_num_batched_tokens = 2048`): **2048 is already that
  minimum**, so keep it — the 5.1–26× at small n is then structural rather than a bug. A
  2048-row graph costs 2.597–2.853 ms even for a single token, and the measured gap to a 512-row
  graph is **−1.92 … −2.10 ms / −135 MB per call**, i.e. the same absolute gap as eager
  (−2.02 … −2.18 ms) and a slightly *larger* ratio (3.6–3.9× vs 3.4–3.8×): **oversizing the
  ceiling is device work and does not get cheaper when the frame is captured.** The next knob
  after this constant is therefore *which* ceiling each captured graph uses
  (per-capture-size buckets, priced by the tables above) — and in-graph that knob is worth
  more, not less: a 2048-row graph serving a 1–192-token batch costs 10–26× upstream's device
  work instead of the 4.1–4.8× the eager table suggests, not a smaller single process-wide
  value.

**One guard is worth adding:** `MAX_TOKENS=256` with `CHUNK=512` does not pad to 256 —
`_gate_max_tokens()` silently falls back to `max(chunk, 4096)`, i.e. to the *worst* point on
this curve (measured 5.677 ms at n = 1, identical to the `MAX=4096` arm). A declared value
below `CHUNK` should raise, or floor to `CHUNK`, instead.

### 3.2.2 n = 4096 (beyond the shipped contract)

The shipped configuration rejects 4096 tokens by contract
(`RuntimeError: size of tensor a (2048) must match b (4096)`); the harness records this as
`RAISES` in the results table and as an `error` row in the JSON — it is a result, not a crash.
With `MAX_TOKENS` raised to 4096 the arm runs at **6.356 ms / 600 MB** versus upstream's
**6.750 ms / 1600.3 MB** — 0.94× the time, **0.375× the HBM**. The earlier single-card run put
the same arm at 8.276 ms vs upstream 7.221 ms (1.15×) with the same 600 MB / 1600.3 MB. **The
memory result reproduces exactly; the time ratio does not (0.94× vs 1.15×)** — which is why
§3.2 quotes parity rather than picking either run.

**Fill command** (A3 slot `c0` = die 3; the lock is released on exit):

```bash
# provenance only — no NPU needed
python3 bench/bench_engram_gate_head2head.py --verify-verbatim

# measurement on the A3 (writes into the container's /work = the remote workspace)
bash tools/a3_chip.sh c0 --timeout 900 --name engram-h2 -- \
  python3 /work/bench/bench_engram_gate_head2head.py \
  --sizes 1,8,32,192,2048,4096 --reps 30 --warmup 5 \
  --json /work/agents/<you>/out/engram_gate_20260921.json

# same script on the single-card box (its own lock, releases on exit)
with_chip.sh 1 --name engram-h2 -- python3 bench/bench_engram_gate_head2head.py \
  --sizes 1,8,32,192,2048,4096 --reps 30 --warmup 5 \
  --json out/engram_gate_20260921.json
```

### 3.3 Second comparison — **【MEASURED on A3, 2026-09-21】**

Run on `A3-node1` in an isolated per-die container (**die 3**, `Ascend910_9382`, PCI `19e5:d803`
= 910C class, CANN `9.1.0`, driver `26.1.1`, torch 2.10.0+cpu / torch_npu 2.10.0.post4,
numba 0.67.0). Both scripts load our shipped files **by path** (not embedded copies) and
re-verify every source sha256 via `--verify-verbatim` before measuring.
Raw data: `logs/raw/37-ngram-history-a3prbench-c0.{json,txt}`,
`logs/raw/37-hostmap-ab-a3prbench-c1.{json,txt}`; full write-up in
`logs/37-20260921-ngram-and-hostreg-ab.md`.

| Comparison | Upstream arm | Our arm | Measured on A3 |
|---|---|---|---|
| host table registration | `aclrtHostRegisterV2(MAPPED\|PINNED)` (PR #16925) | `acl.rt.host_register(MAPPED)` + `host_mem_pool` check | **Both `ret=0` and both byte-exact on a device readback** at 128 MiB in the production shape (writable file mmap, payload at +96 B), each followed by a device `torch.index_select` that `torch.equal`s the host bytes. `host_mem_pool=1` on all 16 dies (PCI `19e5:d803` ×16). **No fork in the road at this size** — the divergence #16828 reports needs the **206 GiB full table**, which was **not run**. Registration wall time is *not* a result here: the three arms land in two clusters (~45 ms / ~90 ms) and their ordering flips between the probe's two sections, so the verdict rests on the return code and the readback, not the timer |
| token history update | `PagedNgramHistory.update()` (per-token Python `tolist()`+`zip`) | numba JIT (`patches/files/engram_jit_kernel.py`) | At the production decode shape (n=128 tokens, 64 seqs, block 128): whole update **1.6807 → 0.0736 ms (22.8×)**, `torch.equal` on hashes and mask at every one of 14 sizes. Upstream's own per-token walk alone (verbatim `common.py:112-130`): **10.3053 → 0.0079 ms (1312×)**, of which ~6.5× is the packed-mirror layout and the rest the JIT |

Two calibration notes, both of which we would rather state than have read past:

1. The **0.427 → 0.076 ms** quoted in §3.1/§3.2 is an **in-engine per-step phase timer**
   (`[bneck] hash`), which is *narrower* than this harness's `PagedNgramHistory.update()`.
   Our side reproduces it (**0.0736 vs 0.076 ms**); the upstream side does **not**
   (1.68 ms here vs 0.427 ms there). The *ratio* does reproduce: 22.8× here against the
   23× that the same report measured on a pure-CPU bench. **Use the ratio, not the absolute.**
2. §3.3 above is a **function-level, single-process, no-NPU** measurement. It says nothing
   about step time, TTFT or throughput, and the inputs are synthetic ids (tables, primes,
   multipliers, block table and page contents are production-shaped). Re-running on a second
   die with `OMP_NUM_THREADS=16` instead of 1 moved n≤384 by <10%.

**What is still open on this comparison:** the 206 GiB full-table registration (the actual
#16828 criterion); 8-rank concurrent full-table registration; and any end-to-end effect.

---

## 4. Per-card efficiency — **【CONVERTED, NOT MEASURED】**

This section exists because capacity planning is a real question for the roadmap, and it is
the easiest place in this document to be misread. **Everything in §4 is a conversion
between two different published configurations, not a measurement of two implementations
on one machine.** Do not quote these numbers as a speedup.

### 4.1 The two configurations

| Dimension | Upstream reference | Ours |
|---|---|---|
| Hardware | 2 × A3 node = **32 NPU** | 1 × A3 node = **8 NPU** |
| Topology | DP4 / TP8 / EP32 | TP8 / EP8 |
| Quantization | **W8A8** | **W4A8** |
| `MAX_SEQS` | 32 | 64 |
| Engram | table in CPU, host-register path | table in host DRAM (INT8, 206.0 GiB), device-side lookup |
| Sources | issue #16828, PR #16689, PR #16925 | `EXPECTED_PERF.md §6`, `CHANGELOG.md §0/§9`, `results/bench/conc_dihuo_v8.json` |

### 4.2 Capacity, per card — 【CONVERTED】

| Metric | Upstream PLE off | Upstream PLE on | Ours (A3, release default) | Conversion note |
|---|---|---|---|---|
| KV tokens, instance level | 3,105,872 | 6,110,521 | 2,823,080 (`util=0.92`) | different instance shapes |
| NPUs | 32 | 32 | **8** | — |
| **KV tokens per NPU** | 97,058 | 190,953 | **352,885** | **【converted】** 1.85× vs PLE-on, 3.64× vs PLE-off |
| Weights per NPU | 34.67 GiB | 21.80 GiB | ≈39 GiB (incl. draft + vision) | **not comparable**: different quantisation and different included components; Engram lives entirely in host DRAM on our side |

Cleaner same-topology view (both are TP8 replicas): upstream 6,110,521 ÷ 4 DP replicas =
1.53M tokens per replica; ours 2.82M ⇒ **【converted】** ≈1.85×.

### 4.3 Throughput — **not offered as evidence for any RFC item**

| Concurrency | Upstream total tok/s (32×A3, W8A8) | Upstream arm, stated explicitly | Ours total tok/s (8×A3, W4A8, 256-token output, release default) |
|---|---:|---|---:|
| 1 | 54.56 | PR #16689, VMM arm, 64-token output | 87.1 |
| 4 | 144.13 | PR #16689, VMM arm, 64-token output | 237.7 |
| 8 | 226.65 | PR #16689, VMM arm, 64-token output | 324.4 |
| 32 | 429.70 | issue #16828, PLE **on**, 2K/2K prompts | 583.9 |
| 64 | — | engine died between C24 and C32 with offload on | 719.5 |

Sources: upstream = PR #16689 and issue #16828 (note that the first three rows and the C32
row come from **different upstream runs** — they are not one curve, which is one more reason
not to divide these columns); ours = `CHANGELOG.md §9` (all rows from one sweep, release
default; the same report lists a 108.3 tok/s C1 arm with `DRAFT_GRAPH=1`, deliberately not
used here so that our column is one arm).

Reasons these columns must not be divided by each other, all of which we accept as
deflating our side of the comparison:

1. output length differs (64 vs 256 tokens) — short outputs carry proportionally more
   fixed per-request cost, which lowers tok/s;
2. quantisation differs (W8A8 vs W4A8) — different work, and W4A8 is our design choice;
3. DP4 idles at low concurrency upstream (only one replica works at C1) while TP8 uses all
   8 dies;
4. contexts and prompts differ (2K/2K and 64K/1K upstream; 1,024/256 here).

**No converted per-card ratio is derived from the throughput table at all.** The honest
statement is: *at instance level our sweep did not fall behind across the concurrency range
we measured, and the per-card capacity conversion in §4.2 is 1.85×.*

---

## 5. Ablation table — one patch at a time

> **STATUS: DONE (2026-09-21) — the single-session ablation has been run on eight cards.**
> Twelve arms, one 8×910C box, one harness, one model, one pair of `(A, ms/step)` per arm;
> every arm differs from the **stock** arm by exactly one gate. Full write-up with per-cell
> provenance: [`logs/44-20260921-single-session-ablation.md`](../logs/44-20260921-single-session-ablation.md).
> Raw per-arm results: `results/<run_id>/` on the bench host, trimmed copies in
> `logs/raw/44-ablation-*/`.
>
> **The headline is not a millisecond count.** In this frame the gates do **not** each save
> 0.3–0.6 ms; **`MOE_AG` alone decides whether the speculative accept length is 1.7 or 4.7**,
> and that decides whether single-stream `tok/s` is 48 or 140 (**×2.9**). The other five
> single-gate arms are all inside the noise floor. See §5.2.
>
> ⚠️ **That accept-length gap carries one unresolved caveat — quote before you cite it.**
> The measurement workload is *verbatim copying* ("transcribe the opening of chapter 5 of
> Dream of the Red Chamber"), i.e. a **copy-type task**, and the harness records metrics
> only (**no generated text**) for it. In the separate, text-recording probe frame we ran,
> **both** arms ran away into a repetition loop (the `MOE_AG=0` arm cycled through *different*
> sentences, the `=1` arm repeated *one* sentence). So the gap is consistent with two
> readings we have **not** separated: (a) AllGather makes the **draft model** more accurate,
> or (b) a tighter loop is simply **easier to guess**, so `A` rises without the model being
> better at real work. On a real agent workload we have only the `MOE_AG=1` side
> (`A` median **3.58**, no anomalies) — a value **between** 1.7 and 4.7.
> ⇒ Until an accept-length comparison is repeated on a **non-degenerate** task (or both arms
> are run through the accuracy gates), quote this as *"on the verbatim-copy workload"*,
> **not** as a general "accept rate ×2.8". Details: `logs/44` §5.2.0.
>
> ⚠️ **Frame of the new table** (stated so nobody mixes it with the per-patch table below):
> `CPU_BIND=0` (production is `=1`; on this box `=1` hangs — see §5.4), `DROPCACHE=0`,
> `SP_TOKENS=5`, `DRAFT_GRAPH=0`, `PREFIX=0`, `MAX_SEQS=4`, `GPU_UTIL=0.92`,
> `MODE=quick` (8K + 32K, 6 repetitions each), vision skipped, `REPEATS=6`.
> **Cross-frame numbers are not paired.**

RFC [97] asks for measurements *"against the corresponding baseline"*, and a stack of
eleven gates invites the fair question: which of them actually pays?

**The single-session ablation has now been run** (§5.2) — twelve arms on one 8×910C box,
every arm differing from stock by exactly one gate. The per-patch table below is kept
**as a second, different frame**: those rows are one-card function-level or single-chip
A/Bs, they answer "does this patch change the kernel it targets", and they must not be
added to or compared against the new table's deltas.

### 5.1 Per-patch table — **second frame, function level** (kept from the 2026-09-21 draft)

| # | Gate | Measured effect | Kind of measurement | Where |
|---:|---|---|---|---|
| — | stock baseline | *(reference)* | — | — |
| 0001 | `V41_MOE_COMM_ALLGATHER=1` | **−4.25 ms** at 128K, −1.35 at 32K, −1.23 at 8K; KV pool 3.39M → **4.16M** tokens | same-session paired A/B, 8 chips | `reports/moe-allgather-breakthrough.md` |
| 0002 | `V41_MOE_MASK_RANGE=1` | `Index`+`IndexCheck` 0.927 → **0.414 ms/step**; net **−0.51 ms/step** on decode | per-pass normalized device profile, 8 chips, dummy | `reports/moe-mask-range-verified.md`; **single-card recheck in §3.2 of the PR draft** |
| 0003 | `V41_ROPE_IDXSEL=1` | **−0.45 … −0.62 ms/pass**; 6 kernels → 2 | per-pass device profile + op counts | `reports/rope-idxsel-verified.md` |
| 0004 | `V41_QLI_NO_CANDIDATE=1` | single op 99.3 → **50.3 µs** ⇒ ≈ **−0.49 ms/step** | single-op timing + projection | `reports/qli-no-candidate-verified.md` |
| 0005 | `V41_O_PROJ_2D=1` | **−0.31 … −0.76 ms** | same-session paired A/B | `reports/f3-wo-a-2d.md` |
| 0006 | `V41_ENGRAM_GATE_CHUNK=512` | **−1.56 ms** at 8K, and **peak HBM 0.52×** at BAT 2048 | same-session A/B (device) + single-card function A/B (§3.2) | `patches/README.md` §3; §3.2 above |
| 0007 | `V41_ENGRAM_HOST_RESIDENT=1` | frees HBM to the KV pool and shortens the route step (no single ms/step figure claimed) | device breakdown, 8 chips | `reports/engram-host-breakdown.md` |
| 0008 | `V41_ENGRAM_JIT=1` | hash **0.427 → 0.076 ms**, plan **0.261 → 0.068 ms** | host-side timing, per step | `reports/engram-jit-verified.md` |
| 0009 | `V41_ENGRAM_DEVICE_INDEX=auto` | **29.5 → 28.4 ms/step** (C1), **35.3 → 32.1** (C4); host sync 3.379 → **0.058 ms/step** | same-session A/B, 8 chips | `CHANGELOG.md` v8 §0 |
| 0010 | default policy (A3 on / A2 off) | A2: full-table registration fails `ret=207001`; A3: succeeds | capability probe on both machines | `CHANGELOG.md` v8 §3.2–3.3 |

**What this frame does not contain, stated plainly:** no row shares a baseline with any
other row, and the accept-length column is empty — none of these A/Bs isolated `A` for the
arm, because `A` is a draw per request (see §0.3) and none of the runs was powered to
resolve a difference in it. **That gap is what §5.2 fills** (twelve arms, one baseline,
one `(A, ms/step)` pair per arm, two repeat arms for the noise floor), and §5.2.3 lists
row by row where the two frames agree and where they do not.

**Fill command:**

```bash
MODEL=<model> bash scripts/run_test.sh MODE=full          # per-arm, gate set via env
# What was actually run for §5.2 (twelve arms, one env change each) is scripted in
# agents/T4_ablation/run_arm.sh + finish_arm.sh; the per-arm record is agents/T4_ablation/arms.tsv
```

**Reporting rules for this table** (fixed in advance so the numbers cannot be cherry-picked
afterwards): report medians, always report the accept length `A` next to the timing, keep
the raw log path per row, and state the session id — cross-session numbers are not paired.

### 5.2 ★ Single-session ablation, measured (2026-09-21) — **[MEASURED]**

Twelve arms, **one 8×910C box** (`A3-node1`, Phy-ID 8–15), one image
(`quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`, `PATCH_MODE=mount`),
one model (W4A8 + Engram-int8, DSpark S=5), one harness (`scripts/run_test.sh`,
`MODE=quick` = single-stream 8K + 32K, 6 repetitions each). **Every arm differs from the
`stock` arm by exactly one gate.** Session: 2026-09-21 16:22–20:3x CST.

**A0 `stock`** = `MOE_AG=0 MOE_MASK=0 ROPE_IDXSEL=0 QLI_NOCAND=0 O_PROJ_2D=0 ENGRAM_JIT=0
ENGRAM_DEVICE_INDEX=0 DRAFT_GRAPH=0`, with `GATE_CHUNK=0 GATE_MAX_TOKENS=2048` (the
production gate values — see the note under the table).

| arm | what it is | `ms/step` 8K | `A` 8K | `tok/s` 8K | `ms/step` 32K | `A` 32K | `tok/s` 32K | peak KV | run_id |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **A0** | stock (all six gates off) | 34.72 | 1.738 | 50.0 | 36.83 | 2.899 | 78.9 | 3,241,932 | `a0_20260921_165927` |
| **A0b** | *repeat of A0* (noise floor) | 34.42 | 1.676 | 48.3 | 35.45 | 2.881 | 80.5 | 3,241,442 | `a0b_20260921_171229` |
| **A1** | shipped (all defaults) | **29.86** | **4.787** | **157.0** | **30.47** | **4.691** | **151.7** | 3,842,534 | `a1_20260921_172649` |
| **A1r** | *repeat of A1* | 30.35 | 4.727 | 150.3 | 31.04 | 4.726 | 150.6 | 3,842,534 | `a1r_20260921_183507` |
| **A2** | A1 with `ENGRAM_DEVICE_INDEX=0` | 32.65 | 4.771 | 143.0 | 33.43 | 4.727 | 137.4 | 3,844,493 | `a2_20260921_175024` |
| **A3** | A1 with `ENGRAM_JIT=0` | 31.70 | 4.815 | 149.6 | 30.66 | 4.631 | 150.2 | 3,842,657 | `a3_20260921_180419` |
| **A4** | A1 with `QLI_NOCAND=0` | 30.11 | 4.815 | 156.5 | 31.09 | 4.640 | 146.2 | 3,842,534 | `a4_20260921_182034` |
| **A5** | A0 + `MOE_MASK=1` | 34.91 | 1.717 | 48.6 | 35.68 | 2.876 | 80.4 | 3,241,564 | `a5_20260921_185117` |
| **A6** | A0 + `QLI_NOCAND=1` | 34.49 | 1.655 | 47.8 | 35.44 | 2.850 | 79.2 | 3,241,687 | `a6_20260921_190549` |
| **A7** | **A0 + `MOE_AG=1`** | **33.59** | **4.743** | **139.6** | **32.29** | **4.727** | **143.9** | **3,844,493** | `a7_20260921_191947` |
| **A8** | A0 + `ROPE_IDXSEL=1` | 34.84 | 1.655 | 46.2 | 36.88 | 2.872 | 76.3 | 3,241,687 | `a8_20260921_193105` |
| **A9** | A0 + `O_PROJ_2D=1` | 34.52 | 1.744 | 48.7 | 35.56 | 2.876 | 79.9 | 3,241,687 | `a9_20260921_194140` |
| **A10** | A0 + `ENGRAM_JIT=1` | 35.50 | 1.746 | 48.9 | 34.87 | 2.866 | 81.8 | 3,241,564 | `a10_20260921_195535` |
| **A11** | A1 + `GATE_CHUNK=512` | 30.68 | 4.771 | 153.3 | 30.61 | 4.658 | 149.0 | 3,842,901 | `a11_20260921_200615` |

*(medians of 6 usable points; A0 is 2 points and carries the note in §5.2.1. Every cell's raw
file is `results/<run_id>/p42_t4_quote_{8192_quote_8k,32768_quote_32k}.jsonl`; peak KV is
`results/<run_id>/env.txt`. `static_kernel.py:650` hits = **0 on every arm**.)*

#### 5.2.1 ★ Result 1 — only `MOE_AG` moves the accept length

| gate (A0 + only this one) | Δ `ms/step` 8K | Δ `A` 8K | Δ `ms/step` 32K | Δ `A` 32K |
|---|---:|---:|---:|---:|
| `MOE_MASK=1` (A5) | +0.49 | +0.041 | +0.23 | −0.005 |
| `QLI_NOCAND=1` (A6) | +0.07 | −0.021 | −0.01 | −0.031 |
| **`MOE_AG=1` (A7)** | **−0.83** | **+3.067 (×2.83)** | **−3.16** | **+1.846 (×1.64)** |
| `ROPE_IDXSEL=1` (A8) | +0.42 | −0.021 | +1.43 | −0.009 |
| `O_PROJ_2D=1` (A9) | +0.10 | +0.068 | +0.11 | −0.005 |
| `ENGRAM_JIT=1` (A10) | +1.08 | +0.070 | −0.58 | −0.015 |

**Measured noise floor** (same config, two back-to-back sessions): 8K `ms/step` **0.30 ms**,
32K `ms/step` **1.37 ms**, `A` 0.06 (8K) / 0.02 (32K). ⇒ **the five non-`MOE_AG` rows are all
inside the floor.** Only `MOE_AG` is distinguishable:

> **`MOE_AG` alone takes the accept length from 1.68 to 4.74 and single-stream `tok/s`
> from 48 to 140 (×2.9).** No other gate in the stack changes `A` at all, in either
> direction (A5–A10 vs A1–A4).

**Peak KV pool, same session**: the gate that sets it is `MOE_AG`, not
`ENGRAM_DEVICE_INDEX` — A0 and A2 **both** run `ENGRAM_DEVICE_INDEX=0`, and A2 (which has
`MOE_AG=1`) still reaches **3,844,493** tokens vs A0's 3,241,932. A7 (`A0` + only `MOE_AG`)
gets **exactly the same 3,844,493**. This independently reproduces the `MOE_AG` row of the
per-patch table above ("KV pool 3.39M → **4.16M**").

**A third, independent check — the output text.** Each arm also answers one fixed
`temperature=0` request (a stable 1.4 K-token prompt ending in
`…Question: What is the capital of France? Answer: The capital of France is`, 6 repetitions,
recorded as a sha256). The four arms whose gates do **not** move `A` — A6 (`QLI_NOCAND`),
A8 (`ROPE_IDXSEL`), A9 (`O_PROJ_2D`), A10 (`ENGRAM_JIT`) — produce the **same** hash
(`42593d7da3b2f373…`, four different run_ids on four different sessions), while **A7
(`MOE_AG`) produces a different one**. So the text fingerprint, the accept length and the
KV pool all separate the same single gate from the rest. *(A caveat the reader should
carry: this fingerprint discriminates configurations; it is not a correctness oracle — one
arm's repetitions disagreed on where generation stopped, see `logs/44` §4.10.1.)*

#### 5.2.2 Result 2 — the reverse direction (shipped minus one gate)

| removed from A1 | Δ `ms/step` 8K | Δ `A` 8K | Δ `ms/step` 32K | Δ `A` 32K | verdict |
|---|---:|---:|---:|---:|---|
| `ENGRAM_DEVICE_INDEX` (A2) | **+2.79** | −0.016 | **+2.96** | +0.036 | **real (−2.9 ms when on)** |
| `ENGRAM_JIT` (A3) | **+1.84** | +0.028 | +0.19 | −0.060 | real at 8K, not resolvable at 32K |
| `QLI_NOCAND` (A4) | +0.25 | +0.028 | +0.62 | −0.051 | inside the floor |

**The two directions do not contradict each other, and the difference matters**:
removing `ENGRAM_DEVICE_INDEX` from the full stack costs ~2.9 ms, but *adding* it to stock
(which is what A5–A10 measure) is invisible — because in the stock configuration the
accept length is 1.7, so each decode step moves far fewer tokens and every per-step
constant is amortised differently. **A delta measured against stock is not the same
quantity as a delta measured against the shipped stack.**

#### 5.2.3 Result 3 — how the new table relates to the per-patch table above

| per-patch claim | single-session result | agreement |
|---|---|---|
| 0001 `MOE_AG`: −1.23 ms @8K, −1.35 @32K, KV 3.39M→4.16M | A7 vs A0: **−0.83 ms @8K / −3.16 @32K**, KV **3.24M→3.84M**, plus `A` 1.68→**4.74** | **same sign**; the per-patch row **omits the accept-length effect, which is the large one** |
| 0002 `MOE_MASK`: net −0.51 ms/step | A5 vs A0: **+0.49 ms @8K** (inside floor) | **not resolvable** in this frame |
| 0003 `ROPE_IDXSEL`: −0.45…−0.62 ms/pass | A8 vs A0: +0.42 @8K / +1.43 @32K | **not resolvable** |
| 0004 `QLI_NOCAND`: ≈ −0.49 ms/step | A6 vs A0: +0.07 @8K; A4 vs A1: +0.25 @8K | **not resolvable** |
| 0005 `O_PROJ_2D`: −0.31…−0.76 ms | A9 vs A0: +0.10 @8K / +0.11 @32K | **not resolvable** |
| 0008 `ENGRAM_JIT`: hash 0.427→0.076 ms (host) | A10 vs A0: +1.08 @8K; A3 vs A1: **+1.84 @8K**, +0.19 @32K | 8K row agrees in sign; the host-side claim is a different quantity |
| 0009 `ENGRAM_DEVICE_INDEX`: 29.5→28.4 ms/step | A2 vs A1: **−2.79 @8K / −2.96 @32K** | same sign, **larger** than the per-patch row |
| 0006 `GATE_CHUNK=512`: −1.56 ms @8K, HBM 0.52× | A11 vs A1: **+0.82 @8K / +0.14 @32K**, KV **+367** | **time claim does not reproduce end-to-end**; HBM claim does (see §5.3) |

**What the new table does *not* establish** (stated plainly): it is **not** a function-level
measurement, so it cannot confirm or refute the per-patch rows' *mechanisms*; it is one
configuration family (TP8/EP8, no SP/DCP/PD, single stream, no prefix caching) on one
910C box, so its absolute numbers do not transfer to 128K context or to A2; and it does
not isolate `ENGRAM_HOST_RESIDENT` (0007) or the registration policy (0010), which are
held constant in every arm.

#### 5.2.4 Method note — what "one session" can and cannot mean here

The gates are **not** runtime-togglable inside one process: `ENGRAM_GATE_CHUNK` is read at
trace/capture time (`_gate_chunk_tokens()` is evaluated while the graph is captured, and
replaying the graph does not re-read the environment), and `STATIC_KERNEL` / `NPUGRAPH_EX`
/ `DRAFT_GRAPH` are compile-time choices. So "one session" here means: **same box, same
image, same model, same harness, back-to-back, one env change per arm, same
`(A, ms/step)` measurement** — with the two repeat arms (A0b, A1r) quantifying what
cross-session variation remains. That variation is what §5.2.1's noise floor reports.

**Never compare across frames**: the per-patch table, this table, and §3.2's function A/B
were produced on different hardware, different context lengths and different frames.

### 5.3 `GATE_CHUNK=512` end-to-end (G19) — **[MEASURED, and it does not amplify]**

The per-patch row 0006 reports **−1.56 ms at 8K** for the chunked gate and notes that
*"production ships 0 (= stock) because long-context re-measurement is pending"*. That
number is **function level**. The single-session ablation ran the missing end-to-end arm:
**A11** = the shipped stack **plus** `V41_ENGRAM_GATE_CHUNK=512` (`MAX_TOKENS=2048`
unchanged, `a11_20260921_200615`), against **A1** = shipped with `CHUNK=0`.

| | `ms/step` 8K | `A` 8K | `tok/s` 8K | `ms/step` 32K | `A` 32K | `tok/s` 32K | peak KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A1** shipped (`CHUNK=0`) | 29.86 | 4.787 | 157.0 | 30.47 | 4.691 | 151.7 | 3,842,534 |
| **A11** shipped + `CHUNK=512` | 30.68 | 4.771 | 153.3 | 30.61 | 4.658 | 149.0 | 3,842,901 |
| **Δ** | **+0.82** | −0.016 | −3.7 | **+0.14** | −0.033 | −2.7 | **+367** |

**Reading**:

* **The −1.56 ms does not survive into the end-to-end step.** At 8K the point estimate is
  **+0.82 ms** — outside the 8K noise floor (0.60 ms) but only by 1.4×, i.e. one arm is
  not enough to call it a regression; at 32K (+0.14 ms) it is far inside the floor.
  The honest sentence is: **no measurable end-to-end gain at 8K or 32K, point estimate
  slightly negative.**
* **The HBM effect is real and points the same way as the old claim**: the KV pool grows
  by **+367 tokens** (3,842,534 → 3,842,901), consistent with the gate's lower peak
  activation (the function-level "peak HBM 0.52×"). The chunking gate buys **memory, not
  time** — which matches §3.2.1's conclusion that at the production `BAT=2048` the
  chunked path is at time parity with upstream (0.94–1.00×).
* **Therefore keeping `GATE_CHUNK=0` in production is correct on this evidence**, and
  row 0006 should be re-labelled from "−1.56 ms at 8K" to "**−1.56 ms function-level;
  end-to-end 8K/32K: not resolvable, point estimate +0.8 ms; buys peak HBM, not step time**".
* **Still missing**: **128K**. `MODE=full` was not run in this session, so the
  long-context re-measurement that row 0006 waits for is still open. Nothing here says
  the 128K behaviour is the same.

### 5.4 ★ Operational finding: `CPU_BIND=1` hangs the 8-card bring-up on this box — **[MEASURED]**

Every arm in §5.2 runs with **`CPU_BIND=0`**, and that is not a preference — with the
production default (`CPU_BIND=1` → `additional-config.enable_cpu_binding=true`) the server
**never became ready** in 35 minutes. The cause is in `vllm_ascend/cpu_binding.py`'s
`bind_memory()`: it runs `migratepages <pid> <all nodes> <target node>` for every rank,
**without checking the target node's free memory**, and the caller's timeout branch
(`except subprocess.TimeoutExpired: p.kill(); p.communicate()`) calls `communicate()`
**again with no timeout** — so once `migratepages` wedges in the kernel, the "1000 s
protection" becomes a permanent block. On this machine NUMA node 6 was 99.99 % full
(`MemFree ≈ 22 MB`) while each rank held ≈ 90 GB of mapped Engram table, so the migration
never progressed (three `MemFree` samples three minutes apart moved by 24 kB; the two
`migratepages` processes burned 100 % CPU each for ~2.5 h and ignored `SIGKILL` until they
finally exited). Full timeline, the `MemFree`/`AnonPages`/`FilePages` samples and a
suggested fix are in `logs/44-20260921-single-session-ablation.md` §1. **This is an
operational hazard for any shared box with uneven NUMA free memory, not a property of our
patches.**

---

## 6. Boundary data — written as a debugging aid for open issue #16828

RFC [50] and [77] both require boundary behaviour to be pinned down. This section records
what we observed, on which machine, and states a **falsifiable hypothesis**. It is not a
claim about anyone's implementation quality, and it does not claim our stack is "more
stable" — the two stacks run different topologies (§4.1), so the observations are not
directly comparable.

### 6.1 What upstream issue #16828 reports

| Observation (from the issue, 2026-09-18) | Value |
|---|---|
| `enable_engram_ple_offload=true` weight saving per rank | 12.87 GiB |
| KV capacity | 1.97× |
| decode impact (their off→on A/B, one controlled repeat, author-flagged as directional) | 2K/2K C4: −29.5% output tok/s, TPOT +32.4%; 64K/1K C4: TPOT +75.4% |
| concurrency 24–32 with offload on | engine dies: `507011` / invalid GM address |
| same config with offload off | stable |

Source: issue #16828, reproduced in `PERF-COMPARE.md §3.2/§4`.

### 6.2 Our data points in the same concurrency range (different configuration)

| Measurement | Ours | Scope / source |
|---|---|---|
| C32 | 583.9 tok/s, `ok=64/64`, service alive | A3 8×910C, W4A8, TP8/EP8, 1,024/256, `CHANGELOG.md §9` |
| C64 | 719.5 tok/s, `ok=64/64`, service alive | same session (`results/bench/conc_dihuo_v8.json`) |
| host-synchronous time per decode step | 3.379 → 0.058 ms | A3, host path → device-index-in-graph, `CHANGELOG.md` v8 §0 |
| host path cost measured by switching it off entirely | 2.77 ms/step (same session, identical output `A`) | `reports/engram-final-quantification.md §3` |
| comm∩compute overlap | **0.000 ms** at 32K decode; cause identified as data dependency, not core selection | `reports/comm-compute-overlap-cannbot.md §1` |

**These are not a rebuttal.** Our instance is 8 dies / W4A8 / single node; theirs is 32 dies /
W8A8 / two nodes with DP4. Different topologies can fail differently for reasons that have
nothing to do with the offload mechanism.

### 6.3 The falsifiable hypothesis we would offer

> **Hypothesis.** Whether the host-table registration path succeeds is decided by the
> driver feature `host_mem_pool`, which is fixed per PCI device id at probe time — not by
> total host memory and not by concurrency. A registration that takes the per-page metadata
> path can still accept a 4 KiB probe and then fail at full-table size, i.e. a capability
> probe that only tests "is the API accepted" is insufficient.

Supporting observations, each with a command:

| Machine | PCI id | CPU↔NPU | `host_mem_pool` | Observed registration behaviour |
|---|---|---|---|---|
| A2 (910B3) | `19e5:d802` | PCIe | **0** | full-table register fails `ret=207001` after ~17 min, while `MemAvailable` was still 703 GiB at the moment of failure; a **single** rank registering the full table succeeded (149.9 s / 159.5 s, two runs) |
| A3 (910C) | `19e5:d803` | HCCS | **1** | 8 ranks × 206 GiB bring-up completes in ~133 s |
| 910C-class dev container, driver 25.5.5 / CANN 9.1.0 | `19e5:d803` | HCCS | **1** | probe reports supported (third data point, different driver version from the two above) |

Sources: `CHANGELOG.md` v8 §3.2/§3.3/§3.5 and `README.md` §1 for the first two rows; the
dev-container row is an internal note (`upstream-v41/OVERNIGHT-PLAN.md` §1.5.2) whose raw
probe output will be published together with the standalone probe. The driver-side reasoning
(per-4 KiB-page metadata, a ~2.06 GiB contiguous kernel allocation per rank, `207001` =
`ACL_ERROR_RT_MEMORY_ALLOCATION`) is a **source-level hypothesis**, not a measurement; it is
testable with the probe below.

Three workarounds we tested and can rule out, so nobody repeats them: writing
`/proc/svm/devN/feature/host_mem_pool` is a no-op; huge pages do not help (page count is
derived from the VA range ÷ 4 KiB); registering fewer ranks is not a production answer
(cost ≈ +1.5–2.3 ms/step and it cannot be captured in a graph).

**Our mitigation (a workaround, not a fix for anyone else's path).** Three layers, all of
them checkable by a user:

| Layer | Behaviour |
|---|---|
| Capability check first | `V41_ENGRAM_DEVICE_INDEX=auto` (default) reads `/proc/svm/devN/feature/host_mem_pool` before touching a single byte of the table; `0` ⇒ do not attempt the full-table registration at all |
| Automatic fallback | On `host_mem_pool=0` machines the stack falls back to the host lookup path: **functionality and accuracy unchanged**, only the device-index acceleration is absent. The fallback prints a full reason into the serve log |
| Explicit override | `=1` forces the attempt and raises on failure (used in acceptance runs so a silent fallback cannot be mistaken for a pass); `=0` disables the feature outright. Both are documented in `README.md` §1 |

In our tests this turns a 17-minute-to-failure startup into an immediate, explained
fallback. Whether the same probe is the right thing to add upstream is a maintainer
decision, and we would rather it be tested against the failing machine from #16828 than
believed on our say-so.

### 6.4 Reproduction (the first two commands need no model weights)

```bash
# 1. Which side of the fork is this machine on?  (1 = A3-class, 0 = A2-class)
cat /proc/svm/dev0/feature/host_mem_pool

# 2. Standalone probe: PCI id, host_mem_pool, and the return codes of both
#    aclrtHostRegister and aclrtHostRegisterV2 (the call upstream #16925 makes),
#    plus a device-side read of a small host-mapped table. Does not import vllm.
python3 tools/probe_engram_hostmap.py --device <free chip> --size-mib 128
#    exit 0 = supported, 3 = unsupported, 1 = probe error

# 3. Full-size registration probe, per rank and optionally 8-way concurrent.
#    Needs a model with engram_int8/; the wrapper starts a throwaway container.
MODEL=<model with engram_int8/> FANOUT=8 PROBE_CHILD_FILES=1 \
  bash tools/run_probe_engram_hostreg.sh      # 8 × 11.4 GiB, then drop PROBE_CHILD_FILES for 8 × 206 GiB
```

The standalone probe in step 2 is a draft at `upstream-v41/pr/probe_engram_hostmap.py`
(wrapper `pr/run_probe_hostmap.sh`) and is intended to be published as
`tools/probe_engram_hostmap.py`. Unlike the per-patch code it does not depend on the V4.1
model code and can land on its own.

**What would falsify the hypothesis:** a machine with `host_mem_pool=0` that registers the
full table successfully at 8 ranks, or a machine with `host_mem_pool=1` that fails with
`207001`. Either result would be worth posting back to #16828.

### 6.5 One known failure mode on our side (so the ledger is symmetric)

Our DSpark graph path has a rare bad state: acceptance length sticks at `1.00` while
accepted throughput is `0.00` (empty output). Observed **once**; 6 subsequent reproduction
attempts (~36 measurement points, 10.4 min of continuous load) failed to reproduce it, and
5 mechanistic hypotheses were falsified. Recovery criterion: two consecutive spec-decoding
metric reads with `Mean acceptance length: 1.00` **and** `Accepted throughput: 0.00` ⇒
restart the service. **This is why the graph path ships `DRAFT_GRAPH=0` by default even
though its measured gain is large.** Source: `CHANGELOG.md §6.1/§6.2`.

---

## 7. Patch index

Eleven code patches (10 `vllm-ascend` + 1 `vllm` core). The two `msmodelslim` quantisation
recipe patches are listed separately in §7.2 because they are not code changes to a
serving stack. **All gates default to off**; no patch changes the stock path unless its env
var is set. Series integrity: `patches/MD5SUMS`, verified machine-identical between the
git series and the whole-file form.

| # | Patch | Gate (default) | Measured effect | Scope of the measurement | RFC item | Status |
|---|---|---|---|---|---|---|
| 0001 | MoE dispatch/combine over AllGather when TP=EP | `V41_MOE_COMM_ALLGATHER=1` (0) | 128K **−4.25 ms/step**, 32K −1.35, 8K −1.23; KV 3.39M → **4.16M** tokens; output byte-identical | A3, 8×910C, TP8/EP8, W4A8, single-stream | [63] [65] | Validated on A3. **Not** validated on W8A8, EP>8, multi-node |
| 0002 | Expert mask by range compare | `V41_MOE_MASK_RANGE=1` (0) | **−0.51 ms/step**; GSM8K 100/100, Vision 23/23 | A3, TP8/EP8 | [63] | Validated; upstream-style unit test + port in progress |
| 0003 | RoPE cos/sin table index fusion | `V41_ROPE_IDXSEL=1` (0) | **−0.45…0.62 ms/pass**; 6 kernels → 2 | A3; op-level counters + wall time | [87] (adjacent) | Validated. Upstream #14428 is merged prior art; ours removes the remaining `expand`/`BroadcastTo` |
| 0004 | QLI fast path when there is no candidate | `V41_QLI_NO_CANDIDATE=1` (0) | 99.3 → 50.3 µs per op ⇒ **−0.49 ms/step** | A3 | [88] [89] (adjacent) | Validated, equivalence checked |
| 0005 | `wo_a` 2D matmul + dummy-shape guard | `V41_O_PROJ_2D=1` (0) | **−0.31 ms/step** (ms scope) / **−0.76 ms** (cli_p50 scope); acceptance length unchanged | A3, 128K single-stream, A/B/A2 pairing | [59] (adjacent) | Validated end-to-end on A3; the port to current `main` has not been re-measured on 8 cards |
| 0006 | Engram chunked gate (no 2048-row padding) | `V41_ENGRAM_GATE_CHUNK=<int>` (0 = stock) | **−1.56 ms/step @ 8K**, KV slightly lower | A3 | [90] | Validated at 8K; production ships `0` (= stock) because long-context re-measurement is pending |
| 0007 | Engram host-resident INT8 tables + local-owner | `V41_ENGRAM_HOST_RESIDENT=1`, `V41_ENGRAM_LOCAL_OWNER=fast` | removes one metadata `all_gather` and ids `all_to_all`; `route` pre-graph 2.462 ms | A3, real weights | [46] [48] [49] | Validated numerically. Superseded in the v8 default by 0009 for the lookup itself |
| 0008 | numba JIT for hash and plan kernels | `V41_ENGRAM_JIT=1` (0) | hash **0.427 → 0.076 ms**, plan **0.261 → 0.068 ms** | A3, per-step, 8 ranks | [47] [90] | Validated. Requires a writable `NUMBA_CACHE_DIR`, else recompiles every start |
| 0009 | Device-side lookup over host-mapped DRAM, captured in ACLGraph | `V41_ENGRAM_DEVICE_INDEX=auto/1/0` (auto) | host-synchronous **3.379 → 0.058 ms/step**; 29.5 → **28.4** ms/step (C1), 35.3 → **32.1** (C4) | A3, decode; prefill still does a full-table gather (16.8 ms) | [46] [47] [50] [77] | Validated on A3 with bit-exact checks. **Unusable on A2** (falls back automatically). Depends on 0007 |
| 0010 | Default policy: on for A3-class, off for A2-class, decided by `host_mem_pool` | `V41_ENGRAM_DEVICE_INDEX=auto` (auto) | prevents a 17-minute-to-failure path on `host_mem_pool=0` machines; no perf number claimed | A2 + A3 | [46] [47] [50] | Validated on both machine classes |
| vllm-1 | Scheduler admission gate (vllm core) | `VLLM_ADMISSION_GATE=1` (unset) | removes multi-hundred-step prefill-only starvation windows; **no RFC item claims this** | A2/A3 | **none** | Validated in our stack. Designed for a single in-flight batch: it conflicts with async scheduling, and vllm #56455 covers overlapping ground — do not propose it without reading #56455 first |

### 7.1 What is missing for these patches to be upstream-ready

Stated so nobody assumes the patches are merge-ready as they sit:

* no `Signed-off-by` (DCO) on the commit series;
* commit subjects use conventional-commit prefixes, not the repository's `[Category]` form;
* some commit bodies are Chinese; upstream review language is English;
* no upstream-style unit tests are attached (end-to-end acceptance runs exist instead);
* browser/large patches 0007 and 0009 (+1,284 / +1,626 lines) are far above the repository's
  typical PR size and would need splitting before review.

### 7.2 Quantisation recipe (separate)

| Patch | What it does | Why it is separate |
|---|---|---|
| `msmodelslim` 0001/0002 | DeepSeek-V4.1 **W4A8** recipe (+ hiaux variant) | The RFC's matrix sets A2/A3 = **W8A8** and A5 = **W4A8**. Our recipe is an off-matrix capability claim, offered as information, **not** as a request to change the matrix. Reproduce from `quant/REPRO_W4A8_QUANT.md` |

---

## 8. Reproducible commands

One command per conclusion. `$MODEL` = a DeepSeek-V4.1 W4A8 checkpoint directory with
`engram_int8/`; `$LOG` = a serve log from `scripts/serve_a3.sh` or `scripts/serve_a2.sh`.
Everything runs from the repository root.

| # | Conclusion | Command |
|---|---|---|
| C1 | Engram table footprint is 206.0 GiB across 4 shards | `du -sBG --apparent-size "$MODEL"/engram_int8/* \| sort -n` |
| C2 | The machine-class criterion is the `host_mem_pool` feature | `cat /proc/svm/dev0/feature/host_mem_pool` |
| C3 | Which PCI id / CPU↔NPU protocol this machine is | `lspci -nn -d 19e5: \| grep -i -E "d802\|d803"` |
| C4 | Full-table host registration succeeds or fails, per rank, 8-way | `MODEL="$MODEL" FANOUT=8 bash tools/run_probe_engram_hostreg.sh` (add `PROBE_CHILD_FILES=1` for the 8 × 11.4 GiB quick step) |
| C5 | Does this box support device-side lookup at all (no model needed) | `python3 tools/probe_engram_hostmap.py --device <free chip> --size-mib 128` (standalone, no vllm import; `tools/probe_a2_hostmap.py --chip <free chip>` is the earlier release-package probe) |
| C6 | host-synchronous time per decode step on the Engram path | `grep -h "\[bneck\]" "$LOG" \| tail -n 20` (fields `d2h`, `hash`, `route`; compare `V41_ENGRAM_DEVICE_INDEX=0` vs `auto`) |
| C7 | JIT effect on hash/plan kernels | enable `V41_ENGRAM_JIT=1` with a writable `NUMBA_CACHE_DIR=./cache/numba`, then `grep -h "\[bneck\]" "$LOG" \| tail -n 20` in both arms |
| C8 | Static kernels were not silently disabled | `grep -ac "static_kernel.py:650" "$LOG"` must print `0` |
| C9 | End-to-end acceptance (8K/32K/128K + Vision + GSM8K) | `MODEL="$MODEL" MODE=full bash scripts/run_test.sh` |
| C10 | Concurrency sweep, C1…C64, TTFT and per-stream/total throughput | `python3 tools/bench_concurrency.py --base-url http://127.0.0.1:8020 --model deepseek-v41 --concurrency 1,2,4,8,16,32,64 --prompt-tokens 1024 --output-tokens 256 --repeats 2` |
| C11 | Production-shaped mixed prefill/decode, prefix on and off | `MODEL="$MODEL" bash tests/multibatch/run_prod_both.sh` |
| C12 | Patch-level A/B (any single patch) | `git apply --check patches/vllm-ascend/0002-*.patch && git apply patches/vllm-ascend/0002-*.patch` then run C9 with the gate on and off |
| C13 | Patch series applies cleanly to the base commit and matches the whole-file form | `cd /vllm-workspace/vllm-ascend && git am --keep-cr /opt/dsv41/patches/vllm-ascend/*.patch && md5sum -c /opt/dsv41/patches/vllm-ascend/MD5SUMS` |
| C14 | DSpark draft: eager vs captured graph, numerical comparison | `bash tools/draft_ab_launch.sh <chip> && bash tools/draft_ab_run.sh` |
| C15 | DSpark graph on/off in one process (no restart) | write `DRAFT_FORCE_EAGER=1` into `/tmp/v41_dspark_flags`, then compare `ms/step` and spec-decoding metrics in the same session |
| C16 | Communication/compute overlap, and the reason it is zero | `PROFILE=1` serve run, then the method in `reports/comm-compute-overlap-cannbot.md` (comm∩compute interval union) |
| C17 | MoE AllGather vs MC2 at TP=EP | serve with and without `V41_MOE_COMM_ALLGATHER=1`, then C9 at 8K/32K/128K and diff the outputs |
| C18 | Expert-mask path cost (operator level) | operator bench with a synthetic `expert_map[384]` and `topk_ids[8,6]`, gate `V41_MOE_MASK_RANGE=1` on/off |
| C19 | Which patches are actually loaded in a running container | `bash tools/verify_baked_tree.sh` |
| C20 | Self-checks really detect the bugs they claim to detect | `bash tools/negative_control.sh` |

---

## Appendix A — Source index

| Claim group | Primary sources under the evidence root |
|---|---|
| Patch inventory, gates, per-patch effects, dependency order | `patches/README.md` §3–§4, `PATCHES.md`, `patches/vllm-ascend/*.patch` |
| Engram footprint / residency / machine criterion / device-index results | `CHANGELOG.md` v8 §0–§3, `quant/README.md` L2, `README.md` §1 |
| Host-path breakdown, route phases, exposure measurement | `reports/engram-host-breakdown.md`, `reports/engram-final-quantification.md` |
| JIT results | `reports/engram-jit-verified.md` |
| MoE dispatch/combine and mask results | `reports/moe-allgather-breakthrough.md`, `reports/moe-mask-range-verified.md` |
| RoPE fusion and QLI fast path | `reports/rope-idxsel-verified.md`, `reports/qli-no-candidate-verified.md` |
| `wo_a` 2D matmul | `reports/f3-wo-a-2d.md` |
| Draft-graph results, per-rank `[bneck]` fields, A2 vs A3 | `CHANGELOG.md §6`, `reports/draft-graph-investigation-20260920.md`, `reports/a2-draft-graph-20260920.md` |
| Static-kernel disable trap | `reports/static-kernel-silent-disable-fix.md` |
| Overlap measurement and its cause | `reports/comm-compute-overlap-cannbot.md` |
| Concurrency sweep raw data | `results/bench/conc_dihuo_v8.json`, `CHANGELOG.md §9` |
| Capacity and HBM accounting | `EXPECTED_PERF.md §6`, `docs/prefill-memory-headroom.md` |
| Probes | `tools/probe_a2_hostmap.py`, `tools/probe_engram_hostreg.py`, `tools/run_probe_engram_hostreg.sh`; standalone probe draft at `upstream-v41/pr/probe_engram_hostmap.py` (to be published as `tools/probe_engram_hostmap.py`) |
| Upstream comparisons | upstream issue #16828, PR #16689, PR #16925, PR #16544 (as cited inline) |

## Appendix B — Claim discipline for anything derived from this document

1. Cite the **item number and its text**, never the number alone.
2. Never turn a same-session paired measurement into a cross-session comparison.
3. Never state a per-card or throughput multiple without the 【converted】 label and the
   caveat list in §4.3.
4. Never describe an observed boundary behaviour as a defect in another implementation.
   State the observation, the scope, the command, and the falsification condition.
5. When a quantity named by an RFC item has not been measured, write "not measured" rather
   than substituting a nearby number.
