`Tracks RFC #16375 items [46][47][48][49][50][77]`

> **Draft implementation-issue body — not posted.** Written in the form RFC #16375
> line 3 invites: *"Release targets and owners can be attached to individual
> implementation issues as they are agreed."*
>
> **Proposed issue title:** `[Feature][dsv4.1][Engram] Host-resident tables with device-side lookup and graph capture`

| Field | Value |
|---|---|
| Target branch | `main` (RFC line 18) |
| Suggested labels | `feature`, `performance`, `dsv4.1` |
| Proposed owner | @chiro2001 — offered, not asserted; maintainers may attach whoever they prefer |
| Evidence root | https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B |
| Hardware behind every number below | 8 × Ascend 910C (A3) and 8 × Ascend 910B3 (A2), TP8/EP8, single node, W4A8 |
| What this issue is **not** | a request to accept a specific implementation, and not a comparative performance claim |

---

## 1. What "done" looks like for these items

Written from the RFC text, not from our implementation. These are the criteria we would
expect a reviewer to apply before any checkbox in this group is ticked.

| Item | Completion criteria |
|---|---|
| **[46]** | CPU/HBM-resident embedding tables work in the V4.1 serving path with **batched** lookups; the staging/transfer design (pinned bounded buffer + async H2D, or another mechanism) is explicit, bounded and documented, including what happens when the bound is exceeded; the lookup cost is hidden or accounted for under realistic concurrency |
| **[47]** | A documented residency policy (what lives in HBM, what in host DRAM, whether hot rows are cached) **plus measured values** for: table footprint, lookup latency, host↔device transfer volume, NUMA/bandwidth sensitivity, and behaviour under realistic concurrency — each with a scope and a command |
| **[48]** | Engram TP ownership is defined: which embedding rows/features and projection weights each rank owns; node-level table sharding is explicitly distinguished from model TP; lookup routing, result redistribution and projection reduction are specified, including the case where sharding is *not* used |
| **[49]** | Duplicate lookups across TP ranks are eliminated or justified; per-layer queries are batched where the RFC asks for it; empty-query ranks and collective ordering are covered by tests, not by assumption |
| **[50]** | Offload + TP validated together with **SP, DCP, PD and graph replay**, covering token history, padding masks, persistent input-buffer refresh, and transfer-event lifetimes |
| **[77]** | The eager/graph boundary for CPU-side Engram work and dynamic communication metadata is written as a rule (what may be captured, what must stay outside), persistent graph inputs are refreshed before replay, and host synchronisation is kept off the captured computation path — with a measurable before/after |

A reviewer should also be able to reproduce each of these on one node; a criterion that
needs a 32-NPU instance to check is not a good acceptance criterion for this group.

## 2. Existing evidence

All numbers are paired (same session, same machine) unless the scope column says otherwise.
Every row carries a command that reproduces it from a published repository.

| Item | Numbers we hold | Scope | One command | Source / repo link |
|---|---|---|---|---|
| **[46]** | CPU-resident INT8 tables **206.0 GiB** (4 shards: 11.4 / 91.6 / 11.4 / 91.6 GiB), batched lookups, lookup executed on device reading host DRAM; removing the H2D/pinned leg moved decode **35.3 → 32.1 ms/step** (C4) and **29.5 → 28.4 ms/step** (C1) | A3, decode, real weights | `du -sBG --apparent-size "$MODEL"/engram_int8/* \| sort -n` | [`quant/README.md` L2](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/quant/README.md), [`CHANGELOG.md` v8 §0](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/CHANGELOG.md) |
| **[47]** | footprint 206.0 GiB; lookup latency hash **0.427 → 0.076 ms**, plan **0.261 → 0.068 ms** (A3, per step); per-rank transfer `d2h` **0.19–3.41 ms**, `route` **1.34–2.89 ms** (A2, host path, 8 ranks); machine split A2 vs A3; concurrency C1…C64 with `ok=64/64` (A3) | A3 and A2, decode; per-rank spread over 8 ranks | `V41_ENGRAM_JIT=1 NUMBA_CACHE_DIR=./cache/numba bash scripts/serve_a3.sh` then `grep -h "\[bneck\]" "$LOG" \| tail -n 20` | [`patches/README.md` §3](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/README.md), [`reports/engram-jit-verified.md`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/engram-jit-verified.md), [`reports/a2-draft-graph-20260920.md` §3.1](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/a2-draft-graph-20260920.md) |
| **[48]** | Deliberate decision to keep **one full table per rank** and index it on device, instead of sharding; the sharded alternative was measured and rejected: `all_to_all` + `broadcast` ≈ **0.5 ms/step**, net gain ceiling ≈ **6%** | A3, decode | `V41_ENGRAM_LOCAL_OWNER=fast bash scripts/serve_a3.sh` and compare against the sharded arm in `reports/engram-host-breakdown.md` §2 | [`patches/files/engram_hbm.py`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/files/engram_hbm.py), [`reports/engram-host-breakdown.md`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/reports/engram-host-breakdown.md) |
| **[49]** | metadata `all_gather` and ids `all_to_all` removed from the decode path; pre-graph device path `route` **2.462 ms**, with the lookup inside the captured graph host enqueue is **0.058 ms** (residual device work 0.695 ms overlaps). Host-synchronous time per step **3.379 → 0.058 ms** | A3, decode, 8 ranks | compare `V41_ENGRAM_DEVICE_INDEX=0` vs `auto`: `grep -h "\[bneck\]" "$LOG" \| tail -n 20` | [`CHANGELOG.md` v8 §0–§1](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/CHANGELOG.md) |
| **[50]** | Graph replay: yes — one ACLGraph per batch shape, zero-copy capture on the model's own buffers, `data_ptr`+shape validation before every replay; decoding with the lookup captured is the shipped A3 default | A3, decode only | `V41_ENGRAM_DEVICE_INDEX=auto bash scripts/serve_a3.sh` then `grep -a "\[DEVICE-INDEX\]" "$LOG"` | [`patches/files/engram_graph.py`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/files/engram_graph.py) |
| **[77]** | Boundary rule written down (persistent buffers and pointer-stable indices may be captured; dynamic collective metadata and changing slot mappings must not be); persistent-input refresh + pointer/shape check before replay; host-synchronous time **3.379 → 0.058 ms/step** | A3, decode | `V41_ENGRAM_DEVICE_INDEX=0` vs `auto`, same session, diff `[bneck]` `total` and step time | [`patches/files/engram_device_index.py`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/files/engram_device_index.py), [`patches/files/engram_graph.py`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/patches/files/engram_graph.py) |

Machine-class criterion that decides whether the device path is available at all
(relevant to [46] and [47]):

| Machine | PCI id | CPU↔NPU | `host_mem_pool` | Observed |
|---|---|---|---|---|
| **A3 (910C), on the production table itself, 3 dies concurrently** (2026-09-21) | `19e5:d803` | HCCS | **1** | one die registers all 206.0 GiB in **122.2 s** (0.580 ms/MiB; 99.9 s warm) and upstream's `aclrtHostRegisterV2(MAPPED\|PINNED)` call in **83.9 s** (0.398 ms/MiB); **three dies registering it at once: 240.9 / 241.3 / 238.9 s — 24 shard-registrations, all `ret=0`, no `207001`, no `507011`**; a concurrent re-registration did **not** disturb the other dies' read bandwidth (≤0.8 %) |
| A3 (910C) | `19e5:d803` | HCCS | **1** | 8 ranks × 206 GiB registered; bring-up time recorded as ≈133 s — **the span that number covers is not documented in our notes, see the cost caveat below** |
| A2 (910B3) | `19e5:d802` | PCIe | **0** | full-table registration fails `ret=207001` after ~17 min (MemAvailable still 703 GiB); single rank succeeded (149.9 s / 159.5 s) |
| 910C-class dev container, driver 25.5.5 | `19e5:d803` | HCCS | **1** | probe reports supported; **device read verified byte-for-byte** |

**Registration cost, newly measured on the third machine** (four sizes spanning 512×, three
repetitions each, anonymous *and* file-backed mappings):

| block | 8 MiB | 128 MiB | 1 GiB | 4 GiB |
|---|---:|---:|---:|---:|
| ms/MiB (anonymous) | 11.08 | 11.38 | 11.46 | 11.45 |
| ms/MiB (file-backed) | 11.27 | 11.63 | 11.49 | 11.84 |

Cost is **linear on that software container** (512× size ⇒ 529× time):
**≈11.4 ms/MiB ⇒ ≈40 min for a 206 GiB table per rank**.

**Re-measured on A3 hardware on 2026-09-21 and the 40-minute figure did not survive:** with a
materialised file and an idle chip on driver 26.1.1 the cost is **0.59–0.78 ms/MiB**
(1/4/8 GiB real files, 100% blocks allocated) ⇒ **206 GiB ≈ 2.0–2.5 minutes per rank**,
matching our A3 (133 s) and A2 (149.9 s) bring-up records. The 18× outlier was the container's
software stack (driver 25.5.5), not the hardware.

Two traps, both of which we fell into: a **sparse** file (built with `ftruncate`) reports
65× cheaper registration than a real one — check `st_blocks`; and the `host_mem_pool` flag
reads `1` on the container *and* on A3 despite the 18× gap, so it does not predict cost.
Measure on the machine you deploy on. Practical upshot: registering 1 GiB predicts the full-table time
only if that 1 GiB is a *real* file on the *deployment* stack.

**Two operational caveats the production-table run surfaced** (both measured, neither is a
bug in anyone's code):

1. **The API dirties what it maps.** A read-only VMA is rejected (`107017
   ACL_ERROR_RT_INVALID_HANDLE`), so the mapping must be writable — and registering it marks
   the pages dirty. A bring-up therefore writes the whole 206 GiB back (`Dirty` reached
   108–126 GiB; shard mtimes moved, **contents stayed byte-identical**). `posix_fadvise(DONTNEED)`
   returns 0 but frees **zero** pages while any rank still maps the file, so "register then
   drop the cache" is not a workaround on this driver.
2. **Row width, not table size, sets the read throughput.** Device reads through the
   registration on the real table (256 B rows): **107 GB/s** contiguous but only **7.55 GB/s**
   for a uniform random row gather — 12.7× below the same code on a synthetic 20480 B-row
   table. **Skew reverses that**: 80 % of queries into 0.1 % of rows gives **2.6–4.3×**, so
   *hot-row caching* (the clause [47] names) is worth answering for real workloads, and a
   wide-row synthetic benchmark cannot answer it.
well enough to tell "slow but progressing" from "stuck" before committing to the wait.

```bash
# curve (≈6 min for the full set, <10 s for the short form)
python bench_host_register_scale.py --sizes-mib 8,128,256 --reps 1
python bench_host_register_scale.py --sizes-mib 8,128,1024,4096 --memory both
```

**Four API-level behaviours we hit while getting the probe to be trustworthy.** Each is a
hypothesis-generating observation for anyone debugging this path — none of them is a
diagnosis of any particular implementation:

1. **`aclrtHostMemMapCapabilities` is not a usable gate.** It returned
   `AIC rc=207000 / AIV rc=207000` (feature-not-support) on a machine where registration
   *and* the device read both succeeded. Gating on that query refuses a configuration that
   works; gate on `host_mem_pool` plus one real device read instead.
2. **`aclrtHostRegisterV2` returns only a status code, never a pointer.** The address comes
   from a separate `aclrtHostGetDevicePointer(ptr, flag=0)`. Code migrated from the legacy
   API that keeps reading the second return value as a pointer silently gets `0`.
3. **A read-only VMA is rejected with a driver-dependent code** — `107017`
   (`ACL_ERROR_RT_INVALID_HANDLE`) here, `507899` in our earlier notes. Either way the fix
   is to open the file `O_RDWR`.
4. **Host-memory kind is irrelevant to cost** (see the table above) — worth knowing before
   spending a day choosing between mmap flavours.

```bash
# the whole matrix in one run: PCI id, host_mem_pool, both APIs x three memory kinds, device read
python probe_engram_hostmap.py
```

```bash
cat /proc/svm/dev0/feature/host_mem_pool          # 1 = A3-class, 0 = A2-class
MODEL="$MODEL" FANOUT=8 PROBE_CHILD_FILES=1 bash tools/run_probe_engram_hostreg.sh
```

## 3. ★ What is still missing

This is the part of the issue we would most like help with. Nothing below is measured by us.

1. **SP, DCP and PD are entirely uncovered for [50].** All evidence above is single-node
   TP8/EP8, decode. Token history, padding masks and transfer-event lifetimes under SP/DCP,
   and Engram N-gram history reconstruction across a P→D boundary, are open.
2. **No NUMA / host-bandwidth sensitivity measurement for [47].** We have a machine-class
   split and per-rank spread; that is not the same quantity the item asks for.
3. **No bounded-pinned-staging design for [46].** We removed the H2D leg instead of
   bounding it. On machines where device-side lookup is unavailable (`host_mem_pool=0`,
   i.e. 910B3-class), the item's mechanism is the only path — and we have not built it.
   That machine class is the one that needs it most.
4. **No row/feature ownership split for [48].** The measured path deliberately keeps a full
   table per rank. If a deployment cannot afford 206 GiB of resident host DRAM per node,
   our result does not apply and sharding (with its measured ≈0.5 ms/step cost) has to be
   designed properly.
5. **Empty-query ranks and collective ordering for [49]** are untested; "batch queries
   across Engram layers" is not implemented — layers are still called per step.
6. **Graph coverage is decode-only, and prefill still gathers the full table (16.8 ms).**
   A segmented prefill alternative was verified (5.36 ms, 2.44×) but is **not wired up**.
7. **The fused INT8 kernel cannot be used on host-mapped memory.**
   `gather_dequantize_engram_int8` rejects host-mapped pointers (the pointer location is
   neither DEVICE nor HOST_NUMA), so the device path falls back to aclnn. At decode scale
   that is only ~0.03 ms, but it means the fusion the RFC names in [90] is not applied on
   this path.
8. **The upstream code line and ours are different lineages.** The measured stack sits on
   `GDzhu01/vllm-ascend-v41-private@46856f89e` (flat `engram_*.py` files); the implementation
   being brought into main (`#16925`) uses an `engram/` subpackage. Porting is real work,
   not a cherry-pick, and we would rather do it against a maintainer-agreed target than
   guess.
9. **No upstream-style unit tests.** Our assets are end-to-end acceptance gates; the
   RFC's items imply focused regression tests for state ownership and transfer lifetimes.

## 4. How we would like to use this issue

* If the direction is right, the natural split is four reviewable PRs: (a) pre-indexer
  kernel fusion, (b) host-resident table + batched lookup, (c) device-side lookup with the
  graph boundary, (d) machine-class capability probe as a standalone tool — the draft
  `tools/probe_engram_hostmap.py` reports the PCI id, `host_mem_pool`, and the return codes
  of both `aclrtHostRegister` and `aclrtHostRegisterV2`, imports neither vllm nor vllm_ascend,
  and can land independently of the V4.1 model code.
* Items 1, 4 and 6 above are the ones where a second pair of hands changes the outcome. If
  an owner for SP/DCP/PD already exists, we would rather hand over the checklist than
  duplicate it.
* If the maintainers prefer the RFC's [46] mechanism (bounded pinned staging + async H2D)
  over a device-side lookup, say so and we will measure that arm instead — we have the
  harness, we simply measured the other fork first.

**Claim discipline for this issue.** No statement in this draft compares our implementation
to any existing implementation. Every number is scoped, sourced and reproducible; where an
RFC-named quantity has not been measured, it is written as "not measured" rather than
replaced by a nearby number.
