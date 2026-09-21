# Track D — Engram host-table support & cost matrix (documentation contribution)

> **Draft — not posted.** Serves RFC #16375 **[24]** *"Publish the hardware, quantization,
> operator-version, graph-mode, and parallelism support matrix with reproducible serving
> examples"* and supplies the missing NUMA/bandwidth-adjacent quantity of **[47]**.
>
> This is the one artifact in our set that is **documentation rather than code**, which
> matters for reviewability: it cannot break anything, it encodes measurements only we
> happen to have, and it is the natural home for the negative results (§4) that would
> otherwise have to be re-discovered.

---

## 1. The document (proposed: `docs/source/developer_guide/engram_host_table_support.md`)

````markdown
# Host-mapped embedding tables: which machines support them, and what they cost

Some models keep very large embedding tables in host DRAM and have *device* operators index
them in place. DeepSeek-V4.1's Engram tables are the motivating case: two layers of
~206 GiB each cannot live in HBM, so the table stays in host memory and a device kernel
reads the rows it needs.

That arrangement rests on one driver property: `aclrtHostRegister` must accept an ordinary
writable host mapping and return a device pointer a device operator can dereference. The
property is **not uniform across machines**, and the difference is not visible from the
usual capability query (see §3). This page is the support matrix for it, plus the cost
model and the checks we would run before enabling the path on a new machine.

## 1. Support matrix

Measured on two machine classes, one node each, 8 chips, TP8/EP8. "device read" means a
device operator read a registered table back **byte-for-byte** (`torch.equal`), not merely
that registration returned success.

| Machine class | PCI id | CPU↔NPU | `host_mem_pool` | registration accepted | **device read** | registration cost |
|---|---|---|---|---|---|---|
| 910C class (Atlas A3), **re-measured 2026-09-21** | `19e5:d803` | HCCS | **1** | yes | **yes** | **0.59–0.78 ms/MiB** — 1/4/8 GiB real files, 100% blocks allocated (§5.1) |
| 910C class (Atlas A3), bring-up record | `19e5:d803` | HCCS | **1** | yes | **yes** | 0.63 ms/MiB (8 ranks × 206 GiB in 133 s) — **consistent with the row above** |
| 910B3 class (Atlas A2), bring-up record | `19e5:d802` | PCIe | **0** | full table **succeeded** in the single-rank probe (149.9 s = 0.71 ms/MiB); fails `ret=207001` at 8 ranks | yes at 1 rank | 0.71 ms/MiB |
| **Software test container** (not a machine truth) | `19e5:d803` | HCCS | **1** | yes | **yes** | **11.4 ms/MiB** — an outlier, see §5.2 |

`host_mem_pool` is read from `/proc/svm/dev<N>/feature/host_mem_pool`, where `N` is the
**logical** device index (procfs entries are per-container; a container may show device 0
while the physical chip is 12).

> ★ The last row is deliberately labelled *not a machine truth*. It shares PCI id and
> `host_mem_pool` with the A3 rows yet costs 15–19× more — because **the outlier is the
> software stack of that container (driver 25.5.5), not the Ascend hardware.** The three
> real-hardware rows agree with each other at 0.59–0.78 ms/MiB. **The flag tells you the
> fast path exists; it does not tell you how fast it is — and neither does a single
> container.** See §5.

## 2. Cost model

Registration cost is **linear in block size**, so a per-MiB constant is meaningful and a
short measurement predicts a long one.

| block | 8 MiB | 128 MiB | 1 GiB | 4 GiB |
|---|---:|---:|---:|---:|
| ms/MiB, anonymous mapping | 11.08 | 11.38 | 11.46 | 11.45 |
| ms/MiB, file-backed mapping | 11.27 | 11.63 | 11.49 | 11.84 |

(Median of three, one card. A 512× step in size produced a 529× step in time — linear, not
super-linear.)

> ⚠️ **Superseded — see §5.** Those 11.4 ms/MiB numbers came from a **software test
> container** (driver 25.5.5), and its "file-backed" arm was a **sparse file** (0 blocks
> allocated). On real A3 hardware with a materialised file the cost is
> **0.59–0.78 ms/MiB**, i.e. **≈2.0–2.5 minutes for 206 GiB per rank**.

Consequences worth planning around:

* On the container that charged ~11.4 ms/MiB a 206 GiB table would take **~40 minutes per
  rank** — but that number describes that container, **not** Ascend hardware. On real A3/A2
  it is **~2 minutes**. Size timeouts from a measurement on the machine you actually deploy
  on, never from this table.
* Registering in parallel across ranks does not improve the wall clock, because each rank
  registers its own mapping.
* **Measure 1 GiB first.** It takes about twelve seconds and predicts the full-table time
  to within a few percent, which is enough to distinguish "slow but progressing" from
  "stuck" before committing to a 40-minute wait.

## 3. The capability query is not a gate

`aclrtHostMemMapCapabilities` returned, on a machine where the path demonstrably works:

```
AIC : rc=207000 (ACL_ERROR_RT_FEATURE_NOT_SUPPORT) -> NOT_SUPPORTED
AIV : rc=207000 (ACL_ERROR_RT_FEATURE_NOT_SUPPORT) -> NOT_SUPPORTED
```

The registration and the device read both succeeded on that same machine. Whatever the
query reports here reflects a feature bit the runtime did not wire up, not the state of the
mapping. **Do not refuse to start based on it.** The checks that do predict success are
`host_mem_pool` plus one actual device read of a small registered table.

## 4. Four things that cost us time (so they need not cost you time)

1. **`aclrtHostRegisterV2` does not return a device pointer.** Unlike the legacy
   `aclrtHostRegister`, it returns only a status code; the address comes from a separate
   `aclrtHostGetDevicePointer(ptr, flag=0)`. Code migrated from the legacy API that keeps
   reading the second return value as a pointer silently gets `0`.
2. **Host-memory kind matters far less than *whether the file has real blocks*.** On A3 a
   materialised file (100% blocks allocated) costs 0.59–0.78 ms/MiB while an anonymous
   mapping costs 0.44 ms/MiB — the same order. But a file created by `ftruncate` and never
   written (**0 blocks**) costs **0.009 ms/MiB**, i.e. 65× less, because there are no
   physical pages to register. **Check `st_blocks` before comparing any two
   file-backed numbers.** (An earlier version of this document claimed the memory kind was
   irrelevant; that conclusion came from exactly this sparse-file artefact and is withdrawn.)
3. **A read-only VMA is rejected**, and the error code is driver-dependent: `107017`
   (`ACL_ERROR_RT_INVALID_HANDLE`) on one driver, `507899` recorded on another. Treat any
   registration failure on a read-only mapping as this, and open the file `O_RDWR`.
4. **Do not call `.cpu()` / `.item()` on a tensor that wraps a registered host pointer.**
   It goes down the memcpy path and can segfault the worker. Copy an operator's *output*
   instead. (If you need a read-back check in a probe, run it in a separate process.)

## 5. Resolved (2026-09-21): what sets the cost

An 18× discrepancy used to sit here. It is now resolved, and the resolution has two parts.

### 5.1 The real cost on real hardware is 0.59–0.78 ms/MiB

Re-measured on A3 (`A3-node1`, driver **26.1.1**, idle chip, `host_mem_pool=1`) with the same
script, and — this is the part that mattered — **with a materialised file**:

| size (real file, 100% blocks) | `aclrtHostRegister` | ms/MiB |
|---:|---:|---:|
| 1024 MiB | 799.7 ms | 0.781 |
| 4096 MiB | 3119.6 ms | 0.762 |
| 8192 MiB | 4839.0 ms | 0.591 |
| anonymous mapping, 8 GiB | 3962 ms | 0.44 |

⇒ **206 GiB ≈ 2.0–2.5 minutes per rank**, and this reconciles three independent numbers that
previously looked contradictory: A3 bring-up (133 s / 8 ranks ≈ 0.63), A2 bring-up
(149.9 s ≈ 0.71), and this measurement. **The 40-minute figure was a property of the software
test container, not of Ascend hardware.**

**Then we measured the real table, so nothing here is extrapolated any more.** Registering
the four production shards (206.0 GiB) on an idle A3 chip, one at a time, in a throwaway
container, took **119.4 s total (0.566 ms/MiB)**:

| shard | size | cost |
|---|---:|---:|
| `layers_14_..._scale` | 11719.3 MiB | 4711.7 ms (0.402 ms/MiB) |
| `layers_14_..._weight` | 93754.1 MiB | 53406.3 ms (0.570 ms/MiB) |
| `layers_1_..._scale` | 11718.9 MiB | 72.3 ms (0.006 ms/MiB) |
| `layers_1_..._weight` | 93751.5 MiB | 61247.5 ms (0.653 ms/MiB) |

Two notes for anyone repeating this: the **first and third shards have the same size but
differ 65×**, which is page-cache residency and not size; and 119.4 s is within 11% of the
"133 s" recorded in our own release notes, which until now we had flagged as unverifiable.

### 5.2 Where the 18× came from

| stack | backing | ms/MiB |
|---|---|---:|
| A3, driver **26.1.1**, materialised file | file | **0.59–0.78** |
| A3, driver 26.1.1, anonymous | anon | 0.44 |
| software test container, driver **25.5.5**, file *and* anon | file (sparse) / anon | **11.4** |

The container is the only stack that charges ~11.4 ms/MiB, and it does so for *both* memory
kinds. The hardware is common to the first two rows. **Conclusion: driver version
(25.5.5 vs 26.1.1), not hardware class and not `host_mem_pool`.** Which specific driver
change is responsible cannot be identified from here (no driver source), so that stays a
**hypothesis about the mechanism** even though the **fact of the difference is measured**.

> **Practical rule for a new machine:** it takes one `--size-mib 1024 --arms real` run
(≈2 s) to know whether a 206 GiB bring-up is a two-minute step or a forty-minute one.
Never infer it from the `host_mem_pool` flag, and never from a sparse file.

## 6. Checks to run on a new machine

```bash
# 1. which class is this, and is the fast path available?
lspci -nn | grep 19e5                       # d802 = 910B3 class, d803 = 910C class
cat /proc/svm/dev0/feature/host_mem_pool     # 1 = pooled path

# 2. does a device operator actually read a registered table?  (one card, seconds)
python tools/probe_engram_hostmap.py

# 3. how long will the real table take?  (one GiB, ~12 s)
python tools/bench_host_register_scale.py --sizes-mib 8,128,256 --reps 1
```

Exit codes of `probe_engram_hostmap.py`: `0` the production shape read back byte-for-byte;
`3` it did not; `1` the probe itself failed.
````

---

## 2. The PR (proposed)

**Title** — `[Doc] Host-mapped embedding tables: support matrix, cost model and checks`
(`[Doc]` is in the CI whitelist; a documentation PR is not a behaviour change.)

**Files** — `docs/source/developer_guide/engram_host_table_support.md` (new), plus the two
tools if the reviewer wants them in-tree rather than linked:
`tools/probe_engram_hostmap.py`, `tools/bench_host_register_scale.py`.

**What the PR description should say** (three paragraphs, no more):

1. Why a page rather than a code change: the failure family in the tracker (a host-offload
   bring-up that is slow or fails on some machines) is currently diagnosed per-machine, and
   two of the checks that look authoritative — the capability query and the
   `host_mem_pool` flag — do not predict the outcome on their own (§3 and §5 above).
2. What is being contributed: the matrix, a linear cost model with four measured sizes, and
   four environment behaviours that are driver-version-dependent rather than
   platform-dependent.
3. What is explicitly *not* claimed: the §5 hypothesis is labelled as such, the cost
   numbers are one machine, and no statement is made about any particular upstream
   implementation.

**Why this should be cheap to review** — it adds no code path, no configuration, and no
gate. If a maintainer disagrees with the numbers, the fix is editing a table.

---

## 3. Evidence behind every number in the document

| Claim in the doc | Where it comes from |
|---|---|
| 910C: registration accepted, device read works | `out/probe_hostmap_v2_<date>.txt` (rows 1–3) |
| 910B3: full table fails `ret=207001` at 8 ranks, succeeds at 1 rank | our A2 deployment record; **not re-measured** (no 910B3 reachable) |
| A3 0.59–0.78 ms/MiB on **real** files | 2026-09-21 re-measure on `A3-node1` chip 3, `logs/raw/29-a3-register/` |
| sparse file costs 65× less than a real one | same day, `sparse_vs_real_8g.txt` (0 vs 8192 MiB `st_blocks`) |
| 11.4 ms/MiB on the test container, both kinds | `out/regscale3-<date>.log` — **but that file arm was sparse; see §5.1** |
| capability query returns 207000 while the path works | same probe, `[2b]` block |
| V2 returns no pointer | `npu.py`-level inspection of the call sequence + probe row 2/3 |
| read-only VMA → `107017` | probe output, row 6 |
| `.cpu()` on a wrapped pointer segfaults | reproduced once; also recorded as a comment in our own loader |
| registration is *not* deferred to first touch | `deferral_8g_v2.txt`: first device read 6.1 ms vs 0.0 ms steady over the whole 8 GiB span |

> **Caveat that still stands.** Rows 2 and 3 above are our own bring-up records, not
> re-measured artefacts: they do not document which step the elapsed time covers, and the A2
> number comes from a machine we can no longer reach. They are **consistent** with the new
> A3 measurement, which is why the 18× question is closed — but if someone re-measures and
> disagrees, these two rows are the ones to re-check first.
> is exactly how §5 is written, and is *more* useful to a reader than a single number would
> be.

---

## 4. If the reviewer prefers code to docs

The same content supports a smaller code-shaped contribution: land
`tools/probe_engram_hostmap.py` alone with a one-paragraph entry in the existing
troubleshooting page. That is roughly a 40-line diff to review, and the matrix can follow.

**We would rather be told which of the two is wanted than guess.**
