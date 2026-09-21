# 38 — Host-DRAM bandwidth for a device-side Engram lookup (RFC #16375 **[47]**)

> 2026-09-21 · sub-agent **B_bw** · A3 (A3-node1, 910C class) · container slots c0/c1/c2
> (dies 3/6/7) · `torch 2.10.0+cpu` + `torch_npu 2.10.0.post4` · driver/npu-smi 26.1.1
>
> **Question this log answers:** for the deployed "Engram table stays in host DRAM and a
> device operator indexes it in place" design, what bandwidth does a die actually get, how
> does it change with NUMA placement and with several dies reading at once, and what does
> HBM cost by comparison?
>
> **Scope limit (read first):** every number here is from a **synthetic** table
> (512 MiB / 2048 MiB anonymous mapping, 20480 B rows), not from the real 206 GiB Engram
> table, and not from a serving run. Claim vocabulary: **【实测】** = measured this session,
> **【推断】** = derived, **【未确认】** = not established.

---

## 1. Design (written before the measurements)

### 1.1 What is measured

| Arm | Operation | Bytes counted as "useful" |
|---|---|---|
| `contiguous_<N>MiB` | `dst_hbm[:N].copy_(host_mapped[:N])` — device copies N MiB straight out of the registered host mapping | N MiB |
| `gather_<R>rows` | `torch.index_select(host_table, 0, ids, out=)` — R random rows x 20480 B, uniform over the whole table | R x 20480 B |
| `gather_hot_<R>rows` | same, but 90 % of the queries fall inside 0.1 % of the rows (hot-row-caching proxy) | R x 20480 B |
| `full_table_h2d` | the whole table copied host -> HBM once (the transfer the design avoids) | table size |
| `hbm_contiguous_<N>MiB`, `hbm_gather_<R>rows` | the same two operations with the table in HBM | as above |

Row width 20480 B = 5120 x 4, i.e. the Engram row shape named in the task. Table size is
an argument (default 2048 MiB; 512 MiB for the NUMA sweep so that nodes with little free
memory can still be measured). `index_select` over a host-mapped tensor is the exact
production mechanism (patch 0009, `V41_ENGRAM_DEVICE_INDEX`), and the same call is used by
`pr/probe_engram_hostmap.py` for its byte-exactness check.

### 1.2 How it is measured

* Registration (`aclrtHostRegister`, flags `MAPPED` = 0x0) is timed **once, outside** every
  timed loop; all quoted bandwidths are steady-state, already-registered numbers, i.e. they
  **exclude registration time**. The JSON repeats this in `units`.
* Each arm runs `--warmup 2` untimed iterations and `--repeat N` timed ones (7 for the
  sweep, 9 for the concurrency comparison). Every iteration is recorded individually, and
  the JSON keeps the raw list; the log quotes median and best.
* Two clocks per iteration: `wall_us` (host launch + device work, what a caller waits for)
  and `event_us` (device-busy time from NPU events). `GB/s` = 1e9 B/s (decimal).
* **Sanity gate before any measurement:** 3 rows at 0 / rows/3 / last row are gathered on
  the device and compared byte-for-byte against the host bytes; a mismatch aborts the run
  instead of producing a number.
* Throughput for gathers counts *useful* bytes (rows x row width). Real DRAM traffic is
  >= that (fetch granularity, TLB), so this is an upper bound on useful bandwidth, not a
  claim about DRAM bus occupancy.

### 1.3 Concurrency

`--barrier-dir` + `--barrier-peers 3`: each of the three container slots writes a ready file
into the shared `/work` tree after its setup (allocation, registration, sanity read) and
then waits for the others; the common go instant is derived from the newest ready file's
mtime, so the three timed loops start within milliseconds of each other. Measured drift is
recorded in each JSON (`concurrency.drift_s`). Phase order: 1 die alone first, then 3 dies
simultaneously, so the reference and the concurrent run are seconds apart.

### 1.4 NUMA: what is possible on this container, and what is not

`numactl --membind=N` (the literal instruction in the task) **does not work in this
container**:

```
set_mempolicy: Operation not permitted
setting membind: Operation not permitted
```

Docker's default seccomp profile permits `set_mempolicy`/`mbind` only with `CAP_SYS_NICE`,
which the benchmark containers do not have. **【实测】** — reproduced on slot c0 at
12:55:25, and it is the reason the first sweep attempt (12:54:15) produced 6 failed runs
before being stopped.

The equivalent measurement that *is* available: restrict CPU affinity to the CPUs of node
`N` (`numactl --cpunodebind=N`, fallback `taskset -c <node cpus>`) and then touch the buffer.
Under the default memory policy, anonymous pages are placed on first touch **on the node of
the touching CPU**, so this places the pages just as `--membind` would — and it needs no
mempolicy syscall. The script does not *assume* this: after the first touch it reads
`/proc/self/numa_maps` and records the kernel's own per-node page counts for the buffer, and
warns if the placement is not the requested node.

**【实测】 placement check** (slot c0, 256 MiB):

| Launch | `Cpus_allowed_list` | `numa_maps` for the buffer |
|---|---|---|
| plain `python3` | 0-639 | `N1=65395 N7=141` (split) |
| `numactl --cpunodebind=4` | 320-399 | `N4=65536` (single node, 256 MiB) |
| `taskset -c 0-79` | 0-79 | `N0=65536` (single node, 256 MiB) |
| `numactl --membind=4` | — | **fails**, `Operation not permitted` |

### 1.5 Hypotheses to test

*(written after the first 3-die run, which had shown one die unaffected and two halved,
but before the NUMA sweep and the concurrency matrix that separate placement from
concurrency)*

1. Contiguous host-DRAM reads should saturate well below HBM and below the die's own copy
   engine; larger blocks should approach an asymptote (per-transfer overhead amortised).
2. Random gathers should be *slower* than contiguous reads of the same volume, because each
   20 KiB row lands in a different DRAM page.
3. If host DRAM is the shared bottleneck, three dies reading at once should each get
   materially less than one die alone. If each die's HCCS host path is the bottleneck,
   three dies should *not* interfere — this is the measurement the RFC's phrase "under
   realistic concurrency" is actually asking about. The first 3-die run (12:56) showed
   one die at the full 107 GB/s and two at ~55 GB/s, which decided nothing on its own:
   the three tables had landed on nodes 4, 0 and 1, so placement and concurrency were
   confounded and had to be separated by the matrix in §2.3.
4. NUMA placement should matter only if the host DRAM interface the die uses is
   socket-local; whether it is cannot be read off the machine (see 1.6), so it has to be
   measured.

### 1.6 Topology evidence (collected 2026-09-21 ~12:49–12:57)

**【实测】 NUMA nodes** — `numactl -H` (host and container agree): **8 nodes**, 2 nodes per
socket, 4 sockets x 80 cores x 2 threads = 640 CPUs; node CPUs are contiguous blocks of 80
(node0 = 0-79 … node7 = 560-639); sizes ~258 GB each; distances 10 intra-node, 15 within a
socket pair, 20 across sockets:

```
node distances:
node   0   1   2   3   4   5   6   7
  0:  10  15  20  20  20  20  20  20
  1:  15  10  20  20  20  20  20  20
  2:  20  20  10  15  20  20  20  20
  3:  20  20  15  10  20  20  20  20
  4:  20  20  20  20  10  15  20  20
  5:  20  20  20  20  15  10  20  20
  6:  20  20  20  20  20  20  10  15
  7:  20  20  20  20  20  20  15  10
```

Free memory is very uneven on this shared box (`node0` ~1.0–2.9 GB, `node6`/`node7`
~0.3 GB, `node4` ~60 GB, `node5` ~43 GB at 12:57). The NUMA sweep therefore checks each
node's free memory before launching and **skips** nodes that cannot hold the table rather
than letting the allocation OOM.

**【实测】 NPU <-> NPU topology** — `npu-smi info -t topo`, 16 dies: every off-diagonal cell
is `HCCS_SW` (HCCS through a switch), the diagonal-adjacent pair `SIO`; **no `SYS`/`PHB`/`PIX`
cell appears anywhere**, i.e. no die reaches another over PCIe or over an SMP interconnect.

**【实测】 host-memory feature** — `/proc/svm/dev0..15/feature/host_mem_pool` = **1** for all
16 devices (A3 class, the fast registration path); 16 x `19e5:d803` accelerators present.

**【实测】 NPU NUMA affinity is not exposed** — for all 16 `19e5:d803` devices,
`/sys/bus/pci/devices/<bdf>/numa_node` reads **-1**. So the machine does not publish which
NUMA node (if any) each die's host path belongs to; any per-node effect has to be found by
measurement, and no result may claim "the die's local node" from sysfs.

**【未确认】** whether the container sees a *subset* of host DRAM: `Mems_allowed_list` is
`0-7`, so no cpuset/mems restriction is applied to the benchmark containers.

### 1.7 Alternative paths considered and rejected

| Path | Why rejected |
|---|---|
| `numactl --membind=N` inside the container | seccomp denies `set_mempolicy` (measured, §1.4) |
| restart the benchmark containers with `--cap-add SYS_NICE` | the three slots are shared with other sub-agents running right now; recreating containers would kill their work |
| run the measurement on the host instead of in the container | the host has no torch_npu/CANN python environment of the image |

---

## 2. Results

Raw data: `logs/raw/38-host-dram-bw-*.json` (one file per run, full per-iteration
timings); the same files also exist under `agents/B_bw/out/` on A3-node1. Every row below
is **【实测】** unless labelled otherwise. `GB/s` = 1e9 B/s of *useful* bytes; a copy moves
those bytes **plus** an equal number of bytes into HBM, so the copy-engine's total DRAM
traffic is about 2x the number quoted.

### 2.1 One die alone, host DRAM vs HBM (table 2048 MiB, rows 20480 B, n = 7)

| Arm | GB/s best | GB/s median | wall µs/iter (median) | device-busy µs (median) |
|---|---:|---:|---:|---:|
| `contiguous_16MiB` | 70.42 | 66.23 | 253.3 | 167.8 |
| `contiguous_64MiB` | 95.10 | 92.73 | 723.7 | 644.6 |
| `contiguous_256MiB` | 104.35 | 104.24 | 2575.3 | 2500.9 |
| `contiguous_1024MiB` | 107.14 | **107.09** | 10026.6 | 9944.3 |
| `gather_512rows` (10 MiB) | 57.78 | **57.08** | 183.7 | 111.5 |
| `gather_hot_512rows` | 60.81 | 56.20 | 186.6 | 111.5 |
| `gather_4096rows` (80 MiB) | 95.33 | **95.17** | 881.4 | 808.6 |
| `gather_hot_4096rows` | 96.12 | 95.82 | 875.4 | 801.1 |
| `full_table_h2d` (2048 MiB) | 107.69 | 107.65 | 19947.9 | 19864.2 |
| **HBM** `contiguous_1024MiB` | 603.42 | **601.18** | 1786.1 | 1710.0 |
| **HBM** `gather_4096rows` | 728.12 | **723.59** | 115.9 | 47.5 |
| **HBM** `gather_512rows` | 90.68 | 86.86 | 120.7 | 39.0 |
| **HBM** `contiguous_64MiB` | 518.46 | 507.09 | 132.3 | 55.7 |
| **HBM** `contiguous_256MiB` | 515.72 | 509.54 | 526.8 | 452.3 |

Readings:

1. **Contiguous host-DRAM reads saturate at ~107 GB/s per die** (104 GB/s already at
   256 MiB, 66 GB/s at 16 MiB). The 16/64 MiB rows are partly launch overhead: the
   device-busy time is 86 µs lower than the wall time at 16 MiB, i.e. a fixed ~85 µs of
   host-side launch cost per iteration, which matters for small gathers.
2. **Random-row gather costs roughly half the contiguous rate at the deployed access
   size**: 57 GB/s at 512 rows x 20480 B (10 MiB), 95 GB/s at 4096 rows (80 MiB). One
   gather of 512 rows = **183.7 µs wall / 111.5 µs device-busy** — that is the per-call
   cost a decode step pays for a 10 MiB lookup out of host DRAM.
3. **The hot-row variant showed no gain** (56.20 vs 57.08 GB/s at 512 rows; 95.82 vs
   95.17 at 4096 rows): concentrating 90 % of 512-row queries into 0.1 % of the rows did
   not make the lookup faster at this table size. **【推断】** the host side is
   bandwidth/launch bound rather than DRAM-page bound at 20 KiB granularity, so a hot-row
   cache would have to be justified by a *latency* argument, not by this throughput.
   Whether it helps at 206 GiB is untested.
4. **HBM anchor: contiguous 601 GB/s, gather-4096 724 GB/s** (both counting useful bytes
   once, same convention as the host rows). Host DRAM is therefore **5.6x** slower than
   HBM for contiguous streaming and **7.6x** for the gather (107.09/601.18 and
   95.17/723.59). That is the price of not copying the table into HBM — and it is the
   number to quote against the 206 GiB table, which does not fit in 64 GB of HBM anyway.
5. `full_table_h2d` puts the whole 2048 MiB table into HBM in **19.95 ms** at the same
   107.65 GB/s. **【推断】** a 206 GiB table (221 GB) at this rate is ≈2.05 s per rank per
   copy — i.e. the copy itself is not the reason the design avoids it; the reason is that
   206 GiB does not fit in 64 GB of HBM.

### 2.2 NUMA sweep: one die, one node at a time (table 512 MiB, n = 7, two passes)

Placement was forced by CPU affinity and verified from `/proc/self/numa_maps` (§1.4); the
`node_mib` column below is the kernel's own accounting. Rows 6 and 7 were skipped — the
host had only ~0.3 GB free there, less than the table.

| node | pass 1 `contiguous_256MiB` | pass 2 | pass 1 `gather_4096rows` | pass 2 | placement |
|---:|---:|---:|---:|---:|---|
| 0 | 104.39 | 104.08 | 94.76 | 95.21 | `{'0': 512.0}` |
| 1 | 104.35 | 104.36 | 95.62 | 95.23 | `{'1': 512.0}` |
| 2 | 104.62 | 104.51 | 95.30 | 95.12 | `{'2': 512.0}` |
| 3 | 104.78 | 104.63 | 96.64 | 96.18 | `{'3': 512.0}` |
| 4 | 104.39 | 104.34 | 95.69 | 95.37 | `{'4': 512.0}` |
| 5 | 104.18 | 104.11 | 95.72 | 94.76 | `{'5': 512.0}` |
| 6, 7 | skipped | skipped | | | < 1.3 GB free at launch |

**A single die's bandwidth does not depend on which NUMA node its pages are on** — spread
0.67 % across six nodes for contiguous, 1.9 % for the gather, and the two passes agree
within 0.5 %. On its own that looks like "no NUMA sensitivity"; §2.3 shows where the
sensitivity actually is.

### 2.3 Under concurrency: 1, 2 and 3 dies reading host DRAM at the same time

Three container slots (dies 3, 6, 7) ran the same script with a shared barrier; measured
start drift ≤ 4.5 ms (`concurrency.drift_s` in each JSON). Table 2048 MiB, rows
20480 B, n = 7. Numbers in GB/s; `placement` is the kernel's page accounting.

| phase | die | NUMA node(s) | socket(s) | `contig_1024` | `gather_4096` | `gather_512` |
|---|---|---|---:|---:|---:|---:|
| 1 die alone (`ref1`) | c0 | 0+1 | 0 | 107.16 | 94.77 | 57.57 |
| 1 die alone, forced (`same1n4`) | c0 | 4 | 2 | 107.13 | 95.34 | 58.35 |
| 2 dies same socket (`sockA_c1c2`) | c1,c2 | 2, 3 | **1, 1** | **60.24 / 54.23** | 54.48 / 96.62 | 38.4 / 38.4 |
| 2 dies same socket (`pair12` / `rep2_pair12`) | c1,c2 | 5, 4 | **2, 2** | **54.38 / 60.98** | 94.97 / 54.16 | 33.4 / 38.0 |
| 2 dies same socket (`rep5_pair01`) | c0,c1 | 1, 1 | **0, 0** | **57.61 / 57.89** | 54.83 / 54.70 | 43.0 / 41.3 |
| 2 dies, 2 sockets (`pair01`) | c0,c1 | 4, 1 | 2, 0 | 107.17 / 107.10 | 94.77 / 95.09 | 58.35 / 54.49 |
| 2 dies, 2 sockets (`pair02`) | c0,c2 | 0, 4 | 0, 2 | 107.06 / 107.12 | 95.24 / 94.88 | 56.10 / 56.22 |
| 2 dies, 2 sockets (`sockX_c1c2`) | c1,c2 | 3, 5 | 1, 2 | 107.26 / 107.15 | 95.31 / 95.20 | 59.24 / 55.46 |
| 3 dies, 3 sockets (`sock3_all`) | c0,c1,c2 | 1, 3, 5 | 0, 1, 2 | **107.14 / 107.24 / 107.07** | 94.14 / 93.38 / 94.49 | 52.9 / 56.1 / 58.1 |
| 3 dies, 3 sockets (`rep3_all3`, natural) | c0,c1,c2 | 5, 0, 3 | 2, 0, 1 | 107.14 / 106.74 / 107.05 | 94.02 / 94.96 / 94.39 | 55.6 / 58.6 / 53.9 |
| 3 dies, 2 sockets (`all3`) | c0,c1,c2 | 4, 0, 1 | 2, 0, 0 | 107.01 / 58.13 / 53.92 | 95.05 / 54.56 / 54.17 | 58.9 / 37.9 / 33.6 |
| 3 dies, 2 sockets (`mix3`) | c0,c1,c2 | 4, 5, 1 | 2, 2, 0 | 60.80 / 54.46 / 107.08 | 54.29 / 95.53 / 95.36 | 37.6 / 56.6 / 57.9 |
| 3 dies, 1 socket (`same3n4`) | c0,c1,c2 | 4, 4, 4 | 2, 2, 2 | **38.82 / 38.97 / 38.77** | 39.78 / 39.97 / 38.82 | 29.9 / 29.0 / 28.1 |
| 3 dies, 1 socket (repeat, `rep4_same3n4`) | c0,c1,c2 | 4, 4, 4 | 2, 2, 2 | **38.84 / 38.81 / 38.78** | 37.61 / 38.31 / 38.42 | 27.8 / 24.7 / 28.1 |

**Aggregates (`contig_1024` summed over the concurrent dies):**

| configuration | aggregate GB/s | per-die GB/s |
|---|---:|---|
| 1 die, any node | 107.1–107.2 | 107 |
| 2 dies, same socket | **114.5 / 115.4 / 115.5** | 54–61 |
| 2 dies, different sockets | **214.2 / 214.3 / 214.4** | 107 each |
| 3 dies, one socket | **116.4 / 116.6** | 38.8 each |
| 3 dies, two sockets | 219.1 / 222.4 | 107 + 54 + 54 |
| 3 dies, three sockets | **320.9 / 321.4** | 107 each |

**The invariant is per socket, not per die.** Two dies reading from *different* CPU sockets
both keep the full single-die rate (214 GB/s aggregate); two dies reading from the *same*
socket share ≈115 GB/s (each about half); three dies on three sockets reach 321 GB/s, and
three dies on one socket collapse to 39 GB/s each. One die alone already consumes ~107 of a
socket's ~115 GB/s, so **a node's memory interface is ~93 % saturated by a single die's
device-side lookup**.

"Socket" here means a NUMA node pair, and that pairing is read off the machine, not
assumed: `numactl -H` gives distance 10 on the diagonal, **15 inside {0,1} {2,3} {4,5}
{6,7}** and 20 across those pairs, while `lscpu` reports **4 sockets x 80 cores** and each
node holds 40 cores (80 CPUs at SMT2). The placement column in the table above is the
kernel's `/proc/self/numa_maps` accounting for the measured buffer, so "same socket" in
that table means "the pages came from two nodes of one `numactl` distance-15 pair".

**How strongly is that supported?** The per-socket reading was *fitted* on the matrix
above (the same-socket pairs were the slow ones; the different-socket pairs were not), so
by itself it would only be a hypothesis. It was therefore tested out of sample: three
phases with placements not used earlier — {2,3} (same socket 1, never measured before),
{3,5} (sockets 1 and 2) and {1,3,5} (three sockets). The predictions were written into
the driver (`agents/B_bw/conc_socket.sh`) before it ran: "≈115 / ≈214 / ≈320 GB/s total";
the measurements were 114.5 / 214.4 / 321.4 GB/s, within 0.5 % of each. Outputs:
`logs/raw/38-host-dram-bw-conc_sock*.json`.

Two consequences worth carrying into a deployment:

* **Placement is worth as much as 2.8x per die.** In the very first 3-die run of this
  session the kernel placed the three tables on nodes 4, 0 and 1 (socket 2, 0, 0) *by
  itself*, and two of the three dies ran at half rate. On this machine that is not a
  corner case: node 0 had 1.0–2.9 GB free while node 4 had 60 GB free, so an unbound
  allocator lands wherever there is room, not where the bandwidth is.
* The allocation between dies on a shared socket is **not** an even split (60.24/54.23,
  60.98/54.38, 38.82/38.97/38.77 — within a phase it varies by up to 11 %, and the
  "winner" changes between phases). **【推断】** there is no fairness guarantee to rely on.
  It is not even a stable per-die property *within* a phase: in `sockA_c1c2` the c2 die
  ran its contiguous arm at half rate (54.23) yet its 4096-row gather at full rate
  (96.62), and in `pair12` the c1 die did the reverse (54.38 contiguous / 94.97 gather).
  In other words the shared socket throttles the two readers *at the moment of the
  transfer*, not the dies themselves. **【实测】** — visible directly in the per-arm
  columns above; the JSONs hold the per-iteration times that show it arm by arm.

**【实测】 what this does *not* show:** these are dies 3/6/7 in throwaway containers with a
synthetic table, while the machine was also running the 8-die `dsv41-a3` deployment and
~30 other containers. The per-socket ≈115 GB/s figure is a property of *this machine in
this state*; a quiet machine may differ, and the 8-rank production case is not measured.

### 2.4 Incidental: registration cost of these buffers

Registration (`aclrtHostRegister`, `MAPPED`) was excluded from every timed loop and
measured once per run: **0.0014–0.470 ms/MiB** for 256/512/2048 MiB buffers (e.g. 2048 MiB
in 2.9 ms = 0.0014 ms/MiB in one run, 963 ms = 0.470 ms/MiB in another). The spread is
**not** explained by size. It is reported only so that nobody reads a "registration is
free" claim out of these runs; the authoritative registration figure for the production
table remains **0.566 ms/MiB** measured on the real 206 GiB shards (`logs/29`, §2.2.1 of
the contribution report). **【未确认】** whether the fast runs are the pool reusing
already-mapped pages.

### 2.5 The same bandwidth through upstream's exact registration call

Every run above registered with the legacy `aclrtHostRegister(…, MAPPED)`. One extra run used
**`aclrtHostRegisterV2(…, MAPPED|PINNED)` (flags `0x10000002`)** — the call upstream PR
#16925 makes — plus `aclrtHostGetDevicePointer` (which returned `0x3ff47e00000`, i.e. the
V2 call really does not hand the pointer back). Same script, same table size, same rows:

| Arm | legacy `MAPPED` | **V2 `MAPPED\|PINNED`** |
|---|---:|---:|
| `contiguous_1024MiB` | 107.09 | **107.04** |
| `contiguous_256MiB` | 104.24 | 103.91 |
| `contiguous_64MiB` | 92.73 | 93.39 |
| `gather_4096rows` | 95.17 | **94.79** |
| `gather_512rows` | 57.08 | 56.36 |

**【实测】** the two registration paths are bandwidth-equivalent here (all arms within 0.9 %).
`--check-first-4k` also verified the V2 path's device output byte-identical. Source:
`logs/raw/38-host-dram-bw-c0_v2pinned.json`.

---

## 3. Conclusions

### 3.1 The numbers RFC [47] asks for

| What [47] names | Value (useful GB/s, or µs where noted) | Label | Where |
|---|---|---|---|
| contiguous host-DRAM read, 1 die | 66 / 93 / 104 / **107** GB/s at 16 / 64 / 256 / 1024 MiB | 【实测】 | §2.1 |
| **random-row gather (the deployed-looking arm)**, 1 die | **57 GB/s** at 512 rows (10 MiB, 183.7 µs/call) and **95 GB/s** at 4096 rows (80 MiB, 881 µs/call), rows 20480 B | 【实测】 | §2.1 |
| hot-row variant (90 % of queries in 0.1 % of rows) | no throughput gain (56.2 vs 57.1 GB/s at 512 rows) | 【实测】 | §2.1 |
| HBM anchor, same two patterns | 601 GB/s contiguous, 724 GB/s gather ⇒ host DRAM is 5.6–7.6x slower | 【实测】 | §2.1 |
| whole-table host→HBM copy (2 GiB) | 107.65 GB/s ⇒ 19.95 ms per 2 GiB (⇒ ≈2.05 s for 206 GiB) | 【实测】/【推断】 | §2.1 |
| **NUMA sensitivity, one die** | flat across nodes 0–5: 104.1–104.8 GB/s contiguous (≤0.7 %), 94.8–96.6 GB/s gather (≤1.9 %), two passes | 【实测】 | §2.2 |
| **bandwidth sensitivity under concurrency** | per **socket** ≈115 GB/s: 1 die 107; 2 dies on 2 sockets 214 (full each), 2 on 1 socket 115 (57 each); 3 dies on 3 sockets 321, 3 on 2 sockets ≈220, 3 on 1 socket 117 (39 each) | 【实测】 | §2.3 |
| worst-case per-die loss from placement | **2.8x** (107 → 39 GB/s) | 【实测】 | §2.3 |
| allocation fairness between dies sharing a socket | uneven, up to 11 % spread; the "winner" changes between phases | 【实测】/【推断】 | §2.3 |

**Headline for the RFC:** a device operator reading a host-DRAM table gets **~107 GB/s** per
die at best and **~57 GB/s** for a realistic 512-row lookup; host DRAM is **5.6–7.6x slower
than HBM**; NUMA placement does not matter for a lone die but matters **2.8x** once several
dies read concurrently, because each CPU *socket* — not each die — caps at **≈115 GB/s**.

### 3.2 Evidence

* Topology: `numactl -H` (8 nodes, 2/socket, distances 10/15/20), `lscpu` (4 sockets x 80
  cores, 640 CPUs), `npu-smi info -t topo` (all off-diagonal `HCCS_SW`, no `SYS`/`PHB`),
  `/proc/svm/dev0..15/feature/host_mem_pool` = 1, 16 × `19e5:d803`,
  `/sys/bus/pci/devices/*/numa_node` = **-1** for every accelerator. Transcribed in §1.6.
* Placement proof: `/proc/self/numa_maps` per-node page counts for the measured buffer,
  stored in every JSON under `numa_run.host_buffer_placement` (`single_node`, `node_pages`,
  `node_mib`). Example: `{'4': 2048.0}` = the whole 2 GiB table on node 4.
* Concurrency proof: `concurrency.barrier_applied` = true, `peers_seen` = 2/3, start drift
  ≤ 4.5 ms in every multi-die JSON.
* Correctness gate: every run aborts unless a device gather of 3 rows is byte-identical to
  the host bytes (`[table] device readback sanity … byte-identical`).
* Script: `pr/bench_host_dram_bw.py` (local) = `bench/bench_host_dram_bw.py` (A3-node1).
  Drivers on A3-node1 under `agents/B_bw/`: `numa_sweep.sh`, `conc_matrix.sh`,
  `conc_repeat.sh`, `conc_socket.sh`, `test_placement.py`; logs in `agents/B_bw/logs/`.
* Raw JSON, one file per run (**54 files**):
  `logs/raw/38-host-dram-bw-c0_1die.json` (the full single-die run incl. HBM anchors),
  `logs/raw/38-host-dram-bw-numa<p>p<pass>.json` (NUMA sweep),
  `logs/raw/38-host-dram-bw-conc_*.json` (the concurrency matrix, repeats and the
  socket prediction tests).

### 3.3 【未确认】 / open

* **The real table.** Everything here uses a 512 MiB / 2 GiB synthetic mapping with 20480 B
  rows. The production table is 206 GiB and file-backed (safetensors `mmap`, page-cache
  pages), and its access pattern comes from the model, not from a uniform RNG. Nothing in
  this log says the 206 GiB table behaves like the synthetic one.
* **8-rank concurrency.** We have 1–3 dies (the three free slots). The deployed rank set is
  8; whether the per-socket cap is shared *within* one rank or across ranks is untested.
* **Machine state.** The box was running the 8-die `dsv41-a3` deployment and ~30 other
  containers. The ≈115 GB/s per-socket figure is a property of this machine in this state.
* ~~Registration API.~~ **Closed during this session** — one run with upstream's exact
  `aclrtHostRegisterV2(MAPPED|PINNED)` call reproduced the legacy numbers within 0.9 %
  (§2.5), so the bandwidth claim is not an artefact of the legacy entry point.
* **Hot-row caching.** The skewed arm showed no *throughput* gain at 2 GiB / 20 KiB rows;
  a hot-row cache is a latency question at 206 GiB, and that was not measured. Speaking to
  [47]'s "hot-row caching" clause, this log supplies no positive evidence for it.
* **No NUMA binding to a socket-local die.** `numa_node` is -1 for every accelerator, so
  "the die's own socket" cannot be identified on this machine; the per-socket effect was
  established by *forcing* placements, not by reading the topology.
* **Registration-cost spread** (0.0014–0.470 ms/MiB) is unexplained; it does not affect any
  bandwidth number (registration is outside every timed loop).

### 3.4 What we would run next, in order

1. **8-rank, real-table bandwidth sweep** on a quiet machine, binding rank r's table to a
   socket chosen round-robin — this is the version of §2.3 that the RFC actually asks for,
   and it needs the production container rather than a throwaway slot.
2. **Repeat the same-socket pair at 2–3x the duration** to see whether the ~115 GB/s cap is
   a hard per-socket limit or a time-sliced arbitration that a longer run would average
   differently, and whether the intra-pair split ever becomes fair.
3. **A latency-shaped hot-row test** (time-to-first-row rather than throughput) if anyone
   wants to argue for hot-row caching.

(The V2 `MAPPED|PINNED` control that used to be item 2 was run in this session — §2.5.)
