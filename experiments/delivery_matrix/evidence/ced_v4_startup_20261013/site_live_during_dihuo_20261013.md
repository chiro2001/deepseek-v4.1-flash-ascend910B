# A3-21 read-only occupancy snapshot during the Dihuo window

Snapshot time: `2026-10-13T08:21:14+08:00`. This was a read-only status check;
no service was started/stopped and no external container was entered.

- A2v4 P container remained `running`, restart count `0`, host PID `1541878`,
  with `127.0.0.1:18960` listening. Its service launcher, EngineCore, and all
  eight TP workers were live. The P container maps `/dev/davinci0–7`.
- The supported full `npu-smi info` process table listed only the A2v4 worker
  PIDs on NPU IDs 0–3, corresponding to physical chips 0–7. NPU IDs 4–7
  (physical chips 8–15) each reported “No running processes found”.
- The physical 0–7 chips had about 55.7–55.9 GB HBM used, consistent with the
  resident A2v4 model; physical 8–15 were near the idle HBM baseline at
  2.87–3.13 GB each. The instantaneous AICore readings were 0% across all 16.
- `lsof` was unavailable; `fuser -v` on `/dev/davinci0–15` returned no entries.
  The Docker mappings for `xp_mk` and `m00933363_catlass_runtime_bz_a3_1` span
  all 16 devices, but mappings indicate access permission, not current use.
  Neither container was touched. The NPU-SMI process table showed no active
  external NPU process in this snapshot. This is not a future-availability
  guarantee.
- The project D container remained running but no processes were reported on
  its physical chips 8–15. A2v4 was not stopped; no D4/D5 service was started.

The complete captured output is
[site_live_during_dihuo_20261013.txt](site_live_during_dihuo_20261013.txt),
SHA-256 `238312ec995c0956d39eb0720989f1b350bf697fe14f50d12f951e38d572f45d`.
