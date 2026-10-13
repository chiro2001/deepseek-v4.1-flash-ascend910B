# A2v3 prelaunch hold snapshot (2026-10-13)

Read-only gate checks ran on `a3-21` at approximately
`2026-10-13T06:59:34+08:00` (`2026-10-12T22:59:34Z`). The approved A2v3 launcher
was **not** executed because the assigned NPU set was not idle. No processes or
containers were signaled, stopped, or restarted; no P/proxy was started and no
HTTP/API request was issued.

## Checks that passed

- D container `dsv41-real-stack-tp8-20261010-a321`, CID
  `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`, remained
  `running`, restart count `0`.
- Host TCP port `18960` had no listener (`ss -ltnp "sport = :18960"` returned
  only the table header).
- Container-visible RAM: total `2.0 TiB`, available `1.9 TiB`.
- All NPU devices reported health `OK`.

## Blocking occupancy

The post-check `npu-smi info` still showed active, unrelated work on the required
8–15 device set:

| NPU | Physical chip | Host PID | Container PID | Process | NPU memory |
| ---: | ---: | ---: | ---: | --- | ---: |
| 6 | 12 | 2856535 | 3971 | `python3 -u /home/x00464572/megakernel/examples/models/qwen3-30B/test_forward.py` | 28,827 MB |
| 7 | 14 | 2857597 | 4074 | `python3 -u examples/models/qwen3-30B/test_forward.py` | 12,539 MB |

Both host PIDs' cgroups identify container
`f11a0b179a1f992c8821781c50cec5862bc6e7a90ef02ecf8bb2c0acad39f1e5`, named
`xp_mk`, which was still running. The first occupancy read showed PID 2856535
using 16,111 MB; the later read showed 28,827 MB, so this is live occupancy,
not a harmless stale process row. Devices 8–11, 13, and 15 had no running NPU
processes at the later read.

## Decision

The NPU-idle precondition failed, so A2v3 remains held at the static source/plan
gate. The staged runner and CPU plan are unchanged; there is no A2v3 runner PID,
service log, service selection, runtime identity, or generation result. Retry
the complete prelaunch gate only after site/root confirms the unrelated chips 12
and 14 are free.
