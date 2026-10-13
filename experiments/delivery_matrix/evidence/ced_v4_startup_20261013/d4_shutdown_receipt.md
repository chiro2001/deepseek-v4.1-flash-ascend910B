# D4 controlled shutdown receipt (2026-10-13)

Post-shutdown checks were collected from `a3-21` at `2026-10-12T22:42:07Z`
(`2026-10-13T06:42:07+08:00`). Only D4 API PID `1609471` in container
`dsv41-real-stack-tp8-20261010-a321` received SIGTERM. The API's parent runner
PID was `1609465`; it was left intact so it could record its exit status. No
request, proxy launch, P4 launch, container stop/restart, or reset was performed.

## Exit and shutdown classification

- The runner wrote `run.exit=0`, then exited. The API, runner, and EngineCore
  PIDs were absent after shutdown.
- The log records HTTP shutdown and FastAPI application shutdown completion.
  EngineCore shutdown used `mode=abort timeout=0s`; the process manager force-killed
  EngineCore PID `1610342`. Eight `VLLM::Worker_TP` children (PIDs `1610383`,
  `1610413`, `1610457`, `1610520`, `1610566`, `1610612`, `1610658`, `1610704`)
  remain as `<defunct>` entries reparented to PID 1. The log also warns of 248
  leaked semaphore objects and 10 shared-memory objects.
- Therefore this is a controlled operator-requested shutdown with runner exit
  code 0, but it is **not** classified as a fully clean, residue-free exit.
  The D container was deliberately left running; no attempt was made to reap
  zombies by restarting it.

Remote artifacts inside the retained D container:

| Artifact | SHA-256 |
| --- | --- |
| `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d4/run.log` | `e15418693755d8d3b8b9562a719e93c8d5ed3d3f5dc34c37bdbeb44cf24c7a74` |
| `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d4/run.exit` (`0\n`) | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d4/runner.pid` (`1609465\n`) | `43f59140f03547531509b572c3c25d8e38de2955ab98484fa2ad0d2429cb1b61` |

The log's shutdown sequence is timestamped `2026-10-12 22:40:23Z`. It reports
API shutdown triggered, EngineCore abort teardown, completion of HTTP application
shutdown, then the resource-tracker warnings described above.

## Container, port, and NPU state

`docker inspect` after shutdown reported both
`dsv41-ced-p0-7-20261013-v2` and `dsv41-real-stack-tp8-20261010-a321` as
`status=running`, `restart=0`, `network=host`.

TCP `connect_ex` checks from both retained containers returned `111`
(`ECONNREFUSED`) for every port in `18990–18992` and `19090–19097`; no P, proxy,
or D4 service listener remained.

Post-shutdown `npu-smi info` reported no running processes on NPU 0–7. Physical
chip HBM usage was back near the idle baseline: chips 0–15 used respectively
`3110, 2886, 3119, 2875, 3112, 2889, 3112, 2887, 3124, 2875, 3122, 2874,
3128, 2874, 3121, 2877 MB`. The earlier D4-ready snapshot showed roughly
60 GB per logical chip on NPU 4–7; this post-check shows that allocation released.
The unrelated NPU3 / physical-chip-6 process noted in the earlier snapshot was
not present in this post-check; it was not signaled or otherwise touched here.

No A2v3 service or HTTP API was started/accessed as part of this shutdown.
