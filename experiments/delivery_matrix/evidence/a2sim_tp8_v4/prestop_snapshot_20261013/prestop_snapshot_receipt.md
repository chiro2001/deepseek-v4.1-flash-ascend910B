# A2v4 pre-stop evidence receipt (2026-10-13)

This is a **pre-stop** snapshot. It does not claim teardown has started or
completed. All reads and GET requests were read-only; no generation was sent.

## Current service identity and exact PID source

- Target container `dsv41-ced-p0-7-20261013-v2`, CID
  `16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`:
  `running`, restart count `0`, host container PID `1541878`.
- `GET /delivery/identity`: HTTP 200, 12,349 bytes; served model
  `dsv41-delivery-a2sim-tp8-v4`, SID
  `9d2ba214732b4eed84dc4f416f27c6ae`, runtime fingerprint
  `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`, 8 rank
  descriptions. SHA-256 `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`.
- PID ownership was checked live immediately before the stop authorization:
  container PID `342558` is the A2v4
  `python3 -u -m delivery_matrix.service --port 18960 --served-model
  dsv41-delivery-a2sim-tp8-v4` process; host `/proc/2898801/status` reports
  `NSpid: 2898801 342558`, and host PID `2898801` runs that same exact command.
  Its parent runner is container PID `342556` / host PID `2898791`, running the
  A2v4 `runner.sh`. `run.exit` was absent before stop. EngineCore is container
  PID `343435` / host PID `2901163`; all eight TP workers are live. The saved
  identity and service selection agree on SID/PID.
- The authorized stop target is **only** the project API, container PID
  `342558` (host PID `2898801`). Leave the runner/container/caches intact; do
  not signal EngineCore/workers directly, Docker-stop/restart the container, or
  touch external containers.

## GET metrics captured before stop

- `GET /metrics`: HTTP 200, 71,684 bytes; SHA-256
  `1614802543c1fd15bab0e385a31894d2b63141a3ad8e060921c71a8e1d87d6a8`.
- `GET /delivery/metrics`: HTTP 200, 193,999 bytes; SHA-256
  `208c47db05a2ee63319228eef1d871ba6d336cd653e689eeba6770b072dbd30e`.
- The trusted `delivery_runtime_metrics_v1` response had the same SID/model,
  all eight ranks, `pending=false` and no rank errors. Each rank retained 33
  request records; cumulative counters were 209 prefill calls, 2,460 decode
  rounds, 2,460 draft-graph calls and 33 eager draft calls per rank. Async worker
  decode-token fields are null by the recorded counter scope; this is not a
  substitute for the benchmark client's usage receipts.

## Host, NPU, memory, and ports

- The full read-only `ss -ltnp`, `docker top`, `docker stats`, `free`,
  `/proc/meminfo`, `npu-smi info`, and container-list output is saved as
  [site_prestop_snapshot.txt](site_prestop_snapshot.txt). It records port 18960
  listening and no listeners on 18990–18992 or 19090–19097 at this snapshot.
- `docker stats` reported the P container at `308.4 GiB / 1.966 TiB`,
  `15.32%`, 7,337 PIDs; host `MemAvailable` was `1,770,563,740 kB`.
- The full `npu-smi info` process table listed the eight A2v4 workers only on
  physical chips 0–7; NPU IDs 4–7 (physical chips 8–15) reported no running
  processes. Physical 8–15 were near the 2.87–3.13 GB HBM idle baseline, while
  0–7 held the A2 model. `lsof` was unavailable; `fuser -v` returned no device
  entries. No external container was entered or modified.
- At the time of this snapshot the P container was running, restart count was
  zero, and all 8 workers were live. The receipt does not yet classify any
  post-SIGTERM teardown or resource release.

## Runtime artifacts archived before stop

The actual A2v4 `runner.sh`, `runner.pid`, 774,676-byte `run.log`, actual
`service/plan.json`, `service_selection.json`, CPU-only plan, and all eight
`sdk_rank*.json` files are archived under `a2_runtime/`. Key hashes:

- `run.log`: `6fafbefe8eefed875bbef5c501fc30efa8c748a71fc4ad56dd502b4d0a185380`
- `runner.sh`: `502aaabfd432208b9ee56a0d46225bde97dfb68bd43680b87d5de255da46f28c`
- actual service plan: `5716d78594ba91c17c14a86b0eebaaf148bef43752c8fa07c1c26a4a60b7b02b`
- service selection: `70548f84722bee9d70ad54df538cdcfe48880aaa2b2c6c028564cde4cb9e2616`
- CPU-only plan: `479c211a1dda8e1e3de3b155e3b3a83fd1c06968846d9ee414f1573814484686`

The complete per-file archive hash list is
[artifact_hashes.sha256](artifact_hashes.sha256), SHA-256
`7f58a186c0a8c38c8a8114b2aaadffe2b8b27dbd4a8f4dc7431d371ff37476fe`.
No source, plan, service output, or prior quality evidence was overwritten.
