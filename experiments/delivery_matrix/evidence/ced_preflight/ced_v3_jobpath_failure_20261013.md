# CED service v3 D job-path attempt receipt (2026-10-13)

This records a failed D-v3 launch attempt after the official pre-build middleware
fix was authorized. P-v2 remained ready and untouched. No generation request,
canary, benchmark, or quality run occurred in this attempt.

## Frozen v3 inputs and plan

- Service package: `/home/chiro/tmp/dsv41-delivery-matrix/ced_delivery_20261013_v3.tgz`,
  166,512 bytes, SHA-256
  `d4850d9e63285885767e54d1321f8eaec82d44b0b17cf8698b5e7df0a940d1d5`.
- Remote extraction: `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/delivery_matrix_ced_20261013_v3/`;
  all 24 regular files matched the archive byte-for-byte.
- D reused container `dsv41-real-stack-tp8-20261010-a321`, full ID
  `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`.
- The D plan records `http_middleware=["delivery_matrix.ced_decode_boundary.ced_decode_guard"]`,
  runtime fingerprint
  `efd677133f6a012b8332112d3e41220d93b71a3de1e7aa3fa7bc654a8f838091`, and the
  unchanged core CED overlay manifest SHA-256
  `7fc7e2dd1aa4014fa5ff04e022de73473f4b0a50dee0b000c585408eee484989`.
- Shared API and KV settings remained D `18991/19091`, served model
  `dsv41-ced-pd-v2`. Port 19091 is the expected P producer TP-rank-1 listener;
  P's log explicitly reports `KVCacheSendingThread ... tp_rank=1` at that port.

## Failure and preserved evidence

The job root `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/decode_18991_v3`
made vLLM's per-process ZeroMQ IPC socket exceed the Linux 107-byte `sun_path`
limit. The process exited with code 1 before EngineCore/model worker startup:

```text
zmq.error.ZMQError: ipc path "/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/decode_18991_v3/tmp/55037643-5a1d-41f9-91d4-627249d49009" is longer than 107 characters
```

The failure is a job-path constraint, not an OPP or middleware failure. Its
`run.log`, `run.exit`, `runner.pid`, and service plan/selection were copied from
the reused D container to the remote collection path
`/home/l00886679/tmp/dsv41-ced-runtime-20261013/collected/decode_18991_v3_ipc_path_failure/`,
then to local
[`ced_d8_v3_jobpath_failure_20261013`](../ced_d8_v3_jobpath_failure_20261013/).
All eight files in those trees have identical relative paths and SHA-256 values.
Primary file hashes:

| File | SHA-256 |
|---|---|
| `run.log` | `520ce35fc7e3cb4c066bf1187b0e5e26e00076cdf14d192cdac4354a5258a839` |
| `run.exit` | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `runner.pid` | `5a41999a6dd0aff44a2d3b55e9270a92cbd0fbc648392d7deb2a82e9eab90287` |
| `service/plan.json` | `494493f0b8343d4e4ba8d4961c090d1ec82077fc8caaba8c1dd8b07473118381` |
| `service/service_selection.json` | `23dd69e0ced16fd93dfdc0c80efc9d1aeb00239c0c323520f080862d62228055` |

The next D-v3 job uses the shorter task-specific root
`/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3`, leaving the failed job and
all v2 evidence untouched. D-v2 worker children remain only as exited `Z`
entries in the old container; no active EngineCore, API server, or D-v3 worker
was present before the retry. The old D container was not restarted or stopped.
