# CED v2 startup attempt receipt (2026-10-13)

This receipt records the first authorized CED v2 startup attempt. No inference,
canary, workload benchmark, or quality evaluation has run. The P service is ready;
the D service exited before opening its API listener. The pre-existing project
container was not restarted or stopped.

## Frozen inputs and device topology

- CED service package: `/home/chiro/tmp/dsv41-delivery-matrix/ced_delivery_20261013_v2.tgz`,
  SHA-256 `45e05870ccd73fadd904832a2f37ad63d5e9080127ae094f52efba7f61292513`.
- Client contract package: `ced_client_cpu_contract_20261013_v2.tar.gz`, SHA-256
  `5de07966b80b1d4535fcdc7b3e5cfd1b1e97b8c81c2744b8507f424a8391b73b`.
- Base image: `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea`.
- P container: `dsv41-ced-p0-7-20261013-v2`, ID
  `16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`, running
  with physical chips 0–7 only.
- D container: existing `dsv41-real-stack-tp8-20261010-a321`, ID
  `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`, remained
  running with restart count 0; D uses its original physical chips 8–15.
- API/KV/proxy ports: P `18990/19090`, D `18991/19091`, proxy `18992`.

The role jobs used distinct task roots under
`/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/`; P's cache/output path is
`prefill_18990`, D's is `decode_18991`. The earlier unexecuted 18960/18961 launch
scripts and their directories were left untouched.

## OPP and runtime provenance

P has one read-only bind mount for the verified `vendors/` parent. The full tree
manifest SHA-256 is
`0f1a8d8206d3fd36ab3012dc6c547f5fafe974d5cdab3c53ace27b4c37c8f2fe` (722
entries, 618 regular files, one symlink, 222,548,156 bytes); the BI child tree
manifest is `3a8b878c9e8f2cd370e881dc248fd05e9bbbf07e7317a0848069b624179feb4a`.
The sibling `config.ini` contains `load_priority=batch_invariant\n`, is 30 bytes,
and SHA-256 is `7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`.
The full `vendors/` manifest file is preserved at
[`opp_vendors_manifest_20261013.json`](opp_vendors_manifest_20261013.json),
SHA-256 `46de30400478eeabcd3141eb2c0f5293476b2757ef533879d4b306819b25a67d`.

P's `/delivery/identity` returned `engine_initialized=true`,
`ASGI_startup_completed=true`, ranks 0–7, role `prefill`, and
`VLLM_BATCH_INVARIANT=1`. All ranks reported the same
`ced_overlay_manifest_sha256=7fc7e2dd1aa4014fa5ff04e022de73473f4b0a50dee0b000c585408eee484989`
and OPP artifact fingerprint
`0167b055e442435b2790f1a77933fe389bdaa45db67a6f2045646606609d4a6b`. The service
instance ID is `53af26c59ea9402ba83283a4865f8057`; runtime fingerprint is
`752b0b6fe6d85b8b502d908dac7ee6040fe5bc75c846f02a7a1105b0bcf03c8c`.
This is an identity/initialization receipt, not a serving or operator-execution
claim.

## D startup result

D runner PID in the existing container's PID namespace was `861311`. It completed
86/86 weight shards, allocated a 3,168,174-token KV cache, compiled 164/164 static
kernels, and completed one FULL decode graph capture. It then exited with code 1
before listening on port 18991. The first failure is the frozen v2 decode-boundary
middleware being installed too late:

```text
ced_service.py:94: app.middleware("http")(ced_decode_guard)
RuntimeError: Cannot add middleware after an application has started
```

The error is unrelated to model loading, OPP checks, or device allocation. The D
service job had `run.exit=1`; there was no listener and no D identity. Its full
diagnostic `run.log`, `run.exit`, `runner.pid`, `service/plan.json`, and initial
`service/service_selection.json` were copied from the old container's internal
job directory to
`/home/l00886679/tmp/dsv41-ced-runtime-20261013/collected/decode_18991_failed_init1/`.
Source and copied SHA-256 values match:

| File | SHA-256 |
|---|---|
| `run.log` | `f33c0b94a535a380b23491e5b1df124fac5fbc13d9a2465921aeed3acab8d` |
| `run.exit` | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `runner.pid` | `69cad2d6ef70969af4185fa9ae6093d482e7ef435d7db3f29ae2f360ca5eeacf` |
| `service/plan.json` | `b6d50a9afe8b1222a467d8e1bf89ac1310d6b04844081dd385b30dc3f1f8d71f` |
| `service/service_selection.json` | `0c6f0571769e82a13a8331eac6482c17c9278e73d72c93053859c8112a66e7b0` |

The failure bundle was also copied from the remote task path above to local
[`ced_d8_v2_failed_startup_20261013`](../ced_d8_v2_failed_startup_20261013/).
All 48 files in the two trees have matching relative paths and SHA-256 values;
the five primary files above are listed explicitly for quick verification.

The exact v2 package was not edited or replaced. The old D container remains
running; no D cleanup/restart was attempted. Proxy port 18992 is not started, and
no generation request or canary has been sent. Any middleware fix requires a new
server candidate/fingerprint and explicit authorization; it must not inherit
quality or performance results from the frozen v2 package.
