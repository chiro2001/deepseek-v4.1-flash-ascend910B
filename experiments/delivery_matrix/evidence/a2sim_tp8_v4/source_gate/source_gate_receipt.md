# A2 TP8 v4 static source and launch preflight (2026-10-13)

Review package for the proposed A2v4 init-only launch. This is static planning
and package/source validation only: the generated runner was **not** executed;
there is no service init, HTTP identity, generation, quality, or performance
result in this receipt.

## Frozen v7 package/source gate

- Archive `/home/chiro/tmp/dsv41-delivery-matrix/generic_delivery_20261013_v7.tgz`,
  388,779 bytes, SHA-256
  `827c8310684b615a815c813600589e9c3da4ab482087587219c197a25fd35873`.
- Archive inspection found 50 members (36 regular files, 14 directories), with
  no absolute or `..` paths, duplicate names, or non-regular entries.
- Extracted to the previously absent host path
  `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/delivery_matrix_generic_20261013_v7`;
  P container sees it at `/work/delivery_matrix_generic_20261013_v7/delivery_matrix`.
  v4/v5/v6 package paths were not overwritten. Tar reported future mtimes for
  four files due to host clock skew; file contents were independently checked.
- [`package_files.sha256`](package_files.sha256), derived from the frozen archive
  bytes, has 36 entries and SHA-256
  `e38cab6bed9b0b53512142d8f350e7d39fc2f02dc30ea7f2da73db81519df0e1`. All 36
  entries passed against the local extraction, remote host extraction, and P
  container view.
- The candidate is `dsv41_mls_async_bi_v1`, fingerprint
  `8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`; its
  contract file SHA-256 is
  `5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`. All 220
  candidate-pinned source files matched; the checkpoint config matched
  `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`.
- Geometry component `tp8_physical_0_7_20261013_v1/manifest.json` SHA-256 is
  `df5461194e84c713352d85e3e92b03459c947c1541b3c045da1e676cb879c554`; its
  derived `worker.py` SHA-256 is
  `7cabe528dc5df34d8851637deb1cd4ea2f13624ba5dbd427455d6d0fe5dc9ad0`. The
  manifest pins original worker SHA-256
  `8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d`, physical
  chips `0,1,2,3,4,5,6,7`, and exactly one hard-physical-guard literal change.
  The frozen build/loader verify that derivation and all four visibility values;
  actual imported alias filepath/hash is **not yet runtime-observed** and must
  be checked in the eight rank descriptions after init.

## CPU-only plan and launcher

- Job root: `/home/l00886679/tmp/dsv41-a2sim-v4-runtime/a4` in the P container;
  it was absent, created mode `0700`, and verified writable as the container's
  launch user (`uid=0`). `DELIVERY_TMPDIR` is
  `/home/l00886679/tmp/dsv41-a2sim-v4-runtime/a4/tmp`.
- The `DELIVERY_TMPDIR` string is 49 bytes. The actual installed vLLM source
  constructs native IPC URIs as `ipc://{base_rpc_path}/{uuid4()}` in
  `vllm/utils/network_utils.py:141-143` (SHA-256
  `83b24f1e0026d1737e036a5abedc2f34ae234df5ccd7f1addab801c3cba55de9`).
  `VLLM_RPC_BASE_PATH` defaults to `tempfile.gettempdir()` in
  `vllm/envs.py:702-704` (SHA-256
  `567c99293678739cdd133897e3c7edfeba3a7fc5c635c43b29aea6bb89251244`). The P
  container had `VLLM_RPC_BASE_PATH`, `TMPDIR`, and `DELIVERY_TMPDIR` unset at
  preflight; frozen `run_service.sh` (SHA-256
  `cc5bcd644c78e5259b82f2cfc8aff3627ab6482b52fb36e8bd7f197ec5a1cdd9`) sets
  `TMPDIR="$DELIVERY_TMPDIR"` before starting Python. A canonical UUID4 is 36
  bytes, so the full filesystem socket path is `49 + 1 + 36 = 86` bytes (the
  `ipc://` URI is 92 bytes including the scheme), 21 bytes below the 107-byte
  `sun_path` limit. Example full path:
  `/home/l00886679/tmp/dsv41-a2sim-v4-runtime/a4/tmp/f58d75f3-164e-4ba4-ab86-036e280cfb73`.
- Planned identity/config: form `a2-tp8-a3sim`, model
  `dsv41-delivery-a2sim-tp8-v4`, port `18960`, TP8/DCP1, selected physical chips
  0–7, max model length `196608`, batch tokens `8192`, one sequence, async
  scheduling, DSpark5, draft graph, Engram device index `0`, empty route probe,
  BI OPP, and both multistream overlap flags enabled. `simulated_hardware=true`;
  this is A3 hardware simulating A2, not an A2 hardware performance result.
- Runtime fingerprint:
  `38063cf61a7167ca5b78753be16ae78a37d6a6324c1d55c65ca5b3ec9adce2ac`.
  CPU-only `plan.json` SHA-256:
  `479c211a1dda8e1e3de3b155e3b3a83fd1c06968846d9ee414f1573814484686`.
- Generated `runner.sh` SHA-256:
  `502aaabfd432208b9ee56a0d46225bde97dfb68bd43680b87d5de255da46f28c`;
  `bash -n` passed. The generator reported `service_launch_requested=false`.
  The runner begins by unsetting inherited CED/DCP and old physical-worker
  component variables, then exports the new component path and exact 0–7 values
  for `STACK_PHYSICAL_CHIPS`, `STACK_EXPECTED_CHIPS`,
  `ASCEND_RT_VISIBLE_DEVICES`, and `ASCEND_VISIBLE_DEVICES`. It passes no DCP
  overlay/draft/reduction flags. Its future command includes `--execute`, but
  this runner was not run.
- Local copies for review: [`candidate_contract.json`](candidate_contract.json),
  [`package_files.sha256`](package_files.sha256), [`plan.json`](plan.json), and
  [`runner.sh`](runner.sh). The remote `a4` directory currently contains only
  the static plan, runner, and the plan's empty `native/` and temp directories;
  it has no runner PID, `service_selection.json`, run log, or exit receipt.

The runtime identity and eight-rank `physical_worker_derivation` must still
prove the actual imported original/derived worker source and manifest. No
`/v1/models` or `/delivery/identity` request, inference, or client scoring has
been made.
