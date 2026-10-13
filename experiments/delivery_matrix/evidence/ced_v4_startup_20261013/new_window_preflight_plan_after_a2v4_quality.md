# CED v4 new-window preflight plan after A2v4 quality

Status: planning/checklist only. It does not authorize stopping A2v4, launching
P/D, opening a proxy, or sending a generation request. Apply only after the
serial Dihuo bench reaches terminal state and root explicitly authorizes the
next transition.

CPU-only p5/d5 recipe and plan artifacts for root review are recorded in
[p5_d5_cpu_preflight_receipt.md](p5_d5_cpu_preflight_receipt.md); neither
recipe has been executed.

## Current boundary

- A2v4 quality completed GSM100, Vision23, and the 147456 four-needle gate;
  its service was left online. The post-quality live audit is in
  `../a2sim_tp8_v4/quality/site_live_audit_20261013.md`.
- A2v4 currently owns P container physical chips 0–7 and listener 18960. Do
  not stop/restart it until root authorizes after Dihuo terminal.
- The retained project D container
  `dsv41-real-stack-tp8-20261010-a321` maps chips 8–15. Its prior D4 CED v4
  service was stopped: the old API/EngineCore are absent and old ports were
  refused, but shutdown recorded eight defunct workers and leaked IPC objects.
  The D container was not restarted. Do not reuse the old D4 job directory,
  old runtime identity, or SID.
- At the post-quality read-only snapshot, external `xp_mk` and
  `m00933363_catlass_runtime_bz_a3_1` each had cgroup mappings for all 16
  `/dev/davinci*` devices. They were not entered or modified. `npu-smi info
  proc` is unsupported on this device, so cgroup and instantaneous usage data
  are not process-exclusive reservation proof. Re-audit device/process/port
  state after Dihuo; if an external owner conflicts, hold and coordinate rather
  than touching it.

## Frozen CED v4 inputs to re-verify

- Package archive SHA-256:
  `3916d72f0fb798dac9ffe3dab243abd4999e2de42e6f0cc83a7b3785ef940aab`.
  The previously extracted package is visible at
  `/work/delivery_matrix_ced_20261013_v4`; its 29-file manifest SHA-256 is
  `d57ed5caf1688e29af9090a9403ef989a704c0c8cf49b9e66ed20642d606c89b`.
- Core CED overlay manifest: `7fc7e2dd1aa4014fa5ff04e022de73473f4b0a50dee0b000c585408eee484989`.
- D-only compressor manifest: `accf27fa871e62a78cb5e7aed3318c789f69feae14a5299db347795519e4e9f0`;
  derived source SHA-256 `ea3ead30e43d1e11e84cfb410a0a2faeec54e03ead01e53478c30e75dca564bc`.
  The original package's narrow four-address clamp remains D-only and must pass
  the same source/manifest gate; do not substitute a different clamp or worker.
- Pinned actual SDK base source SHA-256:
  `0d4d78ebb701c52364308f294dca59c8f94a0cdbf877011e60b31c91b8aa2b39`.
- Previously observed candidate fingerprint:
  `8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`.
  Previously observed BI OPP artifact fingerprint:
  `0167b055e442435b2790f1a77933fe389bdaa45db67a6f2045646606609d4a6b`.
  These are expected source-gate values, not a substitute for rechecking the
  selected containers, full OPP tree, and `config.ini` at the new window.
- The prior `p4_launch.sh` / `d4_launch.sh` were CPU-generated recipes with
  `service_launch_requested=false`. They are not a ready receipt. Their
  manifest and all 29 package files must be rechecked on both actual selected
  containers before use.

## Required gates before any new service launch

1. Wait for all Dihuo tiers to be terminal and confirm its client is gone. Root
   reviews the result and explicitly authorizes the lifecycle transition.
2. Re-read A2 identity/SID/fingerprint and the A3 container/listener state. If
   authorized to stop A2, signal only its service PID using the approved
   procedure; retain its container and all evidence. Verify runner exit,
   process/listener release, and chip state. Do not Docker-stop/restart the
   container, clear caches, or alter another owner's task.
3. Re-run read-only physical mapping, `npu-smi` process/usage snapshot,
   `docker ps`, device cgroups, and port checks. Require exclusive intended
   P=0–7 and D=8–15 use for this pair. The old `npu-smi info proc` failure
   means process attribution must use a supported full `npu-smi info` path or
   other approved read-only evidence; zero utilization alone is insufficient.
4. Root reviews exact new P/D launch commands and fingerprints. Generate fresh
   short job roots `p5`/`d5` (role-specific paths, with `p5` under `jobs/` as in
   the prepared P recipe), plus fresh output/cache paths; keep each `TMPDIR`/IPC
   path below 107 bytes. Do not reuse old `.../d4` outputs, old D4 SID, or
   placeholder `/work/results/q4_placeholder/native` output/cache paths from
   the prepared recipes.
5. Preserve the role boundary: P prefill binds chips0–7 and has no compressor
   overlay/env; D decode binds chips8–15 and alone receives the exact
   `--compressor-overlay` and matching `DELIVERY_CED_COMPRESSOR` path. Keep
   served model `dsv41-ced-pd-v4`, `max_model_len=1048576`, KV bytes
   `15728022528`, `decode-multistream=best` on both roles, and Engram device
   index 0. Keep the existing v4 baseline: no shared-Engram or DRAM candidate,
   no DCP overlay, and no other source changes unless root approves a changed
   candidate.
6. Confirm APIs/ports are free before launch: P API/KV `18990/19090` (producer
   group 19090–19097), D API/KV `18991/19091`. No old proxy is to be reused or
   started by this preflight. P rank 1 normally listens on producer port 19091;
   attribute that listener to P before calling it a conflict with D's consumer
   KV-port setting.
7. After each role initializes, but before any proxy or generation, verify
   engine/ASGI readiness, eight live ranks per role, physical-chip descriptions,
   actual candidate/runtime fingerprints, actual OPP/source hashes, and the
   new SID. Verify the D compressor import on every D rank and verify it is
   absent from P. Save fresh P/D receipts; never merge identities or evidence
   from the old D4 job.
8. Stop at the source/ready gate. A canary, proxy, quality request, or further
   performance run requires its own explicit root authorization.

No CED service, proxy, or generation was started for this plan. `xp_mk` and
other external containers remain out of scope.
