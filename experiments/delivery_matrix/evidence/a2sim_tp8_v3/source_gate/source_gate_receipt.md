# A2 TP8 v3 static source gate and launcher preflight (2026-10-13)

This receipt is for review before launch. It records a frozen-source package gate,
candidate-source hash check, CPU-only runtime plan, and a generated launcher. No
service was started; no HTTP identity, API, generation, quality, or performance
check was performed.

## Frozen package gate

- Source archive: `/home/chiro/tmp/dsv41-delivery-matrix/dcp_delivery_20261013_v6_sources.tgz`,
  380,546 bytes, SHA-256
  `859158d8055c4655a48c4d9c5a430f63842572a0a1ea91b9ae0d00ff890e5cb8`.
- The archive has 42 members (29 regular files, 13 directories), with no absolute
  paths, `..` path components, duplicate names, or non-regular entries.
- Extracted without overwriting into the new host path
  `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/delivery_matrix_generic_20261013_v6`;
  the D container sees it at
  `/work/delivery_matrix_generic_20261013_v6/delivery_matrix`. Existing v4/v5
  package directories were not changed.
- `package_files.sha256` was derived from bytes extracted from the frozen archive
  (29 files), SHA-256
  `028c626fd6e4a58483551a34ea8c9bd584810b7e079c39e485d0e15daea417f2`. All 29
  checks passed against the local extraction, remote host extraction, and D
  container's `/work` view. In particular, the frozen archive's `worker.py` is
  `bfa63afbbcad83539944c6f8324e63a445147a620ced695afb4bba2c6c7cf80a`; the later
  worktree worker was not substituted.

## Candidate source gate

- Candidate: `/work/src_dspark_general_quality_v11/candidate_contract.json`, id
  `dsv41_mls_async_bi_v1`, raw-file SHA-256
  `5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`;
  normalized candidate fingerprint
  `8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`.
- All 220 candidate-pinned source files matched their recorded SHA-256 values;
  the full-checkpoint config matched
  `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`. This did
  not hash the full model-weight contents.

## Planned configuration and launcher

- Form `a2-tp8-a3sim`, served model `dsv41-delivery-a2sim-tp8-v3`, port `18960`,
  physical chips `8–15`, model length `196608`, batched tokens `8192`, one
  sequence, TP `8`, async scheduling enabled, DSpark draft `5`, Engram enabled
  with device index `0`, route probe empty, draft graph enabled, and BI OPP selected.
- The candidate omits `decode_context_parallel_size`; the frozen service parser
  leaves the native vLLM default in place. In the selected container's source,
  `ParallelConfig.decode_context_parallel_size` is `Field(default=1, ge=1)` at
  `vllm/config/parallel.py:342` (file SHA-256
  `5ee765f980d8e371314caa38a27c5a21ab90f3865e697707d48e608ca4d17f3d`), inherited
  by the argument config in `vllm/engine/arg_utils.py:480` (SHA-256
  `165f9ec6e3b067bb7e6c72d68adedde7fa4d5ed7932fa7568064b9d9086e6830`). Thus the
  planned effective DCP is `1`; no actual v3 runtime identity has been checked.
- Runtime fingerprint: `aa1bb4ef4c762741e07bbec0f5631a924329318a2dde70cdede651bec7c14531`.
  CPU-only `plan.json` SHA-256:
  `c310b2c1a3519a55b159859e8eefcbc85e550344a9cdfdc61f65dfdeb364204b`.
- Generated runner: `/work/results/delivery_a2sim_tp8_v3/runner.sh`, SHA-256
  `910782202ee2e87c0b2794144dd02a16364c18500841c10e2e88a615270b8e02`; `bash -n`
  passed. The generator reported `service_launch_requested=false`. The file
  contains a future `--execute` command but was not run.
- The A2 recipe passes no `--dcp-overlay`, `--dcp-draft`, or
  `--dcp-det-reduce`. Neither the base container environment nor candidate
  startup environment sets `V41_DCP`, `DELIVERY_DCP_OVERLAY`,
  `DELIVERY_DCP_DRAFT`, `V41_CED_ROLE`, `DELIVERY_CED_OVERLAY`, or
  `DELIVERY_CED_COMPRESSOR`; the generated runner contains none of those names.
- `actual_quality_passed=false`, `actual_performance_observed=false`, and
  `simulated_hardware=true` remain explicit in the plan. The job directory
  contains only `runner.sh` and `service_plan_cpu_only/plan.json` (plus an empty
  plan `native/` directory); no runner PID, service selection, run log, or exit
  receipt exists.

## Local review artifacts

- [`package_files.sha256`](package_files.sha256)
- [`candidate_contract.json`](candidate_contract.json)
- [`plan.json`](plan.json)
- [`runner.sh`](runner.sh)

The matching remote artifacts remain under `/work/results/delivery_a2sim_tp8_v3`
and the frozen package remains under `/work/delivery_matrix_generic_20261013_v6`.
