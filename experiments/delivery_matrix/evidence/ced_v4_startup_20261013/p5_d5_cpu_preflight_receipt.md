# CED v4 p5/d5 CPU-only preflight receipt

Status: CPU-only planning artifacts for root review. This receipt is not an
authorization to stop A2v4, create the remote job directories, launch either
service, start a proxy, or send a request. No P/D service was initialized by
this work; no external container was entered or changed.

## Frozen inputs and file gates

- Frozen source candidate:
  `experiments/delivery_matrix/evidence/a2sim_tp8_v4/source_gate/candidate_contract.json`,
  SHA-256 `5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`.
  The A3 source-tree `candidate_contract.json` was read-only hash-checked and
  matched byte-for-byte. The candidate file itself was not modified.
- CED v4 archive SHA-256:
  `3916d72f0fb798dac9ffe3dab243abd4999e2de42e6f0cc83a7b3785ef940aab`.
  It was unpacked only into local task scratch
  `/home/chiro/tmp/dsv41-a2sim-v4-quality/ced_v4_package`; all 29 files passed
  the frozen `package_files.sha256` check. The captured 29-line result is
  [p5_d5_package_hashcheck.txt](p5_d5_package_hashcheck.txt), SHA-256
  `baf7cc8ed2676ad459c895571fc8cd23d8522cd5e619e8ac5d602447cba856f3`.
- The package `prepare_ced_launch.py` is SHA-256
  `40d04537653839951e75c0db2189d51748fa0b414c65a688ebd8ed536752fa21` and
  matches the frozen 29-file manifest. The CED v4 `ced_service.py` CPU plan
  generator and `build_ced_compressor.py` were read from that hash-verified
  extraction. The D plan generator verified the exact packaged compressor;
  its four-address clamp/source manifest gate passed.
- Core CED overlay manifest remains
  `7fc7e2dd1aa4014fa5ff04e022de73473f4b0a50dee0b000c585408eee484989`;
  D-only compressor manifest remains
  `accf27fa871e62a78cb5e7aed3318c789f69feae14a5299db347795519e4e9f0`.

## New proposed recipes

The frozen candidate input and CED v4 source package were used to generate
fresh p5/d5 recipes. Both `bash -n` checks passed. Their accompanying
`prepare_ced_launch.py` JSON records say `service_launch_requested=false`.
The shell files contain the future runner's `--execute` argument, but neither
shell file was run.

| Role | Recipe / CPU plan | API / KV port | Physical chips | Prospective remote output |
| --- | --- | --- | --- | --- |
| P5 prefill | [p5_launch.sh](p5_launch.sh) / [p5_recipe_plan_cpu_only.json](p5_recipe_plan_cpu_only.json) | 18990 / 19090 producer group 19090–19097 | 0–7 | `/home/l00886679/tmp/dsv41-ced-runtime-20261013/p5/service` |
| D5 decode | [d5_launch.sh](d5_launch.sh) / [d5_recipe_plan_cpu_only.json](d5_recipe_plan_cpu_only.json) | 18991 / 19091 consumer | 8–15 | `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d5/service` |

P producer rank 1 normally listens on port 19091. The eventual port-ownership
gate must attribute that listener to P; it is not by itself a D consumer-port
conflict.

The exact CPU plans produced by the packaged `ced_service.make_ced_plan` are
[p5_plan_cpu_only.json](p5_plan_cpu_only.json) and
[d5_plan_cpu_only.json](d5_plan_cpu_only.json). Both retain served model
`dsv41-ced-pd-v4`, source-candidate fingerprint
`8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`,
`max_model_len=1048576`, KV bytes `15728022528`, Engram enabled/index `0`, and
best multistream settings. No DCP, shared-Engram, or DRAM candidate was added.

- P5 is prefill-only, has no compressor env/overlay, uses the producer KV role,
  and has draft graph disabled as prescribed by the existing role builder.
- D5 is decode-only, uses the consumer KV role, keeps its graph enabled, and
  alone receives compressor path
  `/work/delivery_matrix_ced_20261013_v4/delivery_matrix/ced_compressor_20261013_v1`
  and manifest `accf27fa…519e4f0`.
- Candidate `startup_env` values `STACK_DSPARK_FORMAL_OUTPUT` and
  `VLLM_CACHE_ROOT` remain frozen `q4_placeholder` inputs in the generated
  launcher recipes. The CPU `make_ced_plan` outputs correctly override them to
  unique p5/d5 service/native and service/cache paths; the candidate was not
  edited.
- Proposed `DELIVERY_TMPDIR` prefixes are 53 bytes each (`.../p5/tmp` and
  `.../d5/tmp`). Assuming the same 36-character UUID suffix, prefix + `/` +
  UUID + NUL is 91 bytes, below the 107-byte limit. The actual process-specific
  IPC socket path must still be measured at the launch gate.
- CPU plan runtime fingerprints are P5
  `457e5d3290db5c4ae89acdc68f0be04d2f691782d14e0b28a1821ee475cc1d2a` and D5
  `1a6f0c8081f7b35d77f7265437abe8df30b78e8f11f179ed831c14652e14db32`.
  Both plans have `native_opp_receipt=null` and empty
  `DELIVERY_CED_OPP_FINGERPRINT`; these are **CPU planning fingerprints, not
  future actual runtime fingerprints**. An authorized execute-time plan must
  re-inspect the actual OPP and will have a different fingerprint.

## Required before any execute

Wait for Dihuo to reach terminal state and for root's controlled A2v4 stop
authorization. Then recheck ports, exact process/device occupancy, all 29
package files in both selected containers, full OPP tree plus `config.ini`,
physical 0–7/8–15 assignment, and the complete IPC path length. Root must review
these exact new recipes and actual execute-time plan fingerprints. Keep A2v4
online until that authorization; do not reuse the old D4 job directory, SID,
or runtime identity. The current P/D source/OPP evidence is not promoted to a
new ready receipt by these CPU-only files.

The latest read-only full NPU process/occupancy snapshot is
[site_live_during_dihuo_20261013.md](site_live_during_dihuo_20261013.md). It
showed no active NPU process on physical chips 8–15 at that instant, but it
does not replace the required post-Dihuo recheck.

Recipe SHA-256: P5 `bdd4b807108b2f05d711ac86737a11be5c7e07dfda75c6a3b36d67cab0d9bb5c`,
D5 `1f7277c1bbe42f20eb5039e9c024c0e73608c7fa0124e793c24a2156228efe99`.
Plan SHA-256: P5 recipe `725295eee98619161f35afec17bd703a3c8d0a7f6b17de443e879ed61c7261aa`,
D5 recipe `ca626e5823c3d40501401b78939505e761ead50185f084c5320af8023326893f`,
P5 CPU plan `a63a85a627e8dd7d2531ca7a41cb950139c901b5c14e2c7be73b929345317ba7`,
D5 CPU plan `99f369082e565b4b58a90844a8f6138120b2c31ca93e8f28b219cf72442d6837`.
