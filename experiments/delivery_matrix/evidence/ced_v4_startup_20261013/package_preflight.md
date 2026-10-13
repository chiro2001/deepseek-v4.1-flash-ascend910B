# CED v4 package/source preflight (2026-10-13)

This receipt covers the v4 archive, both existing containers' view of the new shared
package path, role-specific launch recipes, and static OPP/source checks. It is not
a service-ready receipt or a generation/canary/quality/performance result.

## Archive and two-container file-hash gate

- Local package: `/home/chiro/tmp/dsv41-delivery-matrix/ced_delivery_20261013_v4.tgz`,
  179,474 B, SHA-256
  `3916d72f0fb798dac9ffe3dab243abd4999e2de42e6f0cc83a7b3785ef940aab`.
- Remote transfer copy: `/home/l00886679/tmp/dsv41-ced-runtime-20261013/collected/ced_delivery_20261013_v4.tgz`;
  its SHA-256 matches the local archive. It was extracted into a previously absent path:
  host `/home/l00886679/projects/dsv41-real-stack-tp8-20261010/delivery_matrix_ced_20261013_v4`,
  visible in both existing containers at `/work/delivery_matrix_ced_20261013_v4`.
- The archive contains only relative `delivery_matrix/...` files (no absolute or `..`
  paths). The extracted package has 29 regular files / 769,553 B. The per-file manifest
  [`package_files.sha256`](package_files.sha256) is 29 lines, SHA-256
  `d57ed5caf1688e29af9090a9403ef989a704c0c8cf49b9e66ed20642d606c89b`; each line passed
  `sha256sum -c` on the host and independently in the P and D containers. No prior v2/v3
  package or source directory was overwritten.

## Core, compressor, source, and actual OPP

- Core CED overlay manifest SHA-256 in the package is the expected unchanged
  `7fc7e2dd1aa4014fa5ff04e022de73473f4b0a50dee0b000c585408eee484989`.
- Compressor component manifest SHA-256 is
  `accf27fa871e62a78cb5e7aed3318c789f69feae14a5299db347795519e4e9f0`; its derived
  `compressor_triton.py` SHA-256 is
  `ea3ead30e43d1e11e84cfb410a0a2faeec54e03ead01e53478c30e75dca564bc`. In both P and D
  containers, the CPU `build_ced_compressor.verify()` passed: the pinned actual SDK base
  source SHA is
  `0d4d78ebb701c52364308f294dca59c8f94a0cdbf877011e60b31c91b8aa2b39`, four masked
  projection load addresses are clamped, and original mask/ring selection math is marked
  unchanged. This is a source/manifest check; actual per-rank import proof is still pending
  until D4 rank descriptions return.
- Both containers independently computed candidate fingerprint
  `8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd` and actual A3 BI OPP
  artifact fingerprint `0167b055e442435b2790f1a77933fe389bdaa45db67a6f2045646606609d4a6b`.
  `inspect_opp` covered `version.info` plus the three pinned ABI libraries, not the full
  vendor tree; the separate full-tree/config receipt remains the authority for that scope.

## Role recipes prepared (not yet launched P4)

The package's CPU-only `prepare_ced_launch.py` generated both scripts with
`service_launch_requested=false`; both pass `bash -n`.

- P4 script copy: [`p4_launch.sh`](p4_launch.sh), SHA-256
  `7303043cffa7f06ad41d6d186ddc2be65aa89084ff2b944dee56d6ec3110a39b`. Role prefill,
  `dsv41-ced-pd-v4`, chips 0–7, API/KV `18990/19090`, job root
  `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/p4`. It has no compressor env or
  `--compressor-overlay` option.
- D4 script copy: [`d4_launch.sh`](d4_launch.sh), SHA-256
  `324ab9cfae0c360eb45e1a205fe0450855d6fedbc4e2d4cb8ec50ae4f859995f`. Role decode,
  `dsv41-ced-pd-v4`, chips 8–15, API/KV `18991/19091`, job root
  `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d4`. Its TMPDIR UUID path is 90 chars,
  below the 107-byte ZMQ IPC limit. Before Python/model import, it exports
  `DELIVERY_CED_COMPRESSOR=/work/delivery_matrix_ced_20261013_v4/delivery_matrix/ced_compressor_20261013_v1`;
  the CLI passes the exact same `--compressor-overlay` path. Both roles retain
  `--decode-multistream best`.

At this preflight snapshot, old P and proxy were already SIGTERM-stopped and their
containers retained. P4 remains held because a different container is actively using
host NPU3 / physical chip 6; that process is not part of CED and must not be killed. D4
may initialize independently on its separately verified idle physical chips 8–15. No
proxy or request is part of this preflight.
