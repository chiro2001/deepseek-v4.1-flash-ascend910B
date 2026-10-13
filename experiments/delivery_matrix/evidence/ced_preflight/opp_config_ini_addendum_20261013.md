# CED OPP activation-config addendum (2026-10-13)

This addendum supplements the earlier read-only image/BI-OPP audit. It records the
additional selector file required to reproduce the active batch-invariant OPP
configuration. The existing D container was only read; its service was not
restarted or modified.

## Source file and fixed image

- Source container: `dsv41-real-stack-tp8-20261010-a321`, full ID
  `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`.
- Source path: `/usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini`.
- Exact bytes: `load_priority=batch_invariant\n` (30 bytes); SHA-256
  `7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`.
- Source metadata: mode `0755`, uid/gid `0:0`.
- The fixed image `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea`
  reports `BASE_IMAGE_CONFIG_MISSING` at this path. The existing D container's
  mount inspection reports no bind mount covering `vendors/config.ini`.

## Verified host copy

The first read-only `docker cp` attempt returned `invalid argument` and produced
no partial file. The file and full `vendors/` tree were then copied by a
container-local tar stream to the task-specific host directory
`/home/l00886679/tmp/dsv41-ced-runtime-20261013/opp_source/`.

- `opp_source/config.ini`: 30 bytes, mode `0755`, SHA-256
  `7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`.
- `opp_source/vendors/config.ini`: same 30 bytes and SHA-256.
- Full `vendors/` tree manifest, computed independently in the D container and
  on the host copy: SHA-256
  `0f1a8d8206d3fd36ab3012dc6c547f5fafe974d5cdab3c53ace27b4c37c8f2fe`; 722
  entries, 618 regular files, one symlink, 222,548,156 regular-file bytes.
- `vendors/batch_invariant/` manifest remains SHA-256
  `3a8b878c9e8f2cd370e881dc248fd05e9bbbf07e7317a0848069b624179feb4a`; 720
  entries, 617 regular files, one symlink, 222,548,126 regular-file bytes.
- Existing standalone reference manifest remains SHA-256
  `0b73aa6a0145cb56a3027d0919216f83056854af29c295bdd7c2f1685f4a58fc`.

## P-container mount contract

The new P container must use the fixed image ID above and keep OPP read-only. The
verified `opp_source/vendors/` directory contains both the complete BI subtree
and `config.ini`; a single read-only parent bind makes the selector available
without nested-mount target-path ambiguity:

| Host source | Container destination | Mode |
|---|---|---|
| `/home/l00886679/tmp/dsv41-ced-runtime-20261013/opp_source/vendors` | `/usr/local/Ascend/cann-9.1.0/opp/vendors` | `ro` |

This parent mount supplies both OPP files, with the complete manifest binding
the subtree and selector. After P creation, the in-container full-tree hash,
selector hash/mode, SDK/ACL hashes, and `ced_overlay_manifest_sha256` must be
checked before the P role can be marked ready. This receipt records the verified
source/copy only; it does not claim P was created or that CED inference ran.
