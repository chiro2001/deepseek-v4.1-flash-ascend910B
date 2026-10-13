# CED P0–7 preflight receipt (read-only)

Date: 2026-10-13. This records the original TP8 container and an isolated CPU-only
inspection of its image. No running service was started, stopped, or modified; the
image inspection had `--network none --read-only`, no NPU devices, and imported no
Python/Torch modules. No benchmark was run.

## Image and source identity

| Item | Value |
|---|---|
| Existing container | `dsv41-real-stack-tp8-20261010-a321` (`b1aa15d88003`) |
| Image | `local/dsv41-operator-stack-real:20261010-production` |
| Image ID / created | `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea` / `2026-10-10T02:38:16Z` |
| vLLM / vLLM-Ascend | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` / `e43cf1e9f5d9bead076853aa6bcacb671465de94` |
| Network / IPC / shm | `host` / `shareable` / 16 GiB |

The current container and isolated base image have identical source bytes for the
ACL graph and BI Python wrappers. `docker diff` contains no changed source files
under `/vllm-workspace/vllm` or `vllm-ascend`; changed directories there are import
caches. The exact file hashes were checked in both:

| Source file | SHA256 |
|---|---|
| `vllm_ascend/compilation/acl_graph.py` | `c8ab2fe6aaae034b03e9e34b9c07fd445046564acd5d45e2c24b5ea5232c5a74` |
| `vllm_ascend/worker/model_runner_v1.py` | `67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e` |
| `vllm_ascend/ops/triton/batch_invariant/matmul.py` | `2c42c1ab8110d4bc17bece46d379cc6c83044a36debd0411efe4bc2d292af20c` |
| `vllm_ascend/ops/triton/batch_invariant/mean.py` | `3b47014519d7efc0de7c1c8865ec0c70fd57dc487b47f0695ff7fd06082d8bf2` |
| `vllm_ascend/ops/triton/batch_invariant/rmsnorm.py` | `d71da6c952b76bfde5f29f9ede956b0b3921b64174fd0e25872fff68e85798b9` |
| `vllm_ascend/ops/triton/batch_invariant/softmax.py` | `424749612a70873e091ed82b8042c0719648d00f658511d6f21878e2c98466c7` |

The CED v1 overlay manifest independently binds its actual patched runtime files:
base scheduler `c67bda2886…` → patched `eba971850e…`; base runner `67035d97…` →
patched `bd250a5981…`; DSA source `e1c6bfd81f…`; Mooncake connector source
`e83ed49d50…`. `CEDFinder` verifies the manifest payload before import.

## BI OPP is not in the image

The CPU-only base-image inspection reported `BI_OPP_MISSING` at
`/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant`. In the running TP8
container, `docker diff` shows this tree as added in the writable layer; it is
214 MiB / 617 files, not a host bind mount. Therefore a fresh P container from the
same image will not inherit the installed OPP. `run_ced_service.sh` only sources
CANN/ATB environment scripts; it does not install this package.

Use the A3 package from the existing `/work` bind, not the 910B package:

| Source / installed artifact | Size | SHA256 |
|---|---:|---|
| `/work/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run` | 105,731,094 B | `9fc692978e9420336e3fea03a92c2a85df1b50a65a7df50173e3bf8bedaea70e` |
| installed `version.info` (`custom_opp_compiler_version=9.0.0`) | 34 B | `cc3e7001ed18e748e400d78bbcd3a8c51c6d292bd7f252a3ed9c4268dc1a5371` |
| `op_api/lib/libcust_opapi.so` | 974,520 B | `ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069` |
| `op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so` | 771,960 B | `f8308be503301621117cd421e411240a077dc4fcf04972321e403e7efc818010` |
| `op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so` | 5,347,432 B | `68221652dffb7fa13351616ecdf4e3445425697452fed29ae147f0cc335b02c1` |

The frozen candidate points `ASCEND_CUSTOM_OPP_PATH` to
`/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant` and requires
`VLLM_BATCH_INVARIANT=1`. The site runtime must arrange the A3 installation (or an
exact read-only mount) in the new P container and verify these hashes before
claiming profile reproducibility.

## Safe P0–7 container profile

Current TP8 container maps `/dev/davinci8`–`/dev/davinci15` plus
`/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`. For P, map the same
management devices and `/dev/davinci0`–`/dev/davinci7`, then set both
`ASCEND_RT_VISIBLE_DEVICES` and `ASCEND_VISIBLE_DEVICES` to `0,1,2,3,4,5,6,7`.
The remaining observed non-secret runtime values are `LOCAL_WORLD_SIZE=8`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `HCCL_DETERMINISTIC=strict`.

The original D process also had `VLLM_BATCH_INVARIANT=1`, `DRAFT_GRAPH=1`,
`DSPARK_GRAPH_CAPTURE_METADATA=1`, and `DSPARK_DRAFT_USE_CUDAGRAPH=1`; CED P has no
draft, so its role plan correctly disables the three draft-graph flags. Remove
`V41_DCP=1` for both CED roles. The existing container has host networking and no
`VLLM_HOST_IP` / HCCL-interface override in its process environment. vLLM's
`get_ip()` falls back to route discovery, so this is not a source-level hard
requirement; set `VLLM_HOST_IP` explicitly to the peer-reachable host/NIC address
for deterministic Mooncake P/D discovery.

Relevant original read-only mounts: qrot-MTPQ model, `/etc/hccn.conf`,
`/etc/ascend_install.info`, `/usr/local/Ascend/driver`, firmware, DCMI, and
`npu-smi`; `/work` is the source-project bind. The model path selected by the
candidate is `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq`.

## Engram and KV source receipt

The actual container `vllm_ascend/models/deepseek_v41/engram_hbm.py` was copied
read-only to `/home/chiro/tmp/dsv41-offload-research/ced-preflight/engram_hbm.py`
(61,745 B, SHA256
`b422e9899204b7a55af05832073bdea41e24cdc14900ea739c1dcd82822c77ae`). It matches
the selected source payload `patches/files/engram_hbm.py` byte-for-byte. The
separate `dcp_runtime_20261013_v2/v3` Engram files have different SHA256
`16c7cb674cf28390e42c1032350ced6aeeeca84ba4ea572d18d46d3c7de1541d`; do not use
those as the frozen best original.

KV replication basis in the pinned vLLM-Ascend source:

- `vllm_ascend/models/deepseek_v41/model.py:296-305` creates the canonical
  `DeepseekV41FullSpec` with `num_kv_heads=1`.
- `vllm_ascend/worker/model_runner_v1.py:5385-5398` takes a
  `DeepseekV41CacheLayer` spec directly; it does not divide the head count by TP.
- `vllm_ascend/core/deepseek_v41.py:122-127, 300-304` sizes and views each local
  cache page with `num_kv_heads` in its shape/byte count, without a TP divisor.

Thus each TP rank allocates the complete one-plane canonical MLA KV page; the
shared latent is not an 8-way KV shard. The code expresses this through the
per-rank one-head spec and full-page allocation rather than an explicit
`replicated=True` flag. `upstream/docs/KV-CACHE-ACCOUNTING.md:255-261` states the
same consequence for MLA: full KV copy per TP rank, not 1/TP.
