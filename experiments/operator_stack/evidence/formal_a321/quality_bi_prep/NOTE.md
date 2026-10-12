# Batch-invariant quality arm: vendor-defect repairs (2026-10-13)

Goal context: the DSpark quality run scores GSM8K 99/100 (gsm8k-12 answers 12
where the 1-row decode answers 13). Root cause carried over from the divergence
audit: the 6-row verifier and the 1-row decode differ in last-bit numerics, and
that题目 lands on a near tie. Vendor remedy: `VLLM_BATCH_INVARIANT=1`
(vllm-ascend registers batch-invariant mm/matmul/softmax/attention/rms-norm/
reduce implementations at worker init when the CANN batch-invariant op package
is present).

Three vendor defects had to be repaired to make that mode usable:

1. `vllm_ascend.batch_invariant` shadows `torch.sum`/`Tensor.sum` with
   `reduce_sum(x, dim, keepdim)`, which rejects the NumPy-style `axis=` alias.
   The vendor's own W4A8 weight loader calls `.sum(axis=1)`, so engine init
   crashed. `bi_axis_compat.py` maps `axis`->`dim`. Non-last-dim reductions fall
   back to the stock torch sum: the Ascend batch-invariant reduce kernel only
   implements the last dimension, and those calls are load-time weight
   post-processing, not part of the served forward.

2. The CANN op package that supplies `libopapi_batchinvariant.so` was missing in
   the container. First attempt installed the 910b variant; this node reports
   `get_soc_version()=253` / `Ascend910_9382` (A3 / 910_93 family), which failed
   with `aclnnReduceSumBatchInvariant ... not in libopapi_batchinvariant.so`.
   The 910b install was uninstalled and
   `cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run` was installed with
   `--quiet --nox11 --install-for-all`. The vendor requires the op path via
   `ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant`;
   exporting LD_LIBRARY_PATH instead breaks `libhccl.so` resolution from
   `set_env.sh`. Device check after install: 6-row and 1-row reduce agree
   bit-for-bit.

3. The A3 op package ships Meta (fake-tensor) kernels with wrong output shapes
   (`npu_reduce_sum_batch_invariant` returned the input shape), so Dynamo's
   fake-tensor tracing failed at the engram gate `.sum(-1)` with a broadcast
   error. `bi_axis_compat.install_meta_kernels()` registers correct Meta shapes
   for seven ops (reduce_sum/reduce_mean/softmax/log_softmax/mm/matmul/
   add_rms_norm); in-container check prints
   `[BI-META] installed=7 skipped=` and fake-tensor shapes are correct.

Files:
- `bi_axis_compat.py`: the env-gated shim (imported by the snapshot's
  `sitecustomize.py` only when `VLLM_BATCH_INVARIANT=1`).
- `build_bi_seed_v9.py`: builds the candidate seed from the mls snapshot plus
  the shim, with the engine flags of the <=20ms windows plus
  `VLLM_BATCH_INVARIANT` and `ASCEND_CUSTOM_OPP_PATH`.
- `patch_v9_manifest.py` / `patch_v9_seed.py`: re-pin the embedded source maps
  after editing the shim (the export step cross-checks seed vs manifest).

Status: not yet validated end-to-end. A first launch attempt (v9b) was refused
by the launcher's occupancy gate (physical 14 held by another tenant's qwen3
test), and a retry hit physical 8 held by another tenant. No foreign process
was touched.
