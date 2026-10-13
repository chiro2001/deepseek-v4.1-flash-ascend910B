import vllm_ascend.vllm_ascend_C  # noqa
import time, torch, torch_npu
torch.npu.set_device("npu:0")
op = torch.ops._C_ascend.npu_sparse_flash_mla_metadata
for B in (1, 8, 16, 32):
    cu = torch.arange(B+1, dtype=torch.int32, device="npu:0")
    seq = torch.full((B,), 100000, dtype=torch.int32, device="npu:0")
    cmp = torch.full((B,), 50000, dtype=torch.int32, device="npu:0")
    res = torch.zeros((B,), dtype=torch.int32, device="npu:0")
    def call():
        return op(64, 1, 512, cu_seqlens_q=cu, seqused_ori_kv=seq, seqused_cmp_kv=cmp,
                  cmp_residual_kv=res, batch_size=B, max_seqlen_q=1, max_seqlen_ori_kv=100000,
                  max_seqlen_cmp_kv=50000, ori_topk=0, cmp_topk=1024, cmp_ratio=4,
                  ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
                  layout_q="TND", layout_kv="PA_BBND", has_ori_kv=True, has_cmp_kv=True)
    call(); torch.npu.synchronize()
    N=50
    t0=time.perf_counter()
    for _ in range(N):
        v=call(); torch.npu.synchronize()
    t1=time.perf_counter()
    t0b=time.perf_counter()
    for _ in range(N): v=call()
    t1b=time.perf_counter(); torch.npu.synchronize(); t2b=time.perf_counter()
    print(f"B={B:3d}  sync-per-call {((t1-t0)/N*1e6):6.1f} us   host {((t1b-t0b)/N*1e6):6.1f} us/call  incl-device {((t2b-t0b)/N*1e6):6.1f} us/call")
