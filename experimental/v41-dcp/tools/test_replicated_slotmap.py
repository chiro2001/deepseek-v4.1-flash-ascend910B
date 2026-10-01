"""device 路径 vs numpy 路径的逐值等价测试（在容器内运行）。"""
import types
import numpy as np
import torch
from vllm_ascend.worker.block_table import BlockTable


class Buf:
    def __init__(self, np_arr, gpu_t):
        self.np = np_arr
        self.gpu = gpu_t

    def copy_to_gpu(self, n):
        self.gpu[:n] = torch.from_numpy(np.ascontiguousarray(self.np[:n])).to(self.gpu.device)


def make(nreq, max_blocks, bppb, bs, ntokens, seed):
    rng = np.random.default_rng(seed)
    bt_np = rng.integers(0, 900, size=(nreq, max_blocks * bppb), dtype=np.int32)
    obj = types.SimpleNamespace()
    obj.effective_dcp_world_size = 1
    obj.is_circular_group = False
    obj.block_size = bs
    obj.kernel_sizes = [bs]
    obj.max_num_blocks_per_req = max_blocks
    obj.blocks_per_phys_block = bppb
    obj.block_table = Buf(bt_np, torch.from_numpy(bt_np.copy()).npu())
    obj.slot_mapping = Buf(
        np.zeros(ntokens, np.int32),
        torch.zeros(ntokens, dtype=torch.int32).npu(),
    )
    obj.compute_slot_mapping_draft = types.MethodType(BlockTable.compute_slot_mapping_draft, obj)
    obj._compute_replicated_slot_mapping = types.MethodType(
        BlockTable._compute_replicated_slot_mapping, obj
    )
    return obj, rng


def run_case(nreq, max_blocks, bppb, bs, ntokens, seed, req_dtype, pos_max):
    obj, rng = make(nreq, max_blocks, bppb, bs, ntokens, seed)
    req = rng.integers(0, nreq, size=ntokens, dtype=np.int64)
    pos = rng.integers(0, pos_max, size=ntokens).astype(np.int64)
    # A: numpy 路径
    obj.compute_slot_mapping_draft(req.copy(), pos.copy())
    a = obj.slot_mapping.np[:ntokens].copy()
    # B: device 路径
    obj.slot_mapping.gpu.zero_()
    obj.compute_slot_mapping_draft(
        torch.from_numpy(req.copy()).npu(), torch.from_numpy(pos.copy()).npu()
    )
    b = obj.slot_mapping.gpu[:ntokens].cpu().numpy()
    same = np.array_equal(a, b)
    print(
        "nreq=%d max_blocks=%d bppb=%d bs=%d T=%d seed=%d req_dtype=%s pos_max=%d -> %s"
        % (nreq, max_blocks, bppb, bs, ntokens, seed, req_dtype, pos_max, "SAME" if same else "DIFF")
    )
    if not same:
        bad = np.nonzero(a != b)[0][:8]
        for i in bad:
            print("   idx=%d numpy=%d device=%d req=%d pos=%d" % (i, a[i], b[i], req[i], pos[i]))
        return False
    return True


ok = True
# 覆盖：默认形状、单请求、块表刚好用满、position 跨越 phys block 边界、超长 positions
ok &= run_case(4, 17, 8, 128, 40, 1, "i64", 17 * 8 * 128)
ok &= run_case(1, 5, 1, 128, 9, 2, "i64", 5 * 128)
ok &= run_case(8, 33, 8, 128, 64, 3, "i64", 33 * 8 * 128)
ok &= run_case(3, 40, 8, 128, 50, 4, "i64", 40 * 8 * 128)
ok &= run_case(2, 200, 8, 128, 33, 5, "i64", 200 * 8 * 128)
# 边界：position 恰好落在块首/块尾
obj, _ = make(2, 17, 8, 128, 6, 9)
edge_req = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
edge_pos = np.array([0, 127, 128, 1023, 8191, 16383], dtype=np.int64)
obj.compute_slot_mapping_draft(edge_req.copy(), edge_pos.copy())
a = obj.slot_mapping.np[:6].copy()
obj.slot_mapping.gpu.zero_()
obj.compute_slot_mapping_draft(
    torch.from_numpy(edge_req).npu(), torch.from_numpy(edge_pos).npu()
)
b = obj.slot_mapping.gpu[:6].cpu().numpy()
same = np.array_equal(a, b)
print("边界用例 -> %s  numpy=%s device=%s" % ("SAME" if same else "DIFF", a.tolist(), b.tolist()))
ok &= bool(same)

print("RESULT:", "ALL_SAME" if ok else "MISMATCH")
