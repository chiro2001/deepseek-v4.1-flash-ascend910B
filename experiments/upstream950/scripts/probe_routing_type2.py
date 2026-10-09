"""Check whether the installed router can emit sparse GMM pairs directly."""
import json
from pathlib import Path

import torch
import torch_npu

torch.npu.set_device(0)
torch.manual_seed(20261009)
records = []
for ids in [[[2, 5]], [[5, 2]], [[7, 0]], [[2, 5], [5, 2], [0, 7], [2, 7]]]:
    x = torch.randn(len(ids), 5120, dtype=torch.bfloat16, device="npu")
    topk = torch.tensor(ids, dtype=torch.int32, device="npu")
    kwargs = dict(active_num=len(ids)*2, expert_num=8, expert_tokens_num_flag=True,
                  active_expert_range=[0, 8], quant_mode=-1)
    native = torch_npu.npu_moe_init_routing_v2(x, topk, expert_tokens_num_type=1, **kwargs)
    row = {"ids": ids, "count_shape": list(native[2].shape)}
    try:
        sparse = torch_npu.npu_moe_init_routing_v2(x, topk, expert_tokens_num_type=2, **kwargs)
        torch.npu.synchronize()
        row.update({"supported": True, "sparse_shape": list(sparse[2].shape),
                    "pairs": sparse[2].cpu().tolist(),
                    "expanded_x_equal": torch.equal(native[0], sparse[0]),
                    "row_idx_equal": torch.equal(native[1], sparse[1])})
    except (RuntimeError, ValueError) as error:
        row.update({"supported": False, "error": str(error)})
    records.append(row)
    if not row["supported"]:
        break
result = {"physical_chip": 5, "cases": records}
Path('/work/results/routing_type2_probe.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
