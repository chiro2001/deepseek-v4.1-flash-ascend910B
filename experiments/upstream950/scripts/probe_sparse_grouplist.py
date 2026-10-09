"""Check installed BF16 GMM type-2 support; this is not a performance test."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument('--weight-format', choices=['ND', 'NZ'], default='ND')
args = parser.parse_args()
if args.weight_format == 'NZ':
    torch_npu.npu.config.allow_internal_format = True
torch.npu.set_device(0)
torch.manual_seed(20261009)
records = []
for k, n in [(5120, 512), (256, 5120)]:
    for active in [(2, 5), (0, 7)]:
        x = torch.randn((2, k), device="npu", dtype=torch.bfloat16) * 0.05
        w = torch.randn((8, k, n), device="npu", dtype=torch.bfloat16) * 0.05
        if args.weight_format == 'NZ':
            w = torch_npu.npu_format_cast(w, 29)
        sizes = [int(i in active) for i in range(8)]
        cumulative = []
        total = 0
        for size in sizes:
            total += size
            cumulative.append(total)
        dense = torch.tensor(cumulative, device="npu", dtype=torch.int64)
        pairs = [[i, 1] for i in active] + [[i, 0] for i in range(8) if i not in active]
        sparse = torch.tensor(pairs, device="npu", dtype=torch.int64)
        native = torch_npu.npu_grouped_matmul([x], [w], group_list=dense,
                                            split_item=2, group_type=0,
                                            group_list_type=0)[0]
        record = {"k": k, "n": n, "active_experts": active,
                  "format": args.weight_format, "actual_format": torch_npu.get_npu_format(w),
                  "split_item": 2}
        try:
            candidate = torch_npu.npu_grouped_matmul([x], [w], group_list=sparse,
                                                   split_item=2, group_type=0,
                                                   group_list_type=2)[0]
            torch.npu.synchronize()
            record.update({"supported": True, "bitwise_equal": torch.equal(native, candidate),
                           "max_abs": (native.float() - candidate.float()).abs().max().item()})
            assert record["bitwise_equal"], record
        except RuntimeError as error:
            record.update({"supported": False, "error": str(error)})
            records.append(record)
            break
        records.append(record)
    if not records[-1]["supported"]:
        break
result = {"physical_chip": 5, "dtype": "bfloat16", "experts": 8,
          "test_kind": "API compatibility and output comparison only", "cases": records}
filename = ('sparse_grouplist_compatibility.json' if args.weight_format == 'ND'
            else 'sparse_grouplist_nz_compatibility.json')
Path('/work/results', filename).write_text(json.dumps(result, indent=2))
print(json.dumps(result))
