"""Compare inverse RoPE and eight-group wo_a on random BF16 inputs/weights."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from epilogue_prepare import inverse_rope_pack,inverse_rope_inplace
from epilogue_patches import grouped_wo_a

parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--output',required=True)
args=parser.parse_args()
torch.npu.set_device(0);enable_custom_op()
records=[]
for rows in [1,4]:
    for dtype in [torch.float32,torch.bfloat16]:
        for seed in range(8):
            torch.manual_seed(20261010+seed)
            raw=torch.randn(rows,64,512,dtype=torch.bfloat16,device='npu')
            weight=torch.randn(8,4096,512,dtype=torch.bfloat16,device='npu')/(4096**.5)
            theta=torch.randn(rows,64,dtype=torch.float32,device='npu')
            cos=theta.cos().to(dtype).view(rows,1,1,64)
            sin=theta.sin().to(dtype).view(rows,1,1,64)
            native=raw.clone()
            torch.ops._C_ascend.inplace_partial_rotary_mul(native.unsqueeze(1),cos,-sin,
                rotary_mode='interleave',partial_slice=[448,512])
            reference=torch_npu.npu_transpose_batchmatmul(native.view(rows,8,4096),weight,
                bias=None,scale=None,perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2),batch_split_factor=1)
            packed=inverse_rope_pack(raw,cos,sin)
            inplace=inverse_rope_inplace(raw.clone(),cos,sin)
            grouped=grouped_wo_a(native,weight).view(rows,8,512)
            actual=torch.bmm(packed,weight).transpose(0,1).contiguous()
            torch.npu.synchronize()
            expected_pack=native.view(rows,8,4096).transpose(0,1)
            diff=(actual.float()-reference.float()).abs()
            record={'rows':rows,'trig_dtype':str(dtype),'seed':seed,
                    'inverse_rope_bitwise_equal':torch.equal(packed,expected_pack),
                    'inplace_rope_bitwise_equal':torch.equal(inplace,native),
                    'gmm_wo_a_bitwise_equal':torch.equal(grouped,reference),
                    'wo_a_bitwise_equal':torch.equal(actual,reference),
                    'wo_a_different':int((actual!=reference).sum().item()),
                    'wo_a_max_abs':float(diff.max().item()),
                    'wo_a_rms_error':float(diff.square().mean().sqrt().item())}
            records.append(record);print('CASE',json.dumps(record),flush=True)
            Path(args.output).write_text(json.dumps(records,indent=2))
assert all(r['inverse_rope_bitwise_equal'] and r['inplace_rope_bitwise_equal'] and r['wo_a_bitwise_equal'] and r['gmm_wo_a_bitwise_equal'] for r in records), 'Precision gate failed'
print('COMPLETE',flush=True)
