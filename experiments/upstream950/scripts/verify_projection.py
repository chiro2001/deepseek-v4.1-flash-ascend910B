"""Random BF16 joint QA/KV, NZ q_b, and packed-panel precision gate."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
import torch.nn.functional as F
from projection_panel import joint_weight,pack_weight,q_b_panel,invalidate,nz_pack,matmul_validation

parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--output',required=True)
args=parser.parse_args()
torch.npu.set_device(0)
records=[]
for rows in [1,4]:
    for seed in range(8):
        torch.manual_seed(20261010+seed);invalidate()
        hidden=torch.randn(rows,5120,dtype=torch.bfloat16,device='npu')
        qa=torch.randn(512,5120,dtype=torch.bfloat16,device='npu')/(5120**.5)
        kv=torch.randn_like(qa)/(5120**.5)
        joined=F.linear(hidden,joint_weight(qa,kv))
        projected_qa=F.linear(hidden,qa);projected_kv=F.linear(hidden,kv)
        qr=torch.randn(rows,512,dtype=torch.bfloat16,device='npu')
        qb=torch.randn(32768,512,dtype=torch.bfloat16,device='npu')/(512**.5)
        expected=F.linear(qr,qb)
        nz=nz_pack(qb)
        nz_output=F.linear(qr,nz)
        packed=pack_weight(qb)
        actual=q_b_panel(qr,packed)
        prefetched=q_b_panel(qr,packed,prefetch=True)
        torch.npu.synchronize()
        difference=(actual.float()-expected.float()).abs()
        record={'rows':rows,'seed':seed,'joint_qa_equal':torch.equal(joined[:,:512],projected_qa),
                'joint_kv_equal':torch.equal(joined[:,512:],projected_kv),
                'nz_format':int(torch_npu.get_npu_format(nz)),'nz_equal':torch.equal(nz_output,expected),
                'panel_bitwise_equal':torch.equal(actual,expected),
                'panel_different':int((actual!=expected).sum().item()),
                'panel_max_abs':float(difference.max().item()),
                'panel_rms_error':float(difference.square().mean().sqrt().item()),
                'panel_bf16_close':bool(torch.allclose(actual,expected,rtol=.0078125,atol=.000244140625))}
        record['joint_qa']=matmul_validation(joined[:,:512],projected_qa)
        record['joint_kv']=matmul_validation(joined[:,512:],projected_kv)
        record['prefetch']=matmul_validation(prefetched,expected)
        records.append(record);print('CASE',json.dumps(record),flush=True)
        Path(args.output).write_text(json.dumps(records,indent=2))
assert all(r['joint_qa']['bf16_close'] and r['joint_kv']['bf16_close'] and r['nz_format']==29 and r['nz_equal'] and r['panel_bf16_close'] and r['prefetch']['bf16_close'] for r in records), 'Projection precision gate failed'
print('COMPLETE',flush=True)
