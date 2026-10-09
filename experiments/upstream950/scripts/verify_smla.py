"""Boundary/random decode comparisons for native, private control and prefetch."""
import argparse
import json
from pathlib import Path

import smla_private as ops
import torch

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--output',default='/work/results/smla_correctness_v1.json')
parser.add_argument('--quick',action='store_true')
parser.add_argument('--arm',choices=['all','native','control','prefetch'],default='all')
args = parser.parse_args()
torch.npu.set_device(0)
torch.manual_seed(20261009)
cases = [(2,2048,512,False)] if args.quick else [
    (ratio,length,topk,holes)
    for ratio in [1,2] for length in [127,128,129,2048,2049]
    for topk,holes in [(512,False),(512,True),(1024,False)]]
records = []
for ratio,length,topk,holes in cases:
    page = 128
    cmp_length = length//ratio
    nori = (length+page-1)//page
    ncmp = (cmp_length+page-1)//page
    # Shuffled physical pages detect accidental contiguous-address assumptions.
    ori_table = torch.randperm(nori,device='npu',dtype=torch.int32).unsqueeze(0)
    cmp_table = torch.randperm(ncmp,device='npu',dtype=torch.int32).unsqueeze(0)
    q = torch.randn(1,64,512,dtype=torch.bfloat16,device='npu')*.1
    ori = torch.randn(nori,page,1,512,dtype=torch.bfloat16,device='npu')*.1
    cmp = torch.randn(ncmp,page,1,512,dtype=torch.bfloat16,device='npu')*.1
    indices = torch.full((1,1,topk),-1,dtype=torch.int32,device='npu')
    count = min(topk,cmp_length)
    indices[0,0,:count] = torch.randperm(cmp_length,device='npu',dtype=torch.int32)[:count]
    if holes:
        indices[0,0,::7] = -1
    # Queries span all 64 heads with independent sink values.
    sink = torch.randn(64,dtype=torch.float32,device='npu')*.1
    cu_q = torch.tensor([0,1],dtype=torch.int32,device='npu')
    ori_len = torch.tensor([length],dtype=torch.int32,device='npu')
    cmp_len = torch.tensor([cmp_length],dtype=torch.int32,device='npu')
    residual = torch.tensor([length%ratio],dtype=torch.int32,device='npu')
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        64,1,512,cu_seqlens_q=cu_q,seqused_ori_kv=ori_len,seqused_cmp_kv=cmp_len,
        cmp_residual_kv=residual,batch_size=1,max_seqlen_q=1,
        max_seqlen_ori_kv=length,max_seqlen_cmp_kv=cmp_length,
        ori_topk=0,cmp_topk=topk,cmp_ratio=ratio,ori_mask_mode=4,cmp_mask_mode=3,
        ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_BBND',
        has_ori_kv=True,has_cmp_kv=True)
    kwargs = dict(ori_kv=ori,cmp_kv=cmp,cmp_sparse_indices=indices,
                  ori_block_table=ori_table,cmp_block_table=cmp_table,
                  cu_seqlens_q=cu_q,seqused_ori_kv=ori_len,seqused_cmp_kv=cmp_len,
                  cmp_residual_kv=residual,sinks=sink,metadata=metadata,
                  softmax_scale=512**-.5,cmp_ratio=ratio,ori_mask_mode=4,cmp_mask_mode=3,
                  ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_BBND',
                  topk_value_mode=1,return_softmax_lse=True)
    print('STAGE native',flush=True)
    expected = ops.native(q,**kwargs)
    torch.npu.synchronize()
    print('NATIVE_PASS',flush=True)
    row = {'ratio':ratio,'length':length,'topk':topk,'holes':holes,'arms':{}}
    for name,fn in [('control',ops.control),('prefetch',ops.prefetch)]:
        if args.arm not in ['all',name]:
            continue
        print('STAGE',name,flush=True)
        actual = fn(q,**kwargs)
        torch.npu.synchronize()
        errors = []
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            errors.append((a.float()-b.float()).abs().max().item())
        row['arms'][name] = {'bitwise_equal':True,'max_abs':errors}
    records.append(row)
    Path(args.output).write_text(json.dumps({'physical_chip':5,'cases':records},indent=2))
    print('CASE',json.dumps(row),flush=True)
print('COMPLETE',len(records),flush=True)
