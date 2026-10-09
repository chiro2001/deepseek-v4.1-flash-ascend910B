"""Compare all rounding stages and complete strided cache storage."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from indexer_post import indexer_post


def scatter_cache_sk(cache,coords,values):
    # Same installed API and layout preparation as dsa_v41.scatter_cache_sk;
    # keep the independent test free of model-module import cycles.
    torch.ops._C_ascend.npu_scatter_nd_update_sk(
        cache.squeeze(-2),coords,values.to(cache.dtype).contiguous())

torch.npu.set_device(0)
enable_custom_op()
torch.manual_seed(20261010)
parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--stress',action='store_true')
parser.add_argument('--layouts',action='store_true')
parser.add_argument('--output',default='/work/results/indexer_post_precision_v1.json')
args=parser.parse_args()
records=[]
seeds=range(16) if args.stress else range(1)
cases=[(rows,mode,seed) for rows in [1,4] for seed in seeds
       for mode in (['random','random-small','random-large'] if args.stress
                    else ['random','zero','constant'])]
if args.stress:
    cases += [(rows,mode,0) for rows in [1,4] for mode in ['zero','constant','quant-tie']]
if args.layouts:
    cases=[(rows,'layout-'+layout,seed) for rows in [1,4] for seed in range(8)
           for layout in ['float32-rope','mixed-coords','strided-input']]
for rows,mode,seed in cases:
    torch.manual_seed(20261010+seed)
    x=torch.randn(rows,128,dtype=torch.bfloat16,device='npu')
    if mode=='zero':x.zero_()
    if mode=='constant':x.fill_(1)
    if mode=='random-small':x*=1e-5
    if mode=='random-large':x*=1e5
    gamma=torch.randn(128,dtype=torch.bfloat16,device='npu')*.1+1
    theta=torch.randn(rows,64,dtype=torch.float32,device='npu')
    cos=theta.cos().bfloat16();sin=theta.sin().bfloat16()
    if mode=='layout-float32-rope':
        cos=theta.cos();sin=theta.sin()
    if mode=='layout-strided-input':
        x=x.T.contiguous().T
        cos=cos.T.contiguous().T;sin=sin.T.contiguous().T
    if mode=='quant-tie':
        x.fill_(1)
        gamma=(torch.arange(128,dtype=torch.float32,device='npu')-126.5).bfloat16()
        gamma[0]=127
        cos.fill_(1);sin.zero_()
    coords=torch.tensor([[1,3],[0,1],[-1,-1],[2,7]][:rows],dtype=torch.int32,device='npu')
    if mode=='layout-mixed-coords':
        coords=torch.tensor([[-1,0],[0,-1],[2,0],[0,7]][:rows],dtype=torch.int64,device='npu').T.contiguous().T
    # Layer-outermost parent creates non-contiguous physical page stride.
    key_parent=torch.full((3,2,8,1,128),31,dtype=torch.int8,device='npu')
    scale_parent=torch.full((3,2,8,1,1),-7,dtype=torch.float16,device='npu')
    native_key=key_parent.clone();native_scale=scale_parent.clone()
    # The low-level native RoPE takes contiguous GM tensors. Use the logical
    # contiguous values as reference for intentionally strided test inputs;
    # the production wk output is already contiguous.
    norm=torch_npu.npu_rms_norm(x.contiguous(),gamma,epsilon=1e-20)[0]
    rot=norm.contiguous().clone()
    torch.ops._C_ascend.inplace_partial_rotary_mul(rot.view(rows,1,1,128),
            cos.contiguous().view(rows,1,1,64),sin.contiguous().view(rows,1,1,64),
            rotary_mode='interleave',partial_slice=[64,128])
    quant,scale=torch_npu.npu_dynamic_quant(rot.view(rows,1,128),dst_type=torch.int8)
    quant=quant.view(rows,128);scale=scale.view(rows)
    scatter_cache_sk(native_key[:,0],coords,quant)
    scatter_cache_sk(native_scale[:,0],coords,scale.unsqueeze(-1).to(torch.float16))
    actual=indexer_post(x,gamma,cos,sin,coords,key_parent[:,0],scale_parent[:,0],1e-20,debug=True)
    torch.npu.synchronize()
    row={'rows':rows,'mode':mode,'seed':seed,'stages':{}}
    for name,a,b in zip(['norm','rope','quant','scale'],actual,[norm,rot,quant,scale]):
        delta=(a.float()-b.float()).abs()
        row['stages'][name]={'equal':torch.equal(a,b),'different':int((a!=b).sum().item()),
                             'max_abs':float(delta.max().item())}
    row['key_cache_equal']=torch.equal(key_parent,native_key)
    row['scale_cache_equal']=torch.equal(scale_parent,native_scale)
    if not row['stages']['quant']['equal']:
        disagreements=[]
        for rr,cc in (actual[2]!=quant).nonzero().cpu().tolist()[:8]:
            value=float(rot[rr,cc].item())
            maximum=float(rot[rr].float().abs().max().item())
            ideal=value*127.0/maximum
            disagreements.append({'row':rr,'col':cc,'value':value,'maximum':maximum,
                                  'scale':float(scale[rr].item()),'ideal':ideal,
                                  'native':int(quant[rr,cc].item()),
                                  'candidate':int(actual[2][rr,cc].item())})
        row['quant_disagreements']=disagreements
    records.append(row)
    print('CASE',json.dumps(row),flush=True)
    Path(args.output).write_text(json.dumps(records,indent=2))
print('COMPLETE',flush=True)
assert all(all(stage['equal'] for stage in row['stages'].values())
           and row['key_cache_equal'] and row['scale_cache_equal'] for row in records), 'Precision gate failed'
