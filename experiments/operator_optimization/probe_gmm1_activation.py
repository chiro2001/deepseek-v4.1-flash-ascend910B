"""Compare native and proven two-task path against a fused routed epilogue."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch_npu

from clamped_swiglu import clamped_swiglu
from gmm1_activation import gmm1_activation
from selected_gemv import gemv
from microbench import capture,paired


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--rotations',type=int,default=40);args=p.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    torch.npu.set_device(0);torch.manual_seed(20261010)
    x=torch.randn((2,5120),device='npu',dtype=torch.bfloat16)
    weights=[(torch.randn((8,5120,512),device='npu')/5120**.5).to(torch.bfloat16) for _ in range(args.rotations)]
    nk=[w.transpose(1,2).contiguous() for w in weights]
    counts=torch.tensor([0,1,0,0,0,0,1,0],device='npu',dtype=torch.int64)
    def native(sample,w,limit):
        raw=torch_npu.npu_grouped_matmul([sample],[w],group_list=counts,
                   split_item=2,group_type=0,group_list_type=1)[0]
        gate,up=raw.chunk(2,-1);gate.clamp_(max=limit);up.clamp_(min=-limit,max=limit)
        return raw,torch_npu.npu_swiglu(raw)
    def proven(sample,w,limit):
        raw=gemv(sample,w,counts=counts,kind='vector',bn=8,bk=512,nk_layout=True)
        return raw,clamped_swiglu(raw,limit,mutate_input=True)
    count=max(24,args.rotations)
    refs={'native':capture(lambda i:native(x,weights[i%args.rotations],7.),count),
          'proven':capture(lambda i:proven(x,nk[i%args.rotations],7.),count)}
    result={'rotations':args.rotations,'bf16_boundary':True,'results':[],
            'gate':'all-element BF16 rtol=1/64 atol=1/64, max-error recorded; model audit required'}
    for bn,bk in [(4,512),(8,512),(4,1024),(8,1024),(16,256),(16,512)]:
        row={'bn':bn,'bk':bk}
        try:
            worst=[0.,0.]
            for factor,limit in itertools.product([0.,.001,1.,100.],[1.,7.,7.9]):
                expected=native(x*factor,weights[0],limit)
                actual=gmm1_activation(x*factor,nk[0],counts,limit,bn=bn,bk=bk)
                for index,(a,b) in enumerate(zip(actual,expected)):
                    torch.testing.assert_close(a,b,rtol=1/64,atol=1/64)
                    worst[index]=max(worst[index],float((a-b).abs().max()))
            candidate=capture(lambda i:gmm1_activation(x,nk[i%args.rotations],counts,7.,bn=bn,bk=bk),count)
            row.update(paired({**refs,'candidate':candidate},repeats=20))
            row.update(status='passed_screen',max_abs_vs_native=worst);del candidate
        except Exception as exc:row.update(status='rejected',error=str(exc))
        result['results'].append(row);target.write_text(json.dumps(result,indent=2)+'\n')
        print('GMM_ACT',json.dumps({k:(v[-500:] if k=='error' else v) for k,v in row.items() if k!='samples_us'}),flush=True)


if __name__=='__main__':main()
