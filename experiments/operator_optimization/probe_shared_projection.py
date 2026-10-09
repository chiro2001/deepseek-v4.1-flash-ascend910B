"""Match actual F.linear N-major layout before considering shared fusion."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch_npu

from clamped_swiglu import clamped_swiglu
from shared_projection_activation import shared_projection_activation
from microbench import capture,paired


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    torch.npu.set_device(0);torch.manual_seed(20261010)
    x=torch.randn((1,5120),device='npu',dtype=torch.bfloat16)
    weights=[(torch.randn((512,5120),device='npu')/5120**.5).to(torch.bfloat16) for _ in range(40)]
    def native(sample,w,limit,alpha,beta):
        raw=torch.nn.functional.linear(sample,w)
        return raw,clamped_swiglu(raw,limit,alpha=alpha,beta=beta,staged_rounding=True)
    base=capture(lambda i:native(x,weights[i%40],7.,1.,0.),40)
    result={'layout':'actual F.linear [N,K], physical ND','rotations':40,'results':[]}
    for bn,bk in [(4,512),(8,512),(16,512),(16,256),(8,1024),(16,1024),(32,256)]:
        row={'bn':bn,'bk':bk}
        try:
            worst=[0.,0.]
            for factor,(limit,alpha,beta) in itertools.product([0.,.001,1.,100.],[(1.,1.,0.),(7.,1.,0.),(7.9,1.702,1.)]):
                ref=native(x*factor,weights[0],limit,alpha,beta)
                actual=shared_projection_activation(x*factor,weights[0],limit,alpha,beta,bn=bn,bk=bk)
                for index,(a,b) in enumerate(zip(actual,ref)):
                    torch.testing.assert_close(a,b,rtol=1/64,atol=1/64)
                    worst[index]=max(worst[index],float((a-b).abs().max()))
            bank=capture(lambda i:shared_projection_activation(x,weights[i%40],7.,1.,0.,bn=bn,bk=bk),40)
            row.update(paired({'native':base,'candidate':bank},repeats=25))
            row.update(status='passed_screen',max_abs=worst);del bank
        except Exception as exc:row.update(status='rejected',error=str(exc))
        result['results'].append(row);target.write_text(json.dumps(result,indent=2)+'\n')
        print('SHARED_PROJ',json.dumps({k:(v[-500:] if k=='error' else v) for k,v in row.items() if k!='samples_us'}),flush=True)


if __name__=='__main__':main()
