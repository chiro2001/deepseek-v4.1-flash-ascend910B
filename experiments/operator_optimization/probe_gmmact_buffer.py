"""Single-variable multibuffer comparison with identical arithmetic/tiling."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

from gmm1_activation import gmm1_activation
from microbench import capture,paired


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    torch.npu.set_device(0);torch.manual_seed(20261010)
    x=torch.randn((2,5120),device='npu',dtype=torch.bfloat16)
    weights=[(torch.randn((8,512,5120),device='npu')/5120**.5).to(torch.bfloat16) for _ in range(40)]
    counts=torch.tensor([0,1,0,0,0,0,1,0],device='npu',dtype=torch.int64)
    graphs={};result={'same_tiling':{'bn':16,'bk':512},'rotations':40,'results':[]}
    for mode,value in [('default',None),('off',False),('on',True)]:
        row={'mode':mode,'multibuffer':value}
        try:
            for scale in [.001,1.,100.]:
                reference=gmm1_activation(x*scale,weights[0],counts,7.,bn=16,bk=512)
                candidate=gmm1_activation(x*scale,weights[0],counts,7.,bn=16,bk=512,multibuffer=value)
                for a,b in zip(candidate,reference):torch.testing.assert_close(a,b,rtol=0,atol=0)
            graphs[mode]=capture(lambda i:gmm1_activation(x,weights[i%40],counts,7.,bn=16,bk=512,multibuffer=value),40)
            row['status']='precision_bit_equal'
        except Exception as exc:row.update(status='rejected',error=str(exc))
        result['results'].append(row);target.write_text(json.dumps(result,indent=2)+'\n')
        print('BUFFER_CONFIG',json.dumps({k:(v[-500:] if k=='error' else v) for k,v in row.items()}),flush=True)
    if len(graphs)>1:
        result['timing']=paired(graphs,pairs=10,repeats=30)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('BUFFER_RESULT',json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
