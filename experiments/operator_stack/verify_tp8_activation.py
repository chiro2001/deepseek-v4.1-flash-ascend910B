"""Retain the existing all-element/one-ULP gate on TP8 row/width shapes."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from clamped_swiglu import clamped_swiglu
from verify_activation import compare, native_routed, native_shared


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--formal',action='store_true',help='W4A8 TP8-compatible BF16 activation shapes; synthetic inputs')
    args=p.parse_args()
    torch.npu.set_device(0);torch.manual_seed(20261010);rows=[]
    shapes = [(1,576),(2,4608),(8,4608),(16,4608)] if args.formal else [(1,64),(2,512),(8,512),(16,512)]
    limits = [1.,10.,10.1] if args.formal else [1.,7.,7.9]
    for m,width in shapes:
        for magnitude in [0.,.001,1.,100.]:
            for limit in limits:
                x=(torch.randn((m,width),device='npu')*magnitude).to(torch.bfloat16)
                for kind in ['routed','shared']:
                    if kind=='routed':
                        actual=clamped_swiglu(x.clone(),limit,mutate_input=True)
                        reference=native_routed(x.clone(),limit)
                    else:
                        actual=clamped_swiglu(x,limit,alpha=1.,beta=1.,staged_rounding=True)
                        reference=native_shared(x,limit,1.,1.)
                    metric=compare(actual,reference);assert metric['passed'],(m,width,kind,metric)
                    rows.append({'shape':[m,width],'magnitude':magnitude,'limit':limit,'kind':kind,**metric})
    result={'cases':len(rows),'all_passed':True,'formal_shapes':args.formal,
            'inputs':'synthetic; full-model coverage requires TP8 audit',
            'max_bf16_ulp':max(r['max_bf16_ulp'] for r in rows),'results':rows}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print('TP8_ACTIVATION_PRECISION',json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
