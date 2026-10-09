"""Numeric/UB/performance gate for full-width HC + RMS/RmsNormCast fusion."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env,enable_custom_op

from hc_static import hc_static,round_hf32
from hc_prenorm import hc_prenorm,finish_prenorm
from microbench import capture,paired


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    bootstrap_custom_op_env();enable_custom_op();torch.npu.set_device(0);torch.manual_seed(20261010)
    x=torch.randn((1,4,5120),device='npu',dtype=torch.bfloat16)
    fn=round_hf32(torch.randn((24,20480),device='npu')/20480**.5)
    scale=torch.tensor([.5,1.,2.],device='npu');base=torch.randn((24,),device='npu')*.1
    weight=torch.randn((5120,),device='npu',dtype=torch.bfloat16)
    mix=torch.randn((1,4),device='npu')
    def reference(sample,pmix,emit):
        hc=hc_static(sample,fn,scale,base,pmix)
        if emit:
            norm,route=torch.ops._C_ascend.npu_rms_norm_cast(hc[0],weight,1e-6)
        else:
            norm=torch_npu.npu_rms_norm(hc[0],weight,epsilon=1e-6)[0]
            route=torch.empty((0,),device=x.device,dtype=torch.float32)
        return *hc,norm,route
    result={'precision_gate':'HC 4e-3/1e-4; norm BF16 1/64; FP32 routing 1e-4 all elements',
            'end_to_end':False,'results':[],
            'loaded_finish_source_sha256':hashlib.sha256(finish_prenorm.src.encode()).hexdigest()}
    print('FINISH_SOURCE_SHA256',result['loaded_finish_source_sha256'],flush=True)
    for by,tree,emit in itertools.product([5120,8192],[True,False],[False,True]):
        row={'by':by,'tree':tree,'emit_fp32':emit}
        try:
            worst=[0.]*6;equal=[True]*6
            for factor,pmix in itertools.product([0.,.001,1.,100.],[None,mix]):
                ref=reference(x*factor,pmix,emit)
                actual=hc_prenorm(x*factor,fn,scale,base,weight,pmix,by=by,tree=tree,emit_fp32=emit)
                for index,(a,b) in enumerate(zip(actual,ref)):
                    if not a.numel():continue
                    tol=4e-3 if index==0 else (1/64 if index==4 else 1e-4)
                    torch.testing.assert_close(a.float(),b.float(),rtol=tol,atol=tol)
                    worst[index]=max(worst[index],float((a.float()-b.float()).abs().max()))
                    equal[index]=equal[index] and bool(torch.equal(a,b))
            banks={'baseline':capture(lambda _:reference(x,mix,emit)),
                   'candidate':capture(lambda _:hc_prenorm(x,fn,scale,base,weight,mix,by=by,tree=tree,emit_fp32=emit))}
            row.update(paired(banks,repeats=30));row.update(status='passed_screen',worst_abs=worst,bit_equal=equal)
            del banks
        except Exception as exc:row.update(status='rejected',error=str(exc))
        result['results'].append(row);target.write_text(json.dumps(result,indent=2)+'\n')
        print('PRENORM',json.dumps({k:(v[-600:] if k=='error' else v) for k,v in row.items() if k!='samples_us'}),flush=True)


if __name__=='__main__':main()
