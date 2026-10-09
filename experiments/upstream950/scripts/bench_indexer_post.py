"""No-profiler, paired NPU-graph timing of the exact native postprocess chain."""
import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from indexer_patches import native_post
from indexer_post import indexer_post

parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--output',required=True)
args=parser.parse_args()
torch.npu.set_device(0)
enable_custom_op()
torch.manual_seed(20261010)
records=[]
for rows in [1,4]:
    x=torch.randn(rows,128,dtype=torch.bfloat16,device='npu')
    gamma=torch.randn(128,dtype=torch.bfloat16,device='npu')*.1+1
    theta=torch.randn(rows,64,dtype=torch.float32,device='npu')
    cos=theta.cos();sin=theta.sin()
    coords=torch.tensor([[1,3],[0,1],[-1,-1],[2,7]][:rows],dtype=torch.int32,device='npu')
    key=torch.zeros((3,2,128,1,128),dtype=torch.int8,device='npu')[:,0]
    scale=torch.zeros((3,2,128,1,1),dtype=torch.float16,device='npu')[:,0]
    class Norm:
        def __call__(self,v):
            return torch_npu.npu_rms_norm(v,gamma,epsilon=1e-20)[0]
    module=SimpleNamespace(width=128,rope_width=64,k_norm=Norm())
    calls={'baseline':lambda:native_post(module,x,coords,cos.view(rows,1,1,64),
                                        sin.view(rows,1,1,64),key,scale),
           'fused':lambda:indexer_post(x,gamma,cos,sin,coords,key,scale,1e-20)}
    banks={}
    for name,fn in calls.items():
        for _ in range(8):fn()
        torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(20):fn()
        banks[name]=graph
    pairs=[]
    for pair in range(12):
        timings={}
        for name in (list(banks) if pair%2==0 else list(reversed(banks))):
            for _ in range(5):banks[name].replay()
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(100):banks[name].replay()
            end.record();end.synchronize()
            timings[name]=start.elapsed_time(end)*1000/(100*20)
        pairs.append({'pair':pair,**{name+'_us':value for name,value in timings.items()},
                      'speedup':timings['baseline']/timings['fused']})
    record={'rows':rows,'profiler':'OFF','same_process':True,'pairs':pairs,
            'baseline_median_us':statistics.median(p['baseline_us'] for p in pairs),
            'fused_median_us':statistics.median(p['fused_us'] for p in pairs),
            'paired_speedup_median':statistics.median(p['speedup'] for p in pairs)}
    records.append(record)
    print(json.dumps(record),flush=True)
Path(args.output).write_text(json.dumps(records,indent=2))
