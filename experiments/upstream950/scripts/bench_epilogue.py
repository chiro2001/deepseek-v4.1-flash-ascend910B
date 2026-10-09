"""Paired device graph timing of inverse RoPE plus eight-group wo_a."""
import argparse
import json
import statistics
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from epilogue_patches import native_epilogue,inplace_epilogue,gmm_epilogue
from epilogue_prepare import grouped_epilogue

parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--output',required=True)
parser.add_argument('--arms',default='baseline,packed,inplace,gmm')
args=parser.parse_args()
arms=args.arms.split(',')
assert arms[0]=='baseline'
torch.npu.set_device(0);enable_custom_op();torch.manual_seed(20261010)
raw=torch.randn(1,64,512,dtype=torch.bfloat16,device='npu')
weight=torch.randn(8,4096,512,dtype=torch.bfloat16,device='npu')/64
theta=torch.randn(1,64,dtype=torch.float32,device='npu')
cos=theta.cos().view(1,1,1,64);sin=theta.sin().view(1,1,1,64)
scratch=raw.clone()
def native():
    # Native RoPE is in-place. Include the same input restore copy in both
    # reported times so graph replay never repeatedly rotates its own output.
    scratch.copy_(raw)
    return native_epilogue(scratch,cos,sin,weight)
def packed():
    scratch.copy_(raw)
    return grouped_epilogue(scratch,cos,sin,weight)
def inplace():
    scratch.copy_(raw)
    return inplace_epilogue(scratch,cos,sin,weight)
def gmm():
    scratch.copy_(raw)
    return gmm_epilogue(scratch,cos,sin,weight)
banks={}
calls={'baseline':native,'packed':packed,'inplace':inplace,'gmm':gmm}
for name in arms:
    fn=calls[name]
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
        for _ in range(50):banks[name].replay()
        end.record();end.synchronize()
        timings[name]=start.elapsed_time(end)*1000/(50*20)
    pairs.append({'pair':pair,**{n+'_us':v for n,v in timings.items()},
                  'speedups':{n:timings['baseline']/timings[n] for n in arms[1:]}})
report={'rows':1,'same_process':True,'profiler':'OFF','common_input_restore_copy':True,
        'pairs':pairs,'baseline_median_us':statistics.median(p['baseline_us'] for p in pairs),
        'arms':{n:{'median_us':statistics.median(p[n+'_us'] for p in pairs),
                   'paired_speedup_median':statistics.median(p['speedups'][n] for p in pairs)}
                for n in arms[1:]}}
Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
