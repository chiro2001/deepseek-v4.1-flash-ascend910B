"""Interleaved graph timings of native and fused activation, profiler OFF."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from clamped_swiglu import clamped_swiglu
from verify_activation import native_routed, native_shared


def capture(func,x):
    for _ in range(10):func(x)
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs=[func(x) for _ in range(20)]
    return graph,outputs


def elapsed(graph):
    begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(50):graph.replay()
    end.record();end.synchronize()
    return begin.elapsed_time(end)*1000/(20*50)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.npu.set_device(0);torch.manual_seed(20261009)
    records=[]
    for kind,m in [('routed',2),('shared',1)]:
        x=torch.randn((m,512),dtype=torch.bfloat16,device='npu')
        funcs={'native':lambda x:native_routed(x,7.0) if kind=='routed' else native_shared(x,7.0),
               'fused':lambda x:clamped_swiglu(x,7.0,mutate_input=kind=='routed',staged_rounding=kind=='shared')}
        captured={name:capture(func,x.clone()) for name,func in funcs.items()}
        samples={name:[] for name in funcs}
        for pair in range(7):
            order=list(funcs) if pair%2==0 else list(reversed(funcs))
            for name in order:samples[name].append(elapsed(captured[name][0]))
        medians={name:statistics.median(values) for name,values in samples.items()}
        result={'kind':kind,'shape':[m,512],'samples_us':samples,'medians_us':medians,'speedup':medians['native']/medians['fused']}
        records.append(result);print('RESULT',json.dumps(result),flush=True)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'profiler':'OFF','method':'20 calls per graph, 50 replays, seven interleaved pairs','records':records},indent=2)+'\n')


if __name__=='__main__':main()
