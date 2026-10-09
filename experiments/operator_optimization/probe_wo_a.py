"""Compare native 8-group wo_a dispatches without changing its mathematics."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu


def transpose_batch(x, weight):
    return torch_npu.npu_transpose_batchmatmul(
        x, weight, bias=None, scale=None,
        perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2),batch_split_factor=1,
    )


def plain_bmm(x, weight):
    # Moving a size-one axis is a view; no weight transpose or per-call copy.
    return torch.bmm(x.transpose(0,1),weight).transpose(0,1)


def capture(func,x,weights):
    for weight in weights:func(x,weight)
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):outputs=[func(x,weight) for weight in weights]
    return graph,outputs


def elapsed(graph,calls,repeats):
    begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):graph.replay()
    end.record();end.synchronize()
    return begin.elapsed_time(end)*1000/(calls*repeats)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.npu.set_device(0);torch.manual_seed(20261009)
    x=torch.randn((1,8,4096),dtype=torch.bfloat16,device='npu')
    weights=[(torch.randn((8,4096,512),device='npu')/4096**.5).to(torch.bfloat16) for _ in range(8)]
    precision=[]
    for scale in [0.0,0.001,1.0,100.0]:
        sample=x*scale
        native=transpose_batch(sample,weights[0]);candidate=plain_bmm(sample,weights[0])
        torch.testing.assert_close(candidate,native,rtol=2**-6,atol=2**-6)
        golden=torch.bmm(sample.cpu().double().transpose(0,1),weights[0].cpu().double()).transpose(0,1)
        errors={name:float((value.cpu().double()-golden).abs().max()) for name,value in [('native',native),('bmm',candidate)]}
        precision.append({'scale':scale,'max_abs_vs_fp64':errors,'native_max_abs_delta':float((candidate-native).abs().max()),
                          'bit_equal_fraction':float((candidate.cpu().view(torch.int16)==native.cpu().view(torch.int16)).float().mean())})
    timings=[]
    for mode,selected,repeats in [('hot',[weights[0]]*20,50),('rotating_8',weights,125)]:
        graphs={name:capture(func,x,selected) for name,func in [('transpose_batch',transpose_batch),('bmm',plain_bmm)]}
        samples={name:[] for name in graphs}
        for pair in range(7):
            order=list(graphs) if pair%2==0 else list(reversed(graphs))
            for name in order:samples[name].append(elapsed(graphs[name][0],len(selected),repeats))
        medians={name:statistics.median(values) for name,values in samples.items()}
        row={'mode':mode,'samples_us':samples,'medians_us':medians,'speedup':medians['transpose_batch']/medians['bmm']}
        timings.append(row);print('TIMING',json.dumps(row),flush=True)
    result={'shape_x':[1,8,4096],'shape_weight':[8,4096,512],'dtype':'bfloat16','profiler':'OFF',
            'precision':precision,'timings':timings,'rotating_weight_bytes':8*8*4096*512*2,
            'cache_note':'rotating weights exceed 192 MiB L2; this is not a measured cache-miss guarantee',
            'deployment':'diagnostic dispatch comparison; not selected without matched model validation'}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
