"""Independent exact-mask probe on all eight formal EP ranges; no E2E claim."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
from formal_moe_mask import local_probs


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-chips',required=True)
    args=p.parse_args()
    assert args.physical_chips==os.environ['ASCEND_RT_VISIBLE_DEVICES']=='8,9,10,11,12,13,14,15'
    args.output.mkdir(parents=True,exist_ok=True)
    cfg_bytes=(args.model/'config.json').read_bytes()
    cfg=json.loads(cfg_bytes)['text_config']
    assert (cfg['num_hidden_layers'],cfg['n_routed_experts'],cfg['num_experts_per_tok'])==(40,384,6)
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
    source=inspect.getsource(TokenDispatcherWithAllGather.token_dispatch)
    assert 'topk_weights.masked_fill(' in source and '_i32_scalar(first_expert_idx' in source
    combine=inspect.getsource(TokenDispatcherWithAllGather.token_combine)
    assert 'combine_metadata.topk_weights.to(hidden_states.dtype)' in combine
    torch.npu.set_device(0);torch.manual_seed(20261011)
    result={'scope':'Independent exact mask and BF16 unpermute-probability boundary; synthetic inputs, no full-model claim',
            'formal_weights':False,'executing_chip':8,'physical_chips':args.physical_chips,
            'checkpoint_config_sha256':hashlib.sha256(cfg_bytes).hexdigest(),
            'dispatch_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'combine_sha256':hashlib.sha256(combine.encode()).hexdigest(),
            'kernel_sha256':hashlib.sha256(Path(inspect.getfile(local_probs)).read_bytes()).hexdigest(),
            'precision_gate':'Bitwise equal, including sign of zero; no floating tolerance',
            'cases':[],'pairs':[],'profiler_during_timing':'OFF','completed':False}
    save=lambda:(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def native(ids,weights,low,high,cast):
        output=weights.masked_fill((ids<low)|(ids>=high),0.)
        return output.to(torch.bfloat16) if cast else output
    for rank in range(8):
        first,last=rank*48,(rank+1)*48
        low=torch.tensor(first,device='npu',dtype=torch.int32)
        high=torch.tensor(last,device='npu',dtype=torch.int32)
        for rows in (1,2,8,16):
            for style in range(3):
                ids=torch.randint(0,384,(rows,6),device='npu',dtype=torch.int32)
                ids[0]=torch.tensor([first-1,first,last-1,last,0,383],device='npu',dtype=torch.int32)
                weights=torch.rand((rows,6),device='npu',dtype=torch.float32)
                if style==1:weights=weights*2-1
                if style==2:
                    weights[0]=torch.tensor([0.,-0.,1.+2.**-8,1.+3.*2.**-8,.125,-.125],device='npu')
                record={'rank_range':rank,'rows':rows,'style':style,'variants':{}}
                for cast in (False,True):
                    expected=native(ids,weights,low,high,cast)
                    actual=local_probs(ids,weights,first,last,consumer_bf16=cast)
                    bits=torch.int16 if cast else torch.int32
                    equal=bool(torch.equal(actual.view(bits),expected.view(bits)))
                    record['variants']['consumer_bf16' if cast else 'float32']={'bitwise_equal':equal}
                    assert equal,record
                result['cases'].append(record);save()
        # Real decode M=1; all ranges keep both local and masked-out slots.
        ids=torch.tensor([[first,first+7,first+47,last,0,383]],device='npu',dtype=torch.int32)
        weights=torch.rand((1,6),device='npu',dtype=torch.float32)
        functions={'native_float32':lambda ids,weights:native(ids,weights,low,high,False),
                   'fused_float32':lambda ids,weights:local_probs(ids,weights,first,last),
                   'native_consumer_bf16':lambda ids,weights:native(ids,weights,low,high,True),
                   'fused_consumer_bf16':lambda ids,weights:local_probs(ids,weights,first,last,consumer_bf16=True)}
        # Tiny graphs can become limited by CPU replay submission. Use enough
        # real device work and distinct buffers to exclude repeated-input CSE.
        graphs={};calls=512;replays=8
        inputs=[(ids.clone(),weights.clone()) for _ in range(calls)]
        result['timing_contract']={'calls_per_graph':calls,'replays_per_measurement':replays,
                                   'distinct_input_buffers_per_graph':calls,
                                   'minimum_device_to_submission_ratio':2.}
        for name,fn in functions.items():
            for _ in range(3):fn(ids,weights)
            torch.npu.synchronize();graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for input_ids,input_weights in inputs:fn(input_ids,input_weights)
            graphs[name]=graph
        for pair in range(8):
            timings={};submissions={}
            names=list(graphs) if pair%2==0 else list(reversed(graphs))
            for name in names:
                begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                begin.record()
                submitted=time.perf_counter()
                for _ in range(replays):graphs[name].replay()
                submissions[name]=(time.perf_counter()-submitted)*1e6/replays
                end.record();end.synchronize()
                timings[name]=begin.elapsed_time(end)*1000/(calls*replays)
            result['pairs'].append({'rank_range':rank,'pair':pair,'microseconds':timings,
                                   'cpu_submission_us_per_graph':submissions})
        save()
    assert len(result['cases'])==96 and len(result['pairs'])==64
    result['summary']={}
    for variant in ('float32','consumer_bf16'):
        native_values=[r['microseconds']['native_'+variant] for r in result['pairs']]
        fused_values=[r['microseconds']['fused_'+variant] for r in result['pairs']]
        speedups=[n/f for n,f in zip(native_values,fused_values)]
        result['summary'][variant]={'native_us':statistics.median(native_values),
                                   'fused_us':statistics.median(fused_values),
                                   'paired_speedup_median':statistics.median(speedups),
                                   'faster_pairs':sum(s>1 for s in speedups)}
    ratios={name:[r['microseconds'][name]*calls/r['cpu_submission_us_per_graph'][name]
                  for r in result['pairs']] for name in functions}
    result['submission_headroom']={name:{'minimum_device_to_submission_ratio':min(values),
                                        'median_device_to_submission_ratio':statistics.median(values)}
                                  for name,values in ratios.items()}
    result['timing_validated']=all(min(values)>=2 for values in ratios.values())
    result.update(completed=True,precision_passed=True);save()
    print('FORMAL_MASK_PROBE_COMPLETE',json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
