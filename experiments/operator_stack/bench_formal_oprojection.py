"""Formal TP8 single-group wo_a GEMV trials; no full-model performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
from safetensors import safe_open
from selected_gemv import gemv


def bf16_order(value):
    bits=value.view(torch.int16).to(torch.int32)&65535
    return torch.where((bits&32768)!=0,32768-(bits&32767),32768+bits)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-chips',required=True)
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    assert args.physical_chips==os.environ['ASCEND_RT_VISIBLE_DEVICES']
    assert args.physical_chips=='8,9,10,11,12,13,14,15'
    torch.npu.set_device(0)
    torch.npu.set_op_timeout_ms(30000)
    torch.manual_seed(20261011)
    cfg_bytes=(args.model/'config.json').read_bytes();cfg=json.loads(cfg_bytes)
    text=cfg['text_config'];assert (text['hidden_size'],text['num_hidden_layers'],text['o_groups'],text['o_lora_rank'])==(5120,40,8,1024)
    wm=json.loads((args.model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    variants={'cube64_256':dict(kind='cube',bn=64,bk=256),
              'cube128_512':dict(kind='cube',bn=128,bk=512),
              'vector8_512':dict(kind='vector',bn=8,bk=512,nk_layout=True),
              'vector16_256':dict(kind='vector',bn=16,bk=256,nk_layout=True)}
    result={'checkpoint_config_sha256':hashlib.sha256(cfg_bytes).hexdigest(),
            'model':str(args.model),'physical_chips':args.physical_chips,'executing_chip':8,
            'scope':'Independent formal TP8 rank-shard wo_a; synthetic inputs, no routing or E2E claim',
            'formal_weights':False,'source_weights':'Unaltered formal BF16 rank shards',
            'input_shape':[1,4096],'weight_shape':[4096,1024],
            'precision_gate':'Every finite BF16 output at most one representable ULP from native; no tolerance relaxation',
            'profiler_during_timing':'OFF','candidates':{},'weight_sha256':{},'cases':[]}
    for name in variants:result['candidates'][name]={'passed':True,'checked_cases':0,'failures':[],'timing':[]}
    # Two distinct real layers × all8 rank shards. Whole-model sharding of
    # ColumnParallelLinear slices1024 output rows, then transposes to K×N.
    for layer in (0,20):
        key=f'layers.{layer}.attn.wo_a.weight'
        with safe_open(str(args.model/wm[key]),framework='pt',device='cpu') as archive:
            sliced=archive.get_slice(key);assert sliced.get_shape()==[8192,4096]
            for rank in range(8):
                cpu=sliced[rank*1024:(rank+1)*1024].contiguous()
                result['weight_sha256'][f'{layer}:{rank}']=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
                w=cpu.t().contiguous().to('npu');nk=cpu.to('npu')
                active=[n for n,v in result['candidates'].items() if v['passed']]
                if not active:break
                for magnitude in (.001,1.,100.):
                    for seed in (0,1):
                        torch.manual_seed(layer*101+rank*7+seed)
                        x=(torch.randn((1,4096),device='npu')*magnitude).to(torch.bfloat16)
                        reference=torch.matmul(x,w)
                        assert torch.isfinite(reference).all()
                        for name in list(active):
                            params=variants[name];begin=time.monotonic()
                            try:
                                actual=gemv(x,nk if params.get('nk_layout') else w,**params)
                                torch.npu.synchronize()
                                finite=bool(torch.isfinite(actual).all())
                                ulp=(bf16_order(actual)-bf16_order(reference)).abs()
                                maximum=int(ulp.max());bad=int((ulp>1).sum())
                                passed=finite and bad==0
                                receipt={'candidate':name,'layer':layer,'rank':rank,'magnitude':magnitude,
                                         'seed':seed,'max_bf16_ulp':maximum,'elements_above_one_ulp':bad,
                                         'finite':finite,'passed':passed,'first_call_seconds':time.monotonic()-begin}
                            except Exception as exc:
                                receipt={'candidate':name,'layer':layer,'rank':rank,'magnitude':magnitude,
                                         'seed':seed,'passed':False,'error':str(exc)[:1200]}
                            result['cases'].append(receipt);state=result['candidates'][name];state['checked_cases']+=1
                            if not receipt['passed']:
                                state['passed']=False;state['failures'].append(receipt);active.remove(name)
                            print('FORMAL_WOA_PRECISION',json.dumps(receipt),flush=True)
                        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                # One short graph pairing per real shard, only for candidates
                # that have passed every case encountered. Overall adoption
                # still requires all96 cases, not the earliest passing shard.
                if not active:continue
                x=torch.randn((1,4096),device='npu',dtype=torch.bfloat16)
                functions={'native':lambda:torch.matmul(x,w)}
                for name in active:
                    params=variants[name];weight=nk if params.get('nk_layout') else w
                    functions[name]=lambda params=params,weight=weight:gemv(x,weight,**params)
                graphs={};calls=16;replays=32
                for name,fn in functions.items():
                    for _ in range(3):fn()
                    torch.npu.synchronize();g=torch.npu.NPUGraph()
                    with torch.npu.graph(g):
                        for _ in range(calls):fn()
                    graphs[name]=g
                for pair in range(4):
                    measurements={}
                    order=list(graphs) if pair%2==0 else list(reversed(graphs))
                    for name in order:
                        begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                        begin.record()
                        for _ in range(replays):graphs[name].replay()
                        end.record();end.synchronize()
                        measurements[name]=begin.elapsed_time(end)*1000/(calls*replays)
                    for name in active:
                        result['candidates'][name]['timing'].append({'layer':layer,'rank':rank,'pair':pair,
                            'native_us':measurements['native'],'candidate_us':measurements[name],
                            'speedup':measurements['native']/measurements[name]})
                del graphs,functions,w,nk,cpu
    for name,state in result['candidates'].items():
        state['full_precision_passed']=state['passed'] and state['checked_cases']==96
        if state['timing']:
            rows=state['timing'];state['median_native_us']=statistics.median(r['native_us'] for r in rows)
            state['median_candidate_us']=statistics.median(r['candidate_us'] for r in rows)
            state['paired_speedup_median']=statistics.median(r['speedup'] for r in rows)
            state['faster_pairs']=sum(r['speedup']>1 for r in rows)
        state['eligible_for_model_trial']=state['full_precision_passed'] and state.get('paired_speedup_median',0)>1.01
    result['completed']=True
    result['eligible_candidates']=[n for n,v in result['candidates'].items() if v['eligible_for_model_trial']]
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FORMAL_WOA_COMPLETED',json.dumps({n:{k:v for k,v in s.items() if k not in ['timing','failures']} for n,s in result['candidates'].items()}),flush=True)


if __name__=='__main__':main()
