"""Formal INT8 projection probe: native NZ loader, exact integer accumulation."""
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
from safetensors import safe_open
from formal_quant_gemv import quant_gemv


def bf16_order(x):
    bits=x.view(torch.int16).to(torch.int32)&65535
    return torch.where((bits&32768)!=0,32768-(bits&32767),32768+bits)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-chips',required=True)
    args=p.parse_args()
    assert args.physical_chips==os.environ['ASCEND_RT_VISIBLE_DEVICES']=='8,9,10,11,12,13,14,15'
    args.output.mkdir(parents=True,exist_ok=True);torch.npu.set_device(0)
    # vLLM's NPU platform enables internal layouts during worker init. This
    # standalone probe must establish the same contract before NZ conversion.
    torch_npu.npu.config.allow_internal_format=True
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod as Native
    source=inspect.getsource(Native.process_weights_after_loading)
    assert 'transpose(0, 1).contiguous()' in source and 'maybe_trans_nz' in source
    cfg_bytes=(args.model/'config.json').read_bytes();cfg=json.loads(cfg_bytes)['text_config']
    assert (cfg['hidden_size'],cfg['q_lora_rank'],cfg['head_dim'],cfg['num_attention_heads'])==(5120,1280,512,64)
    wm=json.loads((args.model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    variants={f'{layout}_64_256_order{order}':dict(nz=layout=='nz',bn=64,bk=256,order=order)
              for layout in ('nd','nz') for order in (0,1,2)}
    variants.update({f'nz_{bn}_{bk}_order0':dict(nz=True,bn=bn,bk=bk,order=0)
                     for bn,bk in ((64,512),(128,256))})
    result={'scope':'Independent unaltered formal W8A8 attention projections; synthetic activations, no TP8 or client performance claim',
            'executing_chip':8,'physical_chips':args.physical_chips,'model':str(args.model),
            'checkpoint_config_sha256':hashlib.sha256(cfg_bytes).hexdigest(),
            'loader_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'kernel_sha256':hashlib.sha256(Path(inspect.getfile(quant_gemv)).read_bytes()).hexdigest(),
            'precision_gate':'Every finite BF16 element within original one-ULP gate; exact-match count also reported',
            'profiler_during_timing':'OFF','candidates':{n:{'passed':True,'cases':0,'failures':[],'pairs':[]} for n in variants},
            'cases':[],'weights':[],'completed':False}
    save=lambda:(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def load(key,start,end):
        with safe_open(str(args.model/wm[key]),framework='pt',device='cpu') as f:
            return f.get_slice(key)[start:end].contiguous()
    for layer in (0,20):
        for part,k,n,ranks in (('wq_a',5120,1280,(0,)),('wkv',5120,512,(0,)),('wq_b',1280,4096,range(8))):
            for rank in ranks:
                key=f'layers.{layer}.attn.{part}.weight';first=rank*n
                cpu=load(key,first,first+n);assert cpu.shape==(n,k) and cpu.dtype==torch.int8
                scpu=load(key+'_scale',first,first+n).flatten().to(torch.bfloat16)
                nd=cpu.t().contiguous().to('npu');nz=torch_npu.npu_format_cast(nd,29);scale=scpu.to('npu')
                assert torch_npu.get_npu_format(nz)==29
                assert torch.equal(torch_npu.npu_format_cast(nz,2).cpu(),cpu.t().contiguous())
                result['weights'].append({'layer':layer,'projection':part,'rank_shard':rank,'K':k,'N':n,
                    'weight_sha256':hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                    'loaded_bf16_scale_sha256':hashlib.sha256(scpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    'native_format':torch_npu.get_npu_format(nz),'native_logical_shape':list(nz.shape)})
                for style in range(3):
                    for seed in (0,1):
                        torch.manual_seed(20261011+layer*97+rank*7+seed)
                        if style==0:x=torch.randint(-128,128,(1,k),device='npu',dtype=torch.int8)
                        elif style==1:x=torch.full((1,k),127 if seed else -128,device='npu',dtype=torch.int8)
                        else:x=torch.where(torch.arange(k,device='npu')%2==0,127,-128).to(torch.int8).reshape(1,k)
                        xs=torch.tensor([(.001,1.,100.)[style]],device='npu',dtype=torch.float32)
                        expected=torch_npu.npu_quant_matmul(x,nz,scale,pertoken_scale=xs,output_dtype=torch.bfloat16)
                        assert torch.isfinite(expected).all()
                        for name,parameters in variants.items():
                            state=result['candidates'][name]
                            if not state['passed']:continue
                            try:
                                actual=quant_gemv(x,nz if parameters['nz'] else nd,scale,xs,**parameters)
                                finite=bool(torch.isfinite(actual).all());ulp=(bf16_order(actual)-bf16_order(expected)).abs()
                                maximum=int(ulp.max());bad=int((ulp>1).sum());exact=bool(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
                                row={'candidate':name,'layer':layer,'projection':part,'rank_shard':rank,'style':style,'seed':seed,
                                     'max_bf16_ulp':maximum,'bad_elements':bad,'all_bitwise_equal':exact,'passed':finite and bad==0}
                            except Exception as exc:
                                row={'candidate':name,'layer':layer,'projection':part,'rank_shard':rank,'style':style,'seed':seed,
                                     'passed':False,'error':str(exc)[:1500]}
                            state['cases']+=1;result['cases'].append(row)
                            if not row['passed']:state['passed']=False;state['failures'].append(row)
                            print('QUANT_GEMV_PRECISION',json.dumps(row),flush=True)
                        save()
                active={n:v for n,v in variants.items() if result['candidates'][n]['passed']}
                if active:
                    x=torch.randint(-128,128,(1,k),device='npu',dtype=torch.int8)
                    xs=torch.tensor([.01],device='npu',dtype=torch.float32)
                    functions={'native':lambda x:torch_npu.npu_quant_matmul(x,nz,scale,pertoken_scale=xs,output_dtype=torch.bfloat16)}
                    for name,parameters in active.items():
                        weight=nz if parameters['nz'] else nd
                        functions[name]=lambda x,p=parameters,w=weight:quant_gemv(x,w,scale,xs,**p)
                    calls=128;replays=16;inputs=[x.clone() for _ in range(calls)];graphs={};outputs={}
                    for name,fn in functions.items():
                        for _ in range(3):fn(x)
                        torch.npu.synchronize();g=torch.npu.NPUGraph();refs=[]
                        with torch.npu.graph(g):
                            for inp in inputs:refs.append(fn(inp))
                        graphs[name]=g;outputs[name]=refs
                    for pair in range(4):
                        times={};submissions={}
                        for name in (list(graphs) if pair%2==0 else list(reversed(graphs))):
                            begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                            begin.record();start=time.perf_counter()
                            for _ in range(replays):graphs[name].replay()
                            submissions[name]=(time.perf_counter()-start)*1e6/replays
                            end.record();end.synchronize();times[name]=begin.elapsed_time(end)*1000/(calls*replays)
                        for name in active:
                            result['candidates'][name]['pairs'].append({'projection':part,'layer':layer,'rank_shard':rank,'pair':pair,
                                'native_us':times['native'],'candidate_us':times[name],'speedup':times['native']/times[name],
                                'native_device_to_submission_ratio':times['native']*calls/submissions['native'],
                                'candidate_device_to_submission_ratio':times[name]*calls/submissions[name]})
                    del graphs,outputs,functions,inputs
                del nd,nz,scale,cpu,scpu
    for name,state in result['candidates'].items():
        state['full_precision_passed']=state['passed'] and state['cases']==120
        state['exact_cases']=sum(r.get('all_bitwise_equal',False) for r in result['cases'] if r['candidate']==name)
        rows=state['pairs']
        if rows:
            state['summary']={'native_us':statistics.median(r['native_us'] for r in rows),
                'candidate_us':statistics.median(r['candidate_us'] for r in rows),'paired_speedup_median':statistics.median(r['speedup'] for r in rows),
                'faster_pairs':sum(r['speedup']>1 for r in rows),'pairs':len(rows),
                'minimum_device_to_submission_ratio':min(min(r['native_device_to_submission_ratio'],r['candidate_device_to_submission_ratio']) for r in rows)}
    result['completed']=True;save();print('QUANT_GEMV_COMPLETE',json.dumps({n:{k:v for k,v in s.items() if k not in ('pairs','failures')} for n,s in result['candidates'].items()}),flush=True)


if __name__=='__main__':main()
