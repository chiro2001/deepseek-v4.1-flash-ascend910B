"""Validate and time batch1 slot conversions independently of model timing."""
import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

import goal20_patches as goal
from metadata_kernels import prepare_slots as reference
from tp8_slot_batches import SlotBatches


@torch.inference_mode()
def run_dtype(dtype):
    registry=SlotBatches();pos=torch.zeros(1,device='npu',dtype=torch.int64)
    query=torch.tensor([0,1],device='npu',dtype=torch.int32)
    groups=[]
    for i in range(12):
        builder=SimpleNamespace(_slot_mapping_2d=torch.full((8,2),-777,device='npu',dtype=dtype))
        common=SimpleNamespace(slot_mapping=torch.zeros(1,device='npu',dtype=dtype),
                               query_start_loc=query,num_reqs=1)
        groups.append((builder,common,bool(i%3),1 if i%3==0 else 2,(1,64,128,256)[i%4]))
    cases=[];reference_discrepancies=[]
    large=2**31-128 if dtype==torch.int32 else 2**31+127
    for raw,pos_value in [(-1,0),(0,0),(1,1),(127,127),(128,128),(129,129),(255,255),(256,256),(large,511)]:
        for actual_reqs in (0,1):
            for actual_tokens in (0,1):
                for skip in (False,True):
                    pos.fill_(pos_value)
                    for i,(_,common,*_) in enumerate(groups):common.slot_mapping.fill_(raw if raw<0 else raw+i)
                    goal.ARM='slot-reference';batch={}
                    for b,c,compressed,ratio,bs in groups:
                        registry.prepare(b,c,pos,1,actual_reqs,actual_tokens,compressed,ratio,bs,skip,batch)
                    native_outputs=[b._slot_mapping_2d[:1].cpu().tolist() for b,*_ in groups]
                    expected=[]
                    for i,(_,_,compressed,ratio,bs) in enumerate(groups):
                        value=raw if raw<0 else raw+i
                        valid=value>=0
                        physical=max(value,0)
                        if compressed and ratio!=1:
                            valid=valid and (physical+1)%ratio==0
                            physical//=ratio
                        if compressed and ratio==2:
                            valid=valid and not skip and actual_reqs==1 and actual_tokens==1 and pos_value%2==1
                        coordinates=[physical//bs,physical%bs] if valid else [-1,-1]
                        gold=torch.tensor([coordinates],dtype=dtype)
                        if native_outputs[i]!=gold.tolist():
                            discrepancy={'dtype':str(dtype),'raw':value,'position':pos_value,'group':i,
                                         'native':native_outputs[i],'cpu_integer_reference':gold.tolist()}
                            reference_discrepancies.append(discrepancy)
                            print('EXISTING_SLOT_REFERENCE_DISCREPANCY',json.dumps(discrepancy),flush=True)
                        if value<2**24:
                            assert native_outputs[i]==gold.tolist(), 'Reference mismatch within formal pool range'
                        expected.append(gold.to('npu'))
                    for b,*_ in groups:b._slot_mapping_2d.fill_(-777)
                    goal.ARM='slot-probe';batch={}
                    for b,c,compressed,ratio,bs in groups:
                        registry.prepare(b,c,pos,1,actual_reqs,actual_tokens,compressed,ratio,bs,skip,batch)
                    for group_index,((b,*_),gold) in enumerate(zip(groups,expected)):
                        if not torch.equal(b._slot_mapping_2d[:1],gold):
                            diagnostic={'slot_dtype':str(dtype),'raw':raw,'position':pos_value,
                                'actual_reqs':actual_reqs,'actual_tokens':actual_tokens,'skip':skip,
                                'group':group_index,'actual':b._slot_mapping_2d[:1].cpu().tolist(),
                                'expected':gold.cpu().tolist(),'descriptors':registry.descriptors.cpu().tolist(),
                                'group_inputs':[int(c.slot_mapping.cpu()[0]) for _,c,*_ in groups]}
                            print('SLOT_COORDINATE_MISMATCH',json.dumps(diagnostic),flush=True)
                        torch.testing.assert_close(b._slot_mapping_2d[:1],gold,rtol=0,atol=0)
                        assert torch.all(b._slot_mapping_2d[1:]==-777).item(),'Output tail overwritten'
                    registry.audit()
                    cases.append({'raw':raw,'position':pos_value,'actual_reqs':actual_reqs,
                                  'actual_tokens':actual_tokens,'skip':skip,'groups':12,'exact':True})
    # Warm fixed scalar capture arguments; vary tensor contents after capture.
    pos.fill_(129);query.copy_(torch.tensor([0,1],device='npu',dtype=torch.int32))
    def fused():
        batch={}
        for b,c,compressed,ratio,bs in groups:
            registry.prepare(b,c,pos,1,1,1,compressed,ratio,bs,False,batch)
    fused();torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):fused()
    for rep in range(3):
        pos.fill_(127+rep)
        for i,(_,c,*_) in enumerate(groups):c.slot_mapping.fill_(127+rep+i)
        graph.replay();torch.npu.synchronize();registry.audit()
    def native():
        for b,c,compressed,ratio,bs in groups:
            reference(b,c,pos,1,1,1,compressed,ratio,bs,False)
    for _ in range(3):native();fused()
    torch.npu.synchronize()
    samples=[]
    for trial in range(8):
        order=(('native_12_launches',native),('fused_1_launch',fused))
        if trial%2:order=tuple(reversed(order))
        for name,fn in order:
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record();begin=time.perf_counter()
            for _ in range(100):fn()
            end.record();end.synchronize()
            samples.append({'trial':trial,'arm':name,'device_event_us':start.elapsed_time(end)*10,
                            'host_fenced_us':(time.perf_counter()-begin)*10000})
    result={'scope':'Independent 12-group batch1 coordinate conversion; not model E2E',
            'physical_chip':8,'slot_dtype':str(dtype),'positions_dtype':'torch.int64',
            'mathematical_cases':len(cases),'all_exact':True,
            'all_output_tails_unchanged':True,'changed_input_graph_replays':3,
            'registry':registry.status(),'samples':samples,
            'medians':{name:{key:statistics.median(r[key] for r in samples if r['arm']==name)
                              for key in ('device_event_us','host_fenced_us')}
                       for name in ('native_12_launches','fused_1_launch')},
            'performance_claim':None,'cases':cases}
    result['integer_reference']='CPU exact floor/remainder; no floating-point conversion'
    result['existing_reference_discrepancies_outside_formal_pool_range']=reference_discrepancies
    result['existing_reference_matches_below_2pow24']=True
    print('FORMAL_SLOT_PROBE',json.dumps({k:v for k,v in result.items() if k not in ('cases','samples')}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--physical-chips',required=True)
    args=p.parse_args();assert args.physical_chips=='8,9,10,11,12,13,14,15'
    torch.npu.set_device(0)
    goal.CONFIGS['slot-probe']={'metadata_manyslots'};goal.CONFIGS['slot-reference']=set()
    variants=[run_dtype(dtype) for dtype in (torch.int32,torch.int64)]
    result={'scope':'Independent coordinate conversion; not model E2E',
            'formal_int32_contract_tested':True,'all_exact':all(v['all_exact'] for v in variants),
            'mathematical_cases':sum(v['mathematical_cases'] for v in variants),
            'changed_input_graph_replays':sum(v['changed_input_graph_replays'] for v in variants),
            'performance_claim':None,'variants':variants}
    args.output.mkdir(exist_ok=True,parents=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
