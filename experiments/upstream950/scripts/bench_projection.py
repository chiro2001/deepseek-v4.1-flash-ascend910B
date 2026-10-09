"""Paired device-graph timing including QA/KV, q_norm, and q_b consumption."""
import argparse
import json
import statistics
from pathlib import Path
import torch
import torch_npu
import torch.nn.functional as F
from projection_panel import joint_weight,pack_weight,q_b_panel,nz_pack

parser=argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--output',required=True)
parser.add_argument('--arms',default='baseline,joint,nz,panel')
args=parser.parse_args()
arms=args.arms.split(',');assert arms[0]=='baseline'
torch.npu.set_device(0);torch.manual_seed(20261010)
hidden=torch.randn(1,5120,dtype=torch.bfloat16,device='npu')
qa=torch.randn(512,5120,dtype=torch.bfloat16,device='npu')/(5120**.5)
kv=torch.randn_like(qa)/(5120**.5)
qb=torch.randn(32768,512,dtype=torch.bfloat16,device='npu')/(512**.5)
gamma=torch.randn(512,dtype=torch.bfloat16,device='npu')*.1+1
joined=joint_weight(qa,kv);packed=pack_weight(qb);nz=nz_pack(qb)
def call(name):
    if name=='joint':
        combined=F.linear(hidden,joined);q_a=combined[:,:512];k=combined[:,512:]
    else:q_a=F.linear(hidden,qa)
    qr=torch_npu.npu_rms_norm(q_a,gamma,epsilon=1e-20)[0]
    q=(q_b_panel(qr,packed,prefetch=name=='prefetch') if name in ['panel','prefetch'] else F.linear(qr,nz if name=='nz' else qb))
    if torch_npu.get_npu_format(q)!=2:q=torch_npu.npu_format_cast(q,2)
    if name!='joint':k=F.linear(hidden,kv)
    return q,k
banks={}
for name in arms:
    for _ in range(8):call(name)
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(10):call(name)
    banks[name]=graph
pairs=[]
for pair in range(12):
    timings={}
    for name in (arms if pair%2==0 else list(reversed(arms))):
        for _ in range(3):banks[name].replay()
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(50):banks[name].replay()
        end.record();end.synchronize()
        timings[name]=start.elapsed_time(end)*1000/(50*10)
    pairs.append({'pair':pair,**{n+'_us':v for n,v in timings.items()},
                  'speedups':{n:timings['baseline']/timings[n] for n in arms[1:]}})
report={'rows':1,'same_process':True,'profiler':'OFF','nz_format':int(torch_npu.get_npu_format(nz)),
        'scope':'QA/KV + q_norm + q_b including ND handoff',
        'pairs':pairs,'baseline_median_us':statistics.median(p['baseline_us'] for p in pairs),
        'arms':{n:{'median_us':statistics.median(p[n+'_us'] for p in pairs),
                   'paired_speedup_median':statistics.median(p['speedups'][n] for p in pairs)}
                for n in arms[1:]}}
Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
