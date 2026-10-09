"""Native NPU and CPU-golden audits, including clamp mutation and edge values."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

from clamped_swiglu import clamped_swiglu


def native_routed(x, limit):
    gate, up = x.chunk(2, dim=-1)
    gate.clamp_(max=limit)
    up.clamp_(min=-limit, max=limit)
    return torch_npu.npu_swiglu(x)


def native_shared(x, limit, alpha=1.0, beta=0.0):
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def ordered_bf16(tensor):
    raw = tensor.contiguous().view(torch.int16).to(torch.int32)
    magnitude = raw & 32767
    return torch.where(raw < 0, 32768 - magnitude, 32768 + magnitude)


def compare(actual, expected):
    actual = actual.cpu()
    expected = expected.cpu()
    a, b = actual.float(), expected.float()
    special_equal = (torch.equal(a.isnan(), b.isnan())
                     and torch.equal(a.isposinf(), b.isposinf())
                     and torch.equal(a.isneginf(), b.isneginf()))
    finite = a.isfinite() & b.isfinite()
    abs_error = (a[finite] - b[finite]).abs()
    ulp = (ordered_bf16(actual)[finite] - ordered_bf16(expected)[finite]).abs()
    # Fixed before testing; all finite elements must pass, stricter than the
    # skill's 99% BF16 mixed-tolerance threshold, plus a one-ULP native bound.
    matched = abs_error <= 2**-6 + 2**-6 * b[finite].abs()
    return {
        "passed": special_equal and bool(matched.all()) and (not ulp.numel() or int(ulp.max()) <= 1),
        "special_classification_equal": special_equal,
        "max_abs": float(abs_error.max()) if abs_error.numel() else 0.0,
        "max_bf16_ulp": int(ulp.max()) if ulp.numel() else 0,
        "bit_equal_fraction": float((actual.contiguous().view(torch.int16) == expected.contiguous().view(torch.int16)).float().mean()) if actual.numel() else 1.0,
    }


def make_input(m, d, scale, special=False, strided=False):
    x = (torch.randn((m, 2 * d)) * scale).to(torch.bfloat16)
    edge = torch.tensor([-200, -100, -90, -80, -30, -7.9375, -7.90625, -7.875,
                         -7, -1, -0.0, 0.0, 1, 7, 7.875, 7.90625, 7.9375, 30, 200],
                        dtype=torch.bfloat16)
    if special:
        edge = torch.cat([edge, torch.tensor([float('inf'),float('-inf'),float('nan')],dtype=torch.bfloat16)])
    if x.numel():
        count = min(edge.numel(), x.numel())
        x.flatten()[:count].copy_(edge[:count])
    if strided:
        storage = torch.empty((m, 4 * d),dtype=torch.bfloat16,device='npu')
        view = storage[:, ::2]
        view.copy_(x.npu())
        return view
    return x.npu()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20261009)
    records=[]
    configs=[(1,256),(2,256),(3,384),(17,128),(0,256)]
    for m,d in configs:
        for limit in [1.0,7.0,7.9]:
            for scale in [0.001,1.0,100.0]:
                x=make_input(m,d,scale)
                for kind,alpha,beta in [('routed',1.0,0.0),('shared',1.0,0.0),('shared',1.702,1.0)]:
                    ref_input=x.clone();test_input=x.clone()
                    expected=native_routed(ref_input,limit) if kind=='routed' else native_shared(ref_input,limit,alpha,beta)
                    actual=clamped_swiglu(test_input,limit,alpha=alpha,beta=beta,
                                          mutate_input=kind=='routed',staged_rounding=kind=='shared')
                    metric=compare(actual,expected)
                    metric['input_effect']=compare(test_input,ref_input)
                    # Input clamp side effects must match bit for bit, except NaN payloads.
                    metric['passed'] &= metric['input_effect']['passed'] and metric['input_effect']['max_bf16_ulp']==0
                    cpu=x.cpu().double();g,u=cpu.chunk(2,-1)
                    bound=float(torch.tensor(limit,dtype=torch.bfloat16))
                    g=g.clamp(max=bound);u=u.clamp(min=-bound,max=bound)
                    golden=g*torch.sigmoid(alpha*g)*(u+beta)
                    for name,value in [('native',expected),('candidate',actual)]:
                        error=(value.cpu().double()-golden).abs()
                        metric[name+'_max_abs_vs_fp64']=float(error.max()) if error.numel() else 0.0
                    records.append(dict(kind=kind,shape=[m,2*d],limit=limit,scale=scale,alpha=alpha,beta=beta,**metric))
                    if not metric['passed']:
                        print('FAILED',json.dumps(records[-1]),flush=True)
    for strided in [False,True]:
        x=make_input(2,256,1.0,special=True,strided=strided)
        for kind in ['routed','shared']:
            # Preserve a strided source in the candidate, and compare native side effects.
            ref_input=x.clone();before=x.cpu().clone()
            expected=native_routed(ref_input,7.9) if kind=='routed' else native_shared(ref_input,7.9)
            actual=clamped_swiglu(x,7.9,mutate_input=kind=='routed',staged_rounding=kind=='shared')
            metric=compare(actual,expected)
            metric['input_effect']=compare(x,ref_input)
            metric['passed'] &= metric['input_effect']['passed'] and metric['input_effect']['max_bf16_ulp']==0
            records.append(dict(kind=kind,strided=strided,special_values=True,**metric))
    result={'thresholds':{'rtol':2**-6,'atol':2**-6,'matched_ratio_required':1.0,'native_max_bf16_ulp':1,'input_max_bf16_ulp':0},
            'cases':len(records),'passed':sum(r['passed'] for r in records),'records':records}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    assert result['passed']==result['cases'],'Candidate rejected; see saved cases'


if __name__=='__main__':main()
