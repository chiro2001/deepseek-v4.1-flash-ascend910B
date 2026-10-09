"""Screen exact head-to-batch remapping when static sinks repeat across groups.

Every group keeps the same complete KV/window/index set. This is only eligible
when its sink vector matches the others; arbitrary sink parameters fall back.
Metadata/replication is prepared outside timing to screen intrinsic feasibility.
"""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env,enable_custom_op

from microbench import capture,paired


def prepare(mode,groups,ori,cmp,seq=2050):
    heads=64//groups
    cu=torch.arange(groups+1,device='npu',dtype=torch.int32)
    length=torch.full((groups,),seq,device='npu',dtype=torch.int32)
    cmp_len=torch.full((groups,),seq//(2 if mode=='c2' else 128),device='npu',dtype=torch.int32)
    residual=torch.full((groups,),seq%2,device='npu',dtype=torch.int32) if mode=='c2' else None
    table=torch.arange(64,device='npu',dtype=torch.int32).repeat(groups,1)
    indices=torch.full((groups,1,512),-1,device='npu',dtype=torch.int32)
    active=min(seq//(2 if mode=='c2' else 128),512)
    indices[:,:,:active].copy_(torch.arange(active,device='npu',dtype=torch.int32).view(1,1,-1).expand(groups,1,-1))
    meta=torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        heads,1,512,cu_seqlens_q=cu,seqused_ori_kv=length,seqused_cmp_kv=cmp_len,
        cmp_residual_kv=residual,batch_size=groups,max_seqlen_q=1,max_seqlen_ori_kv=8192,
        max_seqlen_cmp_kv=4096 if mode=='c2' else 64,ori_topk=0,cmp_topk=512,
        cmp_ratio=2 if mode=='c2' else 1,ori_mask_mode=4,cmp_mask_mode=3,
        ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_BBND',
        has_ori_kv=True,has_cmp_kv=True)
    return dict(ori_kv=ori,cmp_kv=cmp,ori_block_table=table,cmp_block_table=table,
                cmp_sparse_indices=indices,cu_seqlens_q=cu,seqused_ori_kv=length,
                seqused_cmp_kv=cmp_len,cmp_residual_kv=residual,metadata=meta,
                softmax_scale=512**-.5,cmp_ratio=2 if mode=='c2' else 1,
                ori_mask_mode=4,cmp_mask_mode=3,ori_win_left=127,ori_win_right=0,
                layout_q='TND',layout_kv='PA_BBND',topk_value_mode=1,return_softmax_lse=False)


def call(q,sink,params,groups):
    return torch.ops._C_ascend.npu_sparse_flash_mla(q.reshape(groups,64//groups,512),
                    sinks=sink[:64//groups],**params)[0].reshape(1,64,512)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['c128','c2'],required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    bootstrap_custom_op_env();enable_custom_op();torch.npu.set_device(0);torch.manual_seed(20261010)
    q=torch.randn((1,64,512),device='npu',dtype=torch.bfloat16)
    ori=torch.empty((7939,128,1,512),device='npu',dtype=torch.bfloat16)
    ori[:64].copy_(torch.randn((64,128,1,512),device='npu',dtype=torch.bfloat16))
    cmp=ori
    if args.mode=='c2':
        cmp=torch.empty((7939,64,1,512),device='npu',dtype=torch.bfloat16)
        cmp[:64].copy_(torch.randn((64,64,1,512),device='npu',dtype=torch.bfloat16))
    # Nonconstant four-value pattern, repeated to satisfy every screened remap.
    sink=torch.randn((4,),device='npu').repeat(16)
    native_params=prepare(args.mode,1,ori,cmp)
    baseline=capture(lambda _:call(q,sink,native_params,1))
    results={'mode':args.mode,'sink_pattern':'four nonconstant values repeated',
             'timing_excludes':'metadata/replication; model validation required','results':[]}
    for groups in [2,4,8,16]:
        row={'groups':groups,'heads_per_group':64//groups}
        try:
            # Static-parameter guard is outside both capture and timed replay.
            assert torch.equal(sink.reshape(groups,-1),sink[:64//groups].expand(groups,-1))
            params=prepare(args.mode,groups,ori,cmp);errors=[]
            for scale in [.001,1.,100.]:
                actual=call(q*scale,sink,params,groups);reference=call(q*scale,sink,native_params,1)
                torch.testing.assert_close(actual,reference,rtol=1/64,atol=1/64)
                errors.append({'scale':scale,'max_abs':float((actual-reference).abs().max()),
                               'bit_equal':bool(torch.equal(actual,reference))})
            bank=capture(lambda _:call(q,sink,params,groups))
            row.update(paired({'native':baseline,'candidate':bank},repeats=30))
            row.update(status='passed_screen',precision=errors);del bank
        except Exception as exc:row.update(status='rejected',error=str(exc))
        results['results'].append(row);target.write_text(json.dumps(results,indent=2)+'\n')
        print('HEAD_SPLIT',json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)


if __name__=='__main__':main()
