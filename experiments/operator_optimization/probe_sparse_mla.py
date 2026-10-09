"""Real tiny SWA-only native replay, numeric audit, and small-M Cube screening."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

from microbench import capture,paired
from swa_attention import swa_attention


def metadata(length,cu,cmp_length=None,mode='swa',residual=None):
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        64,1,512,cu_seqlens_q=cu,seqused_ori_kv=length,seqused_cmp_kv=cmp_length,cmp_residual_kv=residual,
        batch_size=1,max_seqlen_q=1,max_seqlen_ori_kv=8192,
        max_seqlen_cmp_kv=(4096 if mode=='c2' else 64) if cmp_length is not None else 0,
        # V4.1 vendor ABI: 0=SWA, 1=C128, 2=C2.
        ori_topk=0,cmp_topk=512 if cmp_length is not None else 0,
        cmp_ratio=(2 if mode=='c2' else 1) if cmp_length is not None else 0,
        ori_mask_mode=4,cmp_mask_mode=3 if cmp_length is not None else 0,
        ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_BBND',
        has_ori_kv=True,has_cmp_kv=cmp_length is not None)


def native(q,kv,table,length,sink,cu,meta,cmp_length=None,cmp_kv=None,indices=None,mode='swa',residual=None):
    return torch.ops._C_ascend.npu_sparse_flash_mla(
        q,ori_kv=kv,cmp_kv=cmp_kv,cmp_sparse_indices=indices,
        ori_block_table=table,cmp_block_table=table if cmp_length is not None else None,
        cu_seqlens_q=cu,seqused_ori_kv=length,seqused_cmp_kv=cmp_length,cmp_residual_kv=residual,sinks=sink,metadata=meta,
        softmax_scale=512**-.5,cmp_ratio=(2 if mode=='c2' else 1) if cmp_length is not None else 0,
        ori_mask_mode=4,cmp_mask_mode=3 if cmp_length is not None else 0,
        ori_win_left=127,ori_win_right=0,layout_q='TND',layout_kv='PA_BBND',
        topk_value_mode=1,return_softmax_lse=False)[0]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['swa','c128','c2'],default='swa')
    p.add_argument('--native-only',action='store_true');args=p.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    # CANN resolves its operator providers when the device first initializes.
    bootstrap_custom_op_env();enable_custom_op();torch.npu.set_device(0)
    torch.manual_seed(20261010)
    q=torch.randn((1,64,512),device='npu',dtype=torch.bfloat16)
    # Keep the real outer cache shape; initialize the pages reachable by this probe.
    kv=torch.empty((7939,128,1,512),device='npu',dtype=torch.bfloat16)
    kv[:64].copy_(torch.randn((64,128,1,512),device='npu',dtype=torch.bfloat16))
    table=torch.arange(64,device='npu',dtype=torch.int32).unsqueeze(0)
    sink=torch.randn((64,),device='npu')
    length=torch.tensor([2050],device='npu',dtype=torch.int32)
    cu=torch.tensor([0,1],device='npu',dtype=torch.int32)
    cmp_length=None;cmp_kv=None;indices=None;residual=None
    if args.mode!='swa':
        cmp_length=torch.tensor([1025 if args.mode=='c2' else 16],device='npu',dtype=torch.int32)
        indices=torch.full((1,1,512),-1,device='npu',dtype=torch.int32)
        active=512 if args.mode=='c2' else 16
        indices[0,0,:active].copy_(torch.arange(active,device='npu',dtype=torch.int32))
        cmp_kv=kv
        if args.mode=='c2':
            cmp_kv=torch.empty((7939,64,1,512),device='npu',dtype=torch.bfloat16)
            cmp_kv[:64].copy_(torch.randn((64,64,1,512),device='npu',dtype=torch.bfloat16))
            residual=torch.tensor([0],device='npu',dtype=torch.int32)
    meta=metadata(length,cu,cmp_length,args.mode,residual)
    for _ in range(30):output=native(q,kv,table,length,sink,cu,meta,cmp_length,cmp_kv,indices,args.mode,residual)
    torch.npu.synchronize()
    result={'q_shape':list(q.shape),'kv_shape':list(kv.shape),'device':4,
            'mode':args.mode,'native_cmp_ratio_attribute':(2 if args.mode=='c2' else 1) if cmp_length is not None else 0,
            'cmp_shape':list(cmp_kv.shape) if cmp_kv is not None else None,
            'indices_shape':list(indices.shape) if indices is not None else None,
            'window':128,'sinks':True,'results':[]}
    if args.native_only:
        target.write_text(json.dumps(result,indent=2)+'\n');print('NATIVE_REPLAY_COMPLETE',flush=True);return
    assert args.mode=='swa','HCA currently supports native diagnostic replay only'
    baseline=capture(lambda _:native(q,kv,table,length,sink,cu,meta))
    for bq,bn,round_p in itertools.product([16,32,64],[128,256],[True,False]):
        row={'bq':bq,'bn':bn,'round_p':round_p}
        try:
            errors=[]
            for seq in [1,127,128,129,2048,2049,2050,8192]:
                length.fill_(seq);meta=metadata(length,cu)
                reference=native(q,kv,table,length,sink,cu,meta)
                actual=swa_attention(q,kv,table,length,sink,bq=bq,bn=bn,round_p=round_p)
                torch.testing.assert_close(actual,reference,rtol=1/64,atol=1/64)
                keys=kv[:64].cpu().float().reshape(-1,512)[max(0,seq-128):seq].double()
                scores=q.cpu().double().squeeze(0)@keys.T*512**-.5
                logits=torch.cat([scores,sink.cpu().double().unsqueeze(1)],1)
                golden=(logits.softmax(1)[:,:keys.shape[0]]@keys).unsqueeze(0)
                errors.append({'seq':seq,'max_abs_vs_native':float((actual-reference).abs().max()),
                               'candidate_max_abs_fp64':float((actual.cpu().double()-golden).abs().max()),
                               'native_max_abs_fp64':float((reference.cpu().double()-golden).abs().max())})
            length.fill_(2050);meta=metadata(length,cu)
            # Recapture reference with the current metadata address.
            ref=capture(lambda _:native(q,kv,table,length,sink,cu,meta))
            candidate=capture(lambda _:swa_attention(q,kv,table,length,sink,bq=bq,bn=bn,round_p=round_p))
            row.update(paired({'native':ref,'candidate':candidate},repeats=30))
            row.update(status='passed_screen',precision=errors)
            del ref,candidate
        except Exception as exc:
            row.update(status='rejected',error=str(exc))
        result['results'].append(row);target.write_text(json.dumps(result,indent=2)+'\n')
        print('SWA_CONFIG',json.dumps({k:(v[-500:] if k=='error' else v) for k,v in row.items() if k not in ['samples_us','precision']}),flush=True)


if __name__=='__main__':main()
