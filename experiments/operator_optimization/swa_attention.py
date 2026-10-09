"""One-query SWA-only Cube candidate; same 128-token window and sink softmax."""
import torch
import triton
import triton.language as tl


@triton.jit
def swa_cube(q, kv, table, length, sinks, y, SCALE: tl.constexpr,
             BQ: tl.constexpr, BN: tl.constexpr, ROUND_P: tl.constexpr):
    h = tl.program_id(0)*BQ + tl.arange(0,BQ)
    d = tl.arange(0,512)
    seq = tl.load(length)
    low = tl.maximum(seq-128,0)
    pos = low+tl.arange(0,BN)
    valid = (pos<seq)&(pos<low+128)
    page = tl.load(table+pos//128,valid,0)
    offsets = (page*128+pos%128)*512
    queries = tl.load(q+h[:,None]*512+d[None,:],h[:,None]<64,0)
    keys = tl.load(kv+offsets[None,:]+d[:,None],valid[None,:],0)
    scores = tl.dot(queries,keys)*SCALE
    scores = tl.where(valid[None,:],scores,-3.0e38)
    sink = tl.load(sinks+h,h<64,0)
    maximum = tl.maximum(tl.max(scores,1),sink)
    probability = tl.exp(scores-maximum[:,None])
    denominator = tl.sum(probability,1)+tl.exp(sink-maximum)
    # Native Cube FA casts the unnormalized probability to BF16 before PV.
    # The option is explicit so alternative rounding can be rejected by audit.
    if ROUND_P:
        probability_bf16 = probability.to(tl.bfloat16)
    else:
        probability_bf16 = (probability/denominator[:,None]).to(tl.bfloat16)
    values = tl.trans(keys)
    output = tl.dot(probability_bf16,values)
    if ROUND_P:
        output = output/denominator[:,None]
    tl.store(y+h[:,None]*512+d[None,:],output.to(tl.bfloat16),h[:,None]<64)


def swa_attention(q, kv, table, length, sinks, *, scale=512**-.5, bq=16, bn=128, round_p=True):
    assert q.shape==(1,64,512) and kv.shape[-3:]==(128,1,512)
    assert q.dtype==kv.dtype==torch.bfloat16 and q.is_contiguous() and kv.is_contiguous()
    y=torch.empty_like(q)
    swa_cube[(triton.cdiv(64,bq),)](q,kv,table,length,sinks,y,float(scale),bq,bn,round_p,
                                  enable_fp_fusion=False)
    return y
