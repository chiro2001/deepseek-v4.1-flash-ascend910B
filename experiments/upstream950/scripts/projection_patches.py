"""Independent decode graph banks for joint QA/KV and q_b layout changes."""
import os
import torch
import torch_npu
import torch.nn.functional as F
import projection_panel as panel

ARM=os.getenv('UP950_ARM','baseline')
REFS={}
NZ={}
INSTALLED=False


def nz_weight(weight):
    cache_key=panel.key(weight)
    if cache_key not in NZ:NZ[cache_key]=panel.nz_pack(weight)
    return NZ[cache_key]


def eligible(attn,hidden):
    return (hidden.shape==(1,5120) and hidden.dtype==torch.bfloat16
            and attn.n_local_heads==64 and attn.head_dim==512
            and tuple(attn.wq_a.weight.shape)==tuple(attn.wkv.weight.shape)==(512,5120)
            and tuple(attn.wq_b.weight.shape)==(32768,512)
            and all(layer.weight.dtype==torch.bfloat16 and layer.bias is None
                    for layer in [attn.wq_a,attn.wkv,attn.wq_b]))


def install(model):
    global INSTALLED
    if INSTALLED:return
    from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl
    from vllm.forward_context import get_forward_context
    original=DeepseekV41EagerAttentionImpl._project_q_kv
    panel.invalidate();NZ.clear()
    # V4.1 and its V1 compatibility wrapper register the same projections.
    owners=list({(m.wq_a.weight.data_ptr(),m.wq_b.weight.data_ptr(),m.wkv.weight.data_ptr()):m
                 for m in model.modules() if hasattr(m,'wq_a') and hasattr(m,'wq_b') and hasattr(m,'wkv')}.values())
    assert len(owners)==40,len(owners)
    for attn in owners:
        panel.joint_weight(attn.wq_a.weight,attn.wkv.weight)
        panel.pack_weight(attn.wq_b.weight)
        nz_weight(attn.wq_b.weight)

    def project(attn,hidden,cos,sin):
        if not eligible(attn,hidden):return original(attn,hidden,cos,sin)
        if ARM=='joint':
            joined=F.linear(hidden,panel.joint_weight(attn.wq_a.weight,attn.wkv.weight))
            qa=joined[:,:512];kv_raw=joined[:,512:]
        else:qa=attn.wq_a(hidden)
        qr=attn.q_norm(qa)
        qb=(panel.q_b_panel(qr,panel.pack_weight(attn.wq_b.weight),prefetch=ARM=='prefetch') if ARM in ['panel','prefetch'] else
            F.linear(qr,nz_weight(attn.wq_b.weight)) if ARM=='nz' else attn.wq_b(qr))
        # The subsequent low-level RoPE consumes ND GM, independent of the
        # physical format selected by the native MatMul weight layout.
        if torch_npu.get_npu_format(qb)!=2:qb=torch_npu.npu_format_cast(qb,2)
        if ARM!='joint':kv_raw=attn.wkv(hidden)
        q=qb.unflatten(-1,(64,512))
        kv=attn.kv_norm(kv_raw).view(-1,1,512)
        snapshot=get_forward_context().capturing and os.getenv('TINY_PERF_RANDOM_VALIDATION')=='1'
        intermediates=(hidden.clone(),qa.clone(),kv_raw.clone(),qr.clone(),qb.clone()) if snapshot else (
                      hidden,qa,kv_raw,qr,qb)
        torch.ops._C_ascend.inplace_partial_rotary_mul(q.unsqueeze(1),cos,sin,
            rotary_mode='interleave',partial_slice=[448,512])
        torch.ops._C_ascend.inplace_partial_rotary_mul(kv.unsqueeze(1),cos,sin,
            rotary_mode='interleave',partial_slice=[448,512])
        if get_forward_context().capturing:
            tail=(cos.clone(),sin.clone(),q.clone(),kv.clone()) if snapshot else (cos,sin,q,kv)
            REFS.setdefault(ARM,[]).append((attn,*intermediates,*tail,snapshot))
        return q.to(hidden.dtype),qr,kv.squeeze(1)

    DeepseekV41EagerAttentionImpl._project_q_kv=staticmethod(project)
    INSTALLED=True


def coverage(name):
    rows=REFS[name]
    assert len(rows)==40,(name,len(rows))
    return {'projection_calls':40,'joint_calls':40 if name=='joint' else 0,
            'nz_calls':40 if name=='nz' else 0,'panel_calls':40 if name=='panel' else 0,
            'prefetch_calls':40 if name=='prefetch' else 0}


def set_arm(name):
    global ARM
    assert name in ['baseline','joint','nz','panel','prefetch']
    ARM=name


def reset(name):REFS[name]=[]


def prepare(name,baseline_name):
    if name in ['panel','prefetch']:
        for attn,_,_,_,qr,*_ in REFS[baseline_name]:
            panel.q_b_panel(qr,panel.pack_weight(attn.wq_b.weight),prefetch=name=='prefetch')


def audit(worker,name):
    torch.npu.synchronize()
    max_abs=0.0;different=0;stds=[];qa_different=0;kv_different=0;projection_max_abs=0.0
    for attn,hidden,qa,kv_raw,qr,qb,cos,sin,q,kv,snapshot in REFS[name]:
        assert snapshot
        expected_qa=attn.wq_a(hidden);expected_kv=attn.wkv(hidden)
        qa_check=panel.matmul_validation(qa,expected_qa)
        kv_check=panel.matmul_validation(kv_raw,expected_kv)
        assert qa_check['bf16_close'] and kv_check['bf16_close'],(name,'QA/KV BF16',qa_check,kv_check)
        qa_different+=qa_check['different'];kv_different+=kv_check['different']
        projection_max_abs=max(projection_max_abs,qa_check['max_abs'],kv_check['max_abs'])
        expected_qr=attn.q_norm(qa)
        assert torch.equal(qr,expected_qr),(name,'qr')
        expected_qb=attn.wq_b(qr)
        if name in ['panel','prefetch']:
            assert torch.allclose(qb,expected_qb,rtol=.0078125,atol=.000244140625),(name,'q_b BF16')
        else:assert torch.equal(qb,expected_qb),(name,'q_b')
        difference=(qb.float()-expected_qb.float()).abs()
        max_abs=max(max_abs,float(difference.max().item()));different+=int((qb!=expected_qb).sum().item())
        logical=qb if name in ['panel','prefetch'] else expected_qb
        if torch_npu.get_npu_format(logical)!=2:logical=torch_npu.npu_format_cast(logical,2)
        reference_q=logical.clone().view(1,64,512)
        reference_kv=attn.kv_norm(kv_raw).view(1,1,512)
        torch.ops._C_ascend.inplace_partial_rotary_mul(reference_q.unsqueeze(1),cos,sin,
            rotary_mode='interleave',partial_slice=[448,512])
        torch.ops._C_ascend.inplace_partial_rotary_mul(reference_kv.unsqueeze(1),cos,sin,
            rotary_mode='interleave',partial_slice=[448,512])
        assert torch.equal(q,reference_q),(name,'captured Q RoPE')
        assert torch.equal(kv,reference_kv),(name,'captured KV RoPE')
        stds.append(float(hidden.float().std().item()))
    assert min(stds)>1e-4,stds
    return {**coverage(name),'qa_kv_bitwise_equal':qa_different==kv_different==0,
            'qa_kv_bf16_close':True,'qa_different':qa_different,'kv_different':kv_different,
            'qa_kv_max_abs':projection_max_abs,'captured_rope_bitwise_equal':True,
            'q_b_bitwise_equal':different==0,'q_b_bf16_close':True,'q_b_different':different,
            'q_b_max_abs':max_abs,'consumer_point_snapshots':True,'min_hidden_std':min(stds)}
