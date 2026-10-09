"""Same-process native/control/address-prefetch dispatch for tiny decode SMLA."""
import os

import torch
import smla_private as ops

ARM = os.getenv('UP950_ARM','baseline')
REFS = {}
CONTEXT = None
INSTALLED = False


def install(model):
    global INSTALLED
    if INSTALLED:
        return
    from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl
    from vllm.forward_context import get_forward_context
    original_attention = DeepseekV41EagerAttentionImpl._native_attention

    def attention(self,attn,q,metadata,*,source_cache,compressed_indices):
        global CONTEXT
        previous = CONTEXT
        CONTEXT = (attn,metadata)
        try:
            return original_attention(self,attn,q,metadata,source_cache=source_cache,
                                      compressed_indices=compressed_indices)
        finally:
            CONTEXT = previous

    def dispatch(q,**kwargs):
        eligible = (q.shape == (1,64,512) and q.dtype == torch.bfloat16
                    and kwargs.get('layout_q') == 'TND'
                    and kwargs.get('layout_kv') == 'PA_BBND')
        selected = 'native'
        fn = ops.native
        if eligible and ARM in ['control','prefetch']:
            selected = 'prefetch' if ARM == 'prefetch' and kwargs['cmp_ratio'] in [1,2] else 'control'
            fn = ops.prefetch if selected == 'prefetch' else ops.control
        result = fn(q,**kwargs)
        if get_forward_context().capturing and eligible:
            assert CONTEXT is not None
            saved_kwargs=dict(kwargs)
            audit_snapshot=os.getenv('TINY_PERF_RANDOM_VALIDATION')=='1'
            saved_q=q
            saved_result=result
            if audit_snapshot:
                # Later indexer layers overwrite their shared TopK buffer.
                # Preserve the values used by this call; keep copies out of
                # formal performance graphs by enabling them only for audit.
                saved_q=q.clone()
                if saved_kwargs.get('cmp_sparse_indices') is not None:
                    saved_kwargs['cmp_sparse_indices']=saved_kwargs['cmp_sparse_indices'].clone()
                saved_result=(result[0].clone(),result[1].clone())
            REFS.setdefault(ARM,[]).append((saved_q,saved_kwargs,saved_result,CONTEXT,selected,audit_snapshot))
        return result

    DeepseekV41EagerAttentionImpl._native_attention = attention
    torch.ops._C_ascend.npu_sparse_flash_mla = dispatch
    INSTALLED = True


def coverage(name):
    rows = REFS[name]
    assert len(rows) == 40,(name,len(rows))
    counts = {kind:sum(row[4] == kind for row in rows) for kind in ['native','control','prefetch']}
    expected = ({'native':40,'control':0,'prefetch':0} if name == 'baseline'
                else {'native':0,'control':40,'prefetch':0} if name == 'control'
                else {'native':0,'control':2,'prefetch':38})
    assert counts == expected,(name,counts)
    return {'smla_calls':40,**counts}


def set_arm(name):
    global ARM
    ARM = name


def reset(name):
    REFS[name] = []


def prepare(name,baseline_name):
    fn = ops.control if name == 'control' else ops.prefetch
    for q,kwargs,_,_,_,_ in REFS[baseline_name]:
        fn(q,**kwargs)


def audit(worker,name):
    torch.npu.synchronize()
    errors = []
    stds = []
    for q,kwargs,stored,(attn,metadata),selected,audit_snapshot in REFS[name]:
        native = ops.native(q,**kwargs)
        actual = (ops.native if selected == 'native' else
                  ops.control if selected == 'control' else ops.prefetch)(q,**kwargs)
        for a,b in zip(actual,native):
            if not torch.equal(a,b):
                diagnostic={'arm':name,'ratio':kwargs['cmp_ratio'],
                            'query_finite':bool(torch.isfinite(q).all().item()),
                            'a_nan':int(torch.isnan(a).sum().item()),
                            'b_nan':int(torch.isnan(b).sum().item()),
                            'a_inf':int(torch.isinf(a).sum().item()),
                            'b_inf':int(torch.isinf(b).sum().item()),
                            'stored_nan':int(torch.isnan(stored[0]).sum().item()),
                            'ori_len':kwargs['seqused_ori_kv'].cpu().tolist(),
                            'cmp_len':kwargs['seqused_cmp_kv'].cpu().tolist() if kwargs['seqused_cmp_kv'] is not None else None,
                            'sink_finite':bool(torch.isfinite(kwargs['sinks']).all().item()),
                            'indices_range':[int(kwargs['cmp_sparse_indices'].min().item()),int(kwargs['cmp_sparse_indices'].max().item())] if kwargs['cmp_sparse_indices'] is not None else None,
                            'max_abs':float((a.float()-b.float()).abs().max().item()) if a.numel() else 0}
                print('SMLA_AUDIT_DIAGNOSTIC',diagnostic,flush=True)
                raise AssertionError((name,kwargs['cmp_ratio'],'functional output',diagnostic))
        if not audit_snapshot:
            # Without a snapshot the model has inverse-RoPE'd this buffer.
            cos,sin = metadata.rope(attn.rotary_emb.layername,q.shape[0])
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                native[0].unsqueeze(1),cos,-sin,rotary_mode='interleave',
                partial_slice=[attn.nope_head_dim,attn.head_dim])
        assert torch.equal(stored[0],native[0]),(name,kwargs['cmp_ratio'],'captured output')
        errors.append((stored[0].float()-native[0].float()).abs().max().item())
        stds.append(q.float().std().item())
    assert min(stds)>1e-4,stds
    return {**coverage(name),'functional_bitwise_equal':True,
            'captured_output_bitwise_equal':True,'consumer_point_snapshots':True,'max_abs':max(errors),
            'min_query_std':min(stds)}
