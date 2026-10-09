"""Same-process native versus packed inverse-RoPE/eight-group wo_a banks."""
import os
import torch
import torch_npu
from epilogue_prepare import grouped_epilogue,inverse_rope_inplace

ARM=os.getenv('UP950_ARM','baseline')
REFS={}
GROUPS={}
INSTALLED=False


def weight3d(impl):
    weight=impl.wo_a.weight
    if weight.ndim==3:
        return weight
    cached=getattr(impl,'_dummy_wo_a_3d',None)
    if cached is None:
        cached=weight.view(8,512,4096).transpose(2,1).contiguous()
        impl._dummy_wo_a_3d=cached
    return cached


def native_epilogue(raw,cos,sin,weight):
    torch.ops._C_ascend.inplace_partial_rotary_mul(raw.unsqueeze(1),cos,-sin,
        rotary_mode='interleave',partial_slice=[448,512])
    return native_wo_a(raw,weight)


def native_wo_a(raw,weight):
    return torch_npu.npu_transpose_batchmatmul(raw.view(raw.shape[0],8,4096),weight,
        bias=None,scale=None,perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2),
        batch_split_factor=1).reshape(raw.shape[0],4096)


def inplace_epilogue(raw,cos,sin,weight):
    inverse_rope_inplace(raw,cos,sin)
    return native_wo_a(raw,weight)


def grouped_wo_a(rotated,weight):
    rows=rotated.shape[0]
    cache_key=(rows,rotated.device)
    if cache_key not in GROUPS:
        GROUPS[cache_key]=torch.full((8,),rows,dtype=torch.int64,device=rotated.device)
    grouped=rotated.view(rows,8,4096).transpose(0,1).contiguous().reshape(8*rows,4096)
    result=torch_npu.npu_grouped_matmul([grouped],[weight],group_list=GROUPS[cache_key],
        split_item=2,group_type=0,group_list_type=1)[0]
    return result.view(8,rows,512).transpose(0,1).reshape(rows,4096)


def gmm_epilogue(raw,cos,sin,weight):
    torch.ops._C_ascend.inplace_partial_rotary_mul(raw.unsqueeze(1),cos,-sin,
        rotary_mode='interleave',partial_slice=[448,512])
    return grouped_wo_a(raw,weight)


def install(model):
    global INSTALLED
    if INSTALLED:return
    from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl
    from vllm_ascend.attention.dsa_v1 import oproj_tp_enable,olora_tp_enable
    from vllm.forward_context import get_forward_context
    original=DeepseekV41EagerAttentionImpl.forward

    def forward(self,attn,positions,hidden_states,output=None):
        impl=attn.dsa_attn.dsa_attn.impl
        eligible=(hidden_states.shape[0]==1 and attn.n_local_heads==64 and attn.head_dim==512
                  and impl.n_local_groups==8 and impl.o_lora_rank==512
                  and impl.wo_a.weight.dtype==torch.bfloat16 and not oproj_tp_enable()
                  and not olora_tp_enable() and hidden_states.dtype==torch.bfloat16)
        context=get_forward_context()
        if not eligible or context.attn_metadata is None:
            return original(self,attn,positions,hidden_states,output)
        if output is None:output=torch.empty_like(hidden_states)
        metadata=self._get_layer_metadata(context.attn_metadata)
        positions=metadata.positions[:hidden_states.shape[0]]
        cos,sin=metadata.rope(attn.rotary_emb.layername,hidden_states.shape[0])
        preprocess=self.multistream_preprocess if impl.multistream_dsv4_dsa_overlap else self.preprocess
        q,qr=preprocess(attn,hidden_states,cos,sin,metadata.swa)
        if self.role.is_kv_source:
            self._write_compressed_source(attn,hidden_states,positions,cos,sin,metadata)
        indices=self._select_sparse_indices(attn,hidden_states,qr,positions,cos,sin,metadata)
        raw=self._attention(attn,q,metadata,indices)
        snapshot=context.capturing and os.getenv('TINY_PERF_RANDOM_VALIDATION')=='1'
        saved_raw=raw.clone() if snapshot else raw
        weight=weight3d(impl)
        mid=(grouped_epilogue(raw,cos,sin,weight) if ARM=='packed' else
             inplace_epilogue(raw,cos,sin,weight) if ARM=='inplace'
             else gmm_epilogue(raw,cos,sin,weight) if ARM=='gmm'
             else native_epilogue(raw,cos,sin,weight))
        output[...] = impl.wo_b(mid)
        if context.capturing:
            saved=(saved_raw,cos.clone(),sin.clone(),mid.clone(),output.clone()) if snapshot else (
                   saved_raw,cos,sin,mid,output)
            REFS.setdefault(ARM,[]).append((impl,weight,*saved,snapshot))
        return output

    DeepseekV41EagerAttentionImpl.forward=forward
    INSTALLED=True


def coverage(name):
    rows=REFS[name]
    assert len(rows)==40,(name,len(rows))
    return {'epilogue_calls':40,'packed_prepare_calls':40 if name=='packed' else 0,
            'inplace_rope_calls':40 if name=='inplace' else 0,
            'gmm_projection_calls':40 if name=='gmm' else 0,
            'wo_a_groups':8,'wo_a_rank':512}


def set_arm(name):
    global ARM
    assert name in ['baseline','packed','inplace','gmm']
    ARM=name


def reset(name):REFS[name]=[]


def prepare(name,baseline_name):
    assert name in ['packed','inplace','gmm']
    for _,weight,raw,cos,sin,_,_,_ in REFS[baseline_name]:
        if name=='packed':grouped_epilogue(raw,cos,sin,weight)
        elif name=='inplace':inplace_epilogue(raw.clone(),cos,sin,weight)
        else:gmm_epilogue(raw.clone(),cos,sin,weight)


def audit(worker,name):
    torch.npu.synchronize()
    stds=[]
    for impl,weight,raw,cos,sin,mid,output,snapshot in REFS[name]:
        assert snapshot
        reference=native_epilogue(raw.clone(),cos,sin,weight)
        actual=(inplace_epilogue(raw.clone(),cos,sin,weight) if name=='inplace' else
                gmm_epilogue(raw.clone(),cos,sin,weight) if name=='gmm'
                else grouped_epilogue(raw,cos,sin,weight))
        assert torch.equal(actual,reference),(name,'functional wo_a')
        assert torch.equal(mid,reference),(name,'captured wo_a')
        expected=impl.wo_b(reference)
        assert torch.equal(output,expected),(name,'captured wo_b')
        stds.append(float(raw.float().std().item()))
    assert min(stds)>1e-4,stds
    return {**coverage(name),'functional_bitwise_equal':True,'captured_wo_a_bitwise_equal':True,
            'captured_wo_b_bitwise_equal':True,'consumer_point_snapshots':True,
            'min_attention_std':min(stds),'max_abs':0.0}
