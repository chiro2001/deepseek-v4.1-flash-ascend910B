"""Decode K-postprocessing banks; prefill and unsupported layouts use native."""
import os

import torch
import torch_npu
from indexer_post import indexer_post,initialize_coefficients

ARM = os.getenv('UP950_ARM', 'baseline')
REFS = {}
INSTALLED = False


def native_post(module, projected, slots, cos, sin, key_cache, scale_cache):
    def scatter_cache_sk(cache,coords,values):
        torch.ops._C_ascend.npu_scatter_nd_update_sk(
            cache.squeeze(-2),coords,values.to(cache.dtype).contiguous())
    key = module.k_norm(projected).view(-1, 1, module.width)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        key.unsqueeze(1), cos, sin, rotary_mode='interleave',
        partial_slice=[module.width-module.rope_width, module.width])
    quant, scale = torch_npu.npu_dynamic_quant(key.squeeze(1), dst_type=torch.int8)
    scatter_cache_sk(key_cache, slots, quant)
    scatter_cache_sk(scale_cache, slots, scale.unsqueeze(-1).to(torch.float16))
    return quant.reshape(-1,128), scale.reshape(-1)


def install(model):
    global INSTALLED
    if INSTALLED:
        return
    from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer
    from vllm.forward_context import get_forward_context
    original = DeepseekV41Indexer.update_keys
    owners=[m for m in model.modules() if isinstance(m,DeepseekV41Indexer) and m.owns_k]
    assert len(owners)==4
    for module in owners:initialize_coefficients(module.wk.weight.device)

    def update(self, latent, slots, cos, sin):
        if not self.owns_k or latent.shape[0] == 0:
            return original(self,latent,slots,cos,sin)
        key_cache, scale_cache = self.k_cache.kv_cache[0]
        eligible = (latent.shape[0] == 1 and self.width == 128 and self.rope_width == 64
                    and latent.dtype == self.wk.weight.dtype == self.k_norm.weight.dtype == torch.bfloat16
                    and slots.ndim == 2 and slots.shape[1] == 2
                    and key_cache.dtype == torch.int8 and scale_cache.dtype == torch.float16
                    and key_cache.ndim == scale_cache.ndim == 4
                    and tuple(key_cache.shape[2:]) == (1,128)
                    and tuple(scale_cache.shape[2:]) == (1,1))
        if not eligible:
            return original(self,latent,slots,cos,sin)
        projected = self.wk(latent)
        if ARM == 'fused':
            indexer_post(projected,self.k_norm.weight,cos.reshape(-1,64),sin.reshape(-1,64),
                         slots,key_cache,scale_cache,self.k_norm.eps)
        else:
            native_post(self,projected,slots,cos,sin,key_cache,scale_cache)
        if get_forward_context().capturing:
            snapshot = os.getenv('TINY_PERF_RANDOM_VALIDATION') == '1'
            saved = (projected,slots,cos,sin)
            written = None
            if snapshot:
                saved = tuple(t.clone() for t in saved)
                # Save only the rows this invocation writes, at the consumer
                # point, because caches/coordinates are overwritten on replay.
                blocks=slots[:,0].long().clamp_min(0)
                positions=slots[:,1].long().clamp_min(0)
                written=(key_cache[blocks,positions,0,:].clone(),
                         scale_cache[blocks,positions,0,0].clone())
            REFS.setdefault(ARM,[]).append((self,*saved,key_cache,scale_cache,written,snapshot))

    DeepseekV41Indexer.update_keys = update
    INSTALLED = True


def coverage(name):
    rows=REFS[name]
    assert len(rows)==4,(name,len(rows))
    return {'k_source_calls':4,'fused_post_calls':4 if name=='fused' else 0,
            'cache_layouts':[{'k_shape':list(r[5].shape),'k_stride':list(r[5].stride()),
                             'scale_stride':list(r[6].stride()),'cos_dtype':str(r[3].dtype)}
                            for r in rows]}


def set_arm(name):
    global ARM
    assert name in ['baseline','fused']
    ARM=name


def reset(name):
    REFS[name]=[]


def prepare(name,baseline_name):
    assert name=='fused'
    for module,projected,slots,cos,sin,key_cache,scale_cache,_,_ in REFS[baseline_name]:
        indexer_post(projected,module.k_norm.weight,cos.reshape(-1,64),sin.reshape(-1,64),
                     slots,key_cache,scale_cache,module.k_norm.eps)


def audit(worker,name):
    torch.npu.synchronize()
    deviations=[]
    stds=[]
    valid_rows=0
    for module,projected,slots,cos,sin,key_cache,scale_cache,written,snapshot in REFS[name]:
        assert snapshot and written is not None
        # Recompute without touching the live model cache. Buffers need only
        # contain the four captured rows; actual physical strides were used
        # by the captured candidate and were separately validated end to end.
        coordinates=torch.stack((torch.zeros_like(slots[:,0]),torch.arange(
            slots.shape[0],device=slots.device,dtype=slots.dtype)),dim=1)
        reference_key=torch.empty((1,slots.shape[0],1,128),dtype=torch.int8,device=projected.device)
        reference_scale=torch.empty((1,slots.shape[0],1,1),dtype=torch.float16,device=projected.device)
        quant,scale=native_post(module,projected,coordinates,cos,sin,reference_key,reference_scale)
        debug=indexer_post(projected,module.k_norm.weight,cos.reshape(-1,64),sin.reshape(-1,64),
                           coordinates,reference_key,reference_scale,module.k_norm.eps,debug=True)
        assert torch.equal(debug[2],quant),(name,'functional quant')
        assert torch.equal(debug[3],scale),(name,'functional scale')
        valid=(slots[:,0]>=0)&(slots[:,1]>=0)
        valid_rows+=int(valid.sum().item())
        assert torch.equal(written[0][valid],quant[valid]),(name,'captured cache key')
        assert torch.equal(written[1][valid],scale.to(torch.float16)[valid]),(name,'captured cache scale')
        deviations.append(float((written[0][valid].float()-quant[valid].float()).abs().max().item())
                          if valid.any().item() else 0.0)
        if valid.any().item():
            stds.append(float(projected[valid].float().std().item()))
    assert stds and min(stds)>1e-4,stds
    return {**coverage(name),'functional_bitwise_equal':True,'captured_cache_bitwise_equal':True,
            'consumer_point_snapshots':True,'valid_rows':valid_rows,'max_abs':max(deviations),
            'min_projected_std':min(stds)}
