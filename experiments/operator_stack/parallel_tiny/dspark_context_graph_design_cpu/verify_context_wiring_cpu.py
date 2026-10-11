"""Execute frozen native helper AST with CPU stubs; test wiring and failures.

The stubs check order/shape/identity only. They do not compute native floating
point outputs, prove ScatterSk sentinels or implement any device graph.
"""
from collections import OrderedDict
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import types
from context_wiring_cpu import TensorBinding,CacheGroupBinding,ContextBinding,admission,dispatch_original_site,ReplayLease,LeaseError,ComponentFailure
from context_adapter_factory_cpu import ContextAdapters,PreparedEntry

ROOT=Path(__file__).resolve().parent
GUARD_PATH=ROOT/'evidence/context_source_guards.json'
if not GUARD_PATH.exists():GUARD_PATH=ROOT.parent/'context_source_guards.json'
GUARD=json.loads(GUARD_PATH.read_text())
OPS=[]

class Tensor:
    def __init__(self,shape,tag='',dtype='BF16'):self.shape=tuple(shape);self.tag=tag;self.dtype=dtype
    @property
    def ndim(self):return len(self.shape)
    def numel(self):return math.prod(self.shape)
    def split(self,widths,dim):
        OPS.append(('split',self.tag,tuple(widths)));assert sum(widths)==self.shape[-1]
        return tuple(Tensor(self.shape[:-1]+(w,),self.tag,self.dtype) for w in widths)
    def unsqueeze(self,dim):
        dim=dim if dim>=0 else len(self.shape)+dim+1;s=list(self.shape);s.insert(dim,1);return Tensor(s,self.tag,self.dtype)
    def squeeze(self,dim):
        s=list(self.shape);assert s[dim]==1;s.pop(dim);return Tensor(s,self.tag,self.dtype)
    def view(self,*shape):
        s=list(shape)
        if -1 in s:s[s.index(-1)]=self.numel()//math.prod(n for n in s if n!=-1)
        assert math.prod(s)==self.numel();return Tensor(s,self.tag,self.dtype)
    def contiguous(self):return self
    def repeat(self,*repeats):return Tensor(tuple(n*r for n,r in zip(self.shape,repeats)),self.tag,self.dtype)
    def new_zeros(self,*shape,dtype):OPS.append(('premix_zero',));return Tensor(shape,'premix0',dtype)
    def __setitem__(self,key,value):OPS.append(('premix_seed',value))

class Slots:
    def __init__(self,values,tag):self.values=values;self.tag=tag
    @property
    def ndim(self):return 2 if self.values and isinstance(self.values[0],tuple) else 1
    @property
    def shape(self):return (len(self.values),2) if self.ndim==2 else (len(self.values),)
    def numel(self):return math.prod(self.shape)
    def __ge__(self,value):return [v>=value for v in self.values]
    def __mod__(self,value):return Slots([v%value for v in self.values],self.tag)
    def to(self,dtype):return self

class Cache:
    def __init__(self,name):self.name=name;self.rows={}
    def scatter(self,slots,x):
        assert x.shape==(6,1,512) and slots.shape==(6,2)
        OPS.append(('store',self.name,slots.tag))
        for index,(block,offset) in enumerate(slots.values):
            if block>=0:self.rows[block,offset]=(x.tag,index)

def cat(tensors,dim):
    assert dim==-1;shape=tensors[0].shape[:-1]+(sum(t.shape[-1] for t in tensors),)
    OPS.append(('cat',tensors[0].tag));return Tensor(shape,tensors[0].tag)

torch=types.SimpleNamespace(Tensor=Tensor,float32='FP32',int32='INT32',cat=cat,
    full_like=lambda x,v:Slots([v]*len(x.values),x.tag),
    div=lambda x,v,rounding_mode:Slots([a//v for a in x.values],x.tag),
    where=lambda valid,x,y:Slots([a if ok else b for ok,a,b in zip(valid,x.values,y.values)],x.tag),
    stack=lambda xs,dim:Slots(list(zip(xs[0].values,xs[1].values)),xs[0].tag))
torch.ops=types.SimpleNamespace(_C_ascend=types.SimpleNamespace(npu_scatter_nd_update_sk=lambda cache,slots,x:cache.scatter(slots,x)))

namespace={'torch':torch,'nn':types.SimpleNamespace(Module=object)}
def compile_native(relative,qualified):
    text=GUARD['sources'][relative]['functions'][qualified]['source']
    exec(text,namespace);return namespace[qualified.rsplit('.',1)[-1]]

native_combine=compile_native('models/deepseek_v4/dspark.py','DeepseekV4DSparkModel.combine_hidden_states')
native_project=compile_native('models/deepseek_v4/dspark.py','DeepseekV4DSparkModel._project_shared_kv')
native_store=compile_native('models/deepseek_v4/dspark.py','DeepseekV4DSparkModel._store_standard_swa_kv')
native_kv=compile_native('models/deepseek_v4/dspark.py','DeepseekV4DSparkModel.precompute_and_store_context_kv')
native_forward=compile_native('models/deepseek_v41/dspark.py','DeepseekV41DSparkModel.forward')
native_rope=compile_native('models/deepseek_v4/dspark.py','_apply_dsv4_rope')
native_format=compile_native('attention/dsa_attn_kv_plan.py','DsaAttnKvPlan.format_dsa_slot_mapping')
native_scatter=compile_native('attention/dsa_attn_kv_plan.py','DsaAttnKvPlan.dsa_kv_compress_scatter')
namespace['get_cos_and_sin_dsa']=lambda positions:({'L'+str(i):None for i in [40,41,42]},{'L'+str(i):None for i in [40,41,42]})

class Plan:
    requires_block_offset_slots=True;uses_sparse_flash_mla=False;uses_kv_compress_epilog=False
    format_dsa_slot_mapping=native_format;dsa_kv_compress_scatter=native_scatter

class Layer:
    def __init__(self,index):
        self.index=index;self.cache=Cache('cache'+str(index))
        def wkv(x):OPS.append(('wkv',index));assert x.shape==(6,5120);return Tensor((6,512),str(index))
        def norm(x):OPS.append(('kv_norm',index));return x
        class Rotary:
            layername='L'+str(index)
            def __call__(self,x,cos,sin):OPS.append(('rope',index));assert x.shape==(6,1,64);return x
        self.self_attn=types.SimpleNamespace(wkv=wkv,kv_norm=norm,nope_head_dim=448,rope_head_dim=64,head_dim=512,
            rotary_emb=Rotary(),dsa_attn=types.SimpleNamespace(swa_cache_layer=types.SimpleNamespace(kv_cache=[self.cache],block_size=128)))
    def __call__(self,positions,hidden,premix,**kwargs):
        expected='embed' if self.index==40 else 'hidden'+str(self.index-1)
        assert hidden.tag==expected;OPS.append(('MTP',self.index))
        return Tensor(hidden.shape,'hidden'+str(self.index)),Tensor(premix.shape,'mix'+str(self.index),'FP32')
    def hc_collapse(self,hidden,premix):OPS.append(('hc_collapse',self.index));return Tensor((5,5120),'final')

class Model:
    _project_shared_kv=native_project;_store_standard_swa_kv=native_store
    combine_hidden_states=native_combine;precompute_and_store_context_kv=native_kv;forward=native_forward
    def __init__(self):
        self.layers=OrderedDict((str(i),Layer(i)) for i in [40,41,42]);self.vllm_config=object()
        self.hc_mult=4;self.needs_moe_input_ids=False
        self.main_proj=lambda x:self.proj(x)
        self.main_norm=lambda x:self.norm(x)
        self.embed_tokens=lambda x:Tensor((5,5120),'embed')
    def proj(self,x):OPS.append(('main_proj_and_native_TP_gather',));assert x.shape==(6,15360);return Tensor((6,5120),'combined')
    def norm(self,x):OPS.append(('main_norm',));return x

def binding():
    def tensor(address,shape,dtype,owner):
        stride=[];running=1
        for n in reversed(shape):stride.insert(0,running);running*=n
        return TensorBinding(address,running*(2 if dtype=='BF16' else 4),tuple(shape),tuple(stride),dtype,owner,1)
    groups=[]
    for i in range(3):
        groups.append(CacheGroupBinding(40+i,f'mtp.{i}.self_attn.swa_cache',10+i,128,128,
            tensor(0x10000000000+i*0x100000000,(7939,128,1,512),'BF16','cache'+str(i)),
            tensor(0x300000+i*0x1000,(6,),'INT32','slots'+str(i)),True))
    return ContextBinding(tensor(0x100000,(6,15360),'BF16','aux'),tensor(0x200000,(6,5120),'BF16','context'),
        tensor(0x280000,(6,),'INT32','positions'),tuple(groups),tuple((k,v['sha256']) for k,v in sorted(GUARD['sources'].items())),
        'model',1,'rope',1,producer_dependencies_joined=True,collective_internal_streams_verified=True,
        stream='CPU_FAKE_STREAM',context_identity='CPU_FAKE_CONTEXT',device_identity='CPU_FAKE_DEVICE',
        alias_allocation_domains_verified=True,SDK_capture_gate_passed=True,actual_consumer_gate_passed=True,fresh_native_sources_verified=True)

def main():
    shim=types.ModuleType('vllm_ascend.attention.dsa_attn_kv_plan');shim.get_dsa_attn_kv_plan=lambda config:Plan()
    name=shim.__name__;previous=sys.modules.get(name);sys.modules[name]=shim
    try:
        bundle=binding();cases=[]
        for mode in ['native','synthetic_prepared']:
            OPS.clear();model=Model();slots=[Slots([128*i+j if j!=5 else -1 for j in range(6)],'slots'+str(i)) for i in range(3)]
            current=bundle if mode=='synthetic_prepared' else replace(bundle,SDK_capture_gate_passed=False)
            native=lambda:model.combine_hidden_states(Tensor((6,15360),'aux'))
            combined,receipt=dispatch_original_site('combine',current,current.key('combine'),native,lambda lease:native())
            OPS.append(('native_set_inputs_copy_and_per_gid_slots',))
            kv=lambda:model.precompute_and_store_context_kv(combined,Tensor((6,),'positions','INT32'),slots)
            _,receipt=dispatch_original_site('context_kv',current,current.key('context_kv'),kv,lambda lease:kv())
            model.forward(Tensor((5,),'ids','INT32'),Tensor((5,),'query_positions','INT32'))
            cases.append({'mode':mode,'operator_order':list(OPS),'CPU_cache_rows':[layer.cache.rows for layer in model.layers.values()]})
        assert cases[0]['operator_order']==cases[1]['operator_order']
        assert cases[0]['CPU_cache_rows']==cases[1]['CPU_cache_rows']
        filtered=[op for op in cases[0]['operator_order'] if op[0] in ['wkv','store','MTP']]
        assert filtered==[('wkv',40),('store','cache40','slots0'),('wkv',41),('store','cache41','slots1'),('wkv',42),('store','cache42','slots2'),('MTP',40),('MTP',41),('MTP',42)]
        negatives=[replace(bundle,prefill=True)]
        negatives += [replace(bundle,context_rows=n) for n in [0,1,5,7,2048]]
        negatives += [replace(bundle,outer_capturing=True),replace(bundle,outer_merged_graph_configured=True),
            replace(bundle,asynchronous_refresh=True),replace(bundle,producer_dependencies_joined=False),
            replace(bundle,alias_allocation_domains_verified=False),replace(bundle,actual_consumer_gate_passed=False),
            replace(bundle,fresh_native_sources_verified=False),replace(bundle,model_generation=2),
            replace(bundle,rope_generation=2),replace(bundle,positions=replace(bundle.positions,extent=1)),
            replace(bundle,context=replace(bundle.context,live=False)),replace(bundle,context=replace(bundle.context,dtype='INT32')),
            replace(bundle,groups=bundle.groups[::-1]),replace(bundle,groups=(replace(bundle.groups[0],gid=None),)+bundle.groups[1:])]
        negatives += [replace(bundle,stream=None),replace(bundle,context_identity=None),replace(bundle,device_identity=None),
                      replace(bundle,kv_plan='A5_sparseFlashMla')]
        reasons=[]
        for current in negatives:
            mode,reason=admission('context_kv',current,bundle.key('context_kv'));assert mode=='native';reasons.append(reason)
        assert admission('combine',replace(bundle,collective_internal_streams_verified=False),bundle.key('combine'))[0]=='native'
        assert admission('context_kv',bundle,None)[0]=='native'
        # Same gid is legal when native name-to-gid resolution supplies the same
        # per-group slot buffer; do not require three distinct cache groups.
        same_gid=replace(bundle,groups=tuple(replace(g,gid=10,context_slots=bundle.groups[0].context_slots) for g in bundle.groups))
        assert admission('context_kv',same_gid,same_gid.key('context_kv'))[0]=='prepared_component'
        conflicting=replace(same_gid,groups=(same_gid.groups[0],replace(same_gid.groups[1],context_slots=bundle.groups[1].context_slots),same_gid.groups[2]))
        assert admission('context_kv',conflicting,same_gid.key('context_kv'))[0]=='native'
        lease=ReplayLease('context_kv',bundle)
        for action in [lease.refresh,lease.release]:
            try:action()
            except LeaseError:pass
            else:raise AssertionError('in-flight owner release/refresh accepted')
        lease.mark_writer_stop()
        try:lease.release()
        except LeaseError:pass
        else:raise AssertionError('cache writer stop was treated as MTP42 consumer completion')
        lease.mark_consumer_stop();lease.release();assert not lease.inflight
        # A CPU snapshot can restore synthetic rows; it does not authorize any
        # device rollback. A partial failure must never invoke native retry.
        state={'rows':{},'native_retries':0,'downstream_MTP':0};snapshot=state['rows'].copy()
        def native_retry():state['native_retries']+=1
        def partial(lease):state['rows'][0]='partial';raise RuntimeError('CPU injected partial cache write')
        try:dispatch_original_site('context_kv',bundle,bundle.key('context_kv'),native_retry,partial)
        except ComponentFailure:pass
        else:raise AssertionError('partial failure not terminal')
        assert state['native_retries']==state['downstream_MTP']==0 and state['rows']!=snapshot
        state['rows']=snapshot.copy();assert state['rows']==snapshot
        # The reviewable adapter preserves public native signatures, strong
        # owner pins and default-native behavior without monkeypatching model.
        OPS.clear();model=Model();original=model.combine_hidden_states
        default=ContextAdapters(model.combine_hidden_states,model.precompute_and_store_context_kv)
        out=default.combine_hidden_states(Tensor((6,15360),'aux'))
        assert out.shape==(6,5120) and not default.active_leases
        assert model.combine_hidden_states.__func__ is original.__func__
        invoke=lambda args,lease:model.precompute_and_store_context_kv(*args)
        entries={'context_kv':PreparedEntry(bundle.key('context_kv'),invoke,model)}
        adapter=ContextAdapters(model.combine_hidden_states,model.precompute_and_store_context_kv,lambda stage,args:bundle,entries)
        slots=[Slots(list(range(6)),'slots'+str(i)) for i in range(3)]
        adapter.precompute_and_store_context_kv(out,Tensor((6,),'positions','INT32'),slots)
        assert len(adapter.active_leases)==1;held=adapter.active_leases[0]
        assert held.keepalive[0][2] is slots and held.keepalive[2] is entries['context_kv']
        try:adapter.assert_can_refresh_or_rebind()
        except LeaseError:pass
        else:raise AssertionError('adapter dropped consumer owner pins')
        held.mark_writer_stop();held.mark_consumer_stop();adapter.release_completed_lease(held)
        adapter.assert_can_refresh_or_rebind();assert not held.keepalive
        result={'CPU_only':True,'NPU_calls':0,'actual_native_helper_AST_order_and_shape_passed':True,
            'synthetic_prepared_path_matches_native_operator_order':True,'native_operator_order':cases[0]['operator_order'],
            'admission_negative_cases':len(negatives)+3,'admission_negative_reasons':reasons,
            'same_gid_shared_slot_and_distinct_layer_cache_supported':True,
            'inflight_refresh_early_release_and_writer_only_release_rejected':True,
            'partial_cache_write_stops_without_native_retry_or_downstream_MTP':True,
            'rollback_CPU_snapshot_only_not_device_proof':True,'capture_implementation_or_support_proven':False,
            'native_signature_adapter_factory_default_native_no_model_patch':True,
            'adapter_retains_actual_argument_and_entry_owner_references_until_consumer_stop':True,
            'numeric_or_actual_cache_consumer_proven':False,'performance_claim':None,'passed':True}
        print(json.dumps(result))
    finally:
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name]=previous

if __name__=='__main__':main()
