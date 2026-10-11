"""CPU-only admission/lifetime prototype for two original native call sites.

No torch, graph constructor, device call, monkeypatch or capture implementation.
Opaque callbacks retain the original native arithmetic and operator order.
"""
from dataclasses import dataclass

DTYPE_BYTES={'BF16':2,'INT32':4}

@dataclass(frozen=True)
class TensorBinding:
    address:int
    extent:int
    shape:tuple
    stride:tuple
    dtype:str
    owner:str
    generation:int
    live:bool=True

    def valid(self):
        if not self.live or not self.owner or self.address<=0 or self.generation<0:return False
        if self.dtype not in DTYPE_BYTES or len(self.shape)!=len(self.stride):return False
        if any(n<=0 for n in self.shape) or any(s<=0 for s in self.stride):return False
        needed=(1+sum((n-1)*s for n,s in zip(self.shape,self.stride)))*DTYPE_BYTES[self.dtype]
        return needed<=self.extent and self.address+self.extent<2**48

@dataclass(frozen=True)
class CacheGroupBinding:
    layer:int
    layer_name:str
    gid:int
    kernel_block_size:int
    store_block_size:int
    cache:TensorBinding
    context_slots:TensorBinding
    native_name_to_gid_verified:bool

@dataclass(frozen=True)
class ContextBinding:
    aux:TensorBinding
    context:TensorBinding
    positions:TensorBinding
    groups:tuple
    source_fingerprints:tuple
    model_owner:str
    model_generation:int
    rope_owner:str
    rope_generation:int
    kv_plan:str='A3_PA_ND_BLOCK_OFFSET_ScatterSk'
    stream:int=47
    context_rows:int=6
    query_rows:int=5
    prefill:bool=False
    outer_capturing:bool=False
    outer_merged_graph_configured:bool=False
    asynchronous_refresh:bool=False
    producer_dependencies_joined:bool=False
    collective_internal_streams_verified:bool=False
    alias_allocation_domains_verified:bool=False
    SDK_capture_gate_passed:bool=False
    actual_consumer_gate_passed:bool=False
    fresh_native_sources_verified:bool=False

    def key(self,stage):
        # Include keyword tensors, per-group slots, dtype/strides, owner epochs,
        # model/weights/RoPE/KV-plan identities and the original stream.
        return (stage,self.aux,self.context,self.positions,self.groups,self.source_fingerprints,
                self.model_owner,self.model_generation,self.rope_owner,self.rope_generation,
                self.kv_plan,self.stream,self.context_rows,self.query_rows)

def admission(stage,current,prepared_key):
    if stage not in ('combine','context_kv'):raise ValueError('unknown original call site')
    if current.prefill or current.context_rows!=6 or current.query_rows!=5:
        return 'native','prefill_or_dynamic_extent'
    if current.outer_capturing:return 'native','no_nested_graph'
    if stage=='context_kv' and current.outer_merged_graph_configured:
        return 'native','native_hook_belongs_to_existing_outer_graph'
    if not current.fresh_native_sources_verified or len(current.source_fingerprints)<8:
        return 'native','native_source_fingerprint_guard_pending'
    if not current.model_owner or not current.rope_owner or min(current.model_generation,current.rope_generation)<0:
        return 'native','model_or_RoPE_owner_unbound'
    tensors=[current.aux,current.context,current.positions]
    for group in current.groups:tensors.extend([group.cache,group.context_slots])
    if not all(t.valid() for t in tensors):return 'native','unbound_or_invalid_lifetime_or_extent'
    if (current.aux.shape,current.aux.dtype)!=((6,15360),'BF16') or (current.context.shape,current.context.dtype)!=((6,5120),'BF16'):
        return 'native','hidden_shape_or_dtype_changed'
    if (current.positions.shape,current.positions.dtype)!=((6,),'INT32'):return 'native','position_descriptor_changed'
    expected=[(40+i,f'mtp.{i}.self_attn.swa_cache') for i in range(3)]
    if [(g.layer,g.layer_name) for g in current.groups]!=expected:return 'native','native_layer_order_or_binding_changed'
    by_gid={}
    for group in current.groups:
        if group.gid is None or group.gid<0 or not group.native_name_to_gid_verified:return 'native','actual_name_to_gid_missing'
        if group.kernel_block_size<=0 or group.store_block_size!=128:return 'native','native_logical_or_store_block_size_unbound'
        if group.cache.shape!=(7939,128,1,512) or group.cache.dtype!='BF16':return 'native','cache_geometry_changed'
        if group.context_slots.shape!=(6,) or group.context_slots.dtype!='INT32':return 'native','context_slot_descriptor_changed'
        # Actual groups can repeat. Repeated gid must retain the same native
        # context-slot buffer; different layer cache views still remain bound.
        if group.gid in by_gid and by_gid[group.gid]!=group.context_slots:return 'native','same_gid_has_conflicting_slot_buffers'
        by_gid[group.gid]=group.context_slots
    if current.asynchronous_refresh:return 'native','refresh_can_race_inflight_helper_or_consumer'
    if not current.producer_dependencies_joined:return 'native','producer_stream_join_not_proven'
    if stage=='combine' and not current.collective_internal_streams_verified:return 'native','TP_allgather_capture_or_stream_join_not_proven'
    if not current.alias_allocation_domains_verified:return 'native','cache_alias_domains_not_proven'
    if not current.SDK_capture_gate_passed or not current.actual_consumer_gate_passed:return 'native','SDK_or_actual_consumer_gate_pending'
    if prepared_key!=current.key(stage):return 'native','prepared_entry_signature_or_owner_epoch_changed'
    return 'prepared_component','all_CPU_admission_checks_passed'

class LeaseError(RuntimeError):pass
class ComponentFailure(RuntimeError):pass

class ReplayLease:
    def __init__(self,stage,binding):
        self.stage=stage;self.binding=binding;self.inflight=True
        self.producer_stop=False;self.consumer_stop=False;self.poisoned=False
    def refresh(self):
        if self.inflight:raise LeaseError('cannot refresh/rebind in-flight inputs, slots or cache owners')
    def mark_writer_stop(self):self.producer_stop=True
    def mark_consumer_stop(self):self.consumer_stop=True
    def release(self):
        # Combine output lives through the native context copy; KV input/cache
        # owners live through the serial MTP42 consumer, not only writer stop.
        if not self.producer_stop or not self.consumer_stop:raise LeaseError('owners cannot be released before producer and consumer completion')
        self.inflight=False

def dispatch_original_site(stage,binding,prepared_key,native_call,prepared_call):
    mode,reason=admission(stage,binding,prepared_key)
    if mode=='native':return native_call(),{'mode':mode,'reason':reason,'device_support_proven':False}
    lease=ReplayLease(stage,binding)
    try:return prepared_call(lease),{'mode':mode,'reason':reason,'lease':lease,'device_support_proven':False}
    except BaseException as error:
        lease.poisoned=True
        # Cache writes may already be submitted. Never transparently re-run
        # native after a partial component failure. Device rollback remains an
        # explicit future gate; CPU snapshots cannot authorize device recovery.
        raise ComponentFailure('component failed; stop downstream MTP, no native retry') from error
