"""Private actual-native consumer taps; Tensor math stays in original source."""
from dataclasses import dataclass
import hashlib

try:
    import torch as _ledger_torch
except Exception:  # CPU harnesses always have torch; keep a safe fallback
    _ledger_torch = None


def _ledger_active():
    # The ledger counts real Python executions. Inside torch.dynamo tracing the
    # branch is treated as dead, so the dict mutation never enters a captured
    # graph (v20 broke npugraph compile with a traced dict mutation).
    try:
        return not _ledger_torch.compiler.is_compiling()
    except Exception:
        return True


if __package__:
    from .observer import PointSpec,make_observer_class
else:
    from observer import PointSpec,make_observer_class


@dataclass(frozen=True)
class Owner:
    role: str
    global_layer: int
    native: object


class ConsumerCallbacks:
    def __init__(self,torch,*,device,epoch,native_bindings):
        self.torch = torch;self.device = device;self.epoch = epoch
        self.native_bindings = native_bindings;self.owners = {};self.points = {}
        self.current_owner = None;self.Observer = make_observer_class(torch)
        self.native_functions_installed = [];self.prepack_output_tuples = []
        self.target_valid = torch.zeros((6,),device=device,dtype=torch.bool)
        self.draft_valid = torch.zeros((5,),device=device,dtype=torch.bool)
        self.target_positions = torch.full((6,1),-1,device=device,dtype=torch.int64)
        self.draft_positions = torch.full((5,1),-1,device=device,dtype=torch.int64)
        self.target_root_seen = torch.zeros((1,),device=device,dtype=torch.int64)
        self.draft_root_seen = torch.zeros((1,),device=device,dtype=torch.int64)
        self.required_point_names = set();self.supported_point_names = set()
        # Per-generation call ledger: which instrumented Python functions and
        # taps actually executed (counts only; no device reads).
        self.enter_counts = {};self.tap_counts = {};self.root_entries = {'main':0,'draft':0}
        self.tap_skips = {}
        self.tap_pads = {}
        # Module attribute aliases that satisfy Dynamo's LOAD_GLOBAL guards for
        # installed audit taps, plus any cross-session rebinds observed.
        self.dynamo_alias_modules = {};self.dynamo_alias_rebinds = []

    def register_owner(self,obj,role,layer):
        existing = self.owners.get(id(obj))
        owner = Owner(role,layer,obj)
        if existing is not None and existing != owner:
            # Real shared dispatcher ownership is resolved by the current
            # decoder, never by config/class count or cloned native objects.
            self.owners[id(obj)] = None
        else:self.owners[id(obj)] = owner

    def add_point(self,role,layer,label,width,dtype,*,max_rows=None,position_width=1):
        name = f'{role}.{layer}.{label}'
        if name in self.points:return self.points[name]
        rows = max_rows if max_rows is not None else (6 if role == 'main' else 5)
        spec = PointSpec(name,role,layer,rows,width,dtype,position_width)
        point = self.Observer(spec,device=self.device,epoch=self.epoch)
        self.points[name] = point;self.required_point_names.add(name)
        return point

    def owner(self,obj):
        owner = self.owners.get(id(obj))
        return owner if owner is not None else self.current_owner

    def inputs(self,owner,rows,metadata):
        positions = metadata.get('positions')
        if positions is None:positions = self.target_positions if owner.role == 'main' else self.draft_positions
        positions = positions.reshape(-1,1)[:rows]
        valid = (self.target_valid if owner.role == 'main' else self.draft_valid)[:rows]
        return positions,valid

    def tap(self,obj,label,carrier,payload,metadata,*,coordinates=None,valid=None):
        owner = self.owner(obj)
        if owner is None:return carrier
        name = f'{owner.role}.{owner.global_layer}.{label}'
        point = self.points.get(name)
        if point is None:return carrier  # An unregistered point is never coverage.
        if _ledger_active():self.tap_counts[name] = self.tap_counts.get(name,0)+1
        rows = payload.shape[0]
        positions,mask = self.inputs(owner,rows,metadata)
        if label.startswith('combine.'):
            positions = self.target_positions[:rows]
            mask = self.target_valid[:rows]
        if coordinates is not None:positions = coordinates
        if valid is not None:mask = valid
        rows = payload.shape[0];flat = payload.reshape(rows,-1)
        # The K5 draft pass can run one row more than the five-row draft
        # metadata window (the extra combined row has no stored position), and
        # the layer's own `positions` argument may already carry six rows while
        # the valid mask still has five. Pad positions and mask independently:
        # missing rows stay valid=False with position=-1 instead of silently
        # dropping the whole write (the v13 records showed exactly this split).
        padded_any = False
        if rows > positions.shape[0] and point.spec.max_rows >= rows:
            padded = self.torch.full((rows,point.spec.position_width),-1,device=positions.device,dtype=positions.dtype)
            kept = positions.shape[0]
            if kept:padded[:kept].copy_(positions)
            positions = padded;padded_any = True
        if rows > mask.shape[0] and point.spec.max_rows >= rows:
            pmask = self.torch.zeros((rows,),device=mask.device,dtype=torch.bool)
            kept = mask.shape[0]
            if kept:pmask[:kept].copy_(mask[:kept])
            mask = pmask;padded_any = True
        if padded_any and _ledger_active():
            key = name+':padded'
            self.tap_pads[key] = self.tap_pads.get(key,0)+1
        write_ok = (rows <= point.spec.max_rows and flat.shape[1] == point.spec.width
                    and positions.shape == (rows,point.spec.position_width) and mask.shape == (rows,)
                    and flat.dtype == point.values.dtype
                    and positions.dtype == self.torch.int64 and mask.dtype == self.torch.bool)
        if not write_ok and _ledger_active():
            row = self.tap_skips.get(name) or {'count':0}
            row.update({'count':row['count']+1,'rows':rows,'flat_width':int(flat.shape[1]),
                        'spec_max_rows':point.spec.max_rows,'spec_width':point.spec.width,
                        'positions_shape':list(positions.shape),'valid_shape':list(mask.shape),
                        'dtype':str(flat.dtype)})
            self.tap_skips[name] = row
        return point(carrier,payload,positions,mask)

    def enter(self,obj,qualified,metadata):
        if _ledger_active():
            owner_for_count = self.owners.get(id(obj))
            key = qualified+('@'+owner_for_count.role+'.'+str(owner_for_count.global_layer) if owner_for_count else '')
            self.enter_counts[key] = self.enter_counts.get(key,0)+1
        if qualified == 'DeepseekV41DecoderLayer.forward':
            owner = self.owners.get(id(obj))
            if owner is None:raise ValueError('Decoder is not a retained actual native owner')
            self.current_owner = owner
        elif qualified.endswith('.token_dispatch'):
            data = metadata['token_dispatch_input']
            ids = data.topk_ids;weights = data.topk_weights
            self.tap(obj,'dispatch.ids',ids,ids,{})
            self.tap(obj,'dispatch.weights',weights,weights,{})
        elif qualified.endswith('._attention'):
            attn = metadata['attn'];layer = getattr(attn,'long_kv_cache',None)
            context = metadata['metadata']
            if layer is not None and getattr(context,'compressor',None) is not None:
                # This is a bounded diagnostic of the local written rows.
                # Full SparseFlashMla source-prefix/block-table correspondence
                # remains explicitly uncovered in the first diagnostic pack.
                cache = layer.kv_cache[0];slots = context.compressor.cache.slot_mapping
                self.cache_read(obj,'cache.long.read',metadata['q'],cache,slots,metadata)
        elif qualified.endswith('._forward_attention'):
            # Cache planes differ by actual backend. The caller must bind an
            # exact plane/slot mapping; absence remains an uncovered point.
            layer_metadata = metadata['layer_metadata']
            slots = getattr(layer_metadata,'slot_mapping',None)
            if slots is not None and slots.ndim == 2:
                cache = metadata['kv_cache'][0]
                self.cache_read(obj,'cache.draft.read',metadata['hidden_states'],cache,slots,metadata)

    def value(self,obj,label,value,metadata):
        if label.startswith('shared.'):
            obj = metadata['attn']
        return self.tap(obj,label,value,value,metadata)

    def return_value(self,obj,qualified,value,metadata):
        if qualified == 'DeepseekV41DecoderLayer.forward':
            self.tap(obj,'decoder.output',value[0],value[0],metadata)
        elif qualified == 'AscendFusedTopKRouter._compute_routing':
            weights,ids = value
            self.tap(obj,'route.ids',ids,ids,{})
            self.tap(obj,'route.weights',weights,weights,{})
        elif qualified.endswith('.token_dispatch'):
            self.tap(obj,'dispatch.hidden',value.hidden_states,value.hidden_states,{})
        elif qualified == 'DeepseekV4DSparkModel.combine_hidden_states':
            self.tap(obj,'combine.output',value,value,metadata)
        elif qualified == 'DeepseekV4DSparkModel._project_shared_kv':
            self.tap(metadata['attn'],'shared.kv',value,value,metadata)
        return value

    def before_write(self,obj,qualified,operator,args,kwargs,metadata):
        if operator == 'torch_npu.npu_scatter_nd_update_':cache,slots,updates = args
        elif operator in ('torch.ops._C_ascend.npu_scatter_nd_update_sk','scatter_cache_sk'):
            cache,slots,updates = args
        elif operator == 'torch.ops._C_ascend.kv_compress_epilog':
            cache = kwargs['kv_compress_cache'];updates = kwargs['x'];slots = kwargs['slot_mapping']
        else:raise ValueError('Unknown actual native cache writer')
        if slots.ndim != 2 or slots.shape[1] != 2:return
        valid = (slots[:,0] >= 0)&(slots[:,1] >= 0)
        self.tap(obj,'cache.write',updates,updates,metadata,
                 coordinates=slots.to(self.torch.int64),valid=valid)

    def cache_read(self,obj,label,carrier,cache,slots,metadata):
        if slots.ndim != 2 or slots.shape[1] != 2 or cache.ndim < 3:return carrier
        rows = min(slots.shape[0],6 if self.owner(obj).role == 'main' else 5)
        slots = slots[:rows]
        valid = (slots[:,0] >= 0)&(slots[:,1] >= 0)&(slots[:,0] < cache.shape[0])&(slots[:,1] < cache.shape[1])
        blocks = slots[:,0].long().clamp(0,cache.shape[0]-1)*valid.long()
        offsets = slots[:,1].long().clamp(0,cache.shape[1]-1)*valid.long()
        values = cache[blocks,offsets].reshape(rows,-1)
        return self.tap(obj,label,carrier,values,metadata,coordinates=slots.to(self.torch.int64),valid=valid)

    def all_tensor_outputs(self):
        result = tuple(tensor for point in self.points.values() for tensor in point.tensor_tuple())
        return result+tuple(tensor for values in self.prepack_output_tuples for tensor in values)

    def role_metadata_outputs(self,role):
        if role == 'main':return self.target_positions,self.target_valid,self.epoch,self.target_root_seen
        return self.draft_positions,self.draft_valid,self.epoch,self.draft_root_seen

    def enter_root(self,role,positions):
        if _ledger_active():self.root_entries[role] = self.root_entries.get(role,0)+1
        # Tensor-only copies of ORIGINAL root inputs, including graph replay.
        destination = self.target_positions if role == 'main' else self.draft_positions
        marker = self.target_root_seen if role == 'main' else self.draft_root_seen
        rows = positions.numel()
        if positions.ndim != 1 or rows > destination.shape[0]:return
        destination.fill_(-1)
        destination[:rows].copy_(positions.reshape(-1,1))
        marker.add_(1)

    def begin_after_prior_fence(self,generation,*,actual_target_rows):
        # Only the caller at the normal runner boundary invokes this outside
        # capture and AFTER a real full-rank prior fence.
        for point in self.points.values():point.reset_after_fence()
        self.epoch.fill_(generation)
        self.target_valid.zero_();self.target_valid[:actual_target_rows].fill_(True)
        self.draft_valid.fill_(True)
        self.target_positions.fill_(-1)
        self.draft_positions.fill_(-1)
        self.target_root_seen.zero_();self.draft_root_seen.zero_()

    def export_after_fence(self,generation,raw_path):
        records = {};missing = [];raw = {}
        for name,point in self.points.items():
            values,pos,valid,epoch,count,rows = point.tensor_tuple()
            values_cpu = values.detach().cpu();positions_cpu = pos.detach().cpu();valid_cpu = valid.detach().cpu()
            raw[name] = {'values':values_cpu,'positions':positions_cpu,'valid':valid_cpu}
            payload = values_cpu.contiguous().view(self.torch.uint8).numpy().tobytes()
            record = {'point':name,'epoch':int(epoch.item()),'copy_count':int(count.item()),'rows':int(rows.item()),
                      'values_sha256':hashlib.sha256(payload).hexdigest(),'value_shape':list(values_cpu.shape),
                      'value_dtype':str(values_cpu.dtype),'positions':positions_cpu.tolist(),'valid':valid_cpu.tolist(),
                      'owned_layout':{'ptr':values.data_ptr(),'shape':list(values.shape),'stride':list(values.stride()),
                                      'dtype':str(values.dtype),'device':str(values.device)}}
            records[name] = record
            if record['epoch'] != generation or record['copy_count'] != 1:missing.append(name)
        self.torch.save(raw,raw_path)
        data = raw_path.read_bytes()
        return {'generation':generation,'records':records,'missing_or_duplicate_points':missing,
                'raw_device_export':{'path':str(raw_path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                                     'retain_remote_only_if_at_least_1MB':len(data) >= 1_000_000},
                'complete_point_exports':not missing,'registered_points_are_device_coverage':False,
                'device_audit_passed':False,'full_consumer_audit_passed':False,'safe_formal_receipt':False,
                'formal_model_validated':False,'actual_prepack_consumer_order_validated':False}
