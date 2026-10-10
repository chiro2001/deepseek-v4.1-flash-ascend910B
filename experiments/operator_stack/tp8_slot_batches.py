"""Experimental one-launch slot-coordinate conversion for formal batch1.

Only tensor addresses and layout constants are retained. Every launch reads
the current raw slots, positions and query boundaries. Each cache group keeps
its own output buffer. Batch dictionaries are supplied by the native runner
and reset every step; no token/position values are cached across steps.
"""
from collections import Counter
import json

import torch
import triton
import triton.language as tl

import goal20_patches as goal
from metadata_kernels import prepare_slots as native_prepare_slots


@triton.jit
def many_slot_kernel(descriptors, positions, query, actual_reqs, actual_tokens,
                     SKIP: tl.constexpr, INPUT_INT64: tl.constexpr, OUTPUT_INT64: tl.constexpr):
    g = tl.program_id(0)
    inp = tl.load(descriptors + g * 5).to(tl.pointer_type(tl.int64 if INPUT_INT64 else tl.int32))
    out = tl.load(descriptors + g * 5 + 1).to(tl.pointer_type(tl.int64 if OUTPUT_INT64 else tl.int32))
    block_shift = tl.load(descriptors + g * 5 + 2).to(tl.int32)
    ratio_shift = tl.load(descriptors + g * 5 + 3).to(tl.int32)
    c2 = tl.load(descriptors + g * 5 + 4) != 0
    raw = tl.load(inp).to(tl.int64)
    physical = tl.maximum(raw, 0)
    valid = raw >= 0
    valid = valid & ((ratio_shift == 0) | (((physical + 1) & 1) == 0))
    physical = physical >> ratio_shift
    end = tl.minimum(tl.load(query + actual_reqs), actual_tokens)
    pos = tl.load(positions)
    c2_valid = (end > 0) & ((pos & 1) == 1)
    if SKIP:
        c2_valid = False
    valid = valid & ((~c2) | c2_valid)
    # Ascend scalar pointer stores reject a nonzero offset in this compiler.
    # Emit one masked contiguous vector store for the two integer coordinates.
    j=tl.arange(0,16)
    row=tl.where(valid,physical >> block_shift,-1)
    column=tl.where(valid,physical & ((1 << block_shift)-1),-1)
    tl.store(out+j,tl.where(j==0,row,column),j<2)


class SlotBatches:
    def __init__(self, expected_groups=12):
        self.expected_groups = expected_groups
        self.targets = {}
        self.descriptors = None
        self.descriptor_signature = None
        self.counts = Counter()
        self.batch_key = 'tp8:slot_conversion_batch'
        self.contract_snapshots = {}
        self.model_pool_blocks = None

    def bind_pool(self, blocks):
        assert int(blocks)>0
        self.model_pool_blocks=int(blocks)

    def record(self, builder, common, positions, n, compressed, ratio, block_size):
        slots, output = common.slot_mapping, builder._slot_mapping_2d
        query = common.query_start_loc
        if (n != 1 or int(getattr(common, 'num_reqs', query.numel()-1)) != 1 or
                positions is None or ratio not in (1, 2) or block_size <= 0 or
                block_size & (block_size-1) or block_size>2**30 or slots.dtype not in (torch.int32,torch.int64) or
                output.dtype not in (torch.int32,torch.int64) or positions.dtype != torch.int64 or
                not slots.is_contiguous() or not output.is_contiguous() or
                output.ndim != 2 or output.shape[1] != 2 or output.shape[0] < 1 or
                slots.numel() < 1 or positions.numel() < 1 or query.numel() < 2):
            return None
        if positions.device.type!='npu' or any(t.device!=positions.device for t in (slots,output,query)):
            return None
        # This experiment is aligned to the current formal pool. Very large
        # raw coordinates expose a discrepancy in the existing Triton helper;
        # keep such model configurations outside this integration experiment.
        config=getattr(builder,'vllm_config',None)
        if config is not None:
            blocks=self.model_pool_blocks
            if blocks is None:
                blocks=getattr(config.cache_config,'num_gpu_blocks',None)
            spec=getattr(builder,'kv_cache_spec',None)
            logical=getattr(spec,'block_size',None)
            if blocks is None or logical is None or int(blocks)*int(logical)>=2**24:
                return None
        shift = ratio.bit_length()-1 if compressed else 0
        return {'builder': builder, 'slots': slots, 'output': output,
                'positions': positions, 'query': query,
                'layout': (slots.data_ptr(), output.data_ptr(), block_size.bit_length()-1,
                           shift, int(compressed and ratio == 2)),
                'dtype_flags':(slots.dtype==torch.int64,output.dtype==torch.int64),
                'query_ptr': query.data_ptr(), 'positions_ptr': positions.data_ptr()}

    def prepare(self, builder, common, positions, n, actual_reqs, actual_tokens,
                compressed, ratio, block_size, skip, batch_shared):
        # The comparison arm must retain its normal preparation cost once the
        # candidate has learned the layouts. It computes from current inputs
        # directly; candidate-only registry checks are unnecessary on this path.
        if not goal.enabled('metadata_manyslots') and len(self.targets)==self.expected_groups:
            self.counts['fallback:arm_after_learning']+=1
            return native_prepare_slots(builder,common,positions,n,actual_reqs,actual_tokens,
                                        compressed,ratio,block_size,skip)
        def fallback(reason):
            self.counts['fallback:'+reason] += 1
            return native_prepare_slots(builder,common,positions,n,actual_reqs,actual_tokens,
                                        compressed,ratio,block_size,skip)
        rec = self.record(builder,common,positions,n,compressed,ratio,block_size)
        if rec is None:
            if n==1 and not torch.npu.is_current_stream_capturing():
                config=getattr(builder,'vllm_config',None)
                spec=getattr(builder,'kv_cache_spec',None)
                tensors={'slots':common.slot_mapping,'output':builder._slot_mapping_2d,
                         'positions':positions,'query':common.query_start_loc}
                snapshot={'n':n,'num_reqs':getattr(common,'num_reqs',None),
                    'compressed':compressed,'ratio':ratio,'block_size':block_size,
                    'logical_block_size':getattr(spec,'block_size',None),
                    'num_gpu_blocks':getattr(config.cache_config,'num_gpu_blocks',None) if config else None,
                    'bound_model_pool_blocks':self.model_pool_blocks,
                    'tensors':{name:None if t is None else {'shape':list(t.shape),'dtype':str(t.dtype),
                        'device':str(t.device),'device_type':t.device.type,'contiguous':t.is_contiguous()}
                        for name,t in tensors.items()}}
                if self.contract_snapshots.get(id(builder))!=snapshot:
                    self.contract_snapshots[id(builder)]=snapshot
                    print('SLOT_BATCH_CONTRACT_FALLBACK',json.dumps(snapshot),flush=True)
            return fallback('contract')
        rec.update(actual_reqs=actual_reqs,actual_tokens=actual_tokens,skip=skip)
        self.targets[id(builder)] = rec
        # Learn all group-local destinations from the already-audited core
        # path. Pointer changes rebuild the descriptor table; an incomplete
        # registry takes the native path and never guesses missing groups.
        if not goal.enabled('metadata_manyslots'):
            return fallback('arm')
        if batch_shared is None or len(self.targets) != self.expected_groups:
            return fallback('registry')
        targets = list(self.targets.values())
        if len({r['layout'][1] for r in targets}) != self.expected_groups:
            return fallback('output_alias')
        if {r['layout'][0] for r in targets} & {r['layout'][1] for r in targets}:
            return fallback('input_output_alias')
        if any(r['query_ptr']!=rec['query_ptr'] or r['positions_ptr']!=rec['positions_ptr'] for r in targets):
            return fallback('input_identity')
        if any(r['dtype_flags']!=rec['dtype_flags'] for r in targets):
            return fallback('mixed_dtypes')
        signature = tuple(r['layout'] for r in targets)
        # Actual requests/tokens are same-batch scalars supplied by the runner.
        # Reject a changed contract rather than reuse an earlier step's marker.
        key = (signature,rec['dtype_flags'],rec['query_ptr'],rec['positions_ptr'],actual_reqs,actual_tokens,skip)
        existing = batch_shared.get(self.batch_key)
        if existing is not None:
            if existing != key:
                return fallback('batch_identity')
            self.counts['reused_group_output'] += 1
            return builder._slot_mapping_2d[:n]
        if self.descriptor_signature != signature:
            if torch.npu.is_current_stream_capturing():
                return fallback('descriptor_capture')
            self.descriptors = torch.tensor(signature, dtype=torch.int64, device=positions.device)
            self.descriptor_signature = signature
            self.counts['descriptor_builds'] += 1
        many_slot_kernel[(self.expected_groups,)](self.descriptors,positions,common.query_start_loc,
                                                actual_reqs,actual_tokens,skip,*rec['dtype_flags'])
        batch_shared[self.batch_key] = key
        self.counts['fused_launches'] += 1
        return builder._slot_mapping_2d[:n]

    def status(self):
        return {'counts':dict(self.counts),'registered_groups':len(self.targets),
                'expected_groups':self.expected_groups,'dynamic_values_cached':False,
                'bound_model_pool_blocks':self.model_pool_blocks,
                'contract_snapshots':list(self.contract_snapshots.values())}

    def audit(self):
        assert not torch.npu.is_current_stream_capturing()
        rows=[]
        for rec in self.targets.values():
            raw=int(rec['slots'][:1].cpu()[0])
            pos=int(rec['positions'][:1].cpu()[0])
            query=rec['query'].cpu()
            end=min(int(query[rec['actual_reqs']]),rec['actual_tokens'])
            _,_,block_shift,ratio_shift,c2=rec['layout']
            physical=max(raw,0)
            valid=raw>=0 and (ratio_shift==0 or (physical+1)%2==0)
            physical=physical>>ratio_shift
            valid=valid and (not c2 or (not rec['skip'] and end>0 and pos%2==1))
            expected=[physical>>block_shift,physical% (1<<block_shift)] if valid else [-1,-1]
            expected=torch.tensor(expected,dtype=torch.int64).to(rec['output'].dtype).tolist()
            actual=rec['output'][:1].cpu().reshape(-1).tolist()
            assert actual==expected,{'actual':actual,'expected':expected,'layout':rec['layout'][2:]}
            rows.append({'block_size':1<<block_shift,'ratio_shift':ratio_shift,
                         'c2_mask':bool(c2),'exact':True})
        assert len(rows)==self.expected_groups,(len(rows),self.status())
        return {'groups':rows,'all_exact':True,'count':len(rows),'reference':'CPU integer coordinates from current inputs'}


REGISTRY = SlotBatches()
