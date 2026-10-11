"""CPU/source-guard prototype for a FUTURE separate DSpark consumer audit job.

No torch import, live monkey-patch, allocation, capture, device copy or device
receipt is implemented here. Prepared source/bindings are not execution proof.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


HOOKS = [
    ('v41_model.py','DeepseekV41DecoderLayer.forward','main0-39/draft40-42 actual layer output'),
    ('actual_router.py','AscendFusedTopKRouter._compute_routing','actual role/layer top6 route producer'),
    ('token_dispatcher.py','TokenDispatcherWithAllGather.token_dispatch','actual route IDs/weights consumed by dispatch'),
    ('token_dispatcher.py','TokenDispatcherWithMC2.token_dispatch','source guard if actual backend selects MC2; not inferred from config'),
    ('v41_model.py','DeepseekV41Model.forward','actual aux residual entering target layers37-39'),
    ('v4_dspark.py','DeepseekV4DSparkModel.combine_hidden_states','actual aux main_proj/main_norm'),
    ('v4_dspark.py','DeepseekV4DSparkModel._project_shared_kv','three distinct native wkv/norm/RoPE producers'),
    ('v4_dspark.py','DeepseekV4DSparkModel._store_standard_swa_kv','actual draft SWA scatter consumer'),
    ('v41_dspark.py','DeepseekV41DSparkModel.forward','three serial delayed-mHC draft blocks'),
    ('kv_plan.py','DsaAttnKvPlan.dsa_kv_compress_scatter','native formatted valid-slot cache write'),
    ('dsa_v41.py','DeepseekV41EagerAttentionImpl._write_compressed_source','actual target long/indexer KV source'),
    ('dsa_v41.py','DeepseekV41EagerAttentionImpl._attention','actual main SparseFlashMla cache reader'),
    ('dsa_v1.py','AscendDSAImpl._forward_attention','actual draft attention/cache reader'),
    ('stock_dspark_proposer.py','AscendDSparkProposer.set_inputs_first_pass','actual per-group positions/block/slot/seq metadata'),
    ('runner.py','NPUModelRunner._sample','actual native rejection sampler device output'),
    ('runner.py','NPUModelRunner._bookkeeping_sync','existing CPU accepted-prefix bookkeeping'),
    ('scheduler.py','Scheduler.update_from_output','actual rejected-token rollback/stop/clip')]


def qualified_functions(source):
    tree=ast.parse(source)
    result={}
    def visit(nodes,prefix=''):
        for node in nodes:
            if isinstance(node,ast.ClassDef):visit(node.body,prefix+node.name+'.')
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                result[prefix+node.name]=node
    visit(tree.body)
    return result


def prepare_source_plan(discovery):
    discovery=Path(discovery)
    manifest=json.loads((discovery/'source_receipt.json').read_text())
    files={item['name']:item for item in manifest['files'] if not item.get('missing')}
    hooks=[]
    for name,qualified,consumer in HOOKS:
        receipt=files[name]
        source=(discovery/(name+'.txt')).read_text()
        if hashlib.sha256(source.encode()).hexdigest()!=receipt['sha256']:
            raise ValueError('Frozen source bytes changed: '+name)
        functions=qualified_functions(source)
        if qualified not in functions:raise ValueError('Actual SDK hook absent: '+qualified)
        node=functions[qualified]
        segment=ast.get_source_segment(source,node)
        hooks.append({'file':name,'remote_source':receipt['path'],'file_sha256':receipt['sha256'],
            'qualified_function':qualified,'line':node.lineno,
            'function_sha256':hashlib.sha256(segment.encode()).hexdigest(),
            'consumer':consumer,'applied':False})
    return {'scope':'CPU verified source/AST hook plan; no actual device observer installed',
        'formal_source_snapshot':manifest['formal_snapshot'],'hooks':hooks,
        'checkpoint_target_ids':[37,38,39],'runner_aux_ids':[38,39,40],
        'actual_target_entry_layer_indices':[37,38,39],
        'main_layers':list(range(40)),'draft_layers':[40,41,42],
        'capture_policy':{'preallocated_fixed_address_bounded_buffers_only':True,
            'device_epoch_and_position_payload_required':True,'d2h_inside_capture':False,
            'logging_inside_capture':False,'python_budget_count_inside_capture':False,
            'opaque_none_mutation_wait_copy_reused':False,
            'native_aten_copy_epoch_capture_alias_verified':False},
        'export_policy':'real decode/propose/sample generation -> current-rank full device fence -> export before reuse',
        'source_registration_is_device_coverage':False,'device_audit_passed':False,
        'full_consumer_audit_passed':False,'safe_formal_receipt':False,
        'formal_model_validated':False,'performance_claim':None}


@dataclass(frozen=True)
class TensorSeal:
    ptr: int
    storage_ptr: int
    storage_offset: int
    version: int | None
    shape: tuple[int,...]
    stride: tuple[int,...]
    dtype: str
    device: str
    format: int

    @classmethod
    def from_metadata(cls,value):
        required={'ptr','storage_ptr','storage_offset','version','shape','stride','dtype','device','format'}
        if set(value)!=required:raise ValueError('Incomplete actual ptr/version/layout seal')
        if value['ptr']<=0 or value['storage_ptr']<=0 or value['storage_offset']<0:
            raise ValueError('Unallocated or invalid tensor binding')
        if not value['shape'] or len(value['shape'])!=len(value['stride']) or any(v<=0 for v in value['shape']):
            raise ValueError('Empty or invalid tensor layout')
        if any(v<0 for v in value['stride']) or not value['device'].startswith('npu:'):
            raise ValueError('Unexpected device/stride binding')
        return cls(**{**value,'shape':tuple(value['shape']),'stride':tuple(value['stride'])})

    def assert_matches(self,value,*,weight):
        actual=TensorSeal.from_metadata(value)
        fields=('ptr','storage_ptr','storage_offset','shape','stride','dtype','device','format')
        if any(getattr(actual,k)!=getattr(self,k) for k in fields):
            raise ValueError('Actual pointer/storage/layout binding changed')
        if weight and (self.version is None or actual.version!=self.version):
            raise ValueError('Actual weight version unavailable or changed')
        # Cache versions legitimately change on writes; they are recorded, not
        # misinterpreted as the device generation marker.
        return {'cpu_binding_check_passed':True,'device_consumer_proof':False}


def accepted_cache_transition(input_tokens,positions,draft_tokens,sampled_tokens,computed_before,remaining_budget):
    vectors=(input_tokens,positions,draft_tokens,sampled_tokens)
    if any(not isinstance(v,list) or any(type(x)is not int or x<0 for x in v) for v in vectors):
        raise ValueError('Need exact existing CPU integer lists after the device fence')
    if not sampled_tokens or remaining_budget<=0 or input_tokens[1:]!=draft_tokens:
        raise ValueError('Invalid actual verifier input/proposal binding')
    if len(input_tokens)!=1+len(draft_tokens) or positions!=list(range(computed_before,computed_before+len(input_tokens))):
        raise ValueError('Actual target input positions do not match scheduled cache rows')
    accepted=len(sampled_tokens)-1
    if accepted>len(draft_tokens) or sampled_tokens[:accepted]!=draft_tokens[:accepted]:
        raise ValueError('Native accepted tokens are not the actual proposal prefix')
    rejected=len(draft_tokens)-accepted
    usable=sampled_tokens[:remaining_budget]
    return {'raw_accepted_draft_tokens':accepted,'raw_rejected_draft_tokens':rejected,
        'raw_bonus_token':sampled_tokens[-1],'usable_output_tokens':usable,
        'terminal_tokens_trimmed':len(sampled_tokens)-len(usable),
        'expected_computed_after':computed_before+len(input_tokens)-rejected,
        # Cache inputs are last actual token + accepted drafts. The bonus token
        # is predicted by the final committed row and is computed NEXT round.
        'committed_target_input_tokens':input_tokens[:accepted+1],
        'committed_target_positions':positions[:accepted+1],
        'next_round_input_token':sampled_tokens[-1],
        'next_round_input_position':computed_before+accepted+1,
        'cpu_correspondence_check_passed':True,'actual_cache_write_read_proven':False}


class GenerationEnvelope:
    """CPU state/seal protocol. Matching synthetic data never grants a receipt."""
    def __init__(self,model_source_sha256,checkpoint_sha256,rank):
        if not 0<=rank<8:raise ValueError('Expected a real formal TP rank')
        self.model_source_sha256=model_source_sha256
        self.checkpoint_sha256=checkpoint_sha256
        self.rank=rank
        self.phase='idle';self.epoch=0;self.exported=[]

    def begin(self,epoch,external_id,core_id,output_processor_mapping):
        if self.phase not in ('idle','exported') or epoch!=self.epoch+1:
            raise ValueError('Unfenced/unexported generation cannot be reused or skipped')
        if output_processor_mapping.get(external_id)!=[core_id]:
            raise ValueError('External/core request identity needs the actual add_request mapping')
        self.epoch=epoch;self.core_id=core_id;self.external_id=external_id;self.phase='submitted'

    def fence(self,epoch,*,current_rank_device_synchronized):
        if self.phase!='submitted' or epoch!=self.epoch or current_rank_device_synchronized is not True:
            raise ValueError('Actual generation/device completion fence is missing')
        self.phase='fenced'

    def structural_export(self,core_id,device_payload):
        if self.phase!='fenced' or core_id!=self.core_id:
            raise ValueError('Export outside the fenced actual core request generation')
        expected={('main',i) for i in range(40)}|{('draft',i) for i in (40,41,42)}
        found={(row['role'],row['layer']) for row in device_payload}
        if found!=expected or len(device_payload)!=43:
            raise ValueError('Missing/duplicate actual layer payload; registration counts are insufficient')
        if any(row.get('epoch')!=self.epoch or row.get('payload_source')!='fixed-device-buffer' for row in device_payload):
            raise ValueError('Stale epoch or CPU callback-derived layer payload')
        self.phase='exported'
        result={'rank':self.rank,'epoch':self.epoch,'external_id':self.external_id,'core_id':self.core_id,
            'cpu_structure_check_passed':True,'payload':copy.deepcopy(device_payload),
            'source':self.model_source_sha256,'checkpoint':self.checkpoint_sha256,
            'device_exporter_implemented':False,'capture_alias_validated':False,
            'device_audit_passed':False,'full_consumer_audit_passed':False,
            'safe_formal_receipt':False,'performance_claim':None}
        self.exported.append(result)
        return result


def install_in_new_job(*args,**kwargs):
    raise NotImplementedError('CPU/source-guard prototype only: native fixed-address copy/epoch capture and actual cache/route exporter require independent SDK/NPU validation in a new job')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--discovery',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plan=prepare_source_plan(a.discovery)
    a.output.write_text(json.dumps(plan,indent=2)+'\n')
    print('DSPARK_CPU_SOURCE_PLAN',len(plan['hooks']),'hooks; device_audit_passed=false')


if __name__=='__main__':main()
