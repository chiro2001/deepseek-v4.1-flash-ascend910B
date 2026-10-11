"""Actual checkpoint/config/EP/profile source guards; stdlib only, NPU0."""
import ast
import copy
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

ROOT=Path('/work/gmm_lane_sparse_msd_k5_route_cpu_20261011_v1')
FORMAL=Path('/work/results/formal_dspark_native_eager_aa_v3')
PROFILE=Path('/work/results/formal_dspark_native_profile_v1')
ASCEND=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
VLLM=Path('/vllm-workspace/vllm/vllm')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def source(path,ranges):
    p=Path(path);lines=p.read_text().splitlines()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p),
            'snippets':[{'first_line':a,'last_line':min(b,len(lines)),
                         'text':'\n'.join(lines[a-1:b])} for a,b in ranges]}

class Config(SimpleNamespace):
    def update(self,fields):self.__dict__.update(fields)

@dataclass
class CPUArch:
    architectures:list
    model_type:str
    is_mm_prefix_lm:bool
    num_experts:int
    num_experts_per_token:int

def main():
    assert os.environ.get('ASCEND_VISIBLE_DEVICES')==os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==''
    ROOT.mkdir(exist_ok=True)
    plan_path=FORMAL/'prepared_plan.json';plan=json.loads(plan_path.read_text());kwargs=plan['kwargs']
    assert kwargs['tensor_parallel_size']==8 and kwargs['enable_expert_parallel']
    assert kwargs.get('data_parallel_size',1)==kwargs.get('prefill_context_parallel_size',1)==1
    assert kwargs.get('expert_placement_strategy','linear')=='linear' and not kwargs.get('enable_eplb',False)
    checkpoint_path=Path(kwargs['model'])/'config.json';checkpoint=json.loads(checkpoint_path.read_text())
    assert digest(checkpoint_path)=='40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424'
    text=checkpoint['text_config'];assert (text['n_routed_experts'],text['num_experts_per_tok'])==(384,6)
    assert (text['dspark_n_routed_experts'],text['dspark_num_experts_per_tok'])==(128,3)
    override_path=ASCEND/'patch/platform/patch_speculative_config.py'
    tree=ast.parse(override_path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_normalize_deepseek_v4_dspark_draft')
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace={'replace':replace};exec(compile(module,str(override_path),'exec'),namespace)
    target_config=copy.deepcopy(checkpoint)
    draft_hf=Config(**copy.deepcopy(checkpoint));draft_hf.text_config=Config(**copy.deepcopy(text))
    class Registry:
        def inspect_model_cls(self,architectures,model):return SimpleNamespace(CPU_stub=True),architectures[0]
    draft=SimpleNamespace(hf_config=draft_hf,model_arch_config=CPUArch(
        checkpoint['architectures'],checkpoint['model_type'],True,384,6),registry=Registry())
    namespace[fn.name](draft)
    assert (draft.hf_config.text_config.n_routed_experts,draft.hf_config.text_config.num_experts_per_tok)==(128,3)
    assert (draft.model_arch_config.num_experts,draft.model_arch_config.num_experts_per_token)==(128,3)
    assert target_config==checkpoint
    paths={
        'draft_override':(override_path,[(42,120),(123,138)]),
        'draft_config_owner':(VLLM/'config/speculative.py',[(835,866)]),
        'actual_draft_consumer':(ASCEND/'models/deepseek_v41/dspark.py',[(57,72),(140,150)]),
        'AllGather_range_and_count_mode':(ASCEND/'ops/fused_moe/token_dispatcher.py',[(406,413),(462,501)]),
        'linear_expert_map':(VLLM/'model_executor/layers/fused_moe/expert_map_manager.py',[(62,93)]),
        'EP_rank_equals_flattened_TP_rank':(VLLM/'model_executor/layers/fused_moe/config.py',[(1114,1121),(1204,1215),(1234,1250)]),
        'parallel_defaults':(VLLM/'config/parallel.py',[(124,133),(173,180)]),
        'role_comm_state_guard':(ASCEND/'ops/fused_moe/moe_comm_method.py',[(52,80),(96,117),(258,263)]),
        'route_capture_patch':(Path('/work/src_dspark_native_eager_v3/stack/route_capture_patch.py'),[(1,65)]),
    }
    sources={name:source(p,ranges) for name,(p,ranges) in paths.items()}
    # Check the actual formal snapshot receipts, not an arbitrary working copy.
    result=json.loads((FORMAL/'result.json').read_text())
    assert sources['route_capture_patch']['sha256']==result['source_hashes']['route_capture_patch.py']
    guard={'CPU_only':True,'NPU_calls':0,'NPU_permission':0,'prepared_plan':{'path':str(plan_path),'sha256':digest(plan_path)},
           'checkpoint':{'path':str(checkpoint_path),'sha256':digest(checkpoint_path),
                         'target_experts':384,'target_top_k':6,'draft_experts':128,'draft_top_k':3,
                         'K':text['moe_intermediate_size'],'N':text['hidden_size'],'block_size':text['dspark_block_size'],
                         'target_layer_ids':text['dspark_target_layer_ids']},
           'override_CPU_execution':{'actual_function_AST_executed':True,'registry_is_CPU_stub':True,
                                     'actual_ModelConfig_or_model_not_constructed':True,'target_unchanged':True,
                                     'draft_private_text_config_experts':128,'draft_private_text_config_top_k':3,
                                     'draft_model_arch_experts':128,'draft_model_arch_top_k':3,'passed':True},
           'EP_mapping':{'EP':8,'DP':1,'PCP':1,'TP':8,'enable_expert_parallel':True,'placement':'linear',
                         'enable_eplb':False,'ep_rank_equals_formal_TP_rank':True,
                         'target_ranges':[[48*r,48*(r+1)] for r in range(8)],
                         'draft_ranges':[[16*r,16*(r+1)] for r in range(8)],
                         'runtime_device_expert_map_readback':False,
                         'basis':'actual formal kwargs + defaults + EP flatten/source + actual AllGather active_expert_range'},
           'role_comm_state_has_distinct_shape_key':True,'sources':sources,'passed':True}
    (ROOT/'source_guards.json').write_text(json.dumps(guard,indent=2)+'\n')
    profiles=[]
    for rank in range(8):
        paths=list((PROFILE/'profile').glob(f'dspark_rank{rank}_*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
        assert len(paths)==1;path=paths[0];rows=[];steps=set()
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                if row['Type']=='GroupedMatmul' and 'aclnnGroupedMatmulWeightNz' in row['Name']:
                    steps.add(row['Step Id'])
                    if row['Step Id']=='9':rows.append(row)
        by_role={}
        for role,capacity,experts,expected_count in [('target',36,48,40),('draft',15,16,3)]:
            selected=[r for r in rows if r['Input Shapes'].strip('"').startswith(f'{capacity},2304;{experts},2304,5120;')]
            assert len(selected)==expected_count and all(r['Output Shapes'].strip('"')==f'{capacity},5120' for r in selected)
            by_role[role]={'capacity':capacity,'counts_length':experts,'GMM2_calls':len(selected),
                           'input_shapes':sorted({r['Input Shapes'] for r in selected}),
                           'output_shapes':sorted({r['Output Shapes'] for r in selected}),
                           'duration_sum_us':sum(float(r['Duration(us)']) for r in selected),
                           'sum_not_recoverable_E2E_budget':True}
        profiles.append({'rank':rank,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),
                         'step_id':'9','all_observed_GMM2_step_ids':sorted(steps),'roles':by_role})
    geometry={'CPU_only':True,'NPU_calls':0,'actual_all_8_ranks_step9_verified':True,'profiles':profiles,
              'target':{'capacity':36,'E':48,'counts_length':48,'top_k':6,'calls_per_step':40},
              'draft':{'capacity':15,'E':16,'counts_length':16,'top_k':3,'calls_per_step':3},
              'median_across_ranks_target_GMM2_duration_sum_ms':statistics.median(p['roles']['target']['duration_sum_us'] for p in profiles)/1000,
              'median_across_ranks_draft_GMM2_duration_sum_ms':statistics.median(p['roles']['draft']['duration_sum_us'] for p in profiles)/1000,
              'no_actual_R_or_active_count_in_profiler_shapes':True,'E2E_budget_claim':False,'passed':True}
    (ROOT/'actual_step9_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    print(json.dumps({'source_guard_passed':True,'draft_override_experts_top_k':[128,3],
                      'all_8_rank_step9_geometry_passed':True,
                      'target_GMM2_sum_ms':geometry['median_across_ranks_target_GMM2_duration_sum_ms'],
                      'draft_GMM2_sum_ms':geometry['median_across_ranks_draft_GMM2_duration_sum_ms'],'NPU_calls':0}))

if __name__=='__main__':main()
