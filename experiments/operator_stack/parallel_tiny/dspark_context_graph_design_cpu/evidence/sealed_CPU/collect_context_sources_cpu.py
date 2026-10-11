"""Readonly native helper/graph/bank/profile source guards, no torch import."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/work/dspark_context_graph_design_cpu_20261011_v1')
ASCEND=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
RESULTS=Path('/work/results')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def functions(path):
    p=Path(path);text=p.read_text();result={}
    def walk(nodes,prefix=''):
        for n in nodes:
            if isinstance(n,ast.ClassDef):walk(n.body,prefix+n.name+'.')
            elif isinstance(n,ast.FunctionDef):
                source=ast.get_source_segment(text,n)
                result[prefix+n.name]={'first_line':n.lineno,'last_line':n.end_lineno,'source':source,
                                      'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    walk(ast.parse(text).body)
    return result

def main():
    assert os.environ.get('ASCEND_VISIBLE_DEVICES')==os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==''
    ROOT.mkdir(exist_ok=True)
    selections={
        'models/deepseek_v4/dspark.py':['_apply_dsv4_rope','DeepseekV4DSparkModel.combine_hidden_states',
            'DeepseekV4DSparkModel._project_shared_kv','DeepseekV4DSparkModel._store_standard_swa_kv',
            'DeepseekV4DSparkModel.precompute_and_store_context_kv','DeepseekV4DSparkModel.get_draft_kv_cache_layer_names',
            'DSparkDeepseekV4ForCausalLM.combine_hidden_states','DSparkDeepseekV4ForCausalLM.precompute_and_store_context_kv'],
        'models/deepseek_v41/dspark.py':['DeepseekV41DSparkModel.__init__','DeepseekV41DSparkModel.forward'],
        'models/deepseek_v41/model.py':['DeepseekV41DecoderLayer.forward'],
        'models/deepseek_v4/model.py':['DeepseekV4Attention.__init__','DeepseekV4Attention.forward'],
        'ops/dsa.py':['AscendDeepseekSparseAttention.__init__','AscendDeepseekSparseAttention.forward',
                      'dsa_forward','_build_kv_cache','unfold_kvcache'],
        'spec_decode/dspark_proposer.py':['AscendDSparkProposer.__init__','AscendDSparkProposer.initialize_attn_backend',
            'AscendDSparkProposer.set_inputs_first_pass'],
        'spec_decode/dflash_proposer.py':['AscendDflashProposer.build_model_inputs_first_pass'],
        'spec_decode/llm_base_proposer.py':['AscendSpecDecodeBaseProposer._propose','AscendSpecDecodeBaseProposer._run_merged_draft'],
        'attention/dsa_attn_kv_plan.py':['DsaAttnKvPlan.format_dsa_slot_mapping','DsaAttnKvPlan.dsa_kv_compress_scatter','get_dsa_attn_kv_plan'],
        'ops/rope_dsv4.py':['get_cos_and_sin_dsa'],
        'compilation/acl_graph.py':['ACLGraphWrapper.__call__'],
    }
    sources={}
    for relative,names in selections.items():
        path=ASCEND/relative;all_functions=functions(path)
        assert all(n in all_functions for n in names),(relative,set(names)-set(all_functions))
        sources[relative]={'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),
                           'functions':{name:all_functions[name] for name in names},
                           'class_bases':{node.name:[ast.unparse(base) for base in node.bases]
                                          for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,ast.ClassDef)}}
    # The graph wrapper's exact construction site is outside load_model.
    p=ASCEND/'spec_decode/llm_base_proposer.py';lines=p.read_text().splitlines()
    sources['spec_decode/llm_base_proposer.py']['outer_wrapper_snippet']={'first_line':604,'last_line':618,'source':'\n'.join(lines[603:618])}
    banks=[]
    for tag in ['formal_dspark_native_eager_aa_v3','formal_dspark_native_profile_v1']:
        path=RESULTS/tag/'banks.json';rows=json.loads(path.read_text())
        assert len(rows)==8 and sorted(r['rank'] for r in rows)==list(range(8))
        selected=[]
        for row in rows:
            assert not row['draft_use_cuda_graph'] and not row['draft_graph_configured']
            assert row['draft_layer_ids']==[40,41,42] and row['actual_draft_target_layer_ids']==[37,38,39]
            names=row['draft_attn_layer_names'];assert names==[f'mtp.{i}.self_attn.swa_cache' for i in range(3)]
            cache_metadata=row['current_cache_tensor_metadata'];cache=[]
            for i,name in enumerate(names):
                meta=cache_metadata[name][0]
                assert meta['shape'][1:]==[128,1,512] and meta['dtype']=='torch.bfloat16'
                aliases=[k for k,v in cache_metadata.items() if v and k!=name and v[0].get('ptr')==meta['ptr']]
                cache.append({'draft_layer':40+i,'layer_name':name,'cache_metadata':meta,
                              'same_ptr_other_bank_entries':aliases,'actual_gid':None,'actual_context_slot_GM':None,
                              'actual_context_slot_values':None,'live_owner_generation':None,'device_cache_consumer_proof':False})
            selected.append({'rank':row['rank'],'current_draft_wrapper_class':row['current_draft_wrapper_class'],
                             'draft_use_cuda_graph':False,'actual_graph_submission_proof':row['actual_graph_submission_proof'],
                             'cache_metadata_is_device_consumer_proof':row['cache_metadata_is_device_consumer_proof'],'bindings':cache})
        banks.append({'tag':tag,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'ranks':selected})
    profile=[]
    for rank in range(8):
        matches=list((RESULTS/'formal_dspark_native_profile_v1/profile').glob(f'dspark_rank{rank}_*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
        assert len(matches)==1;path=matches[0]
        with path.open(newline='') as f:rows=[r for r in csv.DictReader(f) if r['Step Id']=='9']
        combines=[r for r in rows if '15360' in r['Input Shapes']];assert len(combines)==1
        start=float(combines[0]['Start Time(us)']);selected=[combines[0]]
        scatters=[r for r in rows if float(r['Start Time(us)'])>=start and r['Type']=='ScatterNdUpdateSk' and r['Input Shapes'].strip('"').endswith(';6,2;6,1,512')]
        assert len(scatters)==3;selected+=scatters
        keys=['Name','Type','Start Time(us)','Duration(us)','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Stream ID']
        profile.append({'rank':rank,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),
                        'step_id':'9','selected_shape_records':[{k:r[k] for k in keys} for r in selected],
                        'source_slot_GM_or_gid_attribution':False,'internal_collective_streams_not_proven_by_kernel_stream_column':True})
    config_path=Path('/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq/config.json');cfg=json.loads(config_path.read_text())['text_config']
    assert digest(config_path)=='40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424'
    record={'CPU_only':True,'NPU_calls':0,'NPU_permission':0,'sources':sources,'formal_bank_bindings':banks,'profile_shape_records':profile,
            'geometry':{'observed_context_rows':6,'aux_hidden_width':15360,'combined_hidden_width':5120,
                        'main_proj_local_output_width_TP8':640,'KV_head_dim':512,'nope_dim':448,'rope_dim':64,
                        'serial_MTP_layers':[40,41,42],'target_aux_layers':[37,38,39],'query_tokens_K5':5,
                        'hc_mult':cfg['hc_mult'],'cache_BF16_block_shape':[128,1,512]},
            'capture_finding':{'current_formal_draft_graph_enabled':False,'current_actual_helper_capture_proven':False,
                               'combine_outside_merged_runnable':True,'KV_hook_inside_merged_runnable':True,
                               'future_outer_runnable_capture_would_include_KV_hook':True,'nested_component_graph_forbidden':True},
            'missing':['actual name-to-gid values and per-group logical kernel block sizes',
                       'actual borrowed device/context/aclrtStream identities; profiler Stream ID is not a handle',
                       'current context aux/combined/positions/slot GM descriptors and producer completion ordering',
                       'live owner generations and cache alias allocation/subrange ownership',
                       'query slot/block-table/per-layer metadata GM and consumer owner pins',
                       'native WKV quant method/workspace and RoPE registry/pool lifetime',
                       'TP allgather/internal stream SDK graph support and actual ScatterSk capture/replay',
                       'write-after-cache consumer proof and partial-write rollback or terminal cleanup evidence'],
            'ready_for_NPU_capture':False,'capture_support_claim':False,'performance_claim':None,'passed':True}
    (ROOT/'context_source_guards.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'source_guard_passed':True,'sources':len(sources),'all_8_rank_shape_records':True,
                      'current_formal_helper_capture_proven':False,'combine_outside_KV_inside_merged_runnable':True,
                      'unbound_group_lifetime_fields':True,'NPU_calls':0}))

if __name__=='__main__':main()
