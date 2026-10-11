"""Read-only formal DSpark checkpoint and launch contract, stdlib only."""
import hashlib
import json
from pathlib import Path
import re


PHYSICAL_CHIPS = '8,9,10,11,12,13,14,15'
TARGET_MS = 20.0
GRAPH_ENV = {
    'DRAFT_GRAPH': '1', 'DSPARK_GRAPH_CAPTURE_METADATA': '1',
    'DSPARK_DRAFT_METADATA_MODE': 'sync', 'DSPARK_CAPTURE_VALUE_FIX': '1',
    'DSPARK_SWA_INDICES_RESIDENT': '1', 'DSPARK_CAPTURE_NCTX_FIX': '1',
    'DSPARK_DISPATCH_QUERY_LEN_FIX': '1', 'DSPARK_DRAFT_USE_CUDAGRAPH': '1',
    'DSPARK_HOIST_CONTEXT_KV': '0'}


def inspect_dspark_checkpoint(model, k):
    from formal_model_contract import inspect_checkpoint
    formal = inspect_checkpoint(model)
    root = Path(model)
    config = json.loads((root/'config.json').read_text())
    text = config.get('text_config', config)
    if text.get('num_nextn_predict_layers') != 3 or text.get('dspark_target_layer_ids') != [37,38,39]:
        raise ValueError('Formal DSpark requires the native three MTP layers and target [37,38,39]')
    block = text.get('dspark_block_size')
    if type(block) is not int or block <= 0 or k < block:
        raise ValueError(f'SDK DSpark K must be >= checkpoint block size {block}; got {k}')
    if not isinstance(config.get('vision_config'), dict) or not config['vision_config']:
        raise ValueError('Full formal checkpoint must retain its vision configuration')
    index = json.loads((root/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    mtp = {name:shard for name,shard in index.items() if name.startswith('mtp.')}
    layers = sorted({int(re.match(r'mtp\.(\d+)\.',name)[1]) for name in mtp})
    if layers != [0,1,2]:
        raise ValueError(f'Incomplete checkpoint MTP layers: {layers}')
    required = ['mtp.0.main_proj.weight', 'mtp.0.attn.wq_a.weight',
                'mtp.0.ffn.experts.0.w1.weight']
    if any(name not in mtp for name in required):
        raise ValueError('Native DSpark projection/attention/expert weights are missing')
    description = root/'quant_model_description.json'
    desc = json.loads(description.read_text())
    formal['dspark'] = {'method':'dspark','num_nextn_predict_layers':3,
        'target_layer_ids':[37,38,39],'trained_block_size':block,'num_speculative_tokens':k,
        'mtp_tensor_count':len(mtp),'mtp_layers':layers,
        'mtp_weight_map_sha256':hashlib.sha256(json.dumps(mtp,sort_keys=True).encode()).hexdigest(),
        'quant_description_sha256':hashlib.sha256(description.read_bytes()).hexdigest(),
        'selected_quant_types':{name:desc.get(name) for name in required},
        'draft_checkpoint':'native target checkpoint; stock method=dspark auto-selects it',
        'draft_weight_content_hashed':False,'vision_config_preserved':True,
        'runtime_draft_load_verified':False}
    return formal


def model_kwargs(model, k, *, audit, draft_graph):
    return {'model':model,'tokenizer':model,'load_format':'auto','dtype':'bfloat16',
        'safetensors_load_strategy':'lazy',
        'model_loader_extra_config':{'enable_multithread_load':True,'num_threads':128},
        'worker_cls':'parallel_tiny.dspark_formal_prep.worker.FormalDSparkWorker',
        'tensor_parallel_size':8,'distributed_executor_backend':'mp',
        'enable_expert_parallel':True,'seed':0,'trust_remote_code':True,
        'async_scheduling':False,'limit_mm_per_prompt':{'image':4},
        'max_model_len':8192,'max_num_seqs':1,'max_num_batched_tokens':2048,
        'gpu_memory_utilization':.70,'kv_cache_memory_bytes':4*1024**3,
        'block_size':128,'enable_prefix_caching':False,
        'enable_return_routed_experts':audit,
        'speculative_config':{'method':'dspark','num_speculative_tokens':k,
                              'enforce_eager':not draft_graph},
        'compilation_config':{'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1+k]},
        'additional_config':{'enable_engram':True,'engram_storage':'int8','enable_cpu_binding':False,
            'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':True},
            'multistream_dsv4_dsa_overlap':False}}


def startup_env(output, *, audit, draft_graph):
    env = {'VLLM_ENABLE_V1_MULTIPROCESSING':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
        'VLLM_ALLOW_INSECURE_SERIALIZATION':'1','VLLM_DISABLE_COMPILE_CACHE':'1',
        'VLLM_CACHE_ROOT':str(Path(output)/'cache'),'HCCL_DETERMINISTIC':'strict',
        'STACK_DSPARK_FORMAL_OUTPUT':str(Path(output).absolute()),
        'LOCAL_WORLD_SIZE':'8','STACK_PHYSICAL_CHIPS':PHYSICAL_CHIPS,
        'STACK_REAL_WEIGHTS':'1','STACK_REAL_AUDIT':'1' if audit else '0',
        'STACK_TP8_ARM':'tp8base','STACK_FP32_DECODE_REDUCTION':'0',
        'STACK_TINY_PROFILE':'0','STACK_W4A8_PREFIX_ENABLED':'0',
        'STACK_WOA_CUBE_ENABLED':'0','STACK_QKV_MERGE_ENABLED':'0',
        'STACK_QKV_TAIL_SCHEDULE':'native','STACK_MOE_MASK_ENABLED':'0',
        'STACK_METADATA_MANY_SLOTS_ENABLED':'0',
        'OPT_BLOCKMAP_VERIFY':'0','DRAFT_FORCE_EAGER':'0',
        'V41_ENGRAM_WITH_DUMMY':'0','V41_DUMMY_WO_A_FIX':'0'}
    env.update(GRAPH_ENV)
    if not draft_graph:
        env.update(DRAFT_GRAPH='0',DSPARK_DRAFT_USE_CUDAGRAPH='0')
    return env


def validate_hot_flags(text):
    flags = {}
    for line in text.splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key,value=line.split('=',1);flags[key.strip()]=value.strip()
    if flags.get('DRAFT_FORCE_EAGER','0') != '0':
        raise ValueError('Existing runtime hot flag forces eager draft')
    return flags
