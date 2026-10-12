"""Candidate/quality-job identity and configuration guards; stdlib only."""
import copy
import hashlib
import json
from pathlib import Path
import re

SAFE_ID=re.compile(r'^[A-Za-z0-9_-]{1,96}$')
COUNTER_SHA='da845a5891557f21e80eac06d96aec1955263d1ae93afbc4d84e86dd9585dbec'
# [async arm] Default gates keep the frozen synchronous v4 counter/worker
# bytes. With ``DSPARK_QUALITY_ASYNC_ARM=1`` the same gates apply to the
# async-gated counter/worker pair the fastest served configuration uses; the
# candidate must prove the matching scheduler arm and environment either way.
ASYNC_COUNTER_SHA='4877900b492640e292d9daa9755d8699767deeae7eec17a8c2edcf85659b8e8c'
ASYNC_ARM_ENV='DSPARK_QUALITY_ASYNC_ARM'
JOB_ENV={'STACK_DSPARK_FORMAL_OUTPUT','VLLM_CACHE_ROOT'}


def async_arm():
 import os
 return os.environ.get(ASYNC_ARM_ENV,'0')=='1'


def counter_sha():
 return ASYNC_COUNTER_SHA if async_arm() else COUNTER_SHA


def digest(data):return hashlib.sha256(data).hexdigest()


def fingerprint(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def false_claims():
 return {'actual_general_API_quality_observed':False,'actual_GSM100_passed':False,
  'actual_Vision23_passed':False,'actual_client_quality_passed':False,
  'actual_device_consumer_proof':False,'performance_receipt_granted':False,
  'formal_receipt_granted':False}


def validate_candidate(candidate):
 required={'schema','candidate_id','source_root','source_sha256','model_kwargs','checkpoint_config',
  'sdk_source_manifest_sha256','sdk_pair_provenance','audit_enabled','startup_env'}
 if not required.issubset(candidate) or not candidate['schema'] or not SAFE_ID.fullmatch(candidate['candidate_id']):
  raise ValueError('Candidate identity/source/config contract is incomplete')
 if candidate['audit_enabled'] is not False:
  raise ValueError('Quality must use the audit-disabled view of the same candidate')
 if not candidate['source_sha256'] or not candidate['sdk_source_manifest_sha256'] or not candidate['sdk_pair_provenance']:
  raise ValueError('Candidate SDK/source provenance is missing')
 for path,sha in candidate['source_sha256'].items():
  if Path(path).is_absolute() or '..' in Path(path).parts or not re.fullmatch('[0-9a-f]{64}',sha):
   raise ValueError('Candidate source manifest contains an invalid relative path/SHA')
 kw=candidate['model_kwargs'];spec=kw.get('speculative_config')
 if type(spec) is not dict:raise ValueError('Original DSpark speculative config is required')
 expected={'tensor_parallel_size':8,'enable_expert_parallel':True,
  'enable_prefix_caching':False,'max_num_seqs':1,'dtype':'bfloat16'}
 if any(kw.get(k)!=v for k,v in expected.items()):raise ValueError('Candidate is outside original strict TP8 quality config')
 if kw.get('async_scheduling') is not async_arm():
  raise ValueError('Candidate scheduler arm differs from the quality async gate')
 if (spec.get('method')!='dspark' or spec.get('num_speculative_tokens')!=5 or
  kw.get('load_format')=='dummy' or kw.get('enable_return_routed_experts',False) or
  kw.get('model')!=kw.get('tokenizer') or not kw.get('worker_cls')):
  raise ValueError('Actual DSpark5/real weights/whole checkpoint/tokenizer/worker binding is required')
 env=candidate['startup_env']
 expected_env={'HCCL_DETERMINISTIC':'strict','STACK_FP32_DECODE_REDUCTION':'0',
  'STACK_REAL_AUDIT':'0','STACK_REAL_WEIGHTS':'1','STACK_PHYSICAL_CHIPS':'8,9,10,11,12,13,14,15'}
 if any(env.get(k)!=v for k,v in expected_env.items()):raise ValueError('Candidate strict/native environment differs')
 if async_arm() and (env.get('DSPARK_ASYNC_COUNTER')!='1' or env.get('DSPARK_ASYNC_SCHEDULING')!='1'):
  raise ValueError('Async quality arm requires the deferred async scheduler environment')
 if candidate.get('startup_env_fingerprint') not in (None,fingerprint(env)):
  raise ValueError('Candidate startup environment fingerprint changed')
 cfg=candidate['checkpoint_config']
 if not cfg.get('path') or not re.fullmatch('[0-9a-f]{64}',cfg.get('sha256','')):
  raise ValueError('Exact whole-checkpoint config source is required')
 return candidate


def make_plan(candidate,job_id,port,output,counter_file):
 validate_candidate(candidate)
 if not SAFE_ID.fullmatch(job_id) or type(port) is not int or not 1024<=port<=65535:
  raise ValueError('Explicit independent quality job/port identity is required')
 output=Path(output).resolve();model='dsv41-dspark-quality-'+job_id
 original=copy.deepcopy(candidate['model_kwargs']);kw=copy.deepcopy(original)
 base=kw['worker_cls']
 if 'dspark_general_quality_api_cpu' in base:raise ValueError('Quality worker cannot inherit itself')
 kw['worker_cls']='parallel_tiny.dspark_general_quality_api_cpu.worker_adapter.GeneralQualityDSparkWorker'
 env=copy.deepcopy(candidate['startup_env'])
 env.update(STACK_DSPARK_FORMAL_OUTPUT=str(output/'native'),VLLM_CACHE_ROOT=str(output/'cache'),
  DSPARK_QUALITY_BASE_WORKER=base,DSPARK_API_FROZEN_COUNTER=str(Path(counter_file).resolve()),
  DSPARK_QUALITY_CANDIDATE_CONTRACT=str(output/'candidate_contract.json'))
 comparable=lambda value:{k:v for k,v in value.items() if k not in JOB_ENV}
 assert comparable(candidate['startup_env'])==comparable({k:v for k,v in env.items() if k in candidate['startup_env']})
 return {'schema':'dspark_general_quality_plan_v1','CPU_preparation_only':True,
  'job_id':job_id,'served_model':model,'host':'127.0.0.1','port':port,'output':str(output),
  'candidate_id':candidate['candidate_id'],'candidate_fingerprint':fingerprint(candidate),
  'candidate_source_root':candidate['source_root'],'candidate_source_sha256':candidate['source_sha256'],
  'sdk_source_manifest_sha256':candidate['sdk_source_manifest_sha256'],
  'checkpoint_config':candidate['checkpoint_config'],'original_model_kwargs':original,'model_kwargs':kw,
  'original_model_kwargs_fingerprint':fingerprint(original),'environment':env,
  'same_candidate_environment_except_job_paths_and_quality_description':True,
  'native_AsyncLLM':True,'private_armed_window_adapter':False,'private_route_whitelist':False,
  'counter_algorithm_sha256':counter_sha(),'v4_counter_window_must_remain_inactive':True,
  'async_scheduler_arm':async_arm(),
  'service_started':False,'NPU_calls':0,'HTTP_inference_requests':0,**false_claims()}


def verify_source_files(candidate):
 validate_candidate(candidate);root=Path(candidate['source_root']).resolve()
 for relative,sha in candidate['source_sha256'].items():
  path=(root/relative).resolve()
  if not path.is_relative_to(root) or digest(path.read_bytes())!=sha:raise ValueError('Candidate source differs: '+relative)
 cfg=Path(candidate['checkpoint_config']['path'])
 if digest(cfg.read_bytes())!=candidate['checkpoint_config']['sha256']:raise ValueError('Checkpoint config changed')
 if cfg.resolve()!=Path(candidate['model_kwargs']['model']).joinpath('config.json').resolve():
  raise ValueError('Checkpoint config is not the candidate original model owner')
 config=json.loads(cfg.read_text());text=config.get('text_config',config)
 if (text.get('num_hidden_layers')!=40 or text.get('hidden_size')!=5120 or
  text.get('n_routed_experts')!=384 or text.get('num_experts_per_tok')!=6 or
  text.get('dspark_block_size')!=5 or not config.get('vision_config')):
  raise ValueError('Original full40/W4A8/DSpark/vision config is incomplete')
 return {'candidate_sources_verified':True,'checkpoint_config_verified':True,
  'full_weight_contents_hashed':False,**false_claims()}


def verify_identity(identity,selection,expected_model):
 keys=('job_id','candidate_id','candidate_fingerprint','original_model_kwargs_fingerprint','sdk_source_manifest_sha256','service_instance_id')
 if (identity.get('served_model')!=expected_model or selection.get('served_model')!=expected_model or
  selection.get('service_started') is not True or identity.get('native_AsyncLLM') is not True or
  identity.get('private_armed_window_adapter') is not False or identity.get('private_route_whitelist') is not False or
  identity.get('audit') is not False or identity.get('profiler')!='OFF'):
  raise ValueError('This is not the owned native general quality API')
 for key in keys:
  if identity.get(key)!=selection.get(key) or identity.get(key) is None:raise ValueError('Quality identity changed: '+key)
 verify_rank_descriptions(identity.get('rank_descriptions',[]))
 return True


def verify_rank_descriptions(ranks):
 if len(ranks)!=8 or {r.get('rank') for r in ranks}!=set(range(8)):raise ValueError('Actual TP8 worker identity is incomplete')
 for row in ranks:
  if (row.get('counter_sha256')!=counter_sha() or row.get('counter_active') is not False or
   row.get('audit') is not False or row.get('speculative_method')!='dspark' or row.get('num_speculative_tokens')!=5):
   raise ValueError('General quality requires original DSpark workers with inactive counter windows')
  if row.get('scheduler_async',False) is not async_arm():
   raise ValueError('Worker scheduler arm differs from the quality async gate')
 return True
