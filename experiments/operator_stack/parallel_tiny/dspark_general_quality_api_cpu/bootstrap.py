"""Native general OpenAI lifecycle. Default only writes an offline CPU plan."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from .contract import false_claims,fingerprint,make_plan,verify_source_files,verify_rank_descriptions

HERE=Path(__file__).resolve().parent


def verify_official_sources():
 receipt=json.loads((HERE/'evidence/official_api_sources.json').read_text())
 for item in receipt['files']:
  if any(fn.get('missing_or_ambiguous') for fn in item['functions'].values()):raise ValueError('An official serving ABI is not source-guarded')
  if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Official API source changed: '+item['path'])
 return {'source_guard_sha256':hashlib.sha256((HERE/'evidence/official_api_sources.json').read_bytes()).hexdigest(),'guarded_files':len(receipt['files'])}


def identity_from_selection(selection,ranks):
 return {**{k:selection[k] for k in ('job_id','served_model','candidate_id','candidate_fingerprint',
  'original_model_kwargs_fingerprint','sdk_source_manifest_sha256','service_instance_id')},
  'native_AsyncLLM':True,'private_armed_window_adapter':False,'private_route_whitelist':False,
  'audit':False,'profiler':'OFF','rank_descriptions':ranks,**false_claims()}


@asynccontextmanager
async def observed_native_lifespan(app,original,selection,write):
 # The original FastAPI lifecycle still owns startup/GC/log tasks/shutdown.
 async with original(app):
  selection['ASGI_startup_completed']=True
  selection['service_started']=True
  selection['HTTP_readiness_verified']=False
  write(selection)
  try:yield
  finally:
   selection['service_started']=False;selection['ASGI_shutdown_entered']=True;write(selection)


async def execute_service(args,candidate,plan):
 # Only a later root-allocated real quality job invokes this branch.
 verify_source_files(candidate);official_receipt=verify_official_sources()
 from parallel_tiny.dspark_formal_prep.contract import inspect_dspark_checkpoint
 checkpoint=inspect_dspark_checkpoint(plan['original_model_kwargs']['model'],5)
 output=Path(plan['output']);output.mkdir(parents=True,exist_ok=True)
 (output/'native').mkdir(exist_ok=True)
 os.environ.update(plan['environment'])
 sys.path.insert(0,candidate['source_root'])
 if candidate.get('pythonpath'):
  for path in reversed(candidate['pythonpath']):sys.path.insert(0,path)
 from parallel_tiny.dspark_api_counter_bridge_cpu.runtime_guards import load_frozen_counter
 load_frozen_counter(args.counter_file)
 from vllm.entrypoints.openai import api_server as official
 from vllm.entrypoints.openai.cli_args import make_arg_parser,validate_parsed_serve_args
 from vllm.utils.argparse_utils import FlexibleArgumentParser
 parser=make_arg_parser(FlexibleArgumentParser())
 serving=parser.parse_args(['--model',plan['model_kwargs']['model'],
  '--served-model-name',plan['served_model'],'--host','127.0.0.1','--port',str(plan['port']),
  '--tokenizer-mode','deepseek_v41','--disable-log-stats',
  '--reasoning-parser','deepseek_v41','--tool-call-parser','deepseek_v41',
  '--enable-auto-tool-choice','--default-chat-template-kwargs',json.dumps({'enable_thinking':False})])
 for key,value in plan['model_kwargs'].items():setattr(serving,key,value)
 serving.enable_log_requests=False;serving.middleware=[];serving.api_server_count=1
 validate_parsed_serve_args(serving)
 selection={**plan,'CPU_preparation_only':False,'service_started':False,
  'ASGI_startup_completed':False,'HTTP_readiness_verified':False,'engine_initialized':False,
  'checkpoint':checkpoint,'official_source_receipt':official_receipt,
  'service_instance_id':uuid.uuid4().hex,'pid':os.getpid(),'argv':list(sys.argv),**false_claims()}
 # CPU zero-call counters are not runtime device-work counters.
 selection.pop('NPU_calls');selection.pop('HTTP_inference_requests')
 selection_path=output/'service_selection.json'
 if selection_path.exists():raise FileExistsError('Do not reuse an earlier actual quality selection')
 def write(value):selection_path.write_text(json.dumps(value,indent=2)+'\n')
 write(selection)
 listen,sock=official.setup_server(serving,reuse_port=False)
 try:
  async with official.build_async_engine_client(serving) as engine:
   # This factory hardcodes the ORIGINAL AsyncLLM. No subclass/global patch.
   selection['engine_initialized']=True
   tasks=await engine.get_supported_tasks()
   app=official.build_app(serving,tasks,engine.model_config)
   await official.init_app_state(engine,app.state,serving,tasks)
   ranks=await engine.collective_rpc('dspark_general_quality_v1_describe')
   identity=identity_from_selection(selection,ranks)
   # Verify all eight inactive windows before serving; started stays false here.
   verify_rank_descriptions(ranks)
   selection['rank_descriptions']=ranks;selection['identity']=identity
   write(selection)
   @app.get('/private/dspark-quality/identity')
   async def quality_identity():
    current=await engine.collective_rpc('dspark_general_quality_v1_describe')
    return identity_from_selection(selection,current)
   original=app.router.lifespan_context
   app.router.lifespan_context=lambda actual_app:observed_native_lifespan(actual_app,original,selection,write)
   # Keep original FastAPI routes/state and general completion/chat/vision APIs.
   shutdown_task=await official.serve_http(app,sock=sock,host=serving.host,port=serving.port,
    enable_ssl_refresh=serving.enable_ssl_refresh,log_level=serving.uvicorn_log_level,
    access_log=not serving.disable_uvicorn_access_log,timeout_keep_alive=official.envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
    ssl_keyfile=serving.ssl_keyfile,ssl_certfile=serving.ssl_certfile,
    ssl_ca_certs=serving.ssl_ca_certs,ssl_cert_reqs=serving.ssl_cert_reqs,ssl_ciphers=serving.ssl_ciphers,
    h11_max_incomplete_event_size=serving.h11_max_incomplete_event_size,
    h11_max_header_count=serving.h11_max_header_count)
  await shutdown_task
 finally:
  selection['service_started']=False;write(selection);sock.close()


def main(argv=None):
 parser=argparse.ArgumentParser(allow_abbrev=False)
 parser.add_argument('--candidate',required=True,type=Path)
 parser.add_argument('--job-id',required=True)
 parser.add_argument('--port',required=True,type=int)
 parser.add_argument('--output',required=True,type=Path)
 parser.add_argument('--counter-file',required=True,type=Path)
 parser.add_argument('--execute',action='store_true')
 args=parser.parse_args(argv);candidate=json.loads(args.candidate.read_text())
 plan=make_plan(candidate,args.job_id,args.port,args.output,args.counter_file)
 args.output.mkdir(parents=True,exist_ok=True)
 path=args.output/'CPU_quality_plan.json'
 if path.exists():raise FileExistsError('Use a new independent quality job output directory')
 path.write_text(json.dumps(plan,indent=2)+'\n')
 (args.output/'candidate_contract.json').write_text(json.dumps(candidate,indent=2)+'\n')
 if args.execute:asyncio.run(execute_service(args,candidate,plan))
 else:print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
