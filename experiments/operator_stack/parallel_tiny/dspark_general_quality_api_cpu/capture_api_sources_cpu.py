"""Capture actual official quality-serving source without SDK/device imports."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/vllm-workspace/vllm/vllm')
SELECT={
 'entrypoints/openai/api_server.py':['build_async_engine_client','build_async_engine_client_from_engine_args','build_app','init_app_state','setup_server','build_and_serve'],
 'entrypoints/openai/cli_args.py':['make_arg_parser','validate_parsed_serve_args'],
 'v1/engine/async_llm.py':['AsyncLLM.from_vllm_config','AsyncLLM.collective_rpc'],
 'engine/arg_utils.py':['AsyncEngineArgs.from_cli_args'],
 'engine/protocol.py':['EngineClient.collective_rpc'],
 'entrypoints/launcher.py':['serve_http'],
 'entrypoints/serve/utils/server_utils.py':['lifespan'],
}


def definitions(tree,prefix=''):
 result={}
 for node in ast.iter_child_nodes(tree):
  if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
   name=prefix+node.name;result[name]=node;result.update(definitions(node,name+'.'))
  else:result.update(definitions(node,prefix))
 return result


def main():
 files=[]
 for relative,names in SELECT.items():
  path=ROOT/relative;data=path.read_bytes();source=data.decode();defs=definitions(ast.parse(source));fragments={}
  for name in names:
   selected=[(q,n) for q,n in defs.items() if q==name or q.endswith('.'+name.rsplit('.',1)[-1])]
   exact=defs.get(name)
   if exact is not None:selected=[(name,exact)]
   if len(selected)!=1:
    fragments[name]={'missing_or_ambiguous':True,'candidates':[q for q,_ in selected]};continue
   qualified,node=selected[0];part=ast.get_source_segment(source,node)
   fragments[qualified]={'source':part,'first_line':node.lineno,'last_line':node.end_lineno,'sha256':hashlib.sha256(part.encode()).hexdigest()}
  files.append({'path':str(path),'relative':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'functions':fragments})
 print(json.dumps({'CPU_readonly_source':True,'SDK_imports':0,'NPU_calls':0,'HTTP_requests':0,'files':files},indent=2))


if __name__=='__main__':main()
