"""Strict general API quality runner; default outputs only a CPU command plan."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

from .contract import false_claims,verify_identity
from .scoring import canonical_gold,check_gsm100,check_vision23


def commands(args,model):
 return {'gsm100':[sys.executable,str(args.client_root/'tests/acc_eval.py'),
  '--task','gsm8k','--limit','100','--mode','chat','--conc','1','--max-tokens','512',
  '--serialize-prefill','1','--base-url',args.base_url,'--model',model,
  '--enc-dir',str(args.encoder),'--gsm8k-data-dir',str(args.gsm_data),
  '--out',str(args.output/'gsm100.json'),'--tag','dspark-quality-'+args.job_id],
  'vision23':[sys.executable,str(args.client_root/'tests/vision_accuracy_check.py'),
   '--server',args.base_url,'--model',model,'--images-dir',str(args.images),
   '--official-dir',str(args.official_dir),'--max-tokens','32','--temperature','0',
   '--min-hit-rate','1.0','--expect-dspark','--out',str(args.output/'vision23.raw.json')]}


def request_json(url):
 with urllib.request.urlopen(url,timeout=30) as response:return json.loads(response.read())


def verify_owned_process(selection):
 pid=selection['pid'];path=Path('/proc')/str(pid)/'cmdline'
 argv=[part.decode() for part in path.read_bytes().split(b'\0') if part]
 module='parallel_tiny.dspark_general_quality_api_cpu.bootstrap'
 if module not in argv or '--job-id' not in argv or argv[argv.index('--job-id')+1]!=selection['job_id']:
  raise ValueError('General quality HTTP process is not the exact owned bootstrap job')
 return {'pid':pid,'argv':argv,'cmdline_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_eval_sources(args):
 expected=json.loads((Path(__file__).resolve().parent/'evidence/evaluator_sources.json').read_text())['files']
 for label,name in [('gsm','acc_eval.py'),('vision','vision_accuracy_check.py')]:
  path=args.client_root/'tests'/name
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[label]['sha256']:
   raise ValueError('The original quality script differs from the frozen scorer source: '+name)


def execute(args,selection,command_plan):
 parsed=urllib.parse.urlsplit(args.base_url)
 if (parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or
  parsed.port!=selection['port'] or parsed.path or parsed.query or parsed.fragment or selection['job_id']!=args.job_id):
  raise ValueError('Use the local exact owned general-quality job URL/identity')
 # An entirely new directory prevents any old A1/candidate checkpoint reuse.
 if args.output.exists():raise FileExistsError('Actual quality output must be new; no old .jsonl resumptions')
 verify_eval_sources(args);gold,splits=canonical_gold(args.gsm_data)
 model=selection['served_model'];identity=request_json(args.base_url+'/private/dspark-quality/identity')
 verify_identity(identity,selection,model)
 catalogue=request_json(args.base_url+'/v1/models')
 if [row['id'] for row in catalogue['data']]!=[model]:raise ValueError('General API model ownership mismatch')
 process=verify_owned_process(selection)
 args.output.mkdir(parents=True)
 report={'CPU_preparation_only':False,'job_id':args.job_id,'served_model':model,
  'candidate_id':selection['candidate_id'],'candidate_fingerprint':selection['candidate_fingerprint'],
  'service_instance_id':identity['service_instance_id'],'identity_before':identity,
  'process':process,'canonical_GSM_split_sha256':splits,'commands':[],**false_claims()}
 report_path=args.output/'strict_quality_receipt.json'
 def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
 save()
 try:
  for label,argv in command_plan.items():
   log=args.output/(label+'.log')
   with log.open('w') as stream:
    proc=subprocess.run(argv,cwd=args.client_root,stdout=stream,stderr=subprocess.STDOUT)
   report['commands'].append({'label':label,'argv':argv,'returncode':proc.returncode,
    'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()});save()
   if proc.returncode!=0:raise ValueError('Original quality client failed: '+label)
  gsm=json.loads((args.output/'gsm100.json').read_text())
  vision=json.loads((args.output/'vision23.raw.json').read_text())
  report['gsm_strict']=check_gsm100(gsm,model,gold,splits)
  report['vision_strict']=check_vision23(vision,model)
  after=request_json(args.base_url+'/private/dspark-quality/identity')
  verify_identity(after,selection,model)
  report['identity_after']=after
  if after['service_instance_id']!=identity['service_instance_id']:raise ValueError('Quality service restarted during evaluation')
  report.update(actual_general_API_quality_observed=True,actual_GSM100_passed=True,
   actual_Vision23_passed=True,actual_client_quality_passed=True,
   performance_receipt_granted=False,formal_receipt_granted=False)
  report['performance_or_meanA_used_for_quality_admission']=False
 except BaseException as exc:
  report.update(false_claims());report['failed']=True;report['error_type']=type(exc).__name__;report['error']=str(exc)
  save();raise
 save();return report


def main(argv=None):
 parser=argparse.ArgumentParser(allow_abbrev=False)
 parser.add_argument('--selection',required=True,type=Path)
 parser.add_argument('--job-id',required=True)
 parser.add_argument('--base-url',required=True)
 parser.add_argument('--client-root',required=True,type=Path)
 parser.add_argument('--encoder',required=True,type=Path)
 parser.add_argument('--images',required=True,type=Path)
 parser.add_argument('--official-dir',required=True,type=Path)
 parser.add_argument('--gsm-data',required=True,type=Path)
 parser.add_argument('--output',required=True,type=Path)
 parser.add_argument('--execute',action='store_true')
 args=parser.parse_args(argv);selection=json.loads(args.selection.read_text());plan=commands(args,selection['served_model'])
 if args.execute:
  result=execute(args,selection,plan);print(json.dumps({k:result[k] for k in false_claims()},indent=2))
 else:
  print(json.dumps({'CPU_preparation_only':True,'commands':plan,'HTTP_requests':0,'NPU_calls':0,
   'actual_output_must_be_new':True,'strict_GSM_gate':'canonical8-shot first100/chat/512/SSE,100/100,empty/errors0',
   'strict_Vision_gate':'original20+3/image/SSE,23/23,rawPASS,empty/errors0,mandatorypass',**false_claims()},indent=2))


if __name__=='__main__':main()
