"""Run one owned CPU-only case in the existing formal container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

CONTAINER='dsv41-real-stack-tp8-20261010-a321'
HOST_ROOT=Path('/home/l00886679/projects/dsv41-real-stack-tp8-20261010/offline_tools/initialized_sdk_v8_cpu_keep_owner')
CONTAINER_ROOT='/work/offline_tools/initialized_sdk_v8_cpu_keep_owner'
EXPECTED_HARNESS_SHA='5c967fe0d4330a2eb56f91488f80cad82e44efc81db4e80a2f8677aaa84edfd1'
parser=argparse.ArgumentParser()
args=parser.parse_args()
harness=HOST_ROOT/'in_memory_cleanup_harness.py'
assert hashlib.sha256(harness.read_bytes()).hexdigest()==EXPECTED_HARNESS_SHA
out=HOST_ROOT/'cleanup_stdout.json'
err=HOST_ROOT/'cleanup_stderr.txt'
run_receipt=HOST_ROOT/'cleanup_execution.json'
assert not any(p.exists() for p in [out,err,run_receipt])
env={'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','ASCEND_RT_VISIBLE_DEVICES':'','ASCEND_VISIBLE_DEVICES':'',
     'CUDA_VISIBLE_DEVICES':'','CPU_OUTER_KEEP':'0',
     'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
command=['docker','exec']
for name,value in env.items():command+=['--env',name+'='+value]
command+=[CONTAINER,'python3',CONTAINER_ROOT+'/in_memory_cleanup_harness.py']
result=subprocess.run(command,capture_output=True,timeout=180)
out.write_bytes(result.stdout);err.write_bytes(result.stderr)
receipt={'scope':'owned CPU-only actual Linux child cleanup case execution',
         'container':CONTAINER,'harness_sha256':EXPECTED_HARNESS_SHA,'exitcode':result.returncode,
         'environment':env,'stdout_bytes':len(result.stdout),'stderr_bytes':len(result.stderr),
         'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
         'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),
         'npu_jobs':0,'device_audit_passed':False,'safe_formal_receipt':False}
run_receipt.write_text(json.dumps(receipt,indent=2)+'\n')
summary={k:receipt[k] for k in ['exitcode','stdout_bytes','stderr_bytes','stdout_sha256','stderr_sha256','npu_jobs']}
if result.returncode==0:
 r=json.loads(result.stdout)
 summary.update({k:r[k] for k in ['tests','passed','skipped','normal_success_grace_seconds','failure_uses_success_grace_seconds']})
 summary['delayed_child_exitcodes']=r['delayed_normal_child_proof']['report']['worker_exit_codes']
 summary['delayed_child_elapsed_seconds']=r['delayed_normal_child_proof']['elapsed_seconds']
else:summary['stderr_tail']=result.stderr.decode(errors='replace')[-2000:]
print(json.dumps(summary,indent=2))
raise SystemExit(result.returncode)
