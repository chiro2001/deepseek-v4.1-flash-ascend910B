"""One fresh CPU-only source-pinned run in the existing formal container."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
STACK=HERE.parents[2]
HOST_ROOT='/home/l00886679/projects/dsv41-real-stack-tp8-20261010'
CONTAINER='dsv41-real-stack-tp8-20261010-a321'


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--fresh-name',required=True)
    args=parser.parse_args();assert re.fullmatch(r'full43_v19_CPU_[a-zA-Z0-9_]+',args.fresh_name)
    paths=list(HERE.glob('*.py'))+[HERE/'consumer_plan.json']+list((HERE/'evidence/native_sources').glob('*'))
    paths += [HERE/'evidence/checkpoint_config_CPU_fixture.json',HERE/'evidence/sdk_source_manifest_static.json']
    bridge=HERE.parent/'formal_consumer_sdk_bridge_v9_cpu'
    paths += [bridge/'freeze.json',bridge/'evidence/actual_pair_protocol.json']
    prepack=STACK/'parallel_tiny/dspark_prepack_audit_cpu'
    frozen=json.loads((prepack/'frozen_files.json').read_text())
    paths += [prepack/name for name in frozen['text_files']]+[prepack/'frozen_files.json',prepack/'delivery.json']
    paths += [STACK/'parallel_tiny/dspark_formal_prep/contract.py']
    inputs={'stack/'+str(p.relative_to(STACK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    bundle=HERE/'evidence/CPU_same_version_source.tgz'
    with tarfile.open(bundle,'w:gz') as archive:
        for path in sorted(set(paths)):
            assert path.stat().st_size < 1_000_000
            archive.add(path,arcname='stack/'+str(path.relative_to(STACK)),recursive=False)
    assert bundle.stat().st_size < 1_000_000,'No SSH transfer >=1MB'
    sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
    host=f'{HOST_ROOT}/offline_tools/{args.fresh_name}';container=f'/work/offline_tools/{args.fresh_name}'
    ssh=['ssh','-o','ControlPath=none','-o','BatchMode=yes','a3-21']
    code='from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True,exist_ok=False)'
    subprocess.run([*ssh,shlex.join(['python3','-c',code,host])],check=True,capture_output=True)
    subprocess.run(['scp','-o','ControlPath=none','-o','BatchMode=yes',str(bundle),'a3-21:'+host+'/source.tgz'],check=True,capture_output=True)
    remote_code='''import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]); bundle=root/'source.tgz'
(root/'runtime/tmp').mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(bundle.read_bytes()).hexdigest()==sys.argv[3]
with tarfile.open(bundle) as archive:
 members=archive.getmembers()
 assert all(m.isfile() and m.size<1_000_000 and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 for member in members:
  path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(member).read())
env={'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','ASCEND_RT_VISIBLE_DEVICES':'','ASCEND_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'',
 'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
 'STACK_SRC_ROOT':sys.argv[2],'PYTHONPATH':sys.argv[2]+'/stack','TMPDIR':sys.argv[2]+'/runtime/tmp',
 'TORCHINDUCTOR_CACHE_DIR':sys.argv[2]+'/runtime/torchinductor'}
command=['docker','exec']
for key,value in env.items():command.extend(['--env',key+'='+value])
command.extend(['dsv41-real-stack-tp8-20261010-a321','python3','-m','parallel_tiny.dspark_formal_audit.full43_consumers_v19.test_contracts_cpu'])
result=subprocess.run(command,capture_output=True,timeout=120)
(root/'stdout.json').write_bytes(result.stdout);(root/'stderr.log').write_bytes(result.stderr)
receipt={'scope':'same-version PyTorch CPU-only full43 diagnostic contracts','command':command,'environment':env,
 'bundle_sha256':sys.argv[3],'exitcode':result.returncode,'stdout_bytes':len(result.stdout),'stderr_bytes':len(result.stderr),
 'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),
 'NPU_calls':0,'device_audit_passed':False,'full_consumer_audit_passed':False,'safe_formal_receipt':False}
(root/'execution.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt));sys.exit(result.returncode)
'''
    process=subprocess.run([*ssh,shlex.join(['python3','-c',remote_code,host,container,sha])],capture_output=True,text=True,timeout=160)
    receipts=HERE/'evidence'
    (receipts/'CPU_same_version_run.log').write_text(process.stdout+process.stderr)
    (receipts/'CPU_same_version_input_manifest.json').write_text(json.dumps({'files':inputs,'bundle_sha256':sha,
        'bundle_bytes':bundle.stat().st_size,'fresh_host_root':host,'fresh_container_root':container,'NPU_calls':0},indent=2)+'\n')
    for remote,local in [('stdout.json','CPU_same_version.json'),('stderr.log','CPU_same_version.log'),('execution.json','CPU_same_version_execution.json')]:
        code='from pathlib import Path; import sys; p=Path(sys.argv[1]); data=p.read_bytes(); assert len(data)<1_000_000; sys.stdout.buffer.write(data)'
        result=subprocess.run([*ssh,shlex.join(['python3','-c',code,host+'/'+remote])],capture_output=True,check=True)
        (receipts/local).write_bytes(result.stdout)
    print(json.dumps({'exitcode':process.returncode,'bundle_bytes':bundle.stat().st_size,'bundle_sha256':sha,
        'fresh_host_root':host,'NPU_calls':0}))
    return process.returncode

if __name__=='__main__':raise SystemExit(main())
