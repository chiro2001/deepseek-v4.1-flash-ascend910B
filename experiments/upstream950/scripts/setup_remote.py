"""Create only this lane's model snapshot, cache, container and service."""
import ast
import json
import subprocess
import tarfile
from pathlib import Path

HOST = "a3-21"
REMOTE = "/home/l00886679/projects/dsv41-tiny-upstream950-20261009"
NAME = "dsv41-tiny-upstream950-20261009-c5"
IMAGE = "local/dsv41-a3-tp8:20261005-1001"
root = Path(__file__).resolve().parents[1]
bundle = root / "results/source.tgz"
paths = sorted((root / "scripts").glob("*.py")) + [root / "scripts/serve_tiny.sh"]
for path in paths:
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=str(path))
with tarfile.open(bundle, "w:gz") as archive:
    for path in paths:
        archive.add(path, arcname=path.name)
assert bundle.stat().st_size < 1_000_000, "Use COS for bundles >= 1 MB"
ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST]
subprocess.run(ssh + ["mkdir", "-p", REMOTE], check=True)
subprocess.run(["scp", "-q", str(bundle), HOST + ":" + REMOTE + "/source.tgz"], check=True)
code = r'''
import json, shutil, subprocess, tarfile
from pathlib import Path
root = Path(REMOTE)
check = subprocess.run(['sudo','-n','docker','container','inspect',NAME],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
assert check.returncode != 0, 'Container already exists; inspect it instead of overwriting'
occupancy = subprocess.run(['sudo','-n','fuser','/dev/davinci5'],capture_output=True,text=True)
assert occupancy.returncode == 1, 'chip5 may be occupied or fuser failed: '+occupancy.stdout+occupancy.stderr
for name in ['scripts','logs','results','cache','cache/vllm','cache/triton',
             'cache/ascend','cache/xdg','cache/inductor']:
    (root/name).mkdir(parents=True,exist_ok=True)
model = root/'model'
if not model.exists():
    shutil.copytree('/home/l00886679/models/out/v41-tiny',model)
with tarfile.open(root/'source.tgz') as archive:
    archive.extractall(root/'scripts',filter='data')
command = ['sudo','-n','docker','run','-d','--name',NAME,'--privileged',
           '--shm-size=16g','--network=bridge','-w','/work']
for src,dst,mode in [
    (str(root),'/work','rw'),(str(model),'/model','ro'),
    ('/usr/local/Ascend/driver','/usr/local/Ascend/driver','ro'),
    ('/usr/local/Ascend/firmware','/usr/local/Ascend/firmware','ro'),
    ('/etc/ascend_install.info','/etc/ascend_install.info','ro'),
    ('/usr/local/bin/npu-smi','/usr/local/bin/npu-smi','ro')]:
    command += ['-v',src+':'+dst+':'+mode]
env = {'ASCEND_RT_VISIBLE_DEVICES':'5','ASCEND_VISIBLE_DEVICES':'5',
       'V41_DUMMY_WO_A_FIX':'1','OMP_NUM_THREADS':'8','TINY_PERF_ARM':'both',
       'VLLM_CACHE_ROOT':'/work/cache/vllm','TRITON_CACHE_DIR':'/work/cache/triton',
       'ASCEND_CACHE_PATH':'/work/cache/ascend','XDG_CACHE_HOME':'/work/cache/xdg',
       'TORCHINDUCTOR_CACHE_DIR':'/work/cache/inductor'}
for key,value in env.items():
    command += ['-e',key+'='+value]
command += ['--entrypoint','/bin/bash',IMAGE,'-lc','sleep infinity']
subprocess.run(command,check=True)
subprocess.run(['sudo','-n','docker','exec',NAME,'bash','-lc',
                'python /work/scripts/environment_probe.py'],check=True)
subprocess.run(['sudo','-n','docker','exec','-d',NAME,'bash','-lc',
               'bash /work/scripts/serve_tiny.sh > /work/logs/serve.log 2>&1'],check=True)
manifest={'host':'a3-21','remote_root':str(root),'container':NAME,'image':IMAGE,
          'physical_chip':5,'excluded_chips':[4,14,15],'port':18973,
          'model_id':'dsv41-tiny-upstream950-20261009','baseline_arm':'both',
          'model_snapshot_source':'/home/l00886679/models/out/v41-tiny',
          'model_mount':'read-only independent copy','network':'bridge'}
(root/'results/launch.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
'''
code = "REMOTE=" + repr(REMOTE) + "\nNAME=" + repr(NAME) + "\nIMAGE=" + repr(IMAGE) + "\n" + code
subprocess.run(ssh + ["python3", "-"], input=code, text=True, check=True)
