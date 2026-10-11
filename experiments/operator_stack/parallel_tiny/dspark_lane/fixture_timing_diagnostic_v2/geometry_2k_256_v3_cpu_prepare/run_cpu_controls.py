"""Use available CPU PyTorch in the container with backend autoload disabled."""
import base64
import gzip
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent


def main():
    files = {path.name: base64.b64encode(path.read_bytes()).decode() for path in BASE.glob("*.py")}
    files["cpu_cases/v1_actual_warmup_off.json"] = base64.b64encode((BASE / "cpu_cases/v1_actual_warmup_off.json").read_bytes()).decode()
    files["frozen_quality_shared/graph_precision_audit.py"] = base64.b64encode(
        (BASE / "frozen_quality_shared/graph_precision_audit.py").read_bytes()).decode()
    for path in (BASE / "source_origin_readonly").rglob("*.py"):
        files[str(path.relative_to(BASE))] = base64.b64encode(path.read_bytes()).decode()
    packed = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()
    program = r'''
import base64,gzip,json,pathlib,subprocess,os
files=json.loads(gzip.decompress(base64.b64decode(PACKED)))
p=pathlib.Path('/tmp/geometry2k_cpu_'+str(os.getpid()));p.mkdir()
for name,data in files.items():
 f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(base64.b64decode(data))
env=dict(os.environ);env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';env['ASCEND_RT_VISIBLE_DEVICES']='';env['ASCEND_VISIBLE_DEVICES']='';env['GEOMETRY_CPU_NATIVE_BUILDER']='/work/dspark_lane_20261011_c4_fixture_timing_diagnostic_v2/vendor/vllm_ascend/attention/dsa_v1.py'
r=subprocess.run(['python',str(p/'check_geometry_cpu.py')],env=env,text=True,capture_output=True)
if r.returncode:print(json.dumps({'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}));raise SystemExit(r.returncode)
x=json.loads((p/'cpu_gate.json').read_text());x['remote_cpu_container']='dsv41-tiny-goal20-20261010-c6';x['torch_device_backend_autoload']='0';x['visible_devices_empty']=True;print(json.dumps(x))
'''.replace("PACKED", repr(packed))
    remote = "import subprocess\nr=subprocess.run(['docker','exec','-i','dsv41-tiny-goal20-20261010-c6','python','-'],input=" + repr(program) + ",text=True,capture_output=True)\nprint(r.stdout,end='')\nprint(r.stderr,end='')\nraise SystemExit(r.returncode)\n"
    assert len(remote.encode()) < 1_000_000
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21",
                             "python3 -"], input=remote, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout[-8000:])
        print(result.stderr[-3000:])
        raise SystemExit(result.returncode)
    assert len(result.stdout.encode()) < 1_000_000
    gate = json.loads(result.stdout)
    (BASE / "cpu_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({key: value for key, value in gate.items() if key not in ("controls", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
