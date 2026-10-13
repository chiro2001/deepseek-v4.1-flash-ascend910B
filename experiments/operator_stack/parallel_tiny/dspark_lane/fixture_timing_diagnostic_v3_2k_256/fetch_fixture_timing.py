"""Bounded read-only compact fetch and physical4 occupancy receipts for the 2048/256 timing job."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent

REMOTE = r'''
import base64,gzip,hashlib,json,pathlib,subprocess,time
p=pathlib.Path('/home/l00886679/projects/dsv41-tiny-prof-20261009')/job
assert p.is_dir()
smi=subprocess.check_output(['npu-smi','info'],text=True).splitlines()
lines=[];process_rows=[]
for i,line in enumerate(smi):
 cols=[c.strip() for c in line.split('|')]
 if len(cols)>3 and cols[1].split()==['0','4'] and ':' in cols[2]:lines.extend(smi[max(0,i-1):i+1])
 if len(cols)>4 and cols[1].split()==['2','0'] and cols[2].isdigit():process_rows.append(line)
requests=json.loads((p/'results/requests.json').read_text()) if (p/'results/requests.json').exists() else []
result={'job':job,'time_unix':time.time(),'terminal':(p/'run.exit').exists(),
 'exit':(p/'run.exit').read_text().strip() if (p/'run.exit').exists() else None,
 'requests_completed':len(requests),'last_request_tag':requests[-1]['tag'] if requests else None,
 'physical4_smi_lines':lines,'physical4_process_rows':process_rows,
 'container_processes':subprocess.check_output(['docker','top','dsv41-tiny-goal20-20261010-c6','-eo','pid,ppid,comm,args'],text=True)}
files={}
if not status_only:
 selected=[p/name for name in ('preflight.json','run.log','run.exit','done.json','supervisor.log')]
 selected+=sorted((p/'results').glob('*.json'))
 selected+=[p/'results/cache_gate'/name for name in ('fixed_o_proj_cache_fixture.json','fixture_validation_timing_after_capture.json')]
 for path in selected:
  if not path.is_file():continue
  raw=path.read_bytes()
  if path.name=='run.log' and len(raw)>=900000:raw=raw[-900000:]
  assert len(raw)<1000000,(str(path),len(raw))
  files[str(path.relative_to(p))]={'base64':base64.b64encode(raw).decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 result['files']=files
packed=base64.b64encode(gzip.compress(json.dumps(result).encode())).decode()
assert len(packed)<1000000
print(packed)
'''


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", default="dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_fixture_timing_") and all(char.isalnum() or char == "_" for char in args.job)
    remote = "python3 - <<'PY'\njob=" + repr(args.job) + "\nstatus_only=" + repr(args.status_only) + "\n" + REMOTE + "\nPY"
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            text=True, capture_output=True, check=True)
    assert len(result.stdout.encode()) < 1_000_000
    data = json.loads(gzip.decompress(base64.b64decode(result.stdout)))
    target = BASE / "runtime" / args.job
    manifest = []
    for name, info in data.pop("files", {}).items():
        raw = base64.b64decode(info.pop("base64"))
        assert len(raw) == info["bytes"] and hashlib.sha256(raw).hexdigest() == info["sha256"]
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest.append({"relative_path": name, **info})
    data.update({"files": manifest, "ssh_packed_bytes": len(result.stdout.encode())})
    suffix = "status" if args.status_only else "fetch"
    (BASE / "evidence" / (args.job + "_" + suffix + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({key: value for key, value in data.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
