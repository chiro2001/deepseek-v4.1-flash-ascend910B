"""Bounded read-only status and compact evidence collector for the frozen window."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parents[1]
JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v3"
REMOTE = r'''
import base64,gzip,hashlib,json,pathlib,subprocess,time
p=pathlib.Path('/home/l00886679/projects/dsv41-tiny-prof-20261009')/job
assert p.is_dir()
smi=subprocess.check_output(['npu-smi','info'],text=True).splitlines();lines=[];process_rows=[]
for i,line in enumerate(smi):
 cols=[c.strip() for c in line.split('|')]
 if len(cols)>3 and cols[1].split()==['0','4'] and ':' in cols[2]:lines.extend(smi[max(0,i-1):i+1])
 if len(cols)>4 and cols[1].split()==['2','0'] and cols[2].isdigit():process_rows.append(line)
records=[];partial_requests_json=False
if (p/'results/requests.json').exists():
 try:records=json.loads((p/'results/requests.json').read_text())
 except json.JSONDecodeError:partial_requests_json=True
result={'job':job,'time_unix':time.time(),'terminal':(p/'run.exit').exists(),
 'exit':(p/'run.exit').read_text().strip() if (p/'run.exit').exists() else None,
 'requests_completed':len(records),'last_request_tag':records[-1]['tag'] if records else None,
 'partial_requests_json':partial_requests_json,'physical4_smi_lines':lines,'physical4_process_rows':process_rows,
 'geometry_step_files':len(list((p/'results/audit').glob('geometry2k_*_step_*.json'))),
 'container_processes':subprocess.check_output(['docker','top','dsv41-tiny-goal20-20261010-c6','-eo','pid,ppid,comm,args'],text=True)}
files={}
if not status_only:
 selected=[p/name for name in ('preflight.json','run.log','run.exit','done.json','supervisor.log')]
 selected+=sorted((p/'results').glob('*.json'))
 # Full per-step geometry remains remote; request files contain the hash of
 # every step and exact prefill/first-decode/terminal selected-slot receipts.
 selected += [path for path in sorted((p/'results/audit').glob('*.json')) if not path.name.startswith('geometry2k_') or path.name=='geometry2k_snapshot_entry_host_guard.json']
 skipped=[]
 for path in selected:
  if not path.is_file():continue
  raw=path.read_bytes()
  if path.name=='run.log' and len(raw)>=900000:raw=raw[-900000:]
  if len(raw)>=1000000:skipped.append({'path':str(path.relative_to(p)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});continue
  files[str(path.relative_to(p))]={'base64':base64.b64encode(raw).decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 result['files']=files;result['skipped_large_compact_files']=skipped
packed=base64.b64encode(gzip.compress(json.dumps(result).encode())).decode();assert len(packed)<1000000
print(packed)
'''


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    remote = "python3 - <<'PY'\njob=" + repr(JOB) + "\nstatus_only=" + repr(args.status_only) + "\n" + REMOTE + "\nPY"
    response = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                              text=True, capture_output=True, check=True)
    assert len(response.stdout.encode()) < 1_000_000
    data = json.loads(gzip.decompress(base64.b64decode(response.stdout)))
    target = BASE / "runtime" / JOB
    manifest = []
    for name, entry in data.pop("files", {}).items():
        raw = base64.b64decode(entry.pop("base64"))
        assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest.append({"relative_path": name, **entry})
    data.update({"files": manifest, "ssh_packed_bytes": len(response.stdout.encode())})
    evidence = BASE / "runtime_evidence"
    evidence.mkdir(exist_ok=True)
    suffix = "status" if args.status_only else "fetch"
    (evidence / (JOB + "_" + suffix + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({key: value for key, value in data.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
