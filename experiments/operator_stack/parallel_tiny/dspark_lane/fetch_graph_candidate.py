"""Bounded graph/native diagnostics; tensor exports and large traces stay remote."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--job", required=True)
    args = p.parse_args()
    assert args.job.startswith(("dspark_lane_20261011_c4_graph_", "dspark_lane_20261011_c4_native_diag_"))
    assert all(ch.isalnum() or ch == "_" for ch in args.job)
    source = r'''
from pathlib import Path
import json,base64,gzip,hashlib,subprocess,time
p=Path('/home/l00886679/projects/dsv41-tiny-prof-20261009')/job
selected=[p/n for n in ('preflight.json','run.exit','done.json','run.log','supervisor.log')]
selected.extend(sorted((p/'results').glob('*.json')))
selected.extend(sorted((p/'results/audit').glob('*.json')))
files={}
for f in selected:
 if not f.exists():continue
 raw=f.read_bytes();assert len(raw)<1000000,(str(f),len(raw))
 files[str(f.relative_to(p))]={'base64':base64.b64encode(raw).decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
smi=subprocess.check_output(['npu-smi','info'],text=True).splitlines();lines=[]
for i,line in enumerate(smi):
 cols=[c.strip() for c in line.split('|')]
 if len(cols)>3 and cols[1].split()==['0','4'] and ':' in cols[2]:lines.extend(smi[max(0,i-1):i+1])
 if len(cols)>4 and cols[1].split()==['2','0'] and cols[2].isdigit():lines.append(line)
result={'job':job,'time_unix':time.time(),'terminal':(p/'run.exit').exists(),'exit':(p/'run.exit').read_text().strip() if (p/'run.exit').exists() else None,'physical4_smi_lines':lines,'container_processes':subprocess.check_output(['docker','top','dsv41-tiny-goal20-20261010-c6','-eo','pid,ppid,comm,args'],text=True),'files':files}
packed=base64.b64encode(gzip.compress(json.dumps(result).encode())).decode();assert len(packed)<1000000;print(packed)
'''
    remote = "python3 - <<'PY'\njob=" + repr(args.job) + "\n" + source + "\nPY"
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            text=True, capture_output=True, check=True)
    assert len(result.stdout.encode()) < 1_000_000
    data = json.loads(gzip.decompress(base64.b64decode(result.stdout)))
    base = Path(__file__).resolve().parent
    target = base.parents[1] / "runtime/evidence/dspark_lane" / args.job
    manifest = []
    for name, item in data.pop("files").items():
        raw = base64.b64decode(item.pop("base64"))
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest.append({"relative_path": name, **item})
    data.update({"files": manifest, "ssh_packed_bytes": len(result.stdout.encode())})
    (base / "evidence" / (args.job + "_fetch.json")).write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k: v for k, v in data.items() if k != "files"},indent=2))


if __name__ == "__main__":
    main()
