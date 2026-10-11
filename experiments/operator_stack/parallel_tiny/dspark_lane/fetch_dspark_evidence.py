"""Bounded terminal/status receipt retrieval; tensors/trace stay remote."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", required=True)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_")
    assert all(c.isalnum() or c == "_" for c in args.job)
    code = r'''
import pathlib,json,base64,gzip,hashlib,subprocess,time
p=pathlib.Path("/home/l00886679/projects/dsv41-tiny-prof-20261009")/job
files={}
selected=["preflight.json","run.exit","done.json","supervisor.log","run.log",
            "results/config.json","results/topology.json","results/native_AA.json",
            "results/requests.json","results/summary.json","results/profile_worker_receipt.json"]
selected.extend(str(f.relative_to(p)) for f in sorted((p/"results").glob("raw_request_*.json")))
for rel in selected:
 f=p/rel
 if not f.exists():continue
 raw=f.read_bytes()
 assert len(raw)<1000000, (rel,len(raw))
 files[rel]={"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"base64":base64.b64encode(raw).decode()}
manifest=p/"vendor_source_manifest.json"
if manifest.exists():
 data=json.loads(manifest.read_text());data.pop("sources",None)
 raw=json.dumps(data,indent=2).encode()
 files["vendor_source_compact.json"]={"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"base64":base64.b64encode(raw).decode()}
status={"job":job,"time_unix":time.time(),"remote_root":str(p),"terminal":(p/"run.exit").exists(),
        "exit":(p/"run.exit").read_text().strip() if (p/"run.exit").exists() else None,
        "container_processes":subprocess.check_output(["docker","top","dsv41-tiny-goal20-20261010-c6","-eo","pid,ppid,comm,args"],text=True),
        "files":files}
smi=subprocess.check_output(["npu-smi","info"],text=True).splitlines()
physical4=[]
for i,line in enumerate(smi):
 cols=[c.strip() for c in line.split("|")]
 if len(cols)>3 and cols[1].split()==["0","4"] and ":" in cols[2]:physical4.extend(smi[max(0,i-1):i+1])
 if len(cols)>4 and cols[1].split()==["2","0"] and cols[2].isdigit():physical4.append(line)
status["physical4_smi_lines"]=physical4
packed=base64.b64encode(gzip.compress(json.dumps(status).encode())).decode();assert len(packed)<1000000
print(packed)
'''
    remote = "python3 - <<'PY'\njob=" + repr(args.job) + "\n" + code + "\nPY"
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            capture_output=True, text=True, check=True)
    assert len(result.stdout.encode()) < 1_000_000
    data = json.loads(gzip.decompress(base64.b64decode(result.stdout)))
    if args.terminal:
        assert data["terminal"], "Keep waiting; this job is not terminal"
    base = Path(__file__).resolve().parent
    runtime = base.parents[1] / "runtime/evidence/dspark_lane" / args.job
    runtime.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel, entry in data.pop("files").items():
        raw = base64.b64decode(entry.pop("base64"))
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        dest = runtime / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        manifest.append({"relative_path": rel, **entry})
    data.update({"ssh_packed_bytes": len(result.stdout.encode()), "files": manifest})
    (base / "evidence" / (args.job + "_fetch.json")).write_text(json.dumps(data,indent=2)+"\n")
    summary = runtime / "results/summary.json"
    if summary.exists():
        (base / "evidence" / (args.job + "_summary.json")).write_bytes(summary.read_bytes())
    printed = {k: v for k, v in data.items() if k != "files"}
    printed["artifact_count"] = len(manifest)
    printed["artifact_total_bytes"] = sum(r["bytes"] for r in manifest)
    printed["has_summary"] = summary.exists()
    print(json.dumps(printed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
