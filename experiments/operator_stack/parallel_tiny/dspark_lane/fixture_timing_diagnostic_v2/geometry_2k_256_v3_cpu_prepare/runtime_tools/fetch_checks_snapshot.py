"""Fetch only the original small checks file and bounded failure metadata."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parents[1]
JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v3"
REMOTE = r'''
import base64,gzip,hashlib,json,pathlib,re,time
p=pathlib.Path('/home/l00886679/projects/dsv41-tiny-prof-20261009')/job
raw=(p/'results/checks.json').read_bytes()
assert len(raw)<1000000
checks=json.loads(raw)
log=(p/'run.log').read_text(errors='replace')
failures=[line for line in log.splitlines() if re.search(r'Traceback|AssertionError|RuntimeError|ValueError|\bERROR\b|\bFATAL\b',line)]
data={'job':job,'time_unix':time.time(),'terminal':(p/'run.exit').exists(),
 'exit':(p/'run.exit').read_text().strip() if (p/'run.exit').exists() else None,
 'relative_path':'results/checks.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
 'checks_completed':len(checks),'failure_lines':failures[-100:],
 'nonfinite_files':[str(q.relative_to(p)) for q in (p/'results').rglob('*nonfinite*') if q.is_file()],
 'base64':base64.b64encode(raw).decode()}
packed=base64.b64encode(gzip.compress(json.dumps(data).encode())).decode()
assert len(packed)<900000
print(packed)
'''


def main():
    program = "job=" + repr(JOB) + "\n" + REMOTE
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", "python3 -"],
                            input=program, text=True, capture_output=True, check=True)
    size = len(result.stdout.encode())
    assert size < 900000
    data = json.loads(gzip.decompress(base64.b64decode(result.stdout)))
    raw = base64.b64decode(data.pop("base64"))
    assert len(raw) == data["bytes"] and hashlib.sha256(raw).hexdigest() == data["sha256"]
    checks = json.loads(raw)
    data.update({"ssh_packed_bytes": size, "original_compact_bytes_verified": True})
    output = BASE / "runtime_evidence/live_checks_snapshot"
    output.mkdir(parents=True, exist_ok=True)
    stem = "checks_" + str(data["checks_completed"]) + "_OFF_ON"
    path = output / (stem + ".json")
    if path.exists():
        assert path.read_bytes() == raw
    else:
        path.write_bytes(raw)
    receipt = output / (stem + "_receipt.json")
    if not receipt.exists():
        receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
