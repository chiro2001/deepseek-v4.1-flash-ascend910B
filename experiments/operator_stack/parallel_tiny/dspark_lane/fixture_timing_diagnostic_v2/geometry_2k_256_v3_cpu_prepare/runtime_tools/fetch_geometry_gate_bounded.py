"""Read-only terminal compact collector; every SSH response stays below 1 MB."""
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
import base64,gzip,hashlib,json,pathlib,re,subprocess,time
p=pathlib.Path('/home/l00886679/projects/dsv41-tiny-prof-20261009')/job
assert p.is_dir()
smi=subprocess.check_output(['npu-smi','info'],text=True).splitlines();lines=[];process_rows=[]
for i,line in enumerate(smi):
 cols=[c.strip() for c in line.split('|')]
 if len(cols)>3 and cols[1].split()==['0','4'] and ':' in cols[2]:lines.extend(smi[max(0,i-1):i+1])
 if len(cols)>4 and cols[1].split()==['2','0'] and cols[2].isdigit():process_rows.append(line)
records=[];partial=False
if (p/'results/requests.json').exists():
 try:records=json.loads((p/'results/requests.json').read_text())
 except json.JSONDecodeError:partial=True
checks=[];partial_checks=False
if (p/'results/checks.json').exists():
 try:checks=json.loads((p/'results/checks.json').read_text())
 except json.JSONDecodeError:partial_checks=True
log=(p/'run.log').read_text(errors='replace') if (p/'run.log').exists() else ''
failure_lines=[line for line in log.splitlines()
               if re.search(r'Traceback|AssertionError|RuntimeError|ValueError|\bERROR\b|\bFATAL\b',line)]
result={'job':job,'time_unix':time.time(),'terminal':(p/'run.exit').exists(),
 'exit':(p/'run.exit').read_text().strip() if (p/'run.exit').exists() else None,
 'requests_completed':len(records),'last_request_tag':records[-1]['tag'] if records else None,
 'partial_requests_json':partial,'physical4_smi_lines':lines,'physical4_process_rows':process_rows,
 'numeric_checks_completed':len(checks),'partial_checks_json':partial_checks,
 'failure_lines':failure_lines[-100:],
 'nonfinite_files':[str(q.relative_to(p)) for q in (p/'results').rglob('*nonfinite*') if q.is_file()],
 'numeric_checks':[{'kind':c['kind'],'passed':c['passed'],'max_logprob_delta':c['max_logprob_delta'],
                    'left':c['actual_consumers']['left'],'right':c['actual_consumers']['right']} for c in checks],
 'geometry_step_files':len(list((p/'results/audit').glob('geometry2k_*_step_*.json'))),
 'container_processes':subprocess.check_output(['docker','top','dsv41-tiny-goal20-20261010-c6','-eo','pid,ppid,comm,args'],text=True)}
def compact_raw(path):
 raw=path.read_bytes()
 truncated=path.name=='run.log' and len(raw)>=900000
 return (raw[-900000:] if truncated else raw),truncated
if mode!='status':
 assert result['terminal'],'Only status polling is allowed before terminal'
 selected=[p/name for name in ('preflight.json','run.log','run.exit','done.json','supervisor.log')]
 selected+=sorted((p/'results').glob('*.json'))
 selected+=[q for q in sorted((p/'results/audit').glob('*.json'))
            if not q.name.startswith('geometry2k_') or q.name=='geometry2k_snapshot_entry_host_guard.json']
 allowed={str(q.relative_to(p)):q for q in selected if q.is_file()}
 if mode=='manifest':
  files=[];skipped=[]
  for name,path in allowed.items():
   raw,truncated=compact_raw(path);row={'relative_path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   if truncated:row['run_log_tail_truncated_for_compact']=True
   (skipped if len(raw)>=1000000 else files).append(row)
  result.update({'files':files,'skipped_large_compact_files':skipped})
 else:
  assert mode=='parts' and parts and sum(x['length'] for x in parts)<=450000
  chunks=[]
  for part in parts:
   name=part['relative_path'];assert name in allowed
   raw,truncated=compact_raw(allowed[name]);assert len(raw)==part['file_bytes'] and hashlib.sha256(raw).hexdigest()==part['file_sha256']
   start,end=part['offset'],part['offset']+part['length'];assert 0<=start<=end<=len(raw)
   data=raw[start:end];chunks.append({'relative_path':name,'offset':start,'bytes':len(data),
    'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
  result['chunks']=chunks
packed=base64.b64encode(gzip.compress(json.dumps(result).encode())).decode()
assert len(packed)<900000
print(packed)
'''


def call(mode, parts=None):
    program = "job=" + repr(JOB) + "\nmode=" + repr(mode) + "\nparts=" + repr(parts or []) + "\n" + REMOTE
    assert len(program.encode()) < 900000
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", "python3 -"],
                            input=program, text=True, capture_output=True, check=True)
    response_bytes = len(result.stdout.encode())
    assert response_bytes < 900000
    return json.loads(gzip.decompress(base64.b64decode(result.stdout))), response_bytes


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    evidence = BASE / "runtime_evidence"
    evidence.mkdir(exist_ok=True)
    if args.status_only:
        data, size = call("status")
        data.update({"files": [], "ssh_packed_bytes": size})
        (evidence / (JOB + "_status.json")).write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data, indent=2))
        return
    data, size = call("manifest")
    manifest = data["files"]
    transfer_sizes = [size]
    batches, batch, batch_bytes = [], [], 0
    for row in manifest:
        # Also split individual compact files; original full-file SHA verifies
        # each part on the remote host and the assembled local original bytes.
        for offset in range(0, row["bytes"], 200000):
            length = min(200000, row["bytes"] - offset)
            if batch_bytes + length > 450000:
                batches.append(batch)
                batch, batch_bytes = [], 0
            batch.append({"relative_path": row["relative_path"], "offset": offset, "length": length,
                          "file_bytes": row["bytes"], "file_sha256": row["sha256"]})
            batch_bytes += length
    if batch:
        batches.append(batch)
    chunks = {row["relative_path"]: {} for row in manifest}
    for batch in batches:
        result, size = call("parts", batch)
        transfer_sizes.append(size)
        assert result["terminal"] and result["exit"] == data["exit"]
        returned = result["chunks"]
        assert len(returned) == len(batch)
        for wanted, got in zip(batch, returned):
            raw = base64.b64decode(got.pop("base64"))
            assert got["relative_path"] == wanted["relative_path"] and got["offset"] == wanted["offset"]
            assert len(raw) == got["bytes"] == wanted["length"]
            assert hashlib.sha256(raw).hexdigest() == got["sha256"]
            assert got["offset"] not in chunks[got["relative_path"]]
            chunks[got["relative_path"]][got["offset"]] = raw
    target = BASE / "runtime" / JOB
    for row in manifest:
        parts = chunks[row["relative_path"]]
        raw = b"".join(parts[offset] for offset in sorted(parts))
        assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
        path = target / row["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    data.update({"ssh_packed_bytes": sum(transfer_sizes), "ssh_response_packed_bytes": transfer_sizes,
                 "largest_ssh_response_packed_bytes": max(transfer_sizes),
                 "all_ssh_responses_below_1MB": True, "original_compact_bytes_verified_after_assembly": True})
    (evidence / (JOB + "_fetch.json")).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({key: value for key, value in data.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
