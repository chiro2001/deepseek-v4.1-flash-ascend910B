"""One guarded physical4 diagnostic job; bounded SSH and immutable private source."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

CONTAINER = "dsv41-tiny-goal20-20261010-c6"
REMOTE_ROOT = "/home/l00886679/projects/dsv41-tiny-prof-20261009"
CPU_DISCOVERY = r'''
import base64,gzip,hashlib,json,os,pathlib,re,subprocess,time
container="dsv41-tiny-goal20-20261010-c6"
root=pathlib.Path("/home/l00886679/projects/dsv41-tiny-prof-20261009")
job=payload["job"]
dest=root/job
assert not dest.exists(), "Use a new immutable job destination"
c=json.loads(subprocess.check_output(["docker","inspect",container],text=True))[0]
assert c["State"]["Running"] and c["HostConfig"]["Privileged"]
mounts={m["Destination"]:m for m in c["Mounts"]}
assert mounts["/work"]["Source"]==str(root)
assert mounts["/model"]["Source"]=="/home/l00886679/models/out/v41-tiny" and not mounts["/model"]["RW"]
def fresh_gate():
 top=subprocess.check_output(["docker","top",container,"-eo","pid,ppid,comm,args"],text=True)
 assert len(top.strip().splitlines())==2 and "sleep infinity" in top, "Container already has an active job: "+top
 smi=subprocess.check_output(["npu-smi","info"],text=True)
 health={};processes=[];board=None;board_health=None
 for line in smi.splitlines():
  cols=[s.strip() for s in line.split("|")]
  if len(cols)<5:continue
  tokens=cols[1].split()
  if len(tokens)==2 and tokens[0].isdigit() and tokens[1].startswith("Ascend"):
   board=int(tokens[0]);board_health=cols[2]
  elif len(tokens)==2 and all(t.isdigit() for t in tokens) and board is not None and ":" in cols[2]:
   chip,physical=map(int,tokens);health[physical]={"board":board,"chip":chip,"health":board_health,"line":line}
  elif len(tokens)==2 and all(t.isdigit() for t in tokens) and cols[2].isdigit():
   b,ch=map(int,tokens);processes.append({"physical":2*b+ch,"pid":int(cols[2]),"line":line})
 assert 4 in health and health[4]["health"]=="OK", health.get(4)
 assert not [p for p in processes if p["physical"]==4], processes
 return {"time_unix":time.time(),"physical4":health[4],"physical4_processes":[]}
initial_gate=fresh_gate()
devs=subprocess.check_output(["docker","exec",container,"test","-c","/dev/davinci4"],text=True)
dest.mkdir()
source=dest/"src";source.mkdir()
for name,data in payload["files"].items():
 raw=base64.b64decode(data["base64"]);assert hashlib.sha256(raw).hexdigest()==data["sha256"]
 (source/name).write_bytes(raw)
p=subprocess.run(["python3",str(source/"stage_ids_vendor.py"),"--destination",str(dest/"vendor")],capture_output=True,text=True,timeout=180)
assert p.returncode==0, (p.stdout,p.stderr)
vendor_receipt=json.loads(p.stdout)
model=dest/"model";model.mkdir()
original=pathlib.Path("/home/l00886679/models/out/v41-tiny-dspark")
config=(original/"config.json").read_bytes();cfg=json.loads(config);t=cfg["text_config"]
assert (t["num_hidden_layers"],t["hidden_size"],t["n_routed_experts"],t["num_experts_per_tok"],t["num_nextn_predict_layers"],t["dspark_target_layer_ids"],t["engram_layer_ids"])==(40,5120,8,2,3,[37,38,39],[])
(model/"config.json").write_bytes(config)
for name in ("tokenizer.json","tokenizer_config.json","configuration.json"):(model/name).symlink_to("/model/"+name)
receipt={"time":time.strftime("%Y-%m-%d %H:%M:%S %z"),"physical4":initial_gate["physical4"],"physical4_processes":[],"initial_gate":initial_gate,"container":container,"mounts":[{"source":m["Source"],"target":m["Destination"],"rw":m["RW"]} for m in c["Mounts"] if m["Destination"] in ("/work","/model")],"source_files":{n:{"sha256":d["sha256"],"bytes":d["bytes"]} for n,d in payload["files"].items()},"vendor_source":vendor_receipt,"model_config_sha256":hashlib.sha256(config).hexdigest(),"target_ms":20,"formal_tp8_claim":False,"no_reset_no_dropcache":True}
(dest/"preflight.json").write_text(json.dumps(receipt,indent=2)+"\n")
original_env=dict(item.split("=",1) for item in c["Config"]["Env"])
pythonpath="/work/"+job+"/src:/work/"+job+"/vendor:"+original_env.get("PYTHONPATH","")
env={"ASCEND_RT_VISIBLE_DEVICES":"4","ASCEND_VISIBLE_DEVICES":"4","LOCAL_WORLD_SIZE":"1","HCCL_DETERMINISTIC":"strict","V41_ENGRAM_WITH_DUMMY":"1","V41_DUMMY_WO_A_FIX":"1","DSPARK_DRAFT_METADATA_MODE":"sync","DRAFT_GRAPH":"0","DSPARK_GRAPH_CAPTURE_METADATA":"0","DSPARK_GRAPH_SHADOW_EAGER":"0","DSPARK_PTR_PROBE":"0","DSPARK_ROW_DUMP":"0","V41_CED_ROLE":"","PYTHONPATH":pythonpath,"TMPDIR":"/work/"+job+"/tmp"}
(dest/"tmp").mkdir()
docker=["docker","exec","-i"]
for k,v in env.items():docker.extend(["-e",k+"="+v])
docker.append(container)
receipt["before_sdk_gate"]=fresh_gate()
sdk="import json,os,torch,torch_npu,acl; from vllm_ascend.worker.worker import NPUWorker; assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='4'; assert torch.npu.device_count()==1; torch.npu.set_device(0); x=torch.ones(16,device='npu'); y=x+x; torch.npu.synchronize(); assert y.cpu().tolist()==[2.]*16; print(json.dumps({'device_count':torch.npu.device_count(),'logical_device':torch.npu.current_device(),'physical_device':4,'sdk_ok':True,'acl_and_worker_import':True,'torch':torch.__version__}))"
p=subprocess.run(docker+["python","-c",sdk],capture_output=True,text=True,timeout=90)
receipt["sdk"]={"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
(dest/"preflight.json").write_text(json.dumps(receipt,indent=2)+"\n")
assert p.returncode==0, receipt["sdk"]
receipt["before_model_gate"]=fresh_gate()
(dest/"preflight.json").write_text(json.dumps(receipt,indent=2)+"\n")
script="set -e\n"+"python /work/"+job+"/src/bench_dspark_tiny.py --model /work/"+job+"/model --output /work/"+job+"/results --requests "+str(payload["requests"])+" --tokens "+str(payload["tokens"])+" --prompt-tokens "+str(payload["prompt_tokens"])+" --target-ms 20"+(" --profile" if payload["profile"] else "")+"\n"
(source/"run.sh").write_text(script)
command=docker+["bash","/work/"+job+"/src/run.sh"]
wrapper="import pathlib,subprocess,time,os,json\np=pathlib.Path("+repr(str(dest))+ ")\n(p/'supervisor.pid').write_text(str(os.getpid())+'\\n')\nf=(p/'run.log').open('w')\nproc=subprocess.Popen("+repr(command)+",stdout=f,stderr=subprocess.STDOUT)\n(p/'docker_exec.pid').write_text(str(proc.pid)+'\\n')\ncode=proc.wait()\n(p/'run.exit').write_text(str(code)+'\\n')\n(p/'done.json').write_text(json.dumps({'exit':code,'finished_unix':time.time()}))\n"
(dest/"supervise.py").write_text(wrapper)
log=(dest/"supervisor.log").open("w")
p=subprocess.Popen(["python3",str(dest/"supervise.py")],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({"remote_root":str(dest),"job":job,"supervisor_pid":p.pid,"preflight":receipt},ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", default="dspark_lane_20261011_c4_" + time.strftime("%H%M%S"))
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_") and all(c.isalnum() or c == "_" for c in args.job)
    assert args.requests >= 8 and args.tokens >= 16 and 1 <= args.prompt_tokens <= 256
    base = Path(__file__).resolve().parent
    files = {}
    for name in ("bench_dspark_tiny.py", "dspark_worker.py", "stage_ids_vendor.py"):
        raw = (base / name).read_bytes()
        files[name] = {"base64": base64.b64encode(raw).decode(),
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    payload = vars(args) | {"files": files}
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    program = "import json,base64,gzip\npayload=json.loads(gzip.decompress(base64.b64decode(" + repr(packed) + ")))\n" + CPU_DISCOVERY
    remote = "python3 - <<'PY'\n" + program + "\nPY"
    assert len(remote.encode()) < 1_000_000
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            capture_output=True, text=True, check=True)
    receipt = json.loads(result.stdout)
    (base / "evidence" / (args.job + "_launch.json")).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    (base / "evidence" / "launch.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
