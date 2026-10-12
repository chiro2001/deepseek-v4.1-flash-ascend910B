"""Fresh physical4 gate, complete native capture configuration, quality before timing."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from launch_dspark_tiny import CPU_DISCOVERY


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--job", default="dspark_lane_20261011_c4_graph_" + time.strftime("%H%M%S"))
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--route-diagnostic", action="store_true")
    p.add_argument("--first-value-trace", action="store_true")
    p.add_argument("--cache-fixture", action="store_true")
    args = p.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_graph_")
    assert all(ch.isalnum() or ch == "_" for ch in args.job) and args.pairs >= 8
    base = Path(__file__).resolve().parent
    files = {}
    names = ("bench_draft_graph_candidate.py", "dspark_worker.py", "graph_candidate_worker.py",
             "draft_graph_observer.py", "graph_precision_audit.py", "native_first_value_trace.py", "stage_ids_vendor.py")
    if args.cache_fixture:
        names += ("o_proj_cache_fixture.py", "o_proj_cache_fixture_worker.py", "native_route_diagnostic_worker.py")
    for name in names:
        raw = (base / name).read_bytes()
        ast.parse(raw.decode(), filename=name)
        files[name] = {"base64": base64.b64encode(raw).decode(), "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()}
    guards = json.loads((base / "evidence/graph_candidate_source_manifest.json").read_text())
    guard_by_name = {row["path"].split("/vendor/", 1)[1]: row["sha256"] for row in guards["files"]}
    expected = {"DSPARK_EXPECTED_ACLGRAPH_SHA256": guard_by_name["vllm_ascend/compilation/acl_graph.py"],
                "DSPARK_EXPECTED_DSA_SHA256": guard_by_name["vllm_ascend/attention/dsa_v1.py"],
                "DSPARK_EXPECTED_ROUTER_SHA256": guard_by_name["vllm_ascend/ops/fused_moe/router/fused_topk_router.py"],
                "DSPARK_EXPECTED_PROPOSER_SHA256": "bd1004869551791ae481a3cb7c61b124ee25300905f89097799ab4572439fd67",
                "DSPARK_EXPECTED_BASE_SHA256": "681c76017b1a508b64ce2594960c605a74008b4fd878a082788551e9b8f53db8"}
    payload = {"job": args.job, "files": files, "expected": expected, "pairs": args.pairs}
    code = CPU_DISCOVERY
    needle = '(dest/"tmp").mkdir()'
    assert code.count(needle) == 1
    flags = {"DRAFT_GRAPH": "1", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
             "DSPARK_CAPTURE_VALUE_FIX": "1", "DSPARK_CAPTURE_NCTX_FIX": "1",
             "DSPARK_CAPTURE_DISPATCH": "1", "DSPARK_DRAFT_METADATA_MODE": "sync",
             "DSPARK_DRAFT_USE_CUDAGRAPH": "1", "DSPARK_CAPTURE_PAD_SLOTS": "0",
             "DSPARK_HOIST_CONTEXT_KV": "0", "DSPARK_DRAFT_SYNC_BEFORE": "0",
             "DSPARK_DRAFT_SYNC_AFTER": "0", "DSPARK_CAPTURE_SEQ_LEN": "256",
             "DSPARK_TOKEN_DUMP_STEPS": "0", "DSPARK_RT_FLAGS": "0"}
    if args.route_diagnostic:
        flags["DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC"] = "1"
    if args.first_value_trace:
        flags["DSPARK_FIRST_VALUE_TRACE"] = "1"
    if args.cache_fixture:
        flags.update({"DSPARK_O_PROJ_CACHE_FIXTURE": "1", "DSPARK_FIXTURE_VALIDATE_CONSUMERS": "1"})
        expected["DSPARK_O_PROJ_CACHE_FIXTURE_SHA256"] = hashlib.sha256((base / "o_proj_cache_fixture.py").read_bytes()).hexdigest()
    code = code.replace(needle, "env.update(" + repr(flags) + ")\nenv.update(payload['expected'])\nenv['DSPARK_AUDIT_OUTPUT_PATH']='/work/'+job+'/results/audit'\n" + needle)
    lines = code.splitlines()
    script_index = next(index for index, line in enumerate(lines) if line.startswith('script="set -e'))
    lines[script_index] = 'script="set -e\\npython /work/"+job+"/src/bench_draft_graph_candidate.py --model /work/"+job+"/model --output /work/"+job+"/results --pairs "+str(payload["pairs"])+" --tokens 32\\n"'
    # The outer code needs real escaped newlines in the remote Python string.
    lines[script_index] = lines[script_index].replace("\\\\n", "\\n")
    code = "\n".join(lines)
    compile(code, "graph_remote_discovery", "exec")
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    remote = "python3 - <<'PY'\nimport json,gzip,base64\npayload=json.loads(gzip.decompress(base64.b64decode(" + repr(packed) + ")))\n" + code + "\nPY"
    assert len(remote.encode()) < 1_000_000
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            text=True, capture_output=True)
    if result.returncode:
        rejected = {"job": args.job, "launch_exit": result.returncode,
                    "stdout_tail": result.stdout[-12000:], "stderr_tail": result.stderr[-12000:],
                    "no_successful_launch_receipt": True}
        (base / "evidence" / (args.job + "_launch_rejected.json")).write_text(json.dumps(rejected, indent=2) + "\n")
        print(json.dumps(rejected, indent=2))
        raise SystemExit(result.returncode)
    receipt = json.loads(result.stdout)
    receipt.update({"graph_candidate": True, "complete_quality_gate_required_before_performance": True,
                    "constructor_bool_flip": False})
    (base / "evidence" / (args.job + "_launch.json")).write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "preflight"}, indent=2))


if __name__ == "__main__":
    main()
