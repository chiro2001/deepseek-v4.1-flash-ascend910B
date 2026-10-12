"""Fresh physical4 eager-only diagnostic; target graph stays at frozen v7."""
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


def build_program(payload):
    code = CPU_DISCOVERY
    needle = '(dest/"tmp").mkdir()'
    assert code.count(needle) == 1
    flags = {"DRAFT_GRAPH": "0", "DSPARK_GRAPH_CAPTURE_METADATA": "0",
             "DSPARK_CAPTURE_VALUE_FIX": "0", "DSPARK_CAPTURE_NCTX_FIX": "0",
             "DSPARK_CAPTURE_DISPATCH": "0", "DSPARK_DRAFT_USE_CUDAGRAPH": "0",
             "DSPARK_DRAFT_METADATA_MODE": "sync", "DSPARK_CAPTURE_PAD_SLOTS": "0",
             "DSPARK_HOIST_CONTEXT_KV": "0", "DSPARK_DRAFT_SYNC_BEFORE": "0",
             "DSPARK_DRAFT_SYNC_AFTER": "0", "DSPARK_TOKEN_DUMP_STEPS": "0",
             "DSPARK_RT_FLAGS": "0", "DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC": "1"}
    flags["DSPARK_FIRST_VALUE_TRACE"] = "1" if payload.get("first_value_trace", False) else "0"
    if payload.get("cache_fixture", False):
        flags.update({"DSPARK_O_PROJ_CACHE_FIXTURE": "1", "DSPARK_FIXTURE_VALIDATE_CONSUMERS": "1"})
    code = code.replace(needle, "env.update(" + repr(flags) + ")\nenv.update(payload['expected'])\n"
                        "env['DSPARK_AUDIT_OUTPUT_PATH']='/work/'+job+'/results/audit'\n" + needle)
    lines = code.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('script="set -e'))
    lines[index] = ('script="set -e\\npython /work/"+job+"/src/native_route_diagnostic.py --model /work/"'
                    '+job+"/model --output /work/"+job+"/results\\n"')
    code = "\n".join(lines)
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    program = ("import json,gzip,base64\npayload=json.loads(gzip.decompress(base64.b64decode(" + repr(packed)
               + ")))\n" + code)
    assert len(program.encode()) < 1_000_000
    compile(program, "native_route_remote", "exec")
    return program


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--job", default="dspark_lane_20261011_c4_native_diag_" + time.strftime("%H%M%S"))
    p.add_argument("--first-value-trace", action="store_true")
    p.add_argument("--cache-fixture", action="store_true")
    args = p.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_native_diag_")
    assert all(ch.isalnum() or ch == "_" for ch in args.job)
    base = Path(__file__).resolve().parent
    files = {}
    names = ("native_route_diagnostic.py", "native_route_diagnostic_worker.py", "graph_precision_audit.py",
             "native_first_value_trace.py", "dspark_worker.py", "stage_ids_vendor.py")
    if args.cache_fixture:
        names += ("o_proj_cache_fixture.py", "o_proj_cache_fixture_worker.py", "graph_candidate_worker.py")
    for name in names:
        raw = (base / name).read_bytes()
        ast.parse(raw.decode(), filename=name)
        assert len(raw) < 1_000_000
        files[name] = {"base64": base64.b64encode(raw).decode(), "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()}
    source = json.loads((base / "evidence/graph_candidate_source_manifest.json").read_text())
    guard = {row["path"].split("/vendor/", 1)[1]: row["sha256"] for row in source["files"]}
    expected = {"DSPARK_EXPECTED_DSA_SHA256": guard["vllm_ascend/attention/dsa_v1.py"],
                "DSPARK_EXPECTED_ROUTER_SHA256": guard["vllm_ascend/ops/fused_moe/router/fused_topk_router.py"],
                "DSPARK_EXPECTED_PROPOSER_SHA256": "bd1004869551791ae481a3cb7c61b124ee25300905f89097799ab4572439fd67",
                "DSPARK_EXPECTED_BASE_SHA256": "681c76017b1a508b64ce2594960c605a74008b4fd878a082788551e9b8f53db8"}
    if args.cache_fixture:
        expected["DSPARK_O_PROJ_CACHE_FIXTURE_SHA256"] = hashlib.sha256((base / "o_proj_cache_fixture.py").read_bytes()).hexdigest()
    program = build_program({"job": args.job, "files": files, "expected": expected,
                             "first_value_trace": args.first_value_trace, "cache_fixture": args.cache_fixture})
    remote = "python3 - <<'PY'\n" + program + "\nPY"
    assert len(remote.encode()) < 1_000_000
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            text=True, capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    receipt.update({"fresh_native_eager_only": True, "draft_capture": False,
                    "target_graph_same_as_v7": True, "diagnostic_only_no_performance_claim": True})
    (base / "evidence" / (args.job + "_launch.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "preflight"}, indent=2))


if __name__ == "__main__":
    main()
