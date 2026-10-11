"""Freeze on CPU, then launch one fresh physical4 timing diagnostic."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_shared.launch_dspark_tiny import CPU_DISCOVERY

BASE = Path(__file__).resolve().parent
SHARED = BASE / "frozen_shared"
RUNTIME_FILES = {
    "bench_fixture_timing.py": BASE / "bench_fixture_timing.py",
    "fixture_timing_worker.py": BASE / "fixture_timing_worker.py",
    "timing_controls.py": BASE / "timing_controls.py",
    "reference_token_ids.json": BASE / "reference_token_ids.json",
    **{name: SHARED / name for name in ("dspark_worker.py", "stage_ids_vendor.py",
                                      "o_proj_cache_fixture.py", "graph_precision_audit.py",
                                      "draft_graph_observer.py")},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(job, pairs):
    assert job.startswith("dspark_lane_20261011_c4_fixture_timing_")
    assert all(char.isalnum() or char == "_" for char in job) and pairs >= 8
    known = json.loads((BASE / "native_sources.json").read_text())
    for name, expected in known["shared_sources"].items():
        assert sha(SHARED / name) == expected, name
    gate_path = BASE / "evidence/cpu_gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["passed"] and gate["cpu_only_no_torch_or_npu_import"]
    files = {}
    for name, path in RUNTIME_FILES.items():
        raw = path.read_bytes()
        assert len(raw) < 1_000_000
        if name.endswith(".py"):
            ast.parse(raw.decode(), filename=name)
        assert gate["runtime_source_sha256"][name] == sha(path), name
        files[name] = {"base64": base64.b64encode(raw).decode(), "bytes": len(raw), "sha256": sha(path)}
    assert gate["launcher_sha256"] == sha(Path(__file__))
    manifest = {"job": job, "pairs": pairs, "measured_requests_per_arm": pairs,
                "input_tokens": 128, "output_tokens": 32,
                "physical_device": 4, "logical_device": 0,
                "cpu_gate_sha256": sha(gate_path), "cpu_controls": gate["controls_passed"],
                "launcher_sha256": sha(Path(__file__)), "native_sources_json_sha256": sha(BASE / "native_sources.json"),
                "runtime_source_files": {name: {key: value for key, value in info.items() if key != "base64"}
                                         for name, info in files.items()},
                "native_sources": known["native_sources"],
                "fixture_source_sha256": known["shared_sources"]["o_proj_cache_fixture.py"],
                "vendor_source_py_tree_sha256": known["vendor_source_py_tree_sha256"],
                "model_config_sha256": known["model_config_sha256"],
                "before_after_full43_source_cache_finite_gate_outside_window": True,
                "diagnostic_d2h_and_hash_forbidden_in_timing_window": True,
                "route_logprob_audit": False, "profiler": False,
                "initial_and_terminal_submitted_draft_included": True,
                "dummy_clean_rate_required": 0.0, "diagnostic_only": True,
                "tiny_candidate_admitted": False, "formal_candidate_admitted": False,
                "formal_tp8_claim": False, "no_reset_no_dropcache_no_unknown_process_stop": True,
                "new_geometry_expansion_authorized_in_this_job": False}
    return known, files, manifest


def remote_program(payload):
    flags = {"DRAFT_GRAPH": "1", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
             "DSPARK_CAPTURE_VALUE_FIX": "1", "DSPARK_CAPTURE_NCTX_FIX": "1",
             "DSPARK_CAPTURE_DISPATCH": "1", "DSPARK_DRAFT_METADATA_MODE": "sync",
             "DSPARK_DRAFT_USE_CUDAGRAPH": "1", "DSPARK_CAPTURE_PAD_SLOTS": "0",
             "DSPARK_HOIST_CONTEXT_KV": "0", "DSPARK_DRAFT_SYNC_BEFORE": "0",
             "DSPARK_DRAFT_SYNC_AFTER": "0", "DSPARK_CAPTURE_SEQ_LEN": "256",
             "DSPARK_TOKEN_DUMP_STEPS": "0", "DSPARK_RT_FLAGS": "0",
             "DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC": "0", "DSPARK_FIRST_VALUE_TRACE": "0",
             "DSPARK_O_PROJ_CACHE_FIXTURE": "1", "DSPARK_FIXTURE_VALIDATE_CONSUMERS": "0"}
    code = CPU_DISCOVERY
    needle = '(dest/"tmp").mkdir()'
    assert code.count(needle) == 1
    code = code.replace(needle, "env.update(" + repr(flags) + ")\n"
                        "env['DSPARK_EXPECTED_ACLGRAPH_SHA256']=payload['known']['native_sources']['vllm_ascend/compilation/acl_graph.py']\n"
                        "env['DSPARK_O_PROJ_CACHE_FIXTURE_SHA256']=payload['known']['shared_sources']['o_proj_cache_fixture.py']\n"
                        "env['DSPARK_TIMING_EXPECTED_SOURCES_JSON']=json.dumps({n:v['sha256'] for n,v in payload['files'].items()})\n"
                        "env['DSPARK_TIMING_EXPECTED_NATIVE_JSON']=json.dumps(payload['known']['native_sources'])\n"
                        "env['DSPARK_AUDIT_OUTPUT_PATH']='/work/'+job+'/results/cache_gate'\n" + needle)
    needle = 'receipt["before_sdk_gate"]=fresh_gate()'
    assert code.count(needle) == 1
    code = code.replace(needle, "assert vendor_receipt['source_py_tree_sha256']==payload['known']['vendor_source_py_tree_sha256']\n"
                        "assert receipt['model_config_sha256']==payload['known']['model_config_sha256']\n"
                        "receipt['timing_freeze_manifest_sha256']=payload['freeze_manifest_sha256']\n" + needle)
    lines = code.splitlines()
    script_index = next(index for index, line in enumerate(lines) if line.startswith('script="set -e'))
    lines[script_index] = 'script="set -e\\npython /work/"+job+"/src/bench_fixture_timing.py --model /work/"+job+"/model --output /work/"+job+"/results --pairs "+str(payload["pairs"])+"\\n"'
    code = "\n".join(lines)
    compile(code, "fixture_timing_remote", "exec")
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    remote = "python3 - <<'PY'\nimport json,gzip,base64\npayload=json.loads(gzip.decompress(base64.b64decode(" + repr(packed) + ")))\n" + code + "\nPY"
    assert len(remote.encode()) < 1_000_000
    return remote


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", default="dspark_lane_20261011_c4_fixture_timing_diagnostic_v2")
    parser.add_argument("--pairs", type=int, default=8)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    known, files, manifest = prepare(args.job, args.pairs)
    freeze = BASE / "evidence" / (args.job + "_freeze.json")
    canonical = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.freeze_only:
        if freeze.exists():
            assert freeze.read_text() == canonical, "Never overwrite a changed freeze"
        else:
            freeze.write_text(canonical)
        remote = remote_program({"job": args.job, "pairs": args.pairs, "known": known,
                                 "files": files, "freeze_manifest_sha256": sha(freeze)})
        print(json.dumps({"job": args.job, "freeze": str(freeze), "sha256": sha(freeze),
                          "remote_script_bytes": len(remote.encode()), "runtime_files": len(files),
                          "cpu_gate_passed": True, "launch_performed": False}, indent=2))
        return
    assert freeze.is_file() and freeze.read_text() == canonical, "CPU freeze must precede launch"
    payload = {"job": args.job, "pairs": args.pairs, "known": known,
               "files": files, "freeze_manifest_sha256": sha(freeze)}
    remote = remote_program(payload)
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            text=True, capture_output=True)
    if result.returncode:
        rejected = {"job": args.job, "exit": result.returncode,
                    "stdout_tail": result.stdout[-12000:], "stderr_tail": result.stderr[-12000:],
                    "no_successful_launch_receipt": True, "freeze_manifest_sha256": sha(freeze)}
        (BASE / "evidence" / (args.job + "_launch_rejected.json")).write_text(json.dumps(rejected, indent=2) + "\n")
        print(json.dumps(rejected, indent=2))
        raise SystemExit(result.returncode)
    receipt = json.loads(result.stdout)
    receipt.update({"freeze_manifest_sha256": sha(freeze), "diagnostic_only": True,
                    "tiny_candidate_admitted": False, "formal_candidate_admitted": False,
                    "formal_tp8_claim": False})
    (BASE / "evidence" / (args.job + "_launch.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "preflight"}, indent=2))


if __name__ == "__main__":
    main()
