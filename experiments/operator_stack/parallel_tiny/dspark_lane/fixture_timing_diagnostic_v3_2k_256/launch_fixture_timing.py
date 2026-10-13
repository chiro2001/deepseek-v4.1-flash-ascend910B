"""Freeze the CPU-checked 2048/256 timing package, then (separately authorized) launch one fresh physical4 job."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from frozen_shared.launch_dspark_tiny import CPU_DISCOVERY

BASE = Path(__file__).resolve().parent
SHARED = BASE / "frozen_shared"
JOB_DEFAULT = "dspark_lane_20261011_c4_fixture_timing_diagnostic_v3_2k_256"
REFERENCE_NAME = "reference_token_ids_2k256.json"
REFERENCE_JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v3"
GEOMETRY = {"input_tokens": 2048, "output_tokens": 256, "verify_scheduled_tokens": 6, "K": 5,
            "prefill_chunks": [2044, 4], "max_model_len": 4096, "max_num_batched_tokens": 2048,
            "capture_representative_seq_len": 3072, "gpu_memory_utilization": 0.70,
            "kv_cache_memory_bytes": 2 * 1024**3, "block_size": 128,
            "physical_device": 4, "logical_device": 0}
RUNTIME_FILES = {
    "bench_fixture_timing.py": BASE / "bench_fixture_timing.py",
    "fixture_timing_worker.py": BASE / "fixture_timing_worker.py",
    "timing_controls.py": BASE / "timing_controls.py",
    **{name: SHARED / name for name in ("dspark_worker.py", "stage_ids_vendor.py",
                                      "o_proj_cache_fixture.py", "graph_precision_audit.py",
                                      "draft_graph_observer.py")},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_entry():
    """Describe the v3-reference state; token bytes are only ever written by build_reference_2k256.py."""
    path = BASE / REFERENCE_NAME
    if not path.is_file():
        return {"status": "pending v3 terminal evidence",
                "expected_package_path": REFERENCE_NAME,
                "expected_source": ("results/requests.json of job " + REFERENCE_JOB +
                                    " fetched to the local repo (runtime/results/requests.json)"),
                "extractor_command": ("python3 build_reference_2k256.py --job " + REFERENCE_JOB +
                                      " --source <local path containing " + REFERENCE_JOB +
                                      ">/results/requests.json --write"),
                "required_schema": {"pending": False, "source_job": REFERENCE_JOB,
                                    "source_requests_path": "must contain the v3 job name",
                                    "source_requests_sha256": "64 hex chars", "extracted_unix": "number",
                                    "seeds": {"0": {"tags": "non-empty list", "token_ids": "256 ints"},
                                              "1": {"tags": "non-empty list", "token_ids": "256 ints"},
                                              "2": {"tags": "non-empty list", "token_ids": "256 ints"}}},
                "fabricated_reference_bytes_forbidden": True}
    data = json.loads(path.read_text())
    return {"status": "present", "bytes": path.stat().st_size, "sha256": sha(path),
            "provenance": {key: data.get(key) for key in ("pending", "source_job", "source_requests_path",
                                                          "source_requests_sha256", "extractor",
                                                          "extracted_unix")}}


def assert_launchable(payload):
    """Single place that decides whether an NPU window may even be rendered for execution."""
    assert not payload.get("reference_pending"), "Pending v3 terminal reference token ids; final freeze required"


def loaded_files(include_reference):
    files = {}
    for name, path in RUNTIME_FILES.items():
        raw = path.read_bytes()
        assert len(raw) < 1_000_000
        if name.endswith(".py"):
            ast.parse(raw.decode(), filename=name)
        files[name] = {"base64": base64.b64encode(raw).decode(), "bytes": len(raw), "sha256": sha(path)}
    if include_reference:
        path = BASE / REFERENCE_NAME
        raw = path.read_bytes()
        assert len(raw) < 1_000_000
        files[REFERENCE_NAME] = {"base64": base64.b64encode(raw).decode(), "bytes": len(raw), "sha256": sha(path)}
    return files


def build_manifest(job, pairs, known, gate_sha256, cpu_controls, files, reference):
    blocked = reference["status"] != "present"
    local_tools = ("analyze_fixture_timing.py", "fetch_fixture_timing.py",
                   "build_reference_2k256.py", "check_fixture_timing_cpu.py")
    return {"job": job, "pairs": pairs, "measured_requests_per_arm": pairs, **GEOMETRY,
            "cpu_gate_sha256": gate_sha256, "cpu_controls": cpu_controls,
            "launcher_sha256": sha(Path(__file__)), "native_sources_json_sha256": sha(BASE / "native_sources.json"),
            "local_tool_files": {name: {"bytes": (BASE / name).stat().st_size, "sha256": sha(BASE / name)}
                                 for name in local_tools},
            "runtime_source_files": {name: {key: value for key, value in info.items() if key != "base64"}
                                     for name, info in files.items()},
            "reference_token_ids_2k256": reference,
            "launch_blocked_by_pending_reference": blocked,
            "native_sources": known["native_sources"],
            "fixture_source_sha256": known["shared_sources"]["o_proj_cache_fixture.py"],
            "vendor_source_py_tree_sha256": known["vendor_source_py_tree_sha256"],
            "model_config_sha256": known["model_config_sha256"],
            "before_after_full43_source_cache_finite_gate_outside_window": True,
            "diagnostic_d2h_and_hash_forbidden_in_timing_window": True,
            "route_logprob_audit": False, "profiler": False, "first_value_trace": False,
            "initial_partial_seed_and_terminal_submitted_draft_included": True,
            "dummy_clean_rate_required": 0.0, "diagnostic_only": True,
            "tiny_candidate_admitted": False, "formal_candidate_admitted": False,
            "formal_tp8_claim": False,
            "formal_full_main_mtp_verify_scheduler_cache_20ms_goal_achieved": False,
            "acceptance_from_dummy_repetition_must_not_be_used_as_benefit": True,
            "no_reset_no_dropcache_no_unknown_process_stop": True,
            "decode_round_count_derived_from_fresh_receipts": True,
            "geometry_cache_gate_v3_numeric_is_external_precondition": True,
            "128_32_timing_results_or_geometry_borrowed": False,
            "npu_authorization_required_from_main_thread": True}


def build_readiness(job, pairs, gate, gate_sha256, reference, freeze_path, freeze_sha):
    pending = reference["status"] != "present"
    return {"job": job, "cpu_ready": True, "cpu_controls": gate["controls_passed"],
            "reference_interface_controls_passed": gate["reference_interface_controls_passed"],
            "cpu_gate_sha256": gate_sha256,
            "checker_sha256": gate["checker_sha256"],
            "freeze_manifest": {"path": str(Path(freeze_path).relative_to(BASE)), "sha256": freeze_sha,
                                "launch_blocked_by_pending_reference": pending},
            "reference_token_ids_2k256": {"status": reference["status"]},
            "pending_items": (["v3 terminal results/requests.json -> build_reference_2k256.py "
                               "(参考 token 未取回前禁止启动)"] if pending else []) +
                              ["主线程另行授权 fresh physical4 窗口后才可 --launch",
                               "终态需再次取回 physical4 健康与释放证据"],
            "future_commands": {
                "1_extract_reference": ("python3 build_reference_2k256.py --job " + REFERENCE_JOB +
                                        " --source <local v3 runtime>/results/requests.json --write"),
                "2_final_freeze": "python3 launch_fixture_timing.py --freeze-only",
                "3_authorized_launch": (f"python3 launch_fixture_timing.py --job {job} "
                                        f"--pairs {pairs} --launch")},
            "npu_authorization": {"authorized": False, "npu_permission": 0, "physical_device": 4,
                                  "logical_device": 0,
                                  "requires_main_thread_fresh_physical4_window": True,
                                  "must_not_touch_remote_job": REFERENCE_JOB},
            "geometry": dict(GEOMETRY),
            "policy": {"dummy_clean_rate_expected": 0.0, "tiny_candidate_admitted": False,
                       "formal_candidate_admitted": False, "formal_tp8_claim": False,
                       "period2_dummy_acceptance_must_not_be_used_as_benefit": True,
                       "results_must_not_be_extrapolated_to_20ms_or_quality": True}}


def remote_program(payload):
    flags = {"DRAFT_GRAPH": "1", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
             "DSPARK_CAPTURE_VALUE_FIX": "1", "DSPARK_CAPTURE_NCTX_FIX": "1",
             "DSPARK_CAPTURE_DISPATCH": "1", "DSPARK_DRAFT_METADATA_MODE": "sync",
             "DSPARK_DRAFT_USE_CUDAGRAPH": "1", "DSPARK_CAPTURE_PAD_SLOTS": "0",
             "DSPARK_HOIST_CONTEXT_KV": "0", "DSPARK_DRAFT_SYNC_BEFORE": "0",
             "DSPARK_DRAFT_SYNC_AFTER": "0", "DSPARK_CAPTURE_SEQ_LEN": "3072",
             "DSPARK_TOKEN_DUMP_STEPS": "0", "DSPARK_RT_FLAGS": "0",
             "DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC": "0", "DSPARK_FIRST_VALUE_TRACE": "0",
             "DSPARK_O_PROJ_CACHE_FIXTURE": "1", "DSPARK_FIXTURE_VALIDATE_CONSUMERS": "0"}
    code = CPU_DISCOVERY
    needle = "initial_gate=fresh_gate()"
    assert code.count(needle) == 1
    code = code.replace(needle, "original_fresh_gate=fresh_gate\n"
                        "def fresh_gate():\n"
                        " g=original_fresh_gate()\n"
                        " pairs=re.findall(r'(\\d+)\\s*/\\s*(\\d+)',g['physical4']['line'])\n"
                        " assert pairs,'Missing current HBM budget evidence'\n"
                        " used,total=map(int,pairs[-1]);free=total-used;required=total*0.70\n"
                        " assert total>0 and 0<=used<=total,('Invalid current HBM budget',used,total)\n"
                        " assert free>=required,('Normal model memory budget is unavailable',used,total,required)\n"
                        " g['normal_memory_budget']={'used_mb':used,'total_mb':total,'free_mb':free,'required_free_mb':required,'gpu_memory_utilization':0.70,'budget_lowered':False}\n"
                        " return g\n" + needle)
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
    lines[script_index] = ('script="set -e\\npython /work/"+job+"/src/bench_fixture_timing.py --model /work/"+job'
                           ' +"/model --output /work/"+job+"/results --pairs "+str(payload["pairs"])'
                           '+" --prompt-tokens 2048 --tokens 256\\n"')
    code = "\n".join(lines)
    compile(code, "fixture_timing2k_remote", "exec")
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    remote = ("python3 - <<'PY'\nimport json,gzip,base64\npayload=json.loads(gzip.decompress(base64.b64decode("
              + repr(packed) + ")))\n"
              "assert not payload.get('reference_pending'), 'Pending v3 terminal reference token ids; final freeze required'\n"
              + code + "\nPY")
    assert len(remote.encode()) < 1_000_000
    return remote


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", default=JOB_DEFAULT)
    parser.add_argument("--pairs", type=int, default=8)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_fixture_timing_") and all(char.isalnum() or char == "_"
                                                                                 for char in args.job)
    assert args.pairs >= 8
    known = json.loads((BASE / "native_sources.json").read_text())
    for name, expected in known["shared_sources"].items():
        assert sha(SHARED / name) == expected, name
    gate_path = BASE / "evidence/cpu_gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["passed"] and gate["cpu_only_no_torch_or_npu_import"]
    assert gate["launcher_sha256"] == sha(Path(__file__))
    assert gate["reference_interface_controls_passed"] >= 8
    for name, path in RUNTIME_FILES.items():
        assert gate["runtime_source_sha256"][name] == sha(path), name
    reference = reference_entry()
    include_reference = reference["status"] == "present"
    if args.launch:
        assert include_reference, "A v3-derived frozen reference is required before any launch"
    files = loaded_files(include_reference)
    manifest = build_manifest(args.job, args.pairs, known, sha(gate_path), gate["controls_passed"],
                              files, reference)
    canonical = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    payload = {"job": args.job, "pairs": args.pairs, "known": known, "files": files,
               "reference_pending": not include_reference}
    if args.freeze_only:
        suffix = "_freeze.json" if include_reference else "_freeze_staged.json"
        freeze = BASE / "evidence" / (args.job + suffix)
        if freeze.exists():
            assert freeze.read_text() == canonical, "Never overwrite a changed freeze"
        else:
            freeze.write_text(canonical)
        payload["freeze_manifest_sha256"] = sha(freeze)
        remote = remote_program(payload)
        readiness = build_readiness(args.job, args.pairs, gate, sha(gate_path), reference, freeze, sha(freeze))
        (BASE / "readiness.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"job": args.job, "freeze": str(freeze), "sha256": sha(freeze),
                          "remote_script_bytes": len(remote.encode()), "runtime_files": len(files),
                          "cpu_gate_passed": True, "reference_status": reference["status"],
                          "readiness": str(BASE / "readiness.json"),
                          "launch_blocked_by_pending_reference": not include_reference,
                          "launch_performed": False}, indent=2))
        return
    freeze = BASE / "evidence" / (args.job + "_freeze.json")
    assert freeze.is_file() and freeze.read_text() == canonical, "Final CPU freeze with v3 reference must precede launch"
    payload["freeze_manifest_sha256"] = sha(freeze)
    assert_launchable(payload)
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
