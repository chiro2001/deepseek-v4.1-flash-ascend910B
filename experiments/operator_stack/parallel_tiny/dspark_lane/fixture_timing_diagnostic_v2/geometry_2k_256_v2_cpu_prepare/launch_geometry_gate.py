"""CPU prepare of the independent gate; --launch is reserved for future authorization."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_quality_shared.launch_dspark_tiny import CPU_DISCOVERY

BASE = Path(__file__).resolve().parent
JOB = "dspark_lane_20261011_c4_graph_geometry2k256_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", default=JOB)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    assert args.job.startswith("dspark_lane_20261011_c4_graph_geometry2k256_") and all(c.isalnum() or c == "_" for c in args.job)
    gate = json.loads((BASE / "cpu_gate.json").read_text())
    assert gate["passed"] and gate["cpu_only_no_npu_sdk_import"]
    paths = {path.name: path for path in (BASE / "frozen_quality_shared").glob("*.py")
             if path.name != "launch_dspark_tiny.py"}
    paths.update({name: BASE / name for name in ("bench_geometry_gate.py", "geometry_contract.py",
                                               "geometry_cache_probe.py", "geometry_quality_worker.py")})
    files = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        ast.parse(raw.decode(), filename=name)
        if path.parent == BASE:
            assert gate["source_sha256"][name] == sha(path)
        assert len(raw) < 1_000_000
        files[name] = {"bytes": len(raw), "sha256": sha(path), "base64": base64.b64encode(raw).decode()}
    assert gate["source_sha256"]["launch_geometry_gate.py"] == sha(Path(__file__))
    expected = {"DSPARK_EXPECTED_ACLGRAPH_SHA256": "c8ab2fe6aaae034b03e9e34b9c07fd445046564acd5d45e2c24b5ea5232c5a74",
                "DSPARK_EXPECTED_DSA_SHA256": "24cb4e1ad3b2fcc9f0524153f22c0c6818aba396d290c4b93cdf4f12ced217d5",
                "DSPARK_EXPECTED_ROUTER_SHA256": "f32950da16926afe7569b1a6daa437860db2e83171d96623ac79f7734e52b004",
                "DSPARK_EXPECTED_PROPOSER_SHA256": "bd1004869551791ae481a3cb7c61b124ee25300905f89097799ab4572439fd67",
                "DSPARK_EXPECTED_BASE_SHA256": "681c76017b1a508b64ce2594960c605a74008b4fd878a082788551e9b8f53db8",
                "DSPARK_O_PROJ_CACHE_FIXTURE_SHA256": "7dfa11336f7be4c8c55d0f09eb302673ef31ac51d582686d31cffc432af6fe78"}
    assert files["o_proj_cache_fixture.py"]["sha256"] == expected["DSPARK_O_PROJ_CACHE_FIXTURE_SHA256"]
    flags = {"DRAFT_GRAPH": "1", "DSPARK_GRAPH_CAPTURE_METADATA": "1",
             "DSPARK_CAPTURE_VALUE_FIX": "1", "DSPARK_CAPTURE_NCTX_FIX": "1",
             "DSPARK_CAPTURE_DISPATCH": "1", "DSPARK_DRAFT_METADATA_MODE": "sync",
             "DSPARK_DRAFT_USE_CUDAGRAPH": "1", "DSPARK_CAPTURE_PAD_SLOTS": "0",
             "DSPARK_HOIST_CONTEXT_KV": "0", "DSPARK_DRAFT_SYNC_BEFORE": "0", "DSPARK_DRAFT_SYNC_AFTER": "0",
             "DSPARK_CAPTURE_SEQ_LEN": "3072", "DSPARK_TOKEN_DUMP_STEPS": "0", "DSPARK_RT_FLAGS": "0",
             "DSPARK_ROUTE_IMMEDIATE_DIAGNOSTIC": "1", "DSPARK_FIRST_VALUE_TRACE": "1",
             "DSPARK_O_PROJ_CACHE_FIXTURE": "1", "DSPARK_FIXTURE_VALIDATE_CONSUMERS": "1"}
    payload = {"job": args.job, "files": files, "expected": expected}
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
    code = code.replace(needle, "env.update(" + repr(flags) + ")\nenv.update(payload['expected'])\n"
                        "env['DSPARK_AUDIT_OUTPUT_PATH']='/work/'+job+'/results/audit'\n" + needle)
    needle = 'receipt["before_sdk_gate"]=fresh_gate()'
    code = code.replace(needle, "assert vendor_receipt['source_py_tree_sha256']=='e552b649eb47a39aeeeb549ffdb0f80d8aad199fe5105df34296b5ba62366ecc'\n"
                        "assert receipt['model_config_sha256']=='ad72656e807079b39a5e94f05f5ffe45116725787239344adb4577fc4a2671c3'\n" + needle)
    lines = code.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('script="set -e'))
    lines[index] = 'script="set -e\\npython /work/"+job+"/src/bench_geometry_gate.py --model /work/"+job+"/model --output /work/"+job+"/results --pairs 8 --prompt-tokens 2048 --tokens 256\\n"'
    code = "\n".join(lines)
    compile(code, "geometry2k_remote", "exec")
    namespace = {"job": "cpu", "payload": {}}
    exec(lines[index], namespace)
    assert namespace["script"].count("\n") == 2 and "--prompt-tokens 2048 --tokens 256\n" in namespace["script"]
    packed = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    remote = "python3 - <<'PY'\nimport json,gzip,base64\npayload=json.loads(gzip.decompress(base64.b64decode(" + repr(packed) + ")))\n" + code + "\nPY"
    assert len(remote.encode()) < 1_000_000
    ready = {"job": args.job, "cpu_ready": True, "cpu_controls": gate["controls_passed"],
             "cpu_gate_sha256": sha(BASE / "cpu_gate.json"), "new_npu_authorization_received": False,
             "native_window_command": f"python3 launch_geometry_gate.py --job {args.job} --launch",
             "sources": {name: {key: value for key, value in value.items() if key != "base64"}
                         for name, value in files.items()}, "physical_device": 4, "logical_device": 0,
             "input_tokens": 2048, "output_tokens": 256, "max_model_len": 4096,
             "max_batched_tokens": 2048, "capture_representative_seq_len": 3072,
             "fresh_memory_budget_required_gpu_fraction": 0.70, "model_memory_budget_lowering_allowed": False,
             "independent_comparisons_required": {"native_off_AA": 3, "off_on": 3, "native_on_AA": 3},
             "actual_decode_round_count_must_be_derived": True,
             "native_chunked_prefill_preserved": True,
             "all_prefill_native_helpers_preserved": True,
             "unique_sampleable_seed_required": True,
             "geometry_snapshots_actual_decode_only": True,
             "v1_numeric_checks_or_geometry_borrowed": False,
             "per_group_cache_NPU_proof_pending": ["Three actual capture SparseAttnSharedkv input bindings",
                                                   "Per-group block table in-place refresh and stable native index addresses",
                                                   "Actual visible logical block columns derived from each real decode reader",
                                                   "All 665 native selected slots finite and bitwise equal per group, on every real step"],
             "SWA128_unretained_prefix_not_scanned": True, "native_null_block_id": 0,
             "capture_device_reads": False, "timing_evaluation": False,
             "tiny_admitted": False, "formal_admitted": False, "formal_tp8_claim": False,
             "remote_script_bytes": len(remote.encode())}
    path = BASE / "readiness.json"
    if args.prepare:
        path.write_text(json.dumps(ready, indent=2) + "\n")
        print(json.dumps({key: value for key, value in ready.items() if key != "sources"}, indent=2))
        return
    # This branch is deliberately not executed during the CPU-only preparation.
    assert path.is_file(), "CPU readiness must precede an explicitly authorized launch"
    result = subprocess.run(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", remote],
                            text=True, capture_output=True)
    if result.returncode:
        rejected = {"job": args.job, "exit": result.returncode, "stdout_tail": result.stdout[-12000:],
                    "stderr_tail": result.stderr[-12000:], "no_successful_launch_receipt": True}
        (BASE / (args.job + "_launch_rejected.json")).write_text(json.dumps(rejected, indent=2) + "\n")
        print(json.dumps(rejected, indent=2))
        raise SystemExit(result.returncode)
    receipt = json.loads(result.stdout)
    (BASE / (args.job + "_launch.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "preflight"}, indent=2))


if __name__ == "__main__":
    main()
