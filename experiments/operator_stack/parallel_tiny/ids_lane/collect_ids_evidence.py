"""Fetch compact terminal receipts only; large models, routes and traces stay remote."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplement", action="store_true", help="Append chip release and compact failed-attempt history")
    args = parser.parse_args()
    assert "ids_lane_private" in args.remote_root
    assert args.job.replace("_", "").isalnum()
    assert args.output.is_relative_to("experiments/operator_stack/runtime/evidence/ids_lane")
    program = "\n".join([
        "from pathlib import Path",
        "import json,hashlib",
        f"root=Path({args.remote_root!r})",
        f"job=root/'results'/{args.job!r}",
        "assert (job/'run.exit').read_text().strip()=='0'",
        "assert json.loads((job/'accuracy_gate.json').read_text())['passed']",
        "assert json.loads((job/'profile_validation.json').read_text())['passed']",
        "names=['config.json','topology.json','banks.json','accuracy_gate.json','native_AA.json','capture_AA.json','candidate_comparisons.json','consumer_witnesses.json','performance_audit_disabled.json','performance_pairs.json','profile_windows.json','profile_validation.json','verification_summary.json','result.json','launch.json','run.exit','npu_before.txt','npu_after.txt','device_mapping.txt']",
        "items={name:(job/name).read_text() for name in names}",
        "for name in ['source_audit.json','source_audit_final.json','instrumentation_receipt.json','fallback_receipt.json','previous_attempt.json']: items[name]=(root/name).read_text()",
        "records=json.loads((job/'requests.json').read_text())",
        "items['request_summary.json']=json.dumps([{k:v for k,v in r.items() if k not in ['step_ms','token_ids','logprobs']} for r in records],indent=2)+'\\n'",
        "items['remote_sources.json']=json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'ids_lane').glob('*.py')},indent=2)+'\\n'",
        "assert all(len(text.encode())<1000000 for text in items.values())",
        "bundle=json.dumps(items)",
        "assert len(bundle.encode())<1000000",
        "print(bundle)",
    ])
    if args.supplement:
        program = "\n".join([
            "from pathlib import Path",
            "import json,hashlib",
            f"root=Path({args.remote_root!r})",
            f"job=root/'results'/{args.job!r}",
            "assert (job/'run.exit').read_text().strip()=='0'",
            "items={name:(job/name).read_text() for name in ['chip4_release.json','chip4_release_npu_smi.txt']}",
            "for version in [1,2]:",
            " old=root.with_name('ids_lane_private_v'+str(version))",
            " previous=old/'results'/('ids_audit_perf_v'+str(version))",
            " assert (previous/'run.exit').read_text().strip()=='1'",
            " for name in ['config.json','launch.json','run.exit','npu_before.txt','accuracy_gate.json','banks.json','native_AA.json','capture_AA.json','candidate_comparisons.json','consumer_witnesses.json','performance_audit_disabled.json']:",
            "  p=previous/name",
            "  if p.exists(): items['history/v'+str(version)+'/'+name]=p.read_text()",
            " items['history/v'+str(version)+'/runner_tail.txt']='\\n'.join((previous/'runner.log').read_text().splitlines()[-55:])+'\\n'",
            " items['history/v'+str(version)+'/runtime_sources.json']=json.dumps({str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old/'ids_lane').glob('*.py')},indent=2)+'\\n'",
            "assert all(len(text.encode())<1000000 for text in items.values())",
            "bundle=json.dumps(items)",
            "assert len(bundle.encode())<1000000",
            "print(bundle)",
        ])
    command = ["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21",
               "python3 -c " + shlex.quote(program)]
    raw = subprocess.check_output(command)
    assert len(raw) < 1000000
    contents = json.loads(raw)
    if args.supplement:
        assert args.output.is_dir()
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, content in contents.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert not (args.output / name).exists()
        (args.output / name).parent.mkdir(parents=True, exist_ok=True)
        (args.output / name).write_text(content)
        manifest[name] = {"sha256": hashlib.sha256(content.encode()).hexdigest(), "bytes": len(content.encode())}
    manifest_name = "history_files.json" if args.supplement else "evidence_files.json"
    (args.output / manifest_name).write_text(json.dumps({"remote_root": args.remote_root,
                                                               "job": args.job, "files": manifest,
                                                               "ssh_bundle_bytes": len(raw)}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "compact_files": len(contents), "ssh_bundle_bytes": len(raw)}))


if __name__ == "__main__":
    main()
