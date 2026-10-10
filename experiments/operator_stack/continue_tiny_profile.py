"""Host controller: full tiny audit gates timing and seven-metric collection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--audit-job', required=True)
    parser.add_argument('--profile-job', required=True)
    parser.add_argument('--controller-job', required=True)
    args = parser.parse_args()
    for name in (args.audit_job, args.profile_job, args.controller_job):
        assert name.replace('_', '').replace('-', '').isalnum()
    root = args.root.resolve()
    assert args.source_dir.is_relative_to('/work')
    source = root / args.source_dir.relative_to('/work')
    audit = root / 'results' / args.audit_job
    out = root / 'results' / args.controller_job
    out.mkdir(parents=True, exist_ok=True)
    launched = json.loads((audit / 'launched.json').read_text())
    assert launched['container'] == args.container
    assert launched['source_dir'] == str(args.source_dir)
    assert launched['source_sha256'] == hashlib.sha256((source / 'stack/bench_tiny_tp8.py').read_bytes()).hexdigest()
    assert launched['env']['STACK_TINY_PROFILE'] == '1'
    deadline = time.monotonic() + 3600
    while not (audit / 'run.exit').exists():
        code = ("from pathlib import Path; p=Path('/work/results/" + args.audit_job +
                "/runner.pid'); pid=p.read_text().strip(); "
                "assert Path('/proc/'+pid).exists(),'Audit runner vanished without terminal receipt'; print(pid)")
        live_pid = subprocess.check_output(['docker', 'exec', args.container, 'python', '-c', code], text=True).strip()
        print('TINY_AUDIT_VERIFIED_LIVE', live_pid, flush=True)
        assert time.monotonic() < deadline, 'Audit observation timed out; do not restart a live model'
        time.sleep(20)
    assert (audit / 'run.exit').read_text().strip() == '0', 'Tiny audit failed; do not profile'
    result = json.loads((audit / 'result.json').read_text())
    assert result['audit'] and result['diagnostic_only'] and not result['formal_weights']
    assert result['source_formal_tensors_byte_exact'] and result['layers'] == result['tensor_parallel_size'] == 8
    assert len(result['native_pairs']) >= 3 and all(r['passed'] for r in result['native_pairs'])
    prefix = result['formal_prefix_comparisons']
    assert len(prefix) >= 7 and {r['seed'] for r in prefix} == {0, 1, 2}
    assert all(r['prefix_routes_equal'] and r['differing_token_layers'] == 0 for r in prefix)
    receipt = {'passed': True, 'audit_result_sha256': hashlib.sha256((audit / 'result.json').read_bytes()).hexdigest(),
               'diagnostic_only': True, 'formal_17ms_claim': False,
               'scope': 'Tiny native A/A and first-four-layer formal prefill routes; no full-model quality equivalence'}
    (out / 'audit_gate.json').write_text(json.dumps(receipt, indent=2) + '\n')
    command = [sys.executable, str(source / 'stack/launch_real_tp8_job.py'),
               '--root', str(root), '--container', args.container,
               '--source-dir', str(args.source_dir), '--job', args.profile_job,
               '--model', '/work/tiny_tp8_8layer_v1', '--chips', '8,9,10,11,12,13,14,15',
               '--tiny-profile', '--tiny-reference-job', 'formal_hostmeta_audit_v1',
               '--arms', 'tp8base', '--pairs', '6', '--profile',
               '--hccl-deterministic', 'strict', '--hccl-npu-socket-port-range', 'auto',
               '--allow-alarm', '--wait-seconds', '60']
    subprocess.run(command, check=True)
    (out / 'profile_started.json').write_text(json.dumps({'argv': command, 'audit_gate': receipt}, indent=2) + '\n')
    print('TINY_PROFILE_GATED_LAUNCH', args.profile_job, flush=True)


if __name__ == '__main__':
    main()
