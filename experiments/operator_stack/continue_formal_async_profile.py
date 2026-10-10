"""Wait for the owned tiny job, parse its trace, then profile formal async banks."""
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
    parser.add_argument('--tiny-job', required=True)
    parser.add_argument('--formal-job', required=True)
    parser.add_argument('--controller-job', required=True)
    args = parser.parse_args()
    for name in (args.tiny_job, args.formal_job, args.controller_job):
        assert name.replace('_', '').replace('-', '').isalnum()
    root = args.root.resolve()
    source = root / args.source_dir.relative_to('/work')
    out = root / 'results' / args.controller_job
    out.mkdir(parents=True, exist_ok=True)
    preflight = root / 'results/async_profile_driver_preflight_v1/result.json'
    assert json.loads(preflight.read_text())['passed']
    audit = root / 'results/tiny_tp8_alignment_audit_v3'
    assert (audit / 'run.exit').read_text().strip() == '0'
    audited = json.loads((audit / 'result.json').read_text())
    assert all(r['passed'] for r in audited['native_pairs'])
    assert all(r['prefix_routes_equal'] for r in audited['formal_prefix_comparisons'])
    job = root / 'results' / args.tiny_job
    launched = json.loads((job / 'launched.json').read_text())
    assert launched['container'] == args.container and launched['env']['STACK_TINY_PROFILE'] == '1'
    deadline = time.monotonic() + 3600
    while not (job / 'run.exit').exists():
        code = ("from pathlib import Path; p=Path('/work/results/" + args.tiny_job +
                "/runner.pid'); pid=p.read_text().strip(); "
                "assert Path('/proc/'+pid).exists(),'Tiny runner vanished'; print(pid)")
        pid = subprocess.check_output(['docker', 'exec', args.container, 'python', '-c', code], text=True).strip()
        print('TINY_PROFILE_VERIFIED_LIVE', pid, flush=True)
        assert time.monotonic() < deadline, 'Observation timed out; do not restart the model'
        time.sleep(20)
    assert (job / 'run.exit').read_text().strip() == '0', 'Tiny profiling failed; inspect before proceeding'
    result = json.loads((job / 'result.json').read_text())
    assert result['diagnostic_only'] and not result['audit'] and result['profiler_during_timing'] == 'OFF'

    def container_run(script, *arguments):
        command = ['docker', 'exec', '-e', 'STACK_SRC_ROOT=' + str(args.source_dir),
                   '-e', 'STACK_TINY_PROFILE=0', args.container, 'bash',
                   str(args.source_dir / 'stack/run_stack.sh'), script, *arguments]
        subprocess.run(command, check=True)

    container_run('parse_formal_profile.py', '--root=/work/results/' + args.tiny_job,
                  '--expected-directories=56', '--jobs=4')
    container_run('analyse_formal_microarch.py', '--root=/work/results/' + args.tiny_job, '--layers=8')
    receipt = {'tiny_result_sha256': hashlib.sha256((job / 'result.json').read_bytes()).hexdigest(),
               'tiny_analysis_sha256': hashlib.sha256((job / 'shape_microarch_summary.json').read_bytes()).hexdigest(),
               'driver_preflight_sha256': hashlib.sha256(preflight.read_bytes()).hexdigest(),
               'scope': 'Tiny alignment and raw seven-metric parse completed; no full-model17ms claim'}
    (out / 'tiny_completed.json').write_text(json.dumps(receipt, indent=2) + '\n')
    command = [sys.executable, str(source / 'stack/launch_real_tp8_job.py'), '--root', str(root),
               '--source-dir', str(args.source_dir), '--container', args.container,
               '--job', args.formal_job,
               '--model', '/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq',
               '--chips', '8,9,10,11,12,13,14,15', '--arms', 'tp8base,tp8metastack',
               '--pairs', '12', '--async-scheduling', '--profile',
               '--hccl-deterministic', 'strict', '--hccl-npu-socket-port-range', 'auto',
               '--allow-alarm', '--wait-seconds', '60']
    subprocess.run(command, check=True)
    (out / 'formal_started.json').write_text(json.dumps({'argv': command, 'completed': receipt}, indent=2) + '\n')
    print('FORMAL_ASYNC_PROFILE_STARTED', args.formal_job, flush=True)


if __name__ == '__main__':
    main()
