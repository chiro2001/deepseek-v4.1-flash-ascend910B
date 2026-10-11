"""One-shot container runner. Full runtime log stays in this private root."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path('/work/gmm_lane_sparse_msd_event_perf_cpu_20261011_v1')


def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == '6'
    assert os.environ.get('TASK_QUEUE_ENABLE') == '0'
    assert not (ROOT / 'runtime_attempt.json').exists() and not (ROOT / 'run_receipt.json').exists(), 'No runtime retry'
    preflight = json.loads((ROOT / 'preflight.json').read_text())
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(preflight['UTC'])).total_seconds()
    assert 0 <= age <= 30
    assert preflight['physical6_processes'] == [] and preflight['health_allowed'] and preflight['container_has_dev6_permission']
    command = ['python3', str(ROOT / 'source/runtime_event_gate.py')]
    attempt = {'command': command, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'physical': 6, 'logical': 0, 'ASCEND_RT_VISIBLE_DEVICES': '6', 'ASCEND_VISIBLE_DEVICES': '6',
               'TASK_QUEUE_ENABLE': '0', 'scope': 'M1-capacity6 event durations only, planner and fresh input included, no fullnet/K5',
               'performance_fullnet_K5': False, 'retries_authorized': False}
    with (ROOT / 'runtime_attempt.json').open('x') as f: f.write(json.dumps(attempt, indent=2) + '\n')
    timeout = False
    with (ROOT / 'runtime.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timeout = True
            process.send_signal(signal.SIGTERM)
            try: code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); code = process.wait()
    data = (ROOT / 'runtime.log').read_bytes()
    terminal_path = ROOT / 'runtime_terminal.json'
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    events = []
    for line in data.decode(errors='replace').splitlines():
        if line.startswith('{'):
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass
    raw_launches = {label: sum(e.get('bridge_stage') == stage and e.get('code') == 0 for e in events)
                    for label, stage in [('raw_native_submit_accepted', 'HostArgs:native'),
                                         ('candidate_submit_accepted', 'HostArgs:candidate'),
                                         ('planner_submit_accepted', 'HostArgs:planner_included'),
                                         ('raw_math_stream_sync_success', 'sync:raw_math')]}
    if terminal:
        raw_launches = {'native_candidate_planner_API_accepted': terminal.get('actual_raw_native_candidate_planner_submit_accepted', []),
                        'cases_completed': terminal.get('cases_completed', [])}
    receipt = {'actual_launch_counters': raw_launches, 'command': command, 'exit_code': code, 'timeout': timeout, 'child_PID': process.pid,
               'terminal': terminal, 'events': events, 'runtime_log_bytes': len(data),
               'runtime_log_sha256': hashlib.sha256(data).hexdigest(), 'runtime_retries': 0,
               'dependency_sha256_after': {record['path']: hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
                                         for record in json.loads((ROOT / 'frozen_dependencies.json').read_text())['frozen_v3']['objects'].values()}}
    (ROOT / 'run_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    summary = {k: v for k, v in receipt.items() if k not in ['events', 'terminal', 'dependency_sha256_after']}
    if terminal:
        summary['terminal'] = {k: v for k, v in terminal.items() if k not in ['all_case_results','partial_case_result','traceback']}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    raise SystemExit(code if 0 <= code <= 255 else 1)


if __name__ == '__main__':
    main()
