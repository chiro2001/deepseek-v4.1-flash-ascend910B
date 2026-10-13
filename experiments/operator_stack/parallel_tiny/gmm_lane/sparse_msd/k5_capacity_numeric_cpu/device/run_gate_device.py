"""One-shot container runner for the K5 capacity physical4 window.

Refuses any second attempt in the same root, re-checks the freshness of the
host-owned preflight snapshot, runs the numeric gate once, and archives the
actual API event counters from the runtime log. The full log stays in this
private root.
"""
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parent
ROOT = LINE.parent
PHYSICAL = 4


def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == str(PHYSICAL)
    assert os.environ.get('TASK_QUEUE_ENABLE') == '0'
    assert os.environ.get('CAPACITY_EXPECTED_PHYSICAL') == str(PHYSICAL)
    assert not (ROOT / 'runtime_attempt.json').exists(), 'No runtime retry'
    assert not (ROOT / 'run_receipt.json').exists(), 'No runtime retry'
    assert not (ROOT / 'runtime.log').exists(), 'No runtime retry'
    preflight = json.loads((ROOT / 'preflight.json').read_text())
    age = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.datetime.fromisoformat(preflight['UTC'])).total_seconds()
    assert 0 <= age <= 30, age
    assert preflight['physical'] == PHYSICAL and preflight['physical_processes'] == []
    assert preflight['health_allowed'] and preflight['container_has_dev_permission']
    script = sys.argv[1] if len(sys.argv) > 1 else 'runtime_capacity_gate.py'
    assert script in ('runtime_capacity_gate.py',)
    command = ['python3', str(HERE / script)]
    attempt = {'command': command, 'script': script,
               'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'physical': PHYSICAL, 'logical': 0,
               'ASCEND_RT_VISIBLE_DEVICES': str(PHYSICAL), 'ASCEND_VISIBLE_DEVICES': str(PHYSICAL),
               'CAPACITY_EXPECTED_PHYSICAL': str(PHYSICAL), 'TASK_QUEUE_ENABLE': '0',
               'K5_CAPACITY_PHASE': os.environ.get('K5_CAPACITY_PHASE', 'numeric_then_timing'),
               'scope': 'K5 target36/E48 and draft15/E16 numeric + minimal operator-window timing',
               'fullnet_K5_verifier': False, 'admission': False, 'retries_authorized': False}
    with (ROOT / 'runtime_attempt.json').open('x') as handle:
        handle.write(json.dumps(attempt, indent=2) + '\n')
    timeout = False
    with (ROOT / 'runtime.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=420)
        except subprocess.TimeoutExpired:
            timeout = True
            process.send_signal(signal.SIGTERM)
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill(); code = process.wait()
    data = (ROOT / 'runtime.log').read_bytes()
    terminal_path = ROOT / 'runtime_terminal.json'
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    events = []
    for line in data.decode(errors='replace').splitlines():
        if line.startswith('{'):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    stages = {}
    for event in events:
        if event.get('code') == 0 and 'bridge_stage' in event:
            stages[event['bridge_stage']] = stages.get(event['bridge_stage'], 0) + 1
    receipt = {'schema': 'k5_capacity_device_gate_v1', 'command': command, 'exit_code': code, 'timeout': timeout,
               'child_PID': process.pid, 'terminal': terminal, 'events': events,
               'accepted_stage_counters': stages, 'runtime_log_bytes': len(data),
               'runtime_log_sha256': hashlib.sha256(data).hexdigest(), 'runtime_retries': 0}
    (ROOT / 'run_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    summary = {k: v for k, v in receipt.items() if k not in ['events', 'terminal']}
    if terminal:
        summary['terminal_passed'] = terminal['passed']
        summary['terminal_keys'] = sorted(terminal.keys())
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    raise SystemExit(code if 0 <= code <= 255 else 1)


if __name__ == '__main__':
    main()
