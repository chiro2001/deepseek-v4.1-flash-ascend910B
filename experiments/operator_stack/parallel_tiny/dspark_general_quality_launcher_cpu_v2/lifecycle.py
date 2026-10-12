"""Exact container process ownership and future graceful-stop guards."""
import json
import os
from pathlib import Path
import signal
import time

from .common import fingerprint

BOOTSTRAP = 'parallel_tiny.dspark_general_quality_api_cpu.bootstrap'


def proc_identity(pid, proc_root=Path('/proc')):
    if type(pid) is not int or pid <= 1:
        raise ValueError('An exact non-init owned process PID is required')
    path = Path(proc_root) / str(pid)
    raw = (path / 'cmdline').read_bytes()
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    argv = [item.decode() for item in raw.split(b'\0') if item]
    uid_line = next(line for line in (path / 'status').read_text().splitlines() if line.startswith('Uid:'))
    return {'pid': pid, 'argv': argv, 'argv_fingerprint': fingerprint(argv),
            'state': fields[0], 'parent_pid': int(fields[1]), 'process_group': int(fields[2]),
            'start_ticks': int(fields[19]), 'effective_uid': int(uid_line.split()[2]),
            'boot_id': (Path(proc_root) / 'sys/kernel/random/boot_id').read_text().strip()}


def argument(argv, key):
    if argv.count(key) != 1 or argv.index(key) + 1 >= len(argv):
        raise ValueError('Exact owned process argument is missing/duplicated: ' + key)
    return argv[argv.index(key) + 1]


def owned_process(selection, source, job_root, job_id, port, candidate, runner_identity, proc_root=Path('/proc')):
    if (selection.get('job_id') != job_id or selection.get('port') != port or
            selection.get('candidate_fingerprint') != fingerprint(candidate) or
            selection.get('candidate_id') != candidate['candidate_id'] or
            selection.get('candidate_source_root') != str(source) or
            selection.get('original_model_kwargs_fingerprint') != fingerprint(candidate['model_kwargs']) or
            selection.get('output') != str(Path(job_root) / 'service') or
            not selection.get('service_instance_id')):
        raise ValueError('Service selection is not this exact source/candidate/job')
    observed = proc_identity(selection['pid'], proc_root)
    argv = observed['argv']
    expected = {'--job-id': job_id, '--port': str(port), '--output': str(Path(job_root) / 'service'),
                '--candidate': str(Path(source) / 'candidate_contract.json'),
                '--counter-file': str(Path(source) / 'stack/parallel_tiny/dspark_general_quality_launcher_cpu/assets/frozen_v4_counter.py')}
    if BOOTSTRAP not in argv or '-m' not in argv or argv[argv.index('-m') + 1] != BOOTSTRAP:
        raise ValueError('The owned native bootstrap must be an exact python -m process')
    if any(argument(argv, key) != value for key, value in expected.items()):
        raise ValueError('Owned process command line belongs to another quality job')
    if observed['state'] == 'Z' or observed['effective_uid'] != os.geteuid():
        raise ValueError('Owned bootstrap is exited or belongs to another container user')
    runner = proc_identity(runner_identity['pid'], proc_root)
    for key in ('pid', 'start_ticks', 'process_group', 'boot_id'):
        if runner.get(key) != runner_identity.get(key):
            raise ValueError('Owned runner PID was reused or restarted: ' + key)
    if (runner['process_group'] != runner['pid'] or observed['parent_pid'] != runner['pid'] or
            observed['process_group'] != runner['process_group']):
        raise ValueError('Bootstrap is not a child of this newly allocated setsid runner')
    if (runner['state'] == 'Z' or runner['effective_uid'] != os.geteuid() or
            not runner['argv'] or runner['argv'][-1] != str(Path(job_root) / 'runner.sh')):
        raise ValueError('Runner is not this exact owned new job script')
    observed['service_instance_id'] = selection['service_instance_id']
    observed['job_id'] = job_id
    return observed


def match_process(expected, current):
    keys = ('pid', 'start_ticks', 'process_group', 'parent_pid', 'effective_uid', 'boot_id', 'argv_fingerprint')
    if any(expected.get(key) != current.get(key) for key in keys):
        raise ValueError('PID identity changed; refuse to signal any process')
    return True


def graceful_stop(expected, job_root, timeout=120, execute=False, proc_root=Path('/proc')):
    job_root = Path(job_root)
    plan = {'job_id': expected['job_id'], 'pid': expected['pid'], 'signal': 'SIGTERM',
            'exact_process_identity_required': True, 'process_group_signaled': False,
            'timeout_s': timeout, 'signal_sent': False, 'actual_service_stopped': False}
    if not execute:
        return plan
    try:
        current = proc_identity(expected['pid'], proc_root)
    except FileNotFoundError:
        current = None
        plan['already_exited'] = True
    if current is not None:
        match_process(expected, current)
    selection = json.loads((job_root / 'service/service_selection.json').read_text())
    if (selection.get('job_id') != expected['job_id'] or
            selection.get('service_instance_id') != expected['service_instance_id'] or
            selection.get('pid') != expected['pid']):
        raise ValueError('Owned selection nonce changed; refuse to signal')
    if current is not None and current['state'] != 'Z':
        os.kill(expected['pid'], signal.SIGTERM)
        plan['signal_sent'] = True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            current = proc_identity(expected['pid'], proc_root)
            match_process(expected, current)
            live = current['state'] != 'Z'
        except FileNotFoundError:
            live = False
        if not live and (job_root / 'run.exit').is_file():
            after = json.loads((job_root / 'service/service_selection.json').read_text())
            if after.get('service_started') is not False:
                raise ValueError('Native service exit did not clear its lifecycle selection')
            plan.update(actual_service_stopped=True, run_exit=int((job_root / 'run.exit').read_text().strip()),
                        native_selection_stopped=True)
            return plan
        time.sleep(1)
    raise TimeoutError('Owned SIGTERM was sent, but the native service has not exited within the allocated timeout')
