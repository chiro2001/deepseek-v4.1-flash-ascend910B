"""Host-side fresh TP8 quality launcher. Default makes an offline command plan."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from parallel_tiny.dspark_general_quality_api_cpu.contract import make_plan, validate_candidate
from .common import SAFE_ID, false_claims, fingerprint, read_pinned, reject_fixture, sha, verify_map
from .resource_gate import CHIPS, resources, verify as verify_resources


def docker_python(container, code, *args):
    return subprocess.check_output(['docker', 'exec', container, 'python', '-c', code, *args], text=True)


def verify_snapshot(root, source, expected):
    root = Path(root).resolve(strict=True)
    source = Path(source)
    if not source.is_absolute() or '..' in source.parts or not source.is_relative_to('/work'):
        raise ValueError('Source must be the exact container-visible /work snapshot')
    host = root / source.relative_to('/work')
    snapshot = read_pinned(host / 'quality_source_snapshot.json', expected)
    if (snapshot.get('schema') != 'dspark_general_quality_source_snapshot_v1' or
            snapshot.get('actual_snapshot_files_written') is not True or
            snapshot.get('source_weights_or_original_candidate_code_modified') is not False):
        raise ValueError('An actual immutable CPU-prepared quality snapshot is required')
    verify_map(host, snapshot['files'])
    candidate = read_pinned(host / 'candidate_contract.json', snapshot['candidate_contract']['sha256'])
    reject_fixture(candidate)
    validate_candidate(candidate)
    if candidate['source_root'] != str(source) or fingerprint(candidate) != snapshot['candidate_fingerprint']:
        raise ValueError('Candidate belongs to another quality source snapshot')
    if candidate['source_sha256'] != snapshot['files']:
        raise ValueError('Snapshot and exact runtime candidate source manifest differ')
    if any(snapshot.get(key) for key in false_claims()):
        raise ValueError('Source snapshots cannot inherit actual quality/formal receipts')
    return host, snapshot, candidate


def command_plan(args, snapshot, candidate):
    if not SAFE_ID.fullmatch(args.job_id) or not 1024 <= args.port <= 65535:
        raise ValueError('An independent explicit job/port is required')
    source = Path(args.source_dir)
    job = Path('/work/results') / args.job_id
    plan = make_plan(candidate, args.job_id, args.port, job / 'service', snapshot['counter_file'])
    wrapper = source / 'stack/parallel_tiny/dspark_general_quality_launcher_cpu/run_general.sh'
    bootstrap = ['bash', str(wrapper), 'parallel_tiny.dspark_general_quality_api_cpu.bootstrap',
                 '--candidate', str(source / 'candidate_contract.json'), '--job-id', args.job_id,
                 '--port', str(args.port), '--output', str(job / 'service'),
                 '--counter-file', snapshot['counter_file'], '--execute']
    evaluate = ['bash', str(wrapper), 'parallel_tiny.dspark_general_quality_launcher_cpu.evaluate_when_ready',
                '--source', str(source), '--snapshot-manifest-sha256', args.snapshot_manifest_sha256,
                '--job-id', args.job_id, '--port', str(args.port), '--job-root', str(job),
                '--ready-timeout', str(args.ready_timeout), '--execute']
    if args.stop_after:
        evaluate.append('--stop-after')
    return {'schema': 'dspark_general_quality_host_launch_plan_v1', 'CPU_preparation_only': True,
            'source_dir': str(source), 'job_root': str(job), 'job_id': args.job_id,
            'port': args.port, 'served_model': plan['served_model'],
            'candidate_fingerprint': plan['candidate_fingerprint'],
            'snapshot_manifest_sha256': args.snapshot_manifest_sha256,
            'environment': plan['environment'], 'bootstrap': bootstrap, 'evaluate_when_ready': evaluate,
            'start_service': ['docker', 'exec', '-d', args.container, 'setsid', 'bash', str(job / 'runner.sh')],
            'start_evaluator': ['docker', 'exec', '-d', args.container, 'bash', str(job / 'evaluator.sh')],
            'fresh_occupancy_checks_required': 2, 'physical_chips': list(CHIPS),
            'allow_only_known_alarm_80C98001': bool(args.allow_known_alarm),
            'stop_after': bool(args.stop_after), 'NPU_calls': 0, 'HTTP_requests': 0,
            'services_started': 0, **false_claims()}


def exports(environment):
    lines = []
    for key, value in sorted(environment.items()):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or not isinstance(value, str):
            raise ValueError('Startup environment must be exact string key/value pairs')
        lines.append('export ' + key + '=' + shlex.quote(value))
    return '\n'.join(lines) + '\n'


def script_body(plan, kind):
    job = plan['job_root']
    command = plan['bootstrap'] if kind == 'service' else plan['evaluate_when_ready']
    prefix = 'runner' if kind == 'service' else 'evaluator'
    identity = (
        'import json,os,pathlib; pid=os.getppid(); p=pathlib.Path("/proc")/str(pid); '
        'parts=(p/"stat").read_text().rsplit(")",1)[1].split(); '
        'out={"pid":pid,"start_ticks":int(parts[19]),"process_group":int(parts[2]),'
        '"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()}; '
        'pathlib.Path(' + repr(job + '/' + prefix + '_identity.json') + ').write_text(json.dumps(out)+"\\n")'
    )
    body = '#!/usr/bin/env bash\nset -uo pipefail\numask 022\n'
    body += exports(plan['environment'])
    body += 'python -c ' + shlex.quote(identity) + '\n'
    body += 'date -u +%FT%TZ > ' + shlex.quote(job + '/' + prefix + '_start.utc') + '\n'
    logfile = 'run.log' if kind == 'service' else 'evaluator_runner.log'
    body += shlex.join(command) + ' > ' + shlex.quote(job + '/' + logfile) + ' 2>&1\n'
    body += 'quality_status=$?\nprintf "%s\\n" "$quality_status" > ' + shlex.quote(
        job + ('/run.exit' if kind == 'service' else '/evaluator.exit')) + '\nexit "$quality_status"\n'
    return body


def execute(args, host_source, snapshot, candidate, plan):
    # This branch is for a later root-allocated job. Child CPU preparation never invokes it.
    container = json.loads(subprocess.check_output(['docker', 'inspect', args.container], text=True))[0]
    env = dict(value.split('=', 1) for value in container['Config']['Env'])
    visible = ','.join(map(str, CHIPS))
    if (not container['State']['Running'] or
            any(env.get(key) != visible for key in ('ASCEND_RT_VISIBLE_DEVICES', 'STACK_PHYSICAL_CHIPS')) or
            not any(item['Destination'] == '/work' and Path(item['Source']).resolve() == Path(args.root).resolve()
                    for item in container['Mounts'])):
        raise ValueError('Container/chips/source mount differ from this root-owned allocation')
    source = str(args.source_dir)
    cpu_guard = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root/'stack'))
from parallel_tiny.dspark_general_quality_launcher_cpu.common import checkpoint_metadata,validate_real_candidate
candidate=json.loads((root/'candidate_contract.json').read_text())
validate_real_candidate(candidate);checkpoint_metadata(candidate)
from parallel_tiny.dspark_general_quality_launcher_cpu.assets import verify
verify(json.loads((root/'quality_assets.json').read_text()))
from parallel_tiny.dspark_general_quality_api_cpu.bootstrap import verify_official_sources
official=verify_official_sources()
print(json.dumps({'CPU_source_checkpoint_assets_verified':True,'official_source_receipt':official}))
'''
    guard = json.loads(docker_python(args.container, cpu_guard, source))
    port_code = 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()'
    docker_python(args.container, port_code, str(args.port))
    job = Path(args.root) / 'results' / args.job_id
    if job.exists():
        raise FileExistsError('Quality job/service/client directory must be completely new')
    raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    initial = resources(raw)
    details = {chip: subprocess.check_output(['npu-smi', 'info', '-t', 'health', '-i', str(chip // 2),
                                              '-c', str(chip % 2)], text=True) for chip in CHIPS}
    verify_resources(initial, details, args.allow_known_alarm)
    job.mkdir(parents=True)
    (job / 'runner.sh').write_text(script_body(plan, 'service'))
    (job / 'evaluator.sh').write_text(script_body(plan, 'evaluator'))
    (job / 'resource_prelaunch.txt').write_text(raw)
    (job / 'CPU_launch_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    receipt = {**plan, 'CPU_preparation_only': False, 'container': args.container,
               'container_id': container['Id'], 'container_environment': env,
               'prelaunch_CPU_guard': guard, 'initial_resources': initial,
               'health_details': details, 'runner_sha256': sha(job / 'runner.sh'),
               'evaluator_sha256': sha(job / 'evaluator.sh'), 'service_started': False,
               'HTTP_readiness_verified': False, 'launch_requested': False}
    for key in ('NPU_calls', 'HTTP_requests', 'services_started'):
        receipt.pop(key)
    receipt_path = job / 'launched.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    # Recheck exact files, port and occupancy immediately before requesting the launch.
    verify_map(host_source, snapshot['files'])
    docker_python(args.container, port_code, str(args.port))
    fresh_details = {chip: subprocess.check_output(['npu-smi', 'info', '-t', 'health', '-i', str(chip // 2),
                                                    '-c', str(chip % 2)], text=True) for chip in CHIPS}
    fresh_raw = subprocess.check_output(['npu-smi', 'info'], text=True)
    fresh = resources(fresh_raw)
    verify_resources(fresh, fresh_details, args.allow_known_alarm, previous=initial)
    receipt.update(fresh_resources=fresh, fresh_health_details=fresh_details,
                   fresh_occupancy_timestamp_unix=time.time(), launch_requested=True)
    (job / 'resource_fresh_prelaunch.txt').write_text(fresh_raw)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    subprocess.run(plan['start_service'], check=True)
    # Waiting/readiness/quality flags belong to actual evaluator receipts, never this launch request.
    subprocess.run(plan['start_evaluator'], check=True)
    return {'job_id': args.job_id, 'served_model': plan['served_model'], 'port': args.port,
            'service_launch_requested': True, 'evaluator_launch_requested': True,
            'actual_HTTP_ready_observed': False, **false_claims()}


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('root', 'source-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('container', 'job-id', 'snapshot-manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--ready-timeout', type=float, default=1800)
    parser.add_argument('--allow-known-alarm', action='store_true')
    parser.add_argument('--stop-after', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    host, snapshot, candidate = verify_snapshot(args.root, args.source_dir, args.snapshot_manifest_sha256)
    plan = command_plan(args, snapshot, candidate)
    result = execute(args, host, snapshot, candidate, plan) if args.execute else plan
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
