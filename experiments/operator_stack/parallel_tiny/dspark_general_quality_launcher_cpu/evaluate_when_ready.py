"""Future exact owned readiness, original strict evaluator, optional graceful stop."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error

from parallel_tiny.dspark_general_quality_api_cpu.contract import verify_identity
from parallel_tiny.dspark_general_quality_api_cpu.orchestrator import request_json
from . import assets
from .common import (false_claims, fingerprint, read_pinned, sha, validate_real_candidate, verify_map)
from .lifecycle import graceful_stop, owned_process


def evaluator_command(source, job_root, job_id, port, asset_manifest):
    paths = asset_manifest['paths']
    return [sys.executable, '-u', '-m', 'parallel_tiny.dspark_general_quality_api_cpu.orchestrator',
            '--selection', str(Path(job_root) / 'service/service_selection.json'), '--job-id', job_id,
            '--base-url', 'http://127.0.0.1:' + str(port), '--client-root', paths['client_root'],
            '--encoder', paths['encoder'], '--images', paths['images'], '--official-dir', paths['official_dir'],
            '--gsm-data', paths['gsm_data'], '--output', str(Path(job_root) / 'client'), '--execute']


def verify_rank_source_owners(identity, source, candidate):
    source = Path(source).resolve(strict=True)
    for rank in identity['rank_descriptions']:
        path = Path(rank['base_worker_source_path']).resolve(strict=True)
        if not path.is_relative_to(source):
            raise ValueError('Actual base worker was imported from another source snapshot')
        relative = str(path.relative_to(source))
        if (rank.get('original_candidate_base_worker') != candidate['model_kwargs']['worker_cls'] or
                rank.get('base_worker_source_sha256') != candidate['source_sha256'].get(relative)):
            raise ValueError('Actual base-worker source/config owner differs from the frozen candidate')
    return True


def ready_once(selection, source, job_root, job_id, port, candidate, runner_identity, request=request_json,
               process_reader=owned_process):
    process = process_reader(selection, source, job_root, job_id, port, candidate, runner_identity)
    if (selection.get('service_started') is not True or selection.get('ASGI_startup_completed') is not True or
            selection.get('engine_initialized') is not True or selection.get('ASGI_shutdown_entered')):
        return {'ready': False, 'owned_process': process, 'reason': 'Native ASGI startup is incomplete'}
    base = 'http://127.0.0.1:' + str(port)
    identity = request(base + '/private/dspark-quality/identity')
    verify_identity(identity, selection, selection['served_model'])
    verify_rank_source_owners(identity, source, candidate)
    catalogue = request(base + '/v1/models')
    if [row.get('id') for row in catalogue.get('data', [])] != [selection['served_model']]:
        raise ValueError('HTTP model catalogue is not the single exact owned quality model')
    return {'ready': True, 'HTTP_readiness_verified': True, 'identity': identity,
            'models': catalogue, 'owned_process': process}


def wait_ready(args, candidate, report, save):
    job = args.job_root
    selection_path = job / 'service/service_selection.json'
    deadline = time.monotonic() + args.ready_timeout
    last_print = 0
    while time.monotonic() < deadline:
        if (job / 'run.exit').is_file():
            raise RuntimeError('Owned service runner exited before actual HTTP readiness')
        try:
            selection = json.loads(selection_path.read_text())
            runner = json.loads((job / 'runner_identity.json').read_text())
            # Validate process ownership before any HTTP request, including incomplete ASGI startup.
            report['owned_process'] = owned_process(selection, args.source, job, args.job_id,
                                                       args.port, candidate, runner)
            result = ready_once(selection, args.source, job, args.job_id, args.port, candidate, runner)
            if result['ready']:
                report['readiness'] = result
                save()
                return selection
            report['last_wait_reason'] = result['reason']
        except (FileNotFoundError, json.JSONDecodeError, urllib.error.URLError,
                urllib.error.HTTPError, ConnectionError, TimeoutError) as exc:
            report['last_wait_reason'] = type(exc).__name__ + ': ' + str(exc)
        now = time.monotonic()
        if now - last_print >= 20:
            print(json.dumps({'job_id': args.job_id, 'waiting': report.get('last_wait_reason', 'No selection yet')}), flush=True)
            save()
            last_print = now
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise TimeoutError('Owned native general API did not become ready within the root-allocated timeout')


def run(args, snapshot, candidate, asset_manifest, command):
    path = args.job_root / 'general_quality_lifecycle.json'
    if path.exists() or (args.job_root / 'client').exists():
        raise FileExistsError('Use a fresh lifecycle receipt and client output')
    launched = json.loads((args.job_root / 'launched.json').read_text())
    if (launched.get('job_id') != args.job_id or launched.get('port') != args.port or
            launched.get('snapshot_manifest_sha256') != args.snapshot_manifest_sha256 or
            launched.get('candidate_fingerprint') != fingerprint(candidate) or
            launched.get('launch_requested') is not True):
        raise ValueError('No exact fresh root-owned launch receipt for this source/job')
    report = {'schema': 'dspark_general_quality_lifecycle_v1', 'CPU_preparation_only': False,
              'job_id': args.job_id, 'port': args.port, 'source': str(args.source),
              'candidate_fingerprint': fingerprint(candidate), 'snapshot_manifest_sha256': args.snapshot_manifest_sha256,
              'evaluator_command': command, 'HTTP_readiness_verified': False, **false_claims()}
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        selection = wait_ready(args, candidate, report, save)
        report['HTTP_readiness_verified'] = True
        save()
        with (args.job_root / 'strict_evaluator.log').open('w') as log:
            process = subprocess.run(command, cwd=asset_manifest['paths']['client_root'],
                                     stdout=log, stderr=subprocess.STDOUT)
        report['strict_evaluator_returncode'] = process.returncode
        if process.returncode != 0:
            raise ValueError('Original strict GSM100/Vision23 orchestrator failed')
        strict = json.loads((args.job_root / 'client/strict_quality_receipt.json').read_text())
        required = ('actual_general_API_quality_observed', 'actual_GSM100_passed',
                    'actual_Vision23_passed', 'actual_client_quality_passed')
        if (any(strict.get(key) is not True for key in required) or
                strict.get('candidate_fingerprint') != fingerprint(candidate) or
                strict.get('service_instance_id') != selection['service_instance_id'] or
                strict.get('performance_receipt_granted') is not False or strict.get('formal_receipt_granted') is not False):
            raise ValueError('Strict quality receipt belongs to another candidate/nonce or is incomplete')
        report.update({key: strict[key] for key in required})
        report['strict_quality_receipt_sha256'] = sha(args.job_root / 'client/strict_quality_receipt.json')
    except BaseException as exc:
        report.update(false_claims(), failed=True, error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    finally:
        if args.stop_after and report.get('owned_process'):
            try:
                report['owned_stop'] = graceful_stop(report['owned_process'], args.job_root,
                                                      timeout=args.stop_timeout, execute=True)
            except BaseException as exc:
                report.update(owned_stop_failed=True, owned_stop_error_type=type(exc).__name__,
                              owned_stop_error=str(exc))
                save()
                raise
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('source', 'job-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--snapshot-manifest-sha256', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--ready-timeout', type=float, default=1800)
    parser.add_argument('--stop-timeout', type=float, default=120)
    parser.add_argument('--stop-after', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    snapshot = read_pinned(args.source / 'quality_source_snapshot.json', args.snapshot_manifest_sha256)
    verify_map(args.source, snapshot['files'])
    candidate = read_pinned(args.source / 'candidate_contract.json', snapshot['candidate_contract']['sha256'])
    validate_real_candidate(candidate)
    asset_manifest = read_pinned(args.source / 'quality_assets.json', snapshot['asset_manifest_sha256'])
    assets.verify(asset_manifest)
    command = evaluator_command(args.source, args.job_root, args.job_id, args.port, asset_manifest)
    if not args.execute:
        print(json.dumps({'CPU_preparation_only': True, 'job_id': args.job_id, 'command': command,
                          'stop_after': args.stop_after, 'NPU_calls': 0, 'HTTP_requests': 0,
                          'services_started': 0, **false_claims()}, indent=2))
        return
    report = run(args, snapshot, candidate, asset_manifest, command)
    print(json.dumps({key: report[key] for key in false_claims()}, indent=2))


if __name__ == '__main__':
    main()
