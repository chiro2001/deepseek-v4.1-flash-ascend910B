"""Host controller: launch timing only after the complete prefix audit passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--container', required=True)
    p.add_argument('--source-dir', required=True)
    p.add_argument('--audit-job', required=True)
    p.add_argument('--perf-job', required=True)
    p.add_argument('--probe-job', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--chips', required=True)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    for name in (args.audit_job, args.perf_job, args.probe_job):
        assert name.replace('_', '').replace('-', '').isalnum()
    assert args.chips == '8,9,10,11,12,13,14,15'
    audit_root = args.root/'results'/args.audit_job
    deadline = time.monotonic()+3600
    print('PREFIX_PERF_WAIT_FOR_COMPLETE_AUDIT', str(audit_root), flush=True)
    while not (audit_root/'run.exit').exists():
        assert time.monotonic() < deadline, 'Audit wait timed out'
        time.sleep(10)
    assert (audit_root/'run.exit').read_text().strip() == '0', 'Audit failed; do not launch performance'
    result = json.loads((audit_root/'result.json').read_text())
    assert result['audit'] and result['formal_weights'] and result['tensor_parallel_size'] == 8
    assert result['model_path'] == args.model and result['hccl_deterministic_env'] == 'strict'
    assert not result['fp32_decode_reduction'] and not result['speculative_decoding']
    arms = ['tp8base', 'tp8metastack', 'tp8prefix', 'tp8prefixroute']
    assert set(result['arms']) == set(arms)
    for arm in arms[1:]:
        pairs = [r for r in result['pairs'] if r['arm'] == arm]
        assert len(pairs) >= 3
        assert all(r['tokens_equal'] and r['actual_routes_equal'] and r['max_logprob_delta'] < 1e-3 for r in pairs)
    controls = json.loads((audit_root/'native_controls.json').read_text())
    assert len(controls) >= 4 and all(r['passed'] for r in controls)
    banks = json.loads((audit_root/'banks.json').read_text())
    requests = json.loads((audit_root/'requests.json').read_text())
    checked = {}
    for arm in ('tp8prefix', 'tp8prefixroute'):
        assert len(banks[arm]) == 8
        for bank in banks[arm]:
            assert bank['w4a8_prefix']['selected'] and bank['w4a8_prefix']['consumer_snapshots'] == 80
            assert bank['w4a8_prefix']['captured_gmm_calls'] == {'apply_gmm1_act_quant': 40, 'apply_gmm2': 40}
        selected = [r for r in requests if r['arm'] == arm and r['tag'].startswith('pair-')]
        assert len(selected) >= 3
        for r in selected:
            assert len(r['rank_audit']) == 8
            for audit in r['rank_audit']:
                assert audit['w4a8_prefix']['passed']
                assert audit['metadata_slots']['all_exact']
                assert len(audit['w4a8_prefix']['native_counts_consumer_comparisons']) == 80
        checked[arm] = {'paired_requests': len(selected), 'ranks_per_request': 8, 'gmm_consumers_per_rank': 80}
    source = Path(args.source_dir)
    assert source.is_absolute() and source.is_relative_to('/work') and '..' not in source.parts
    launcher = args.root/source.relative_to('/work')/'stack/launch_real_tp8_job.py'
    command = [sys.executable, str(launcher), '--root', str(args.root), '--container', args.container,
        '--source-dir', args.source_dir, '--job', args.perf_job, '--model', args.model, '--chips', args.chips,
        '--pairs', '12', '--arms', ','.join(arms), '--prefix-evidence-job', args.probe_job,
        '--hccl-deterministic', 'strict', '--hccl-npu-socket-port-range', 'auto', '--allow-alarm',
        '--wait-seconds', '600']
    if args.profile:
        command.append('--profile')
    receipt = {'full_audit_passed': True, 'checked': checked, 'argv': command,
        'profile_after_unprofiled_timing': args.profile, 'time_unix': time.time()}
    (audit_root/'validated_for_prefix_perf.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('PREFIX_PERF_AUDIT_VALIDATED', json.dumps(receipt), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
