"""Gate twelve formal QKV pairings on the complete forty-layer audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--container', required=True)
    p.add_argument('--audit-job', required=True)
    p.add_argument('--perf-job', required=True)
    p.add_argument('--probe-job', required=True)
    p.add_argument('--mask-probe-job', required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()
    for name in (args.audit_job, args.perf_job, args.probe_job, args.mask_probe_job):
        assert name.replace('_', '').replace('-', '').isalnum()
    assert args.source_dir.is_relative_to('/work') and '..' not in args.source_dir.parts
    out = args.root / 'results' / args.audit_job
    deadline = time.monotonic() + 3600
    while not (out / 'run.exit').exists():
        assert time.monotonic() < deadline, 'Audit wait expired; inspect existing job'
        time.sleep(10)
    assert (out / 'run.exit').read_text().strip() == '0', 'Audit failed; reject performance launch'
    result = json.loads((out / 'result.json').read_text())
    arms = ['tp8base', 'tp8mask', 'tp8qkv']
    assert set(result['arms']) == set(arms) and result['model_path'] == args.model
    assert result['formal_weights'] and result['audit'] and result['async_scheduling']
    assert result['hccl_deterministic_env'] == 'strict' and result['profiler'] == 'OFF'
    assert result['physical_chips'] == list(range(8, 16)) and result['tensor_parallel_size'] == 8
    assert result['same_model_instance'] and result['same_processes_per_rank']
    assert not result['fp32_decode_reduction'] and not result['speculative_decoding']
    controls = json.loads((out / 'native_controls.json').read_text())
    assert len(controls) >= 3 and all(r['passed'] for r in controls)
    banks = json.loads((out / 'banks.json').read_text())
    requests = json.loads((out / 'requests.json').read_text())
    for arm in arms[1:]:
        comparisons = [r for r in result['pairs'] if r['arm'] == arm]
        assert len(comparisons) >= 3 and all(r['tokens_equal'] and r['actual_routes_equal'] and
                                           r['max_logprob_delta'] < 1e-3 for r in comparisons)
        assert len(banks[arm]) == 8 and {r['rank'] for r in banks[arm]} == set(range(8))
        assert all(r['moe_mask']['selected_calls'] == 40 and r['moe_mask']['unique_layers'] == 40 for r in banks[arm])
        selected = [r for r in requests if r['arm'] == arm and r['tag'].startswith('pair-')]
        assert len(selected) >= 3
        for request in selected:
            assert len(request['rank_audit']) == 8
            for rank in request['rank_audit']:
                assert rank['metadata_slots']['all_exact']
                assert rank['moe_mask']['passed'] and rank['moe_mask']['all_bitwise_equal']
                assert rank['moe_mask']['consumer_calls'] == rank['moe_mask']['unique_layers'] == 40
                if arm == 'tp8qkv':
                    assert rank['qkv_merge']['passed'] and rank['qkv_merge']['all_bitwise_equal']
                    assert rank['qkv_merge']['consumer_calls'] == rank['qkv_merge']['unique_layers'] == 40
        if arm == 'tp8qkv':
            assert all(r['qkv_merge']['selected_calls'] == r['qkv_merge']['unique_layers'] == 40 for r in banks[arm])
    source = args.root / args.source_dir.relative_to('/work')
    command = [sys.executable, str(source / 'stack/launch_real_tp8_job.py'),
               '--root', str(args.root), '--source-dir', str(args.source_dir), '--container', args.container,
               '--job', args.perf_job, '--model', args.model, '--chips', '8,9,10,11,12,13,14,15',
               '--arms', ','.join(arms), '--pairs', '12', '--async-scheduling',
               '--mask-evidence-job', args.mask_probe_job, '--qkv-evidence-job', args.probe_job,
               '--hccl-deterministic', 'strict', '--hccl-npu-socket-port-range', 'auto',
               '--allow-alarm', '--wait-seconds', '600']
    receipt = {'passed': True, 'argv': command, 'arms': arms,
               'ranks': 8, 'unique_consumer_layers_per_rank': 40}
    (out / 'validated_for_qkv_perf.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('FORMAL_QKV_FULL_AUDIT_VALIDATED', flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
