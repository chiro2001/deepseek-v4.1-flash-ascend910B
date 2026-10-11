"""Validate actual worker receipts and analyse raw profiles in non-daemon CPUs.

The collector/source snapshot is read only. Derived profiler outputs are
generated in private real directories whose raw inputs are remote-local copies.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            value.update(chunk)
    return value.hexdigest()


def validate(job):
    assert (job / 'run.exit').read_text().strip() == '0', 'Collection failed; do not restart'
    selection = json.loads((job / 'profile_selection.json').read_text())
    launched = json.loads((job / 'launched.json').read_text())
    result = json.loads((job / 'result.json').read_text())
    chips = launched['chips']
    assert chips == list(range(8, 16))
    assert selection['arms'] == ['tp8mask', 'tp8qkv']
    assert selection['metrics'] == ['PipeUtilization']
    assert result['formal_weights'] and result['async_scheduling'] and not result['audit']
    assert launched['env']['HCCL_DETERMINISTIC'] == 'strict'
    banks = []
    for arm in selection['arms']:
        for metric in selection['metrics']:
            config = job / f'profile_receipt_{arm}_{metric}.json'
            steps = job / f'profile_steps_{arm}_{metric}.json'
            configs, schedules = json.loads(config.read_text()), json.loads(steps.read_text())
            assert len(configs) == len(schedules) == len(chips)
            assert {r['rank'] for r in configs} == {r['rank'] for r in schedules} == set(range(len(chips)))
            for row in schedules:
                cfg = next(r for r in configs if r['rank'] == row['rank'])
                assert cfg['metric'] == metric and cfg['expected_physical_chip'] == chips[row['rank']]
                assert row['driver'] == cfg['schedule_driver'] == 'worker.execute_model'
                assert row['manual_calls'] == 0 and row['worker_calls'] >= row['required_schedule_steps']
                assert row['warmup'] == cfg['warmup'] == 5 and row['active'] == cfg['active'] == 10
                assert row['schedule_steps'] == row['required_schedule_steps'] == cfg['required_schedule_steps'] == 16
                assert row['complete']
            paths = sorted((job / 'prof' / arm / metric).rglob('*_ascend_pt'))
            assert len(paths) == len(chips), (arm, metric, len(paths))
            ranks = []
            for path in paths:
                match = re.search(r'formal_tp8_rank(\d+)_chip(\d+)', str(path))
                assert match
                rank, chip = map(int, match.groups())
                assert chip == chips[rank]
                ranks.append(rank)
                banks.append({'arm': arm, 'metric': metric, 'rank': rank, 'chip': chip,
                              'profile_path': str(path), 'active_steps': 10,
                              'config_receipt_sha256': digest(config), 'step_receipt_sha256': digest(steps),
                              'worker_schedule': next(r for r in schedules if r['rank'] == rank)})
            assert set(ranks) == set(range(len(chips)))
    expected = len(chips) * len(selection['arms']) * len(selection['metrics'])
    assert len(banks) == expected
    return {'passed_worker_receipts': True, 'expected_directories': expected,
            'source_job': str(job), 'source_snapshot': launched['source_dir'],
            'source_snapshot_sha256': launched['source_sha256'],
            'launched_sha256': digest(job / 'launched.json'), 'result_sha256': digest(job / 'result.json'),
            'selection': selection, 'banks': banks,
            'scope': 'Actual worker schedule accepted; complete device windows are checked after offline export'}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    assert 1 <= args.jobs <= 4
    accepted = validate(args.job)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'worker_validation.json').write_text(json.dumps(accepted, indent=2) + '\n')
    print('QKV_WORKER_WINDOWS_ACCEPTED', json.dumps({'banks': len(accepted['banks']),
                                                    'expected_directories': accepted['expected_directories']}), flush=True)

    def parse(bank):
        source = Path(bank['profile_path'])
        originals = {str(p): digest(p) for p in source.rglob('*.csv') if p.is_file()}
        source_output = source / 'ASCEND_PROFILER_OUTPUT'
        required = ('kernel_details.csv', 'task_time.csv', 'trace_view.json')
        existing = all((source_output / name).is_file() and (source_output / name).stat().st_size > 100 for name in required)
        log = args.output / 'parse_logs' / f"{bank['arm']}_{bank['metric']}_rank{bank['rank']}.log"
        log.parent.mkdir(exist_ok=True)
        if existing:
            output = source_output
            returncode = 0
            log.write_text('Existing source CSV/trace reused read only.\n')
        else:
            staged = args.output / 'staging' / bank['arm'] / bank['metric'] / source.name
            staged.mkdir(parents=True, exist_ok=False)
            for entry in source.iterdir():
                if entry.name == 'ASCEND_PROFILER_OUTPUT':
                    continue
                target = staged / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, symlinks=False)
                else:
                    shutil.copy2(entry, target, follow_symlinks=True)
            code = ('import torch,torch_npu; '
                    'assert not torch.npu.is_initialized(); '
                    'torch_npu.profiler.profiler.analyse(' + repr(str(staged)) +
                    ',max_process_number=1,export_type="text"); '
                    'assert not torch.npu.is_initialized(); print("OFFLINE_CPU_NO_NPU_INIT")')
            with log.open('w') as stream:
                proc = subprocess.run([sys.executable, '-u', '-c', code], stdout=stream, stderr=subprocess.STDOUT)
            returncode = proc.returncode
            output = staged / 'ASCEND_PROFILER_OUTPUT'
        assert all(Path(path).exists() and digest(Path(path)) == sha for path, sha in originals.items()), 'Original CSV changed'
        row = {**bank, 'output_path': str(output), 'returncode': returncode,
               'used_existing': existing, 'original_csv_sha256': originals,
               'original_csv_unchanged': True, 'parse_log': str(log),
               'required_outputs': {name: (output / name).is_file() for name in required}}
        print('QKV_OFFLINE_BANK_PARSED', json.dumps({'arm': bank['arm'], 'rank': bank['rank'],
                                                   'returncode': returncode, 'outputs': row['required_outputs']}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        receipts = list(pool.map(parse, accepted['banks']))
    (args.output / 'parse_manifest.json').write_text(json.dumps(receipts, indent=2) + '\n')
    assert all(r['returncode'] == 0 and all(r['required_outputs'].values()) for r in receipts), 'Offline parse failed; retain terminal logs'
    print('QKV_OFFLINE_PARSE_COMPLETE', json.dumps({'banks': len(receipts), 'jobs': args.jobs}), flush=True)


if __name__ == '__main__':
    main()
