"""Sequential single-chip screens; never run NPU benchmarks concurrently."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--suite', choices=['initial', 'retry'], default='initial')
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '4'
    root = Path(__file__).parent
    jobs = [('short', 'probe_short_ops.py', []),
            ('woa_cube', 'probe_selected_gemv.py', ['--shape=wo_a', '--kind=cube', '--rotations=8'])]
    jobs += [(shape, 'probe_selected_gemv.py', ['--shape='+shape, '--rotations='+str(rotations)])
             for shape, rotations in [('gmm1', 40), ('gmm2', 40), ('shared1', 40),
                                       ('shared2', 80), ('q_b', 8), ('wo_b', 8)]]
    if args.suite == 'retry':
        jobs = [('short', 'probe_short_ops.py', []),
                ('woa_cube', 'probe_selected_gemv.py', ['--shape=wo_a', '--kind=cube', '--rotations=8'])]
        jobs += [(shape, 'probe_selected_gemv.py', ['--shape='+shape, '--kind=vector',
                 '--rotations='+str(rotations)]) for shape, rotations in
                 [('gmm2', 40), ('shared2', 80), ('q_b', 8)]]
    rows = []
    for name, script, options in jobs:
        log = out/(name+'.log')
        with log.open('w') as handle:
            result = subprocess.run([sys.executable, '-u', str(root/script),
                                     '--output='+str(out/(name+'.json')), *options],
                                    stdout=handle, stderr=subprocess.STDOUT)
        row = {'job': name, 'exit_code': result.returncode, 'log': str(log)}
        rows.append(row); (out/'batch.json').write_text(json.dumps(rows, indent=2)+'\n')
        print('SCREEN', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
