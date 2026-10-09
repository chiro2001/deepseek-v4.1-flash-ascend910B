"""Copy small private integration sources without changing either running lane."""
import argparse
import ast
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--container', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--ssh-control-path', default='/home/chiro/.ssh/cm/goal20-final-a3-21')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    operator = root.parent / 'operator_optimization'
    upstream = root.parent / 'upstream950' / 'scripts'
    files = [(f, Path('stack') / f.name) for f in sorted(root.glob('*.py')) + sorted(root.glob('*.sh'))]
    files += [(f, Path('operator') / f.relative_to(operator)) for f in
              sorted(operator.glob('*.py')) + sorted(operator.glob('*.sh')) + sorted((operator / 'baseline').glob('*.py'))]
    files += [(upstream / name, Path('up950') / name) for name in ['indexer_post.py', 'indexer_patches.py']]
    runtime = root / 'runtime'; runtime.mkdir(exist_ok=True)
    bundle = runtime / (args.container + '_source.tgz')
    with tarfile.open(bundle, 'w:gz') as archive:
        for source, destination in files:
            if source.suffix == '.py': ast.parse(source.read_text(), filename=str(source))
            archive.add(source, arcname=str(destination))
    assert bundle.stat().st_size < 1000000, 'Use COS for >=1 MB'
    opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'ControlPath=' + args.ssh_control_path]
    subprocess.run(['scp', '-q', *opts, str(bundle), 'a3-21:' + args.remote_root + '/operator_stack_source.tgz'], check=True)
    code = f"""import tarfile
from pathlib import Path
out=Path({args.destination!r});out.mkdir(parents=True,exist_ok=True)
with tarfile.open('/work/operator_stack_source.tgz') as archive:archive.extractall(out,filter='data')
print('STACK_SOURCE_SYNCED',str(out))
"""
    command = shlex.join(['docker', 'exec', '-i', args.container, 'python', '-'])
    subprocess.run(['ssh', *opts, 'a3-21', command], input=code, text=True, check=True)


if __name__ == '__main__': main()
