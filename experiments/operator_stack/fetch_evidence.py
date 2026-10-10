"""Fetch only compact receipts; retain large trace/request payloads remotely."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--container', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', default='a3-21')
    p.add_argument('--ssh-control-path', default='/home/chiro/.ssh/cm/operator-stack-a3-21')
    p.add_argument('files', nargs='+')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for filename in args.files:
        assert not Path(filename).is_absolute() and '..' not in Path(filename).parts
        code = ("from pathlib import Path\nimport sys\np=Path("+repr(args.remote_root)+")/"+repr(filename)+
                "\nassert p.stat().st_size<1000000,'Use COS for >=1 MB'\nsys.stdout.buffer.write(p.read_bytes())")
        command = shlex.join(['docker', 'exec', args.container, 'python', '-c', code])
        data = subprocess.check_output(['ssh', '-S', args.ssh_control_path,
            '-o', 'ConnectTimeout=30', args.host, command])
        assert len(data) < 1000000
        target = args.output/filename; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        receipts.append({'file': filename, 'host': args.host, 'container': args.container, 'bytes': len(data),
                         'sha256': hashlib.sha256(data).hexdigest()})
        print('FETCHED', json.dumps(receipts[-1]), flush=True)
    manifest = args.output/'fetch_manifest.json'
    old = json.loads(manifest.read_text()) if manifest.is_file() else []
    manifest.write_text(json.dumps([r for r in old if r['file'] not in args.files]+receipts, indent=2)+'\n')


if __name__ == '__main__': main()
