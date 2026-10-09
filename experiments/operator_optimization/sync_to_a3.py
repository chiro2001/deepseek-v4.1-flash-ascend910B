"""Transfer this small source-only experiment into the existing isolated mount."""
import argparse
import ast
import subprocess
import tarfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='a3-21')
    parser.add_argument('--ssh-control-path',help='Use a dedicated task SSH multiplex socket')
    parser.add_argument('--container',help='Extract as root inside this task container when experiment source files are root-owned')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    bundle=runtime/'operator_opt_source.tgz'
    paths=(sorted(root.glob('*.py'))+sorted(root.glob('*.sh'))
           +sorted((root/'baseline').glob('*.py'))+sorted((root/'baseline').glob('*.sh')))
    for path in paths:
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
    with tarfile.open(bundle,'w:gz') as archive:
        for path in paths:archive.add(path,arcname=str(path.relative_to(root)))
    if bundle.stat().st_size>=1000000:
        raise RuntimeError('Use COS for a bundle >=1 MB')
    remote='/home/l00886679/projects/dsv41-tiny-prof-20261009'
    options=['-o','BatchMode=yes','-o','ConnectTimeout=30']
    if args.ssh_control_path:options+=['-o','ControlPath='+args.ssh_control_path]
    subprocess.run(['scp','-q',*options,str(bundle),args.host+':'+remote+'/operator_opt_source.tgz'],check=True)
    destination='/work' if args.container else remote
    code=f"""from pathlib import Path
import tarfile
root=Path({destination!r})
out=root/'operator_opt';out.mkdir(exist_ok=True)
with tarfile.open(root/'operator_opt_source.tgz') as archive:archive.extractall(out,filter='data')
print('Source synced to '+str(out))
"""
    command=['ssh',*options,args.host]
    if args.container:
        import shlex
        command+=['docker exec -i '+shlex.quote(args.container)+' python -']
    else:command+=['python3 -']
    subprocess.run(command,input=code,text=True,check=True)


if __name__=='__main__':main()
