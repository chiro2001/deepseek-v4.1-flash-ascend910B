"""api_bi_rs_v1 = api_bi_v1 + env-gated enable_reduce_sample (V41_RS)."""
import hashlib, json, shutil, sys
from pathlib import Path
R = Path('/home/l00886679/projects/dsv41-real-stack-tp8-20261010')
SRC = R / 'src_dspark_native_graph_api_bi_v1'
DST = R / 'src_dspark_native_graph_api_bi_rs_v1'
CONTRACT = 'stack/parallel_tiny/dspark_formal_prep/contract.py'
PATCHED = Path('/home/l00886679/tmp/q2stage/contract_bi_rs.py')
MANIFEST = 'native_api_snapshot.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    mode = sys.argv[1]
    if mode == 'copy':
        if DST.exists(): shutil.rmtree(DST)
        tmproot = (SRC / 'runtime' / 'tmp').resolve()
        def ignore(d, names):
            out = [n for n in names if n == '__pycache__']
            if Path(d).resolve() == tmproot:
                out.extend(n for n in names if n != '__pycache__')
            return out
        shutil.copytree(SRC, DST, symlinks=True, ignore=ignore)
        tmp = DST / 'runtime' / 'tmp'
        tmp.mkdir(mode=0o1777) if not tmp.exists() else tmp.chmod(0o1777)
        shutil.copyfile(PATCHED, DST / CONTRACT)
        print('api-bi-rs snapshot copied')
        return
    if mode == 'manifest':
        man = json.loads((DST / MANIFEST).read_text())
        man['files'][CONTRACT] = sha(DST / CONTRACT)
        man['reduce_sample_arm'] = {'gate': 'V41_RS', 'base': SRC.name}
        (DST / MANIFEST).write_text(json.dumps(man, indent=1) + '\n')
        print('manifest updated; contract', man['files'][CONTRACT][:16])
        return
    raise SystemExit('usage: copy|manifest')
if __name__ == '__main__':
    main()
