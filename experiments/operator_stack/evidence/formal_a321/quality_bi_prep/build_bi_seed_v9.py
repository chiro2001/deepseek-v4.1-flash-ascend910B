"""Build the batch-invariant candidate v4 (in-container).

Source tree = the mls snapshot + sitecustomize that installs the NumPy-style
``axis`` alias shim for the vendor batch-invariant ``reduce_sum``. Everything
else is byte-identical to the v2/v3 quality candidate.
"""
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

SRC = Path('/work/src_dspark_native_mls_v1')
BI_SRC = Path('/work/src_dspark_native_mls_bi_v1')
OUT = Path('/work/q2_candidate_v9')
MODEL = Path('/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq')
SDK_SRC = Path('/work/q2_candidate/sdk_manifest_v1.json')
COMPAT_SRC = Path('/work/q2_candidate_v3/bi_axis_compat.py')

SITECUSTOMIZE_APPEND = '''

# [BI-AXIS-COMPAT] vllm_ascend.batch_invariant lowers torch.sum to reduce_sum
# which rejects the NumPy-style ``axis`` alias the vendor W4A8 loader uses.
# Only active under the explicit batch-invariant switch.
if os.environ.get('VLLM_BATCH_INVARIANT', '0') == '1':
    try:
        import bi_axis_compat as _bi_axis_compat
        _bi_axis_compat.install()
        _bi_axis_compat.install_meta_kernels()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(2)
'''

ENGINE_FLAGS = {
    'ASCEND_MAX_OP_CACHE_SIZE': '-1',
    'HCCL_BUFFSIZE': '1024',
    'HCCL_NPU_SOCKET_PORT_RANGE': 'auto',
    'HCCL_OP_EXPANSION_MODE': 'AIV',
    'NUMBA_CACHE_DIR': '/work/cache/numba',
    'PYTORCH_NPU_ALLOC_CONF': 'expandable_segments:True',
    'TASK_QUEUE_ENABLE': '1',
    'V41_ENGRAM_DEVICE_FALLBACK': '0',
    'V41_ENGRAM_DEVICE_INDEX': '1',
    'V41_ENGRAM_GATE_CHUNK': '0',
    'V41_ENGRAM_GATE_HOIST': '0',
    'V41_ENGRAM_GATE_MAX_TOKENS': '4096',
    'V41_ENGRAM_HOST_RESIDENT': '1',
    'V41_ENGRAM_JIT': '1',
    'V41_ENGRAM_REUSE_EP_GROUP': '1',
    'V41_ENGRAM_ROUTE_PROBE': '0',
    'V41_MOE_COMM_ALLGATHER': '1',
    'V41_MOE_MASK_RANGE': '1',
    'V41_MOE_ZERO_INVALID': '0',
    'V41_MOE_ZERO_NONFINITE': '0',
    'V41_O_PROJ_2D': '1',
    'V41_QLI_NO_CANDIDATE': '1',
    'V41_ROPE_IDXSEL': '1',
    'VLLM_BATCH_INVARIANT': '1',
    'ASCEND_CUSTOM_OPP_PATH': '/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def copy_tree():
    if BI_SRC.exists():
        shutil.rmtree(BI_SRC)
    tmproot = (SRC / 'runtime' / 'tmp').resolve()

    def ignore(directory, names):
        out = [n for n in names if n == '__pycache__']
        if Path(directory).resolve() == tmproot:
            out.extend(n for n in names if n != '__pycache__')
        return out

    shutil.copytree(SRC, BI_SRC, symlinks=True, ignore=ignore)
    tmpdst = BI_SRC / 'runtime' / 'tmp'
    if not tmpdst.exists():
        tmpdst.mkdir(mode=0o1777)
    else:
        tmpdst.chmod(0o1777)
    shutil.copyfile(COMPAT_SRC, BI_SRC / 'stack/bi_axis_compat.py')
    site = BI_SRC / 'stack/sitecustomize.py'
    site.write_text(site.read_text() + SITECUSTOMIZE_APPEND)
    print('source tree prepared at', BI_SRC)


def source_map(root):
    mapping = {}
    for name in ('stack', 'operator', 'up950'):
        for path in sorted((root / name).rglob('*')):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == '.pyc':
                continue
            if path.suffix == '.safetensors' or not path.is_file() or path.is_symlink():
                raise ValueError('non-source entry: ' + str(path))
            mapping[str(path.relative_to(root))] = sha(path)
    return mapping


def main():
    copy_tree()
    OUT.mkdir(parents=True, exist_ok=True)
    sdk = OUT / 'sdk_manifest_v1.json'
    sdk.write_bytes(SDK_SRC.read_bytes())
    sdk_sha = sha(sdk)
    root = str(BI_SRC)
    spec = importlib.util.spec_from_file_location(
        'q4_contract', BI_SRC / 'stack/parallel_tiny/dspark_formal_prep/contract.py')
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    mapping = source_map(BI_SRC)
    source_manifest = OUT / 'source_manifest_v1.json'
    source_manifest.write_text(json.dumps({'schema': 'dspark_general_quality_source_manifest_v1',
                                           'source_root': root, 'source_sha256': mapping}, indent=2) + '\n')
    os.environ.update(DSPARK_ASYNC_SCHEDULING='1', DSPARK_ASYNC_COUNTER='1',
                      V41_MLS_SHARED_EXPERT='1', V41_MLS_DSA_OVERLAP='1')
    model_kwargs = contract.model_kwargs(str(MODEL), 5, audit=False, draft_graph=True)
    env = contract.startup_env(Path('/work/results/q4_placeholder/native'), audit=False, draft_graph=True)
    env.update(DSPARK_ASYNC_COUNTER='1', DSPARK_ASYNC_SCHEDULING='1',
               V41_MLS_SHARED_EXPERT='1', V41_MLS_DSA_OVERLAP='1',
               DSPARK_QUALITY_ASYNC_ARM='1', STACK_SRC_ROOT=root,
               ASCEND_RT_VISIBLE_DEVICES='8,9,10,11,12,13,14,15',
               ASCEND_VISIBLE_DEVICES='8,9,10,11,12,13,14,15')
    env.update(ENGINE_FLAGS)
    basis = {'qualified_function': 'parallel_tiny.dspark_formal_prep.contract.startup_env',
             'audit': False, 'draft_graph': True,
             'source_path': root + '/stack/parallel_tiny/dspark_formal_prep/contract.py',
             'source_sha256': sha(BI_SRC / 'stack/parallel_tiny/dspark_formal_prep/contract.py')}
    seed = {'schema': 'dspark_general_quality_actual_source_candidate_seed_v1',
            'candidate_id': 'dsv41_mls_async_bi_v1', 'source_root': root, 'source_sha256': mapping,
            'pythonpath': [root + '/stack', root + '/operator', root + '/operator/baseline', root + '/up950'],
            'model_kwargs': model_kwargs,
            'checkpoint_config': {'path': str(MODEL / 'config.json'), 'sha256': sha(MODEL / 'config.json')},
            'sdk_source_manifest_sha256': sdk_sha,
            'sdk_source_manifest': {'path': str(sdk), 'sha256': sdk_sha},
            'sdk_pair_provenance': {'source': 'frozen sdk_rank0 receipt + installed SDK sources'},
            'audit_enabled': False, 'startup_env': env,
            'startup_env_fingerprint': fingerprint(env), 'startup_env_basis': basis,
            'cpu_preparation_only': True}
    seed_path = OUT / 'candidate_seed_v1.json'
    seed_path.write_text(json.dumps(seed, indent=2) + '\n')
    print(json.dumps({'candidate_seed': str(seed_path), 'candidate_seed_sha256': sha(seed_path),
                      'source_manifest_sha256': sha(source_manifest),
                      'sitecustomize_sha256': sha(BI_SRC / 'stack/sitecustomize.py'),
                      'compat_sha256': sha(BI_SRC / 'stack/bi_axis_compat.py'),
                      'source_files': len(mapping)}, indent=1))


if __name__ == '__main__':
    main()
