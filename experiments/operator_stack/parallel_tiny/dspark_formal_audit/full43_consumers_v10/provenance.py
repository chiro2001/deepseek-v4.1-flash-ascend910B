"""Immutable prior SDK evidence and candidate metadata; stdlib-only imports."""
import hashlib
import json
import os
from pathlib import Path


BRIDGE_FREEZE_SHA = '1bbdc613969add1f47d9e678ac9ade9d35ad9e187a9985e8c21b67ff881623e5'


def read_sdk_precondition(base=None):
    base = Path(base) if base else Path(__file__).parent.parent/'formal_consumer_sdk_bridge_v9_cpu'
    raw = (base/'freeze.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != BRIDGE_FREEZE_SHA:raise ValueError('Immutable dual SDK bridge freeze changed')
    freeze = json.loads(raw)
    name = 'evidence/actual_pair_protocol.json';data = (base/name).read_bytes()
    if hashlib.sha256(data).hexdigest() != freeze['files'][name]:raise ValueError('Frozen original SDK pair provenance changed')
    proof = json.loads(data)
    if proof['protocol_passed'] is not True or len(proof['static']['ranks']) != 8 or len(proof['nonstatic']['ranks']) != 8:
        raise ValueError('Both complete actual SDK jobs are required')
    if proof['static']['variant'] != 'static_keep' or proof['nonstatic']['variant'] != 'nonstatic_keep':
        raise ValueError('Immutable SDK pair mode changed')
    return proof,{'path':str((base/name).resolve()),'sha256':hashlib.sha256(data).hexdigest(),
                  'bridge_freeze_sha256':BRIDGE_FREEZE_SHA,'scope':'prior hardware capability prerequisite only'}


def candidate_config(model,output,*,audit_enabled):
    from parallel_tiny.dspark_formal_prep import contract
    root = Path(output)
    kwargs = contract.model_kwargs(model,5,audit=False,draft_graph=False)
    env = contract.startup_env(root,audit=False,draft_graph=False)
    stack = Path(__file__).resolve().parents[3]
    source_root = Path(os.environ.get('STACK_SRC_ROOT',str(stack.parent))).resolve()
    pythonpath = [str(source_root/name) for name in ('stack','operator','operator/baseline','up950')]
    overrides = {}
    if audit_enabled:
        kwargs['worker_cls'] = 'parallel_tiny.dspark_formal_audit.full43_consumers_v10.sdk_worker.Full43AuditWorker'
        kwargs['distributed_executor_backend'] = 'parallel_tiny.dspark_formal_audit.full43_consumers_v10.sdk_executor.Full43AuditExecutor'
        overrides = {'STACK_REAL_AUDIT':'1','V10_AUDIT_ENABLED':'1','V10_AUDIT_OUTPUT':str(root/'rank_exports')}
    env.update(overrides)
    env.update(STACK_SRC_ROOT=str(source_root),ASCEND_RT_VISIBLE_DEVICES=contract.PHYSICAL_CHIPS,
               ASCEND_VISIBLE_DEVICES=contract.PHYSICAL_CHIPS)
    source = Path(contract.__file__).resolve();source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config = Path(model)/'config.json'
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    if digest != '40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424':
        raise ValueError('Formal checkpoint config changed')
    proof,reference = read_sdk_precondition()
    sdk_manifest = Path(__file__).parent/'evidence/sdk_source_manifest_static.json'
    if hashlib.sha256(sdk_manifest.read_bytes()).hexdigest() != proof['static']['source_manifest']['sha256']:
        raise ValueError('Original frozen SDK source manifest bytes changed')
    return {'schema':'full43_consumers_v10','candidate_id':'v10_native_consumer_observation',
        'source_root':str(source_root),'candidate_package_root':str(Path(__file__).parent.resolve()),
        'source_sha256':{str(p.resolve().relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'model_kwargs':kwargs,'startup_env':env,
        'startup_env_basis':{'qualified_function':'parallel_tiny.dspark_formal_prep.contract.startup_env',
                            'source_path':str(source),'source_sha256':source_hash,'audit':False,'draft_graph':False},
        'startup_env_overrides':overrides,'pythonpath':pythonpath,
        'checkpoint_config':{'path':str(config.resolve()),'sha256':digest},
        'sdk_source_manifest_sha256':proof['static']['source_manifest']['sha256'],
        'sdk_source_manifest':{'path':str(sdk_manifest.resolve()),'sha256':proof['static']['source_manifest']['sha256'],
            'bytes':sdk_manifest.stat().st_size,'original_pair_manifest_reference':dict(proof['static']['source_manifest'])},
        'sdk_pair_provenance':proof,'sdk_pair_reference':reference,'audit_enabled':bool(audit_enabled),
        'PYTHONPATH_source_root':str(stack),
        'native_math_and_formal_model_kwargs_preserved':True,
        'device_audit_passed':False,'full_consumer_audit_passed':False,'safe_formal_receipt':False,
        'formal_model_validated':False,'actual_prepack_consumer_order_validated':False,'performance_claim':None}
