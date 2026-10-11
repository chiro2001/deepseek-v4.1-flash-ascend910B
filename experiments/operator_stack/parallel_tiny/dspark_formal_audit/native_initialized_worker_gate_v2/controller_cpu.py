"""Controller-side CPU file builder, using eight NEW worker RPC descriptions."""
import hashlib
import json
from pathlib import Path

if __package__:
    from .contracts import VARIANTS,validate_controller
else:
    from contracts import VARIANTS,validate_controller


def write_controller_bundle(descriptions, *, metadata_root, output_root, checkpoint_config_path,
                            audit_job_id, nonce, variant, physical_chips, generations=32):
    if variant not in VARIANTS or len(descriptions)!=8:
        raise ValueError('Need one variant and eight new normally initialized worker descriptions')
    if {item.get('tp_rank') for item in descriptions}!=set(range(8)):
        raise ValueError('Actual new worker descriptions are missing or duplicate TP ranks')
    manifests=[item.get('source_manifest') for item in descriptions]
    if manifests[0] is None or any(manifest!=manifests[0] for manifest in manifests):
        raise ValueError('Actual candidate/native source and ABI file SHA differ across new workers')
    for item in descriptions:
        if item.get('device_audit_passed') is not False or item.get('safe_formal_receipt') is not False:
            raise ValueError('Worker metadata cannot claim device/formal audit evidence')
    checkpoint_sha=hashlib.sha256(Path(checkpoint_config_path).read_bytes()).hexdigest()
    metadata_root=Path(metadata_root)
    source_text=json.dumps(manifests[0],indent=2)+'\n'
    manifest_path=(metadata_root/'source_manifest.json').resolve()
    controller={'schema':'native_initialized_worker_gate_v2',
                'operation':'run_one_variant_in_new_formal_audit_job',
                'audit_job_id':audit_job_id,'nonce':nonce,'variant':variant,'generations':generations,
                'tp_world_size':8,'physical_chips':list(physical_chips),
                'checkpoint_config_path':str(Path(checkpoint_config_path).resolve()),
                'checkpoint_config_sha256':checkpoint_sha,
                'output_root':str(Path(output_root).resolve()),
                'source_manifest_path':str(manifest_path),
                'source_manifest_sha256':hashlib.sha256(source_text.encode()).hexdigest(),
                'worker_processes':[{key:item[key] for key in ('tp_rank','pid','boot_id','normal_init_return_ns')}
                                    for item in sorted(descriptions,key=lambda item:item['tp_rank'])]}
    validate_controller(controller)
    metadata_root.mkdir(parents=True,exist_ok=False)
    manifest_path.write_text(source_text)
    controller_path=metadata_root/'controller.json'
    controller_text=json.dumps(controller,indent=2)+'\n'
    controller_path.write_text(controller_text)
    return {'controller_path':str(controller_path.resolve()),
            'controller_sha256':hashlib.sha256(controller_text.encode()).hexdigest(),
            'scope':'controller authorization/file preparation only, no SDK/device probe executed',
            'device_audit_passed':False,'safe_formal_receipt':False}
