"""Controller-side CPU file builder, using eight NEW worker RPC descriptions."""
import hashlib
import json
from pathlib import Path

if __package__:
    from .contracts import VARIANTS,validate_controller
    from .mapped_object_proof import verify_proof
    from .runtime_scope import ABI_SYMBOLS
else:
    from contracts import VARIANTS,validate_controller
    from mapped_object_proof import verify_proof
    from runtime_scope import ABI_SYMBOLS


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
    libraries=manifests[0].get('libraries',{})
    if set(libraries)!=set(ABI_SYMBOLS):
        raise ValueError('Need common exact ACL/runtime file identity and separate per-rank proofs')
    proof_texts={}
    for item in descriptions:
        proofs=item.get('loaded_abi_proofs',{})
        if set(proofs)!=set(ABI_SYMBOLS):
            raise ValueError('Each initialized rank needs complete mapped ABI proofs')
        for name,proof in proofs.items():
            if (proof.get('pid')!=item['pid'] or proof.get('resolved_path')!=libraries[name]['path']
                    or proof.get('mapped_file_bytes')!=libraries[name]['bytes']):
                raise ValueError('Mapped ABI proof is not bound to this new rank PID/common file')
            verify_proof(proof,libraries[name]['sha256'],ABI_SYMBOLS[name])
        proof_texts[item['tp_rank']]=json.dumps(proofs,indent=2)+'\n'
    checkpoint_sha=hashlib.sha256(Path(checkpoint_config_path).read_bytes()).hexdigest()
    metadata_root=Path(metadata_root).resolve()
    driver_output_root=Path(output_root).resolve().parent
    audit_job_root=driver_output_root.parent
    controller_path=metadata_root/'controller.json'
    source_text=json.dumps(manifests[0],indent=2)+'\n'
    manifest_path=(metadata_root/'source_manifest.json').resolve()
    controller={'schema':'native_initialized_worker_discovery_v3',
                'operation':'run_one_variant_in_new_formal_audit_job',
                'audit_job_id':audit_job_id,'nonce':nonce,'variant':variant,'generations':generations,
                'audit_job_root':str(audit_job_root),'driver_output_root':str(driver_output_root),
                'controller_path':str(controller_path),
                'tp_world_size':8,'physical_chips':list(physical_chips),
                'checkpoint_config_path':str(Path(checkpoint_config_path).resolve()),
                'checkpoint_config_sha256':checkpoint_sha,
                'output_root':str(Path(output_root).resolve()),
                'source_manifest_path':str(manifest_path),
                'source_manifest_sha256':hashlib.sha256(source_text.encode()).hexdigest(),
                'worker_processes':[{**{key:item[key] for key in ('tp_rank','pid','boot_id','normal_init_return_ns')},
                                     'loaded_abi_proofs_path':str(metadata_root/'loaded_abi_proofs'/f"rank{item['tp_rank']}.json"),
                                     'loaded_abi_proofs_sha256':hashlib.sha256(proof_texts[item['tp_rank']].encode()).hexdigest()}
                                    for item in sorted(descriptions,key=lambda item:item['tp_rank'])]}
    validate_controller(controller)
    metadata_root.mkdir(parents=True,exist_ok=False)
    manifest_path.write_text(source_text)
    proof_root=metadata_root/'loaded_abi_proofs';proof_root.mkdir()
    for rank,proof_text in proof_texts.items():(proof_root/f'rank{rank}.json').write_text(proof_text)
    controller_text=json.dumps(controller,indent=2)+'\n'
    controller_path.write_text(controller_text)
    return {'controller_path':str(controller_path.resolve()),
            'audit_job_id':audit_job_id,'audit_job_root':str(audit_job_root),
            'driver_output_root':str(driver_output_root),'controller_nonce':nonce,
            'controller_sha256':hashlib.sha256(controller_text.encode()).hexdigest(),
            'scope':'controller authorization/file preparation only, no SDK/device probe executed',
            'device_audit_passed':False,'safe_formal_receipt':False}
