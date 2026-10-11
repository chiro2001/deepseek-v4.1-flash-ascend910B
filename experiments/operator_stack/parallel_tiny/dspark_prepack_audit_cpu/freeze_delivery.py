"""Freeze this independent CPU component without changing prior frozen files."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    cpu=json.loads((HERE/'evidence/CPU_validation.json').read_text())
    assert cpu['passed'] and cpu['tests']==16 and cpu['negative_case_count']==41
    assert not any(cpu['actual_claims'].values()) and cpu['actual_NPU_calls']==cpu['actual_HTTP_inference_requests']==0
    sources=json.loads((HERE/'evidence/installed_sources.json').read_text())
    assert len(sources['sources'])==19
    for item in sources['sources'].values():
        for value in item.get('functions',{}).values():
            assert not value.get('missing_or_ambiguous') and sha(value['source'].encode())==value['sha256']
    plan=HERE.parent/'dspark_formal_audit/consumer_source_plan.json'
    assert sha(plan.read_bytes())=='5d6cd23dd434662932b4446435e8d632a18136e2785db861aabec3145bc0455c'
    prior={}
    for rel in ('dspark_api_counter_bridge_cpu/frozen_files.json','dspark_api_workload_cpu/frozen_files.json',
            'gmm_lane/sparse_msd/m1_event_perf_v1_failure/frozen_files.json',
            'gmm_lane/sparse_msd/m1_event_perf_v2/runtime_frozen_files.json',
            'gmm_lane/sparse_msd/k5_route_feasibility_cpu/frozen_files.json',
            'dspark_context_graph_design_cpu/frozen_files.json'):
        path=HERE.parent/rel;manifest=json.loads(path.read_text());root=Path(manifest['root'])
        for name,expected in manifest['text_files'].items():
            raw=(root/name).read_bytes();assert sha(raw)==expected['sha256'] and len(raw)==expected['bytes']
        prior[rel]={'checked_unchanged_texts':len(manifest['text_files']),'manifest_sha256':sha(path.read_bytes())}
    texts={}
    for path in sorted(HERE.rglob('*')):
        if (not path.is_file() or '__pycache__' in path.parts or path.name in ('frozen_files.json','delivery.json') or
                path.suffix not in ('.py','.json','.md','.txt','.log')):continue
        data=path.read_bytes();data.decode('utf-8');assert len(data)<1_000_000
        texts[str(path.relative_to(HERE))]={'bytes':len(data),'sha256':sha(data)}
    manifest={'root':str(HERE),'CPU_only':True,'NPU_permission':0,'NPU_calls':0,'HTTP_inference_requests':0,
        'services_started':0,'live_hooks_installed':False,'CPU_tests':16,'negative_cases':41,
        'actual_source_files_guarded':19,'frozen17_plan_unchanged':True,
        'native_math_or_source_weights_modified':False,'text_files':texts,
        'prior_frozen_packages':prior,'actual_flags':cpu['actual_claims']}
    mp=HERE/'frozen_files.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=HERE/'dspark_prepack_audit_cpu_v1_compact.tgz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w') as tar:
                for name in [*texts,'frozen_files.json']:
                    data=(HERE/name).read_bytes();info=tarfile.TarInfo(HERE.name+'/'+name)
                    info.size,info.mode,info.mtime=len(data),0o644,0;tar.addfile(info,io.BytesIO(data))
    data=archive.read_bytes();assert len(data)<1_000_000
    with tarfile.open(archive) as tar:
        for name,expected in texts.items():assert sha(tar.extractfile(HERE.name+'/'+name).read())==expected['sha256']
    delivery={'owned_root':str(HERE),'compact':str(archive),'compact_bytes':len(data),
        'compact_sha256':sha(data),'frozen_manifest_sha256':sha(mp.read_bytes()),
        'frozen_text_file_count':len(texts),'CPU_tests':16,'negative_cases':41,'actual_source_files':19,
        'frozen17_plan_sha256':sha(plan.read_bytes()),'prior_frozen_texts_checked_unchanged':sum(r['checked_unchanged_texts'] for r in prior.values()),
        'NPU_calls':0,'HTTP_inference_requests':0,'services_started':0,'actual_flags':cpu['actual_claims']}
    (HERE/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps(delivery,indent=2))


if __name__=='__main__':main()
