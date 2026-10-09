"""Check matching current-combo/fused decode windows and replacement counts."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--device',default='4')
    args=p.parse_args();results={}
    for arm in ['combo','gmmact']:
        paths=list((args.root/'prof'/arm).rglob('kernel_details.csv'));assert len(paths)==1,paths
        path=paths[0];rows=list(csv.DictReader(path.open()));steps=collections.Counter(r['Step Id'] for r in rows)
        assert len(steps)==20 and len(set(steps.values()))==1,steps
        assert {r['Device_id'] for r in rows}=={args.device}
        json.loads((path.parent/'trace_view.json').read_text())
        for name in ['op_statistic.csv','api_statistic.csv']:assert (path.parent/name).is_file()
        kinds=collections.Counter(r['Type'] for r in rows)
        expected={'project_static':80,'finish_hc':80,'hc_post_kernel':80,'tiny_router_gemv':40,
                  'route_init_kernel':40,'route_combine_kernel':40,'GroupedMatmul':40,
                  'vector_gemv':40 if arm=='combo' else 0,
                  'clamped_swiglu_kernel':80 if arm=='combo' else 40,
                  'gmm1_activation_kernel':0 if arm=='combo' else 40}
        for k,n in expected.items():assert kinds[k]==20*n,(arm,k,kinds[k],n)
        duration=collections.defaultdict(float)
        for r in rows:
            duration[r['Type']]+=float(r['Duration(us)'])
            if r['Type']=='gmm1_activation_kernel':assert r['Accelerator Core']=='AI_VECTOR_CORE'
        results[arm]={'steps':20,'rows':len(rows),'tasks_per_step':len(rows)/20,
            'device':args.device,'csv':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'trace_valid':True,'structural_checks_passed':True,'kernel_us_per_step':sum(duration.values())/20,
            'types':{k:{'calls_per_step':n/20,'us_per_step':duration[k]/20} for k,n in kinds.items()}}
    assert results['combo']['tasks_per_step']-results['gmmact']['tasks_per_step']==40,results
    (args.root/'profile_validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='types'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
