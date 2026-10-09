"""Verify structural replacement and matched 20-step windows for goal20."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();results={}
    expected={'baseline': {'project_hf32':80,'finish_hc':80,'HcPost':80,'MoeInitRoutingV3':40,
                          'GroupedMatmul':80,'tiny_router_gemv':40,'clamped_swiglu_kernel':80},
              'combo': {'project_static':80,'project_hf32':0,'finish_hc':80,'HcPost':0,
                        'hc_post_kernel':80,'MoeInitRoutingV3':0,'route_init_kernel':40,
                        'route_combine_kernel':40,'vector_gemv':40,'GroupedMatmul':40,
                        'tiny_router_gemv':40,'clamped_swiglu_kernel':80}}
    for arm in ['baseline','combo']:
        paths=list((args.root/'prof'/arm).rglob('kernel_details.csv'));assert len(paths)==1,paths
        path=paths[0];rows=list(csv.DictReader(path.open()))
        steps=collections.Counter(r['Step Id'] for r in rows)
        assert len(steps)==20 and len(set(steps.values()))==1,steps
        assert {r['Device_id'] for r in rows}=={'4'}
        json.loads((path.parent/'trace_view.json').read_text())
        for required in ['op_statistic.csv','api_statistic.csv']:assert (path.parent/required).is_file()
        kinds=collections.Counter(r['Type'] for r in rows)
        for kind,calls in expected[arm].items():assert kinds[kind]==20*calls,(arm,kind,kinds[kind],calls)
        duration=collections.defaultdict(float)
        for r in rows:
            duration[r['Type']]+=float(r['Duration(us)'])
            if r['Type'] in ['project_static','hc_post_kernel','route_init_kernel','route_combine_kernel','vector_gemv']:
                assert r['Accelerator Core']=='AI_VECTOR_CORE',r
        results[arm]={'rows':len(rows),'tasks_per_step':len(rows)/20,'steps':20,'device_id':4,
                      'csv':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'structural_checks_passed':True,'trace_valid':True,
                      'kernel_us_per_step':sum(duration.values())/20,
                      'types':{k:{'calls_per_step':n/20,'us_per_step':duration[k]/20} for k,n in kinds.items()}}
    (args.root/'profile_validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='types'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
