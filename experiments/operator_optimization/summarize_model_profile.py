"""Validate matched profile windows inside the container owning their files."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--arms',default='baseline,fused')
    args=parser.parse_args()
    result={}
    for arm in args.arms.split(','):
        paths=list((args.root/'prof'/arm).rglob('kernel_details.csv'))
        assert len(paths)==1,paths
        path=paths[0]
        rows=list(csv.DictReader(path.open()))
        steps=collections.Counter(row['Step Id'] for row in rows)
        assert len(steps)==20 and len(set(steps.values()))==1,steps
        assert {row['Device_id'] for row in rows}=={'4'}
        json.loads((path.parent/'trace_view.json').read_text())
        for required in ['op_statistic.csv','api_statistic.csv']:assert (path.parent/required).is_file()
        by_type=collections.defaultdict(lambda:[0,0.0])
        blocks=collections.defaultdict(lambda:[0,0.0])
        for row in rows:
            kind=row['Type'];duration=float(row['Duration(us)'])
            by_type[kind][0]+=1;by_type[kind][1]+=duration
            if kind=='clamped_swiglu_kernel':
                assert row['Accelerator Core']=='AI_VECTOR_CORE'
                blocks[row['Block Num']][0]+=1;blocks[row['Block Num']][1]+=duration
        assert by_type['project_hf32'][0]==by_type['finish_hc'][0]==1600
        assert by_type['tiny_router_gemv'][0]==800
        if arm in ['fused','overlap']:
            assert by_type['clamped_swiglu_kernel'][0]==1600
            assert by_type['ViewCopy'][0]==by_type['Slice'][0]==by_type['SwiGlu'][0]==0
            assert blocks['1'][0]==blocks['2'][0]==800
        types={kind:{'calls_per_step':values[0]/20,'us_per_step':values[1]/20} for kind,values in by_type.items()}
        result[arm]={'rows':len(rows),'steps':20,'kernels_per_step':len(rows)/20,
                     'device_ids':['4'],'csv':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'trace_valid':True,'kernel_us_per_step':sum(float(row['Duration(us)']) for row in rows)/20,
                     'streams':dict(collections.Counter(row['Stream ID'] for row in rows)),
                     'types':types,'activation_blocks':{key:{'calls_per_step':values[0]/20,'us_per_step':values[1]/20} for key,values in blocks.items()}}
    output=args.root/'profile_validation.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({arm:{key:value for key,value in values.items() if key not in ['types']} for arm,values in result.items()},indent=2),flush=True)


if __name__=='__main__':main()
