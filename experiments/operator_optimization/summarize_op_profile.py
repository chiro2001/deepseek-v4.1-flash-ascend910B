"""Validate complete msprof op CSVs; preserve per-core counters and overlap caveats."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    args=p.parse_args();roots=list(args.root.glob('OPPROF_*'));assert len(roots)==1,roots
    root=roots[0];basic=list(csv.DictReader((root/'OpBasicInfo.csv').open()))
    assert len(basic)==1 and basic[0]['Device Id']=='4',basic
    summary={'basic':basic[0],'csvs':{},'pipe':{},'note':'Counters overlap; do not sum wait/pipe times. Per-core bandwidth is not whole-chip HBM.'}
    names=['OpBasicInfo','PipeUtilization','Memory','MemoryL0','MemoryUB','L2Cache','ArithmeticUtilization','ResourceConflictRatio']
    for name in names:
        path=root/(name+'.csv');assert path.is_file(),path
        rows=list(csv.DictReader(path.open()));assert rows,(name,'empty')
        summary['csvs'][name]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'path':str(path)}
    rows=list(csv.DictReader((root/'PipeUtilization.csv').open()))
    for prefix in ['aic','aiv']:
        selected=[r for r in rows if r[prefix+'_time(us)']!='NA']
        assert len(selected)==(24 if prefix=='aic' else 48),(prefix,len(selected))
        fields=[k for k in selected[0] if k.startswith(prefix+'_')]
        stats={}
        for key in fields:
            values=[]
            for r in selected:
                try:values.append(float(r[key]))
                except ValueError:pass
            if values:stats[key]={'min':min(values),'median':statistics.median(values),'max':max(values)}
        slow=max(selected,key=lambda r:float(r[prefix+'_time(us)']))
        summary['pipe'][prefix]={'cores':len(selected),'slowest_core':slow,'statistics':stats}
    (args.root/'collection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'basic':summary['basic'],'pipe':{k:{'cores':v['cores'],'slowest_core':v['slowest_core']} for k,v in summary['pipe'].items()}},indent=2))


if __name__=='__main__':main()
