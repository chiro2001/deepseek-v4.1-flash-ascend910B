"""Inspect existing strict profiles, with explicit interval and graph boundaries.

This is a diagnostic report. A gap in kernel_details is not necessarily an
idle NPU: memcpy, event operations and unrecorded tasks can occur there. Host
overlap is descriptive and does not establish the cause of a device wait.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from analyse_formal_microarch import complete_step_rows, distribution, execution_rows, number


def bounds(row):
    start, duration = number(row['Start Time(us)']), number(row['Duration(us)'])
    assert start is not None and duration is not None and duration >= 0
    return start, start + duration


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        assert end >= start
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def covered(intervals):
    return sum(end-start for start, end in merge(intervals))


def overlap(intervals, start, end):
    return covered((max(start, left), min(end, right))
                   for left, right in intervals if right > start and left < end)


def task_receipt(row, origin):
    start, end = bounds(row)
    return {key: row[key] for key in
            ('Name', 'Type', 'Model ID', 'Stream ID', 'Input Shapes')} | {
                'relative_start_us': start-origin, 'duration_us': end-start}


def analyse(path, arm, rank):
    with path.open() as stream:
        all_rows, excluded = execution_rows(list(csv.DictReader(stream)))
    rows,boundary=complete_step_rows(all_rows)
    assert {int(row['Device_id']) for row in rows} == {rank+8}
    steps = sorted({r['Step Id'] for r in rows}, key=int)
    assert len(steps) == 10
    trace_path = path.with_name('trace_view.json')
    trace = json.loads(trace_path.read_text())
    events = trace if isinstance(trace, list) else trace['traceEvents']
    cpu = []
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') not in ('cpu_op', 'enqueue', 'dequeue'):
            continue
        if event['name'].startswith('ProfilerStep#'):
            continue
        start, duration = number(event.get('ts')), number(event.get('dur'))
        assert start is not None and duration is not None and duration >= 0
        cpu.append((start, start+duration, event['name'], event.get('cat')))
    cpu.sort()
    task_rows = list(csv.DictReader(path.with_name('task_time.csv').open()))
    task_types = Counter(row['kernel_type'] for row in task_rows)
    # Show auxiliary tasks inside each kernel gap separately. EVENT_WAIT can
    # occupy a stream without doing useful work, so never count it as busy.
    tasks = defaultdict(list)
    for row in task_rows:
        start, end = number(row['task_start(us)']), number(row['task_stop(us)'])
        assert start is not None and end is not None and end >= start
        tasks[row['kernel_type']].append((start, end))
    window, gap_details, collective_examples = [], [], []
    for step in steps:
        selected = sorted((r for r in rows if r['Step Id'] == step), key=bounds)
        intervals = merge(bounds(r) for r in selected)
        start = min(bounds(r)[0] for r in selected)
        end = max(bounds(r)[1] for r in selected)
        union = covered(intervals)
        gaps = [(left[1], right[0]) for left, right in zip(intervals, intervals[1:])]
        assert abs((end-start)-union-sum(b-a for a,b in gaps)) < .1
        anchors = [r for r in selected if r['Type'] == 'HcPre']
        assert len(anchors) == 80, (arm, rank, step, len(anchors))
        graph_ids = {r['Model ID'] for r in anchors}
        assert len(graph_ids) == 1 and '4294967295' not in graph_ids
        graph = [r for r in selected if r['Model ID'] in graph_ids]
        graph_start = min(bounds(r)[0] for r in graph)
        graph_end = max(bounds(r)[1] for r in graph)
        phases = {'before_main_graph': (start, graph_start),
                  'main_graph': (graph_start, graph_end),
                  'after_main_graph': (graph_end, end)}
        phase_stats = {}
        for name, (left, right) in phases.items():
            active = overlap(intervals, left, right)
            phase_stats[name] = {'span_us': right-left, 'kernel_union_us': active,
                                'kernel_gap_us': right-left-active}
        cpu_intervals = [(a,b) for a,b,_,_ in cpu]
        host_totals = defaultdict(float)
        for left, right, name, _ in cpu:
            if right > start and left < end:
                host_totals[name] += min(right,end)-max(left,start)
        window.append({'step': step, 'span_us': end-start, 'union_us': union,
                       'gap_us': (end-start)-union, 'last_start_sorted_end_shortfall_us':
                       end-bounds(selected[-1])[1], 'graph_model_ids': sorted(graph_ids),
                       'phases': phase_stats, 'gaps_over_50us': sum(b-a>=50 for a,b in gaps),
                       'host_cpu_union_within_kernel_span_us': overlap(cpu_intervals,start,end),
                       'host_top_inclusive_us': sorted(host_totals.items(),key=lambda x:x[1],reverse=True)[:8]})
        for left, right in sorted(gaps, key=lambda x:x[1]-x[0], reverse=True)[:5]:
            before = max((r for r in selected if bounds(r)[1] <= left+.01), key=lambda r:bounds(r)[1])
            after = min((r for r in selected if bounds(r)[0] >= right-.01), key=lambda r:bounds(r)[0])
            host = sorted(((min(b,right)-max(a,left),name,cat) for a,b,name,cat in cpu
                           if b>left and a<right),reverse=True)[:5]
            auxiliary = {kind: value for kind, intervals2 in tasks.items()
                         if (value := overlap(intervals2,left,right)) > 0}
            gap_details.append({'step': step, 'duration_us': right-left,
                                'relative_start_us': left-start,
                                'phase': next(name for name,(a,b) in phases.items() if a<=left<b),
                                'previous': task_receipt(before,start),
                                'next': task_receipt(after,start),
                                'host_overlapping_inclusive_us': host,
                                'other_task_kind_union_in_gap_us': auxiliary})
        collectives = [r for r in selected if r['Name']=='AivKernel' and r['Type']=='hcom_allReduce_']
        assert len(collectives)==82
        if rank==0:
            for ordinal,r in enumerate(collectives[:3]):
                collective_examples.append({'step':step,'ordinal':ordinal,**task_receipt(r,start)})
    return {'arm':arm,'rank':rank,'device_id':rank+8,'steps':steps,
            'source_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            'execution_rows':len(rows),'removed_hccl_envelopes':len(excluded),'boundary_rows':boundary,
            'task_types':dict(task_types),'window':window,
            **{key:distribution(r[key] for r in window) for key in ('span_us','union_us','gap_us')},
            'phase_means':{name:{key:distribution(r['phases'][name][key] for r in window)
                                 for key in ('span_us','kernel_union_us','kernel_gap_us')}
                           for name in phases},
            'largest_gaps':gap_details,'first_collectives':collective_examples}


def write_report(report, root):
    full = root/'timeline_analysis_v2.json'
    full.write_text(json.dumps(report,indent=2)+'\n')
    compact = dict(report)
    compact['full_report_sha256'] = hashlib.sha256(full.read_bytes()).hexdigest()
    compact['records'] = [dict(record, largest_gaps=sorted(record['largest_gaps'],
        key=lambda r:r['duration_us'], reverse=True)[:10] if record['rank']==0 else [])
        for record in report['records']]
    target=root/'timeline_summary_v2.json'
    target.write_text(json.dumps(compact,indent=2)+'\n')
    assert target.stat().st_size<1000000,'Keep detailed analysis remotely or use COS'
    print(json.dumps({'records':len(report['records']),'full_bytes':full.stat().st_size,
                      'summary_bytes':target.stat().st_size,'output':str(target)}))


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--arms',default='tp8base,tp8core')
    args=parser.parse_args()
    arms=args.arms.split(',')
    assert len(arms)==len(set(arms)) and len(arms)>=1
    assert set(arms)<= {'tp8base','tp8core','tp8metastack','tp8hostmeta'}
    records=[]
    for arm in arms:
        paths=sorted((args.root/'prof'/arm/'PipeUtilization').rglob('kernel_details.csv'))
        assert len(paths)==8
        for path in paths:
            match=re.search(r'formal_tp8_rank(\d+)_chip(\d+)',str(path))
            assert match and int(match[2])==int(match[1])+8
            records.append(analyse(path,arm,int(match[1])))
    report={'scope':'Diagnostic kernel interval union with end=max(end) across streams; '
            'not E2E. Other tasks may fill kernel gaps. CPU inclusive overlap and '
            'HCCL starts do not establish causality or recoverable latency. '
            'Main graph ID is inferred from all 80 HcPre anchors in each step.',
            'performance_claim':None,'arms':arms,'records':records}
    write_report(report,args.root)


if __name__=='__main__':main()
