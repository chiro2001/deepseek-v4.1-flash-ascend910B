"""Shape/interval differences from validated formal QKV profiles, CPU only."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            value.update(chunk)
    return value.hexdigest()


def shapes(text):
    if ';' in text:
        return [tuple(map(int, re.findall(r'-?\d+', part))) for part in text.split(';') if part.strip()]
    brackets = re.findall(r'\[([^\[\]]*)\]', text)
    if brackets:
        return [tuple(map(int, re.findall(r'-?\d+', part))) for part in brackets]
    values = tuple(map(int, re.findall(r'-?\d+', text)))
    return [values] if values else []


def family(kind):
    name = re.sub(r'[^a-z]', '', kind.lower())
    if 'quant' in name and 'matmul' in name: return 'QuantMatmul'
    if 'rms' in name and 'norm' in name: return 'RMSNorm'
    if 'viewcopy' in name or 'asstrided' in name: return 'ViewCopy'
    if 'slice' in name: return 'Slice'
    if 'contiguous' in name: return 'Contiguous'
    if 'transdata' in name: return 'LayoutTransData'
    if 'cast' in name: return 'Cast'
    if 'dynamicquant' in name: return 'DynamicQuant'
    return None


def role(row):
    dims = shapes(row['Input Shapes'])
    outputs = shapes(row.get('Output Shapes', ''))
    if family(row['Type']) == 'QuantMatmul' and dims and dims[0] == (1, 5120):
        # Actual WeightNz records expose storage dimensions, not a fictional
        # logical 2D shape: N/32,K/16,16,32. Confirm scale and output width.
        for width, storage, label in [(1280, (40, 320, 16, 32), 'q_a'),
                                       (512, (16, 320, 16, 32), 'kv'),
                                       (1792, (56, 320, 16, 32), 'merged_qkv')]:
            if (storage in dims or (5120, width) in dims or (width, 5120) in dims) and (width,) in dims and (1, width) in outputs:
                return label
    if family(row['Type']) == 'QuantMatmul' and dims and dims[0] == (1, 1280) and (1, 4096) in outputs:
        if (128, 80, 16, 32) in dims or (1280, 4096) in dims or (4096, 1280) in dims:
            return 'q_b_or_indexer_query'
    if family(row['Type']) == 'RMSNorm' and dims:
        if dims[0] == (1, 1280): return 'q_norm_1280'
        if dims[0] == (1, 512): return 'kv_norm_512_or_other'
    return None


def analyze_bank(bank, core, timeline):
    output = Path(bank['output_path'])
    path = output / 'kernel_details.csv'
    before = digest(path)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        raw = list(reader)
    execution, excluded = core.execution_rows(raw)
    rows, completeness = core.complete_step_rows(execution, expected_hc=80)
    assert len({r['Step Id'] for r in rows}) == bank['active_steps'] == 10
    assert {int(r['Device_id']) for r in rows} == {bank['chip']}
    by_shape = defaultdict(list)
    by_family = defaultdict(list)
    by_role = defaultdict(list)
    for row in rows:
        label = family(row['Type'])
        if label:
            by_family[label].append(row)
            key = (label, row['Type'], row['Input Shapes'], row['Input Data Types'], row['Stream ID'],
                   row.get('Input Formats', ''), row.get('Output Shapes', ''), row.get('Output Data Types', ''))
            by_shape[key].append(row)
        label = role(row)
        if label: by_role[label].append(row)
    def record(selected):
        values = [float(r['Duration(us)']) for r in selected]
        return {'calls_per_step': len(values) / 10, 'sum_duration_us_per_step': sum(values) / 10,
                'duration_us': core.distribution(values)}
    focuses = [{'family': key[0], 'type': key[1], 'input_shapes': key[2], 'input_dtypes': key[3],
                'stream_id': key[4], 'input_formats': key[5], 'output_shapes': key[6],
                'output_dtypes': key[7], **record(selected)} for key, selected in by_shape.items()]
    focuses.sort(key=lambda item: item['sum_duration_us_per_step'], reverse=True)
    roles = {key: record(selected) for key, selected in by_role.items()}
    targets = ('q_a', 'kv') if bank['arm'] == 'tp8mask' else ('merged_qkv',)
    assert all(roles.get(key, {}).get('calls_per_step') == 40 for key in targets), roles
    # HcPre alternates attention/FFN and provides all 40 layer intervals.
    layers = []
    missing = []
    for step in sorted({r['Step Id'] for r in rows}, key=int):
        selected = sorted([r for r in rows if r['Step Id'] == step], key=timeline.bounds)
        anchors = [r for r in selected if r['Type'] == 'HcPre']
        assert len(anchors) == 80
        for layer in range(40):
            begin, finish = timeline.bounds(anchors[2 * layer])[0], timeline.bounds(anchors[2 * layer + 1])[0]
            interval = [r for r in selected if begin <= timeline.bounds(r)[0] < finish]
            projections = [r for r in interval if role(r) in ('q_a', 'kv', 'merged_qkv')]
            qnorm = [r for r in interval if role(r) == 'q_norm_1280']
            kvnorm = [r for r in interval if role(r) == 'kv_norm_512_or_other']
            qb = [r for r in interval if role(r) == 'q_b_or_indexer_query']
            ready_projections = [r for r in projections if role(r) in ('kv', 'merged_qkv')]
            if ready_projections:
                kv_ready = max(timeline.bounds(r)[1] for r in ready_projections)
                kvnorm = [r for r in kvnorm if timeline.bounds(r)[0] >= kv_ready]
            if not projections or not qnorm or not kvnorm:
                missing.append({'step': step, 'layer': layer, 'projection_roles': [role(r) for r in projections],
                                'q_norm': len(qnorm), 'kv_norm': len(kvnorm)})
                continue
            q, kv = qnorm[0], kvnorm[0]
            qa, qe = timeline.bounds(q)
            ka, ke = timeline.bounds(kv)
            overlap = max(0, min(qe, ke) - max(qa, ka))
            kv_projects = [r for r in projections if role(r) == 'kv']
            kv_project_qnorm_overlap = timeline.overlap([timeline.bounds(r) for r in kv_projects], qa, qe)
            q_quant = [r for r in interval if family(r['Type']) == 'DynamicQuant'
                       and shapes(r['Input Shapes']) and shapes(r['Input Shapes'])[0] == (1, 1280)
                       and timeline.bounds(r)[0] >= qa]
            project_start = min(timeline.bounds(r)[0] for r in projections)
            project_end = max(timeline.bounds(r)[1] for r in projections)
            copies = [r for r in interval if family(r['Type']) in ('Slice', 'ViewCopy', 'Contiguous', 'Cast', 'LayoutTransData')]
            merged_copies = [r for r in copies if (1, 1792) in shapes(r['Input Shapes'])]
            layers.append({'step': step, 'layer': layer, 'projection_roles': [role(r) for r in projections],
                'projection_union_us': timeline.covered(timeline.bounds(r) for r in projections),
                'projection_span_us': project_end - project_start,
                'q_norm_us': qe - qa, 'kv_norm_us': ke - ka, 'q_kv_norm_overlap_us': overlap,
                'kv_projection_q_norm_overlap_us': kv_project_qnorm_overlap,
                'kv_norm_start_minus_q_norm_end_us': ka - qe,
                'kv_norm_start_minus_q_quant_end_us': ka - timeline.bounds(q_quant[0])[1] if q_quant else None,
                'kv_norm_start_minus_projection_end_us': ka - project_end,
                'projection_to_first_qb_end_us': timeline.bounds(qb[0])[1] - project_start if qb else None,
                'copies_1792_calls': len(merged_copies),
                'copies_1792_sum_us': sum(float(r['Duration(us)']) for r in merged_copies),
                'q_norm_task': timeline.task_receipt(q, project_start),
                'kv_norm_task': timeline.task_receipt(kv, project_start),
                'projections': [timeline.task_receipt(r, project_start) for r in projections],
                'q_quant': [timeline.task_receipt(r, project_start) for r in q_quant],
                'copies_1792': [timeline.task_receipt(r, project_start) for r in merged_copies]})
    detailed_timeline = timeline.analyse(path, bank['arm'], bank['rank'])
    tasks = list(csv.DictReader((output / 'task_time.csv').open()))
    event_types = sorted({r['kernel_type'] for r in tasks if 'EVENT' in r['kernel_type'].upper()})
    event_phases = defaultdict(lambda: defaultdict(list))
    per_window_events = []
    for window in detailed_timeline['window']:
        step = window['step']
        selected = [r for r in rows if r['Step Id'] == step]
        start = min(timeline.bounds(r)[0] for r in selected)
        end = max(timeline.bounds(r)[1] for r in selected)
        graph_ids = set(window['graph_model_ids'])
        graph = [r for r in selected if r['Model ID'] in graph_ids]
        gs = min(timeline.bounds(r)[0] for r in graph)
        ge = max(timeline.bounds(r)[1] for r in graph)
        phase_bounds = {'before_main_graph': (start, gs), 'main_graph': (gs, ge), 'after_main_graph': (ge, end)}
        for kind in event_types:
            source = [r for r in tasks if r['kernel_type'] == kind]
            for phase, (left, right) in phase_bounds.items():
                intervals = [(float(r['task_start(us)']), float(r['task_stop(us)'])) for r in source]
                count = sum(left <= a < right for a, b in intervals)
                union = timeline.overlap(intervals, left, right)
                clipped_sum = sum(max(0, min(right, b) - max(left, a)) for a, b in intervals)
                event_phases[kind][phase].append({'calls_starting_in_phase': count,
                                                'clipped_sum_us': clipped_sum, 'interval_union_us': union})
                if bank['rank'] == 0:
                    per_window_events.append({'step': step, 'kind': kind, 'phase': phase,
                                              'calls': count, 'clipped_sum_us': clipped_sum, 'interval_union_us': union})
    event_stats = {kind: {phase: {key: core.distribution(v[key] for v in values)
                                 for key in ('calls_starting_in_phase', 'clipped_sum_us', 'interval_union_us')}
                         for phase, values in phases.items()} for kind, phases in event_phases.items()}
    assert digest(path) == before, 'CSV modified during analysis'
    layer_summary = {key: core.distribution(r[key] for r in layers if r[key] is not None)
                     for key in ['projection_span_us', 'projection_union_us', 'q_norm_us', 'kv_norm_us',
                                 'q_kv_norm_overlap_us', 'kv_norm_start_minus_q_norm_end_us',
                                 'kv_projection_q_norm_overlap_us', 'kv_norm_start_minus_q_quant_end_us',
                                 'kv_norm_start_minus_projection_end_us', 'projection_to_first_qb_end_us',
                                 'copies_1792_calls', 'copies_1792_sum_us']}
    assert len(layers) == 400 and not missing, ('Incomplete actual layer matching', missing[:5])
    return {**{key: bank[key] for key in ('arm', 'metric', 'rank', 'chip', 'profile_path', 'output_path')},
        'kernel_csv_sha256': before, 'task_csv_sha256': digest(output / 'task_time.csv'),
        'trace_sha256': digest(output / 'trace_view.json'), 'csv_fields': fields,
        'device_complete_window': completeness, 'removed_exact_hccl_envelopes': len(excluded),
        'families': {label: record(selected) for label, selected in by_family.items()}, 'roles': roles,
        'focused_shapes': focuses, 'layer_windows': layers, 'layer_window_missing': missing,
        'layer_window_summary': layer_summary, 'event_phase_stats': event_stats, 'event_windows': per_window_events,
        'timeline': detailed_timeline}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--parsed', type=Path, required=True)
    parser.add_argument('--helpers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.helpers))
    import analyse_formal_microarch as core
    import analyse_formal_timeline as timeline
    banks = json.loads((args.parsed / 'parse_manifest.json').read_text())
    assert banks and all(r['returncode'] == 0 and all(r['required_outputs'].values()) for r in banks)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for bank in banks:
        record = analyze_bank(bank, core, timeline)
        records.append(record)
        print('QKV_REAL_SHAPE_BANK_ANALYZED', json.dumps({'arm': bank['arm'], 'rank': bank['rank'],
              'roles': record['roles'], 'complete_steps': record['device_complete_window']['complete_steps'],
              'layer_windows': len(record['layer_windows']), 'missing': len(record['layer_window_missing'])}), flush=True)
    report = {'scope': 'Diagnostic true-shape and calibrated interval comparison, not E2E or recoverable budget. '
                       'Kernel/event sums overlap; EVENT_WAIT is not useful busy time. CPU inclusive spans are not causal wait proof.',
              'performance_claim': None, 'source_helpers_sha256': {p.name: digest(p) for p in
                      [args.helpers / 'analyse_formal_microarch.py', args.helpers / 'analyse_formal_timeline.py']},
              'analysis_script_sha256': digest(Path(__file__)), 'records': records}
    full = args.output / 'full_analysis.json'
    full.write_text(json.dumps(report, indent=2) + '\n')
    compact = {**report, 'full_analysis_sha256': digest(full)}
    compact['records'] = [{**r, 'layer_windows': r['layer_windows'][:8] if r['rank'] == 0 else [],
                           'layer_window_missing': r['layer_window_missing'][:4],
                           'event_windows': [], 'focused_shapes': r['focused_shapes'] if r['rank'] == 0 else [],
                           'timeline': {k: v for k, v in r['timeline'].items() if k not in ('largest_gaps', 'window', 'first_collectives')}}
                          for r in records]
    target = args.output / 'compact_analysis.json'
    target.write_text(json.dumps(compact, indent=2) + '\n')
    assert target.stat().st_size < 900000
    print('QKV_SHAPE_ANALYSIS_COMPLETE', json.dumps({'banks': len(records), 'full_bytes': full.stat().st_size,
                                                   'compact_bytes': target.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
