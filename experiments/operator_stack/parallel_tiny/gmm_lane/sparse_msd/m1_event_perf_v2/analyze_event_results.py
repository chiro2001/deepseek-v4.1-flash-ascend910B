"""Readonly statistics for all 324 actual v2 samples; never initializes NPU."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/'evidence/runtime_v2_success'
METHODS=('SDK','RAW_NATIVE','CANDIDATE')
FIELDS=('operator_event_ms','fresh_input_and_operator_event_ms','host_submit_ns','host_submit_through_stop_sync_ns')

def quantile(values,q):
    ordered=sorted(values);pos=(len(ordered)-1)*q;low=int(pos);high=min(low+1,len(ordered)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(pos-low)

def distribution(values):
    return {'n':len(values),'min':min(values),'p05':quantile(values,.05),
            'median':statistics.median(values),'p95':quantile(values,.95),'max':max(values)}

def main():
    manifest=json.loads((EVIDENCE/'runtime_file_manifest.json').read_text())
    for name,record in manifest['files'].items():
        p=EVIDENCE/name
        assert p.stat().st_size==record['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
    terminal=json.loads((EVIDENCE/'runtime_terminal.json').read_text())
    release=json.loads((EVIDENCE/'release_fresh_smi.json').read_text())
    receipt=json.loads((EVIDENCE/'run_receipt.json').read_text())
    assert terminal['passed'] and terminal['pre_event_raw_smoke_completed']
    assert terminal['actual_raw_native_candidate_planner_submit_accepted']==[136,135,135]
    assert terminal['actual_sdk_submit_accepted']==terminal['actual_sdk_numeric_completed']==135
    assert terminal['actual_raw_numeric_completed']==136 and terminal['actual_candidate_numeric_completed']==135
    assert terminal['cleanup_code']==0 and not terminal['owner_retained']
    assert release['physical6_released'] and release['health_allowed'] and release['container_global_env_unchanged']
    assert receipt['runtime_retries']==0 and not receipt['timeout'] and receipt['exit_code']==0
    dependencies=json.loads((EVIDENCE/'frozen_dependencies.json').read_text())
    assert receipt['dependency_sha256_after']=={r['path']:r['sha256'] for r in dependencies['frozen_v3']['objects'].values()}
    stages={}
    for event in receipt['events']:
        if event.get('code')==0 and 'bridge_stage' in event:
            stage=event['bridge_stage'];stages[stage]=stages.get(stage,0)+1
    assert stages['D2H:original_parameter_storage']==11*50+406*3
    assert stages['perf:sync_after_stop_event']==stages['workspace_and_y_canary_guard']==271
    assert stages['perf:D2H:frame_after_events']==135
    cases=[];csv_rows=[];max_ulp=0;all_numeric=0;all_descriptor=0
    for name in terminal['cases_completed']:
        actual=json.loads((EVIDENCE/(name+'_event_result.json')).read_text())
        assert actual['passed'] and actual['full_E48_seal_passed'] and actual['same_GM_fresh_x']
        numeric=actual['numeric_checks'];all_numeric+=len(numeric);all_descriptor+=actual['descriptor_after_call_checks']
        max_ulp=max(max_ulp,max(n['max_ULP'] for n in numeric))
        measurements=actual['measurements'];assert len(measurements)==36
        by_round={}
        for sample in measurements:
            assert sample['D2H_oracle_canaries_seals_outside_timing'] and sample['restored_same_GM_x_before_call']
            assert sample['planner_in_operator_window']==(sample['method']=='CANDIDATE')
            assert sample['operator_event_ms']>0 and sample['fresh_input_and_operator_event_ms']>=sample['operator_event_ms']
            assert sample['host_submit_ns']>0 and sample['host_submit_through_stop_sync_ns']>=sample['host_submit_ns']
            by_round.setdefault(sample['round'],{})[sample['method']]=sample
        assert set(by_round)==set(range(12)) and all(set(row)==set(METHODS) for row in by_round.values())
        per_method={m:{f:distribution([row[m][f] for row in by_round.values()]) for f in FIELDS} for m in METHODS}
        paired={}
        for baseline in ['RAW_NATIVE','SDK']:
            paired[baseline]={}
            for field in FIELDS:
                deltas=[row['CANDIDATE'][field]-row[baseline][field] for row in by_round.values()]
                paired[baseline][field]={'candidate_minus_baseline':distribution(deltas),
                                         'candidate_faster_rounds':sum(v<0 for v in deltas),
                                         'candidate_over_baseline_median_ratio':per_method['CANDIDATE'][field]['median']/per_method[baseline][field]['median']}
        case={'case':name,'R':actual['R'],'active_experts':actual['active_experts'],'per_method':per_method,
              'paired_same_round':paired,'numeric_checks':len(numeric),'max_ULP':max(n['max_ULP'] for n in numeric),
              'descriptor_after_call_checks':actual['descriptor_after_call_checks']}
        cases.append(case)
        csv_rows.append({'case':name,'R':actual['R'],'active_experts':len(actual['active_experts']),
                         'SDK_operator_median_us':1000*per_method['SDK']['operator_event_ms']['median'],
                         'RAW_operator_median_us':1000*per_method['RAW_NATIVE']['operator_event_ms']['median'],
                         'CANDIDATE_operator_median_us':1000*per_method['CANDIDATE']['operator_event_ms']['median'],
                         'candidate_vs_RAW_ratio':paired['RAW_NATIVE']['operator_event_ms']['candidate_over_baseline_median_ratio'],
                         'paired_candidate_minus_RAW_median_us':1000*paired['RAW_NATIVE']['operator_event_ms']['candidate_minus_baseline']['median'],
                         'candidate_faster_than_RAW_rounds':paired['RAW_NATIVE']['operator_event_ms']['candidate_faster_rounds'],
                         'candidate_vs_SDK_ratio':paired['SDK']['operator_event_ms']['candidate_over_baseline_median_ratio']})
    assert len(cases)==9 and all_numeric==all_descriptor==406 and max_ulp==0
    faster={baseline:sum(case['per_method']['CANDIDATE']['operator_event_ms']['median']<case['per_method'][baseline]['operator_event_ms']['median'] for case in cases) for baseline in ['RAW_NATIVE','SDK']}
    if faster['RAW_NATIVE']==0:outcome='NO_M1_EVENT_GAIN_CANDIDATE_SLOWER_THAN_RAW_ALL_9_MEDIANS'
    elif faster['RAW_NATIVE']==9:outcome='M1_EVENT_GAIN_OBSERVED_ALL_9_MEDIANS_ONLY'
    else:outcome='MIXED_M1_EVENT_RESULTS_NO_UNIFORM_GAIN'
    result={'runtime_passed':True,'numeric_passed':True,'event_measurements_completed':324,'warmup_samples_completed':81,
            'pre_event_raw_smoke_numeric_completed':1,'numeric_checks_completed':all_numeric,'descriptor_after_call_checks':all_descriptor,
            'max_ULP_all_406_calls_against_frozen_v5_SDK':max_ulp,'candidate_faster_operator_median_cases':faster,
            'full_E48_seal_invocations':11,'actual_parameter_read_API_successes':1768,
            'raw_candidate_stream_sync_and_canary_checks':271,'planner_frames_exactly_equal_oracle':135,
            'four_frozen_dependency_SHA_after_match':True,
            'performance_outcome':outcome,'event_scope':'M1 same-instance operator/queue interval including possible host-submit idle; synthetic activation',
            'pure_core_time_claim':False,'K5_or_fullnet_claim':False,'fullnet_metrics':{'ms_per_step':None,'acceptance_A':None,'tokens_per_second':None},
            'no_sample_exclusions':True,'paired_by_same_case_round':True,'statistics_quantiles':'linear interpolation over all 12 measured samples per method and case',
            'cases':cases,'runtime_log_bytes':receipt['runtime_log_bytes'],'runtime_log_sha256':receipt['runtime_log_sha256']}
    (ROOT/'event_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'event_statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    final={'runtime_passed':True,'numeric_passed':True,'max_ULP':max_ulp,'performance_outcome':outcome,
           'candidate_faster_operator_median_cases':faster,'measured_event_samples':324,'numeric_calls_including_smoke':406,
           'actual_API_accepted':{'SDK':135,'RAW_NATIVE':136,'CANDIDATE':135,'PLANNER':135},
           'numeric_completed':{'SDK':135,'RAW_NATIVE':136,'CANDIDATE':135},'cleanup_code':0,'owner_retained':False,
           'physical6_released':True,'physical6_Health_OK':True,'container_global_env_unchanged':True,
           'runtime_retries':0,'runtime_window_consumed':True,'NPU_permission_after_window':0,
           'K5_or_fullnet_claim':False,'pure_core_time_claim':False,'readonly_runtime_archive':json.loads((ROOT/'runtime_archive_receipt.json').read_text())}
    (ROOT/'runtime_final_status.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final,indent=2))
    for row in csv_rows:print(json.dumps(row))

if __name__=='__main__':main()
