"""Project complete accepted target input routes onto actual EP source ranges.

Read source NPZ only on the remote CPU. Never infer rejected/missing rows,
never call these projected values actual GMM consumer group_list readbacks.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import numpy as np

ROOT=Path('/work/gmm_lane_sparse_msd_k5_route_cpu_20261011_v1')
FORMAL=Path('/work/results/formal_dspark_native_eager_aa_v3')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def summarize(rows):
    assert rows
    fields={key:[row[key] for row in rows] for key in ['R','active_experts','last_eid']}
    return {'observations':len(rows),**{key:{'min':min(v),'median':statistics.median(v),
                'mean':statistics.mean(v),'max':max(v),'histogram':dict(sorted(Counter(v).items()))} for key,v in fields.items()},
            'empty':sum(r['R']==0 for r in rows),'single_active_expert':sum(r['active_experts']==1 for r in rows),
            'single_active_local_e47':sum(r['active_experts']==1 and r['last_eid']==47 for r in rows),
            'single_active_late40_to47':sum(r['active_experts']==1 and r['last_eid']>=40 for r in rows),
            'exact_M1_measured_gain_pattern_R1_R3_R6_e47':sum(r['active_experts']==1 and r['last_eid']==47 and r['R'] in (1,3,6) for r in rows),
            'fraction_exact_M1_measured_gain_pattern':sum(r['active_experts']==1 and r['last_eid']==47 and r['R'] in (1,3,6) for r in rows)/len(rows)}

def main():
    assert os.environ.get('ASCEND_VISIBLE_DEVICES')==os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==''
    guard=json.loads((ROOT/'source_guards.json').read_text());assert guard['passed']
    mapping=guard['EP_mapping'];assert mapping['EP']==8 and mapping['placement']=='linear' and not mapping['enable_eplb']
    for record in guard['sources'].values():assert digest(record['path'])==record['sha256']
    requests_path=FORMAL/'requests.json';requests=json.loads(requests_path.read_text())
    selected_requests=[r for r in requests if r['tag'].startswith('pair-')]
    assert len(selected_requests)==6
    selections=[];file_sources={str(requests_path):{'bytes':requests_path.stat().st_size,'sha256':digest(requests_path)}}
    all_rows=[];unique_rows=[];seen_route_sha=set();per_layer_rank=defaultdict(list);round_source_digests=Counter()
    full_path=ROOT/'projected_counts_full.jsonl'
    with full_path.open('w') as full:
        for request in selected_requests:
            tag=request['tag'];counter_path=FORMAL/(tag+'_counters.json');npz_path=FORMAL/request['routes_path']
            for path in [counter_path,npz_path]:file_sources[str(path)]={'bytes':path.stat().st_size,'sha256':digest(path)}
            ranks=json.loads(counter_path.read_text());assert len(ranks)==8
            assert sorted(r['rank'] for r in ranks)==list(range(8))
            by_rank={r['rank']:r for r in ranks};request_id=request['core_request_id']
            for rank in ranks:
                assert rank['tag']==tag and not rank['errors'] and request_id in rank['requests']
            records={rank:by_rank[rank]['requests'][request_id] for rank in range(8)}
            rounds=records[0]['rounds'];assert len(rounds)==8
            assert all(len(records[r]['rounds'])==len(rounds) for r in range(8))
            with np.load(npz_path,allow_pickle=False) as archive:
                assert archive.files==['routes'];routes=archive['routes']
            assert list(routes.shape)==request['route_shape'] and routes.shape[1:]==(40,6)
            assert routes.dtype==np.uint16 and int(routes.min())>=0 and int(routes.max())<384
            assert hashlib.sha256(routes.tobytes()).hexdigest()==request['route_sha256']
            emitted_offset=len(records[0]['seed_token_ids'])
            assert request['token_ids'][:emitted_offset]==records[0]['seed_token_ids']
            for index,round0 in enumerate(rounds):
                reasons=[];start=round0['computed_before'];input_count=round0['scheduled_tokens']
                expected_output=round0['token_ids'];actual_output=request['token_ids'][emitted_offset:emitted_offset+6]
                if input_count!=6 or round0.get('verified_draft_tokens')!=5 or round0.get('accepted_draft_tokens_raw')!=5 or len(expected_output)!=6:
                    reasons.append('not_complete_K5_acceptance')
                if expected_output[:5]!=round0['scheduled_draft_token_ids']:reasons.append('accepted_draft_ids_not_actual_output_prefix')
                if len(actual_output)!=6 or actual_output!=expected_output:reasons.append('six_actual_emitted_output_ids_missing_or_different')
                input_routes=routes[start:start+6]
                if input_routes.shape!=(6,40,6):reasons.append('six_input_route_rows_not_exported')
                route_sha=hashlib.sha256(input_routes.astype(np.int32).tobytes()).hexdigest()
                receipts=[]
                for rank in range(8):
                    round_r=records[rank]['rounds'][index];receipt=round_r.get('main_route_receipt',{})
                    same_keys=['computed_before','scheduled_tokens','scheduled_draft_token_ids','verified_draft_tokens','token_ids','accepted_draft_tokens_raw','round']
                    if any(round_r.get(k)!=round0.get(k) for k in same_keys):reasons.append('cross_rank_round_metadata_mismatch')
                    if receipt.get('shape')!=[6,40,6] or receipt.get('dtype')!='int32' or not receipt.get('all_layers_present'):
                        reasons.append('rank_route_receipt_geometry_or_layer_coverage_missing')
                    if receipt.get('sha256')!=route_sha:reasons.append('exported_input_slice_SHA_not_rank_receipt')
                    receipts.append({'rank':rank,'main_route_sha256':receipt.get('sha256'),'slice_SHA_equal':receipt.get('sha256')==route_sha})
                reasons=sorted(set(reasons));kept=not reasons
                selections.append({'tag':tag,'request_id':request_id,'round':round0['round'],'computed_before':start,
                                   'scheduled_inputs':input_count,'nominal_accepted_draft_tokens':round0.get('accepted_draft_tokens_raw'),
                                   'actual_output_slice_ids':actual_output,'counter_output_ids':expected_output,
                                   'input_slice_shape':list(input_routes.shape),'int32_route_SHA':route_sha,'all_rank_receipts':receipts,
                                   'selected':kept,'exclusion_reasons':reasons})
                emitted_offset+=len(expected_output)
                if not kept:continue
                unique=route_sha not in seen_route_sha;seen_route_sha.add(route_sha);round_source_digests[route_sha]+=1
                for layer in range(40):
                    global_counts=np.bincount(input_routes[:,layer,:].reshape(-1).astype(np.int64),minlength=384)
                    assert global_counts.shape==(384,) and int(global_counts.sum())==36
                    assert all(len(set(row.tolist()))==6 for row in input_routes[:,layer,:])
                    projected_sum=0
                    for rank,(first,last) in enumerate(mapping['target_ranges']):
                        counts=global_counts[first:last];assert counts.shape==(48,)
                        active=np.flatnonzero(counts);R=int(counts.sum());projected_sum+=R
                        row={'tag':tag,'round':round0['round'],'layer':layer,'rank':rank,
                             'R':R,'active_experts':len(active),'last_eid':int(active[-1]) if len(active) else -1,
                             'counts_48':counts.tolist(),'role':'target','kind':'CPU_projection_of_complete_accepted_input_routes',
                             'actual_GMM_consumer_count_readback':False}
                        full.write(json.dumps(row,separators=(',',':'))+'\n')
                        all_rows.append(row);per_layer_rank[layer,rank].append(row)
                        if unique:unique_rows.append(row)
                    assert projected_sum==36
    kept=sum(r['selected'] for r in selections);assert kept>0 and len(all_rows)==kept*40*8
    grouped=[{'layer':layer,'rank':rank,'global_expert_range':mapping['target_ranges'][rank],**summarize(rows)}
             for (layer,rank),rows in sorted(per_layer_rank.items())]
    summary={'CPU_only':True,'NPU_calls':0,'NPU_permission':0,'input_source_dir':str(FORMAL),'source_guards_ref':'source_guards.json',
             'coverage':{'measured_requests':6,'nominal_target_rounds':len(selections),'fully_accepted_and_all_six_inputs_exported_rounds':kept,
                         'excluded_rounds':len(selections)-kept,'fraction_of_nominal_rounds':kept/len(selections),
                         'unique_int32_input_route_SHA_slices':len(seen_route_sha),'all_8_rank_receipts_matched':True,
                         'warmup_requests_included':False,'partial_or_unaccepted_rounds_inferred':False,
                         'actual_prepack_device_GMM_consumer_proof':False,'draft_route_coverage':0},
             'all_observed_complete_rounds':summarize(all_rows),'deduplicated_route_SHA_slices':summarize(unique_rows),
             'per_layer_rank':grouped,'repeated_AA_route_slice_occurrences':dict(round_source_digests),
             'full_projected_rows_remote_only':{'path':str(full_path),'bytes':full_path.stat().st_size,'sha256':digest(full_path)},
             'source_files':file_sources,'M1_uniform_event_gain':False,
             'performance_transfer_to_K5_or_graph_allowed':False,'SDK_vs_prebound_host_gap_is_not_formal_graph_GMM_gain':True,
             'passed':True}
    (ROOT/'route_distribution_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'round_selection.json').write_text(json.dumps({'CPU_only':True,'NPU_calls':0,'rounds':selections},indent=2)+'\n')
    with (ROOT/'per_layer_rank_summary.csv').open('w',newline='') as f:
        fields=['layer','rank','observations','R_mean','R_median','R_max','active_mean','active_median','active_max','empty',
                'single_active_expert','single_active_local_e47','single_active_late40_to47','exact_M1_gain_pattern','fraction_exact_M1_gain_pattern']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for g in grouped:
            writer.writerow({'layer':g['layer'],'rank':g['rank'],'observations':g['observations'],'R_mean':g['R']['mean'],
                             'R_median':g['R']['median'],'R_max':g['R']['max'],'active_mean':g['active_experts']['mean'],
                             'active_median':g['active_experts']['median'],'active_max':g['active_experts']['max'],'empty':g['empty'],
                             'single_active_expert':g['single_active_expert'],'single_active_local_e47':g['single_active_local_e47'],
                             'single_active_late40_to47':g['single_active_late40_to47'],
                             'exact_M1_gain_pattern':g['exact_M1_measured_gain_pattern_R1_R3_R6_e47'],
                             'fraction_exact_M1_gain_pattern':g['fraction_exact_M1_measured_gain_pattern']})
    assert all((ROOT/name).stat().st_size<1000000 for name in ['route_distribution_summary.json','round_selection.json','per_layer_rank_summary.csv'])
    print(json.dumps({'coverage':summary['coverage'],'aggregate':summary['all_observed_complete_rounds'],
                      'deduplicated_observations':summary['deduplicated_route_SHA_slices']['observations'],
                      'summary_bytes':(ROOT/'route_distribution_summary.json').stat().st_size,'NPU_calls':0}))

if __name__=='__main__':main()
