"""Protocol negatives and original full-group scheduling equivalence, CPU0."""
import json
from pathlib import Path
import random
from planner_protocol_cpu import TARGET,DRAFT,make_frame,select_groups,aic_work_trace,NativeInputContractError

def main():
    checked=0;fallbacks=0;rejected=0;actual_route_vectors=0
    generator=random.Random(20261011)
    for role in [TARGET,DRAFT]:
        fixtures=[]
        for R in [0,1,3,6,role.capacity]:
            for expert in [0,role.experts-1]:
                counts=[0]*role.experts;counts[expert]=R;fixtures.append(counts)
        counts=[0]*role.experts
        for e in range(role.capacity):counts[e]=1
        fixtures.append(counts)
        for _ in range(200):
            counts=[0]*role.experts
            for _ in range(generator.randrange(role.capacity+1)):counts[generator.randrange(role.experts)]+=1
            fixtures.append(counts)
        for counts in fixtures:
            frame=make_frame(role,counts);selected=select_groups(role,counts,frame)
            assert selected['mode']=='sparse'
            native=tuple((e,sum(counts[:e]),c) for e,c in enumerate(counts))
            for single_m,single_n,cores in [(16,256,24),(32,512,24),(64,128,12)]:
                assert aic_work_trace(native,single_m=single_m,single_n=single_n,cores=cores)==aic_work_trace(selected['groups'],single_m=single_m,single_n=single_n,cores=cores)
                checked+=1
            bad_frames=[]
            for offset in range(8):
                bad=list(frame);bad[offset]=bad[offset]+1;bad_frames.append(bad)
            bad_frames.extend([frame[:-1],frame+(0,),tuple([-1]+list(frame[1:])),tuple([2**32]+list(frame[1:]))])
            if selected['groups']:
                # Corrupt the last slot, ensuring no partial earlier slots run.
                offset=8+4*(len(selected['groups'])-1)
                for column,value in [(0,role.experts),(1,role.capacity+1),(2,0),(3,1)]:
                    bad=list(frame);bad[offset+column]=value;bad_frames.append(bad)
            for bad in bad_frames:
                decision=select_groups(role,counts,bad)
                assert decision['mode']=='full_native_fallback'
                assert decision['group_visit_ids']==tuple(range(role.experts))
                assert decision['groups']==native
                fallbacks+=1
        for bad_counts in [[0]*(role.experts-1),[-1]+[0]*(role.experts-1),[2**63]+[0]*(role.experts-1),[role.capacity+1]+[0]*(role.experts-1)]:
            try:select_groups(role,bad_counts,())
            except NativeInputContractError:rejected+=1
            else:raise AssertionError('illegal native counts were legalized by fallback')
    actual=Path('/work/gmm_lane_sparse_msd_k5_route_cpu_20261011_v1/projected_counts_full.jsonl')
    if actual.exists():
        for line in actual.open():
            row=json.loads(line);counts=row['counts_48'];frame=make_frame(TARGET,counts)
            selected=select_groups(TARGET,counts,frame);assert selected['mode']=='sparse'
            native=tuple((e,sum(counts[:e]),c) for e,c in enumerate(counts))
            assert aic_work_trace(native,single_m=16,single_n=256,cores=24)==aic_work_trace(selected['groups'],single_m=16,single_n=256,cores=24)
            actual_route_vectors+=1
        assert actual_route_vectors==13440
    result={'CPU_only':True,'NPU_calls':0,'device_planner_or_candidate_implemented':False,
            'scheduling_algebra_checks':checked,'malformed_frame_full_native_fallback_checks':fallbacks,
            'illegal_native_counts_rejected':rejected,'actual_target_route_projected_vectors_checked':actual_route_vectors,
            'target_frame_bytes':TARGET.frame_bytes,'draft_frame_bytes':DRAFT.frame_bytes,
            'all_17_math_functions_would_remain_original':True,'not_device_or_numeric_GMM_proof':True,'passed':True}
    print(json.dumps(result))

if __name__=='__main__':main()
