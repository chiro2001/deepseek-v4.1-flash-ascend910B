"""Future one-shot physical6 M1 event gate; execute only after new allocation."""
import ctypes
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import traceback

import numpy as np
from event_protocol import interleaved_order, measure_one
from numeric_contract import cases, activation, array_sha, compare_bf16, oracle
from weights_cpu import load_weights

ROOT = Path('/work/gmm_lane_sparse_msd_event_perf_p6_20261011_v2')
NUMERIC = Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v5')
FROZEN = Path('/work/gmm_lane_sparse_msd_build_20261011_v3_repro')
WARMUP_ROUNDS = 3
MEASURE_ROUNDS = 12


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked(stage, code):
    if code: raise RuntimeError(f'{stage}: {code}')
def emit(value): print(json.dumps(value), flush=True)


def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == '6'
    assert os.environ.get('TASK_QUEUE_ENABLE') == '0'
    pre = json.loads((ROOT / 'preflight.json').read_text())
    assert not pre['physical6_processes'] and pre['health_allowed'] and pre['container_has_dev6_permission']
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(pre['UTC'])).total_seconds()
    assert 0 <= age <= 30
    assert not (ROOT / 'runtime_terminal.json').exists()
    for name, sha in json.loads((ROOT / 'source_freeze.json').read_text()).items():
        assert digest(ROOT / 'source' / name) == sha
    build = json.loads((ROOT / 'build_receipt.json').read_text())
    assert build['exit_code'] == 0 and digest(ROOT / 'libm1_event_perf_bridge.so') == build['shared_sha256']
    abi=json.loads((ROOT/'CPU_ABI_contract.json').read_text())
    assert abi['passed'] and abi['compiled_v1_regression_rejected']
    assert abi['compiled_v2_fixture']['v5_actual_payloads_checked']==27
    frozen = json.loads((NUMERIC / 'frozen_dependencies.json').read_text())
    for r in frozen['frozen_v3']['objects'].values(): assert digest(r['path']) == r['sha256']
    for name, sha in frozen['frozen_v3']['source_sha256'].items(): assert digest(FROZEN / 'source' / name) == sha
    for label,root_dir,library in [('loader_v2',Path('/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011/source/runtime_loader_v2'),Path('/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011/build/libgmm2_sparse_msd_loader_v2.so')),('launch_v3',Path('/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011/source'),Path('/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011/build/libgmm2_launch_args_v3.so'))]:
        assert digest(library)==frozen[label]['shared_sha256']
        for name,sha in frozen[label]['source_sha256'].items(): assert digest(root_dir/name)==sha
    numeric_terminal=json.loads((NUMERIC/'runtime_terminal.json').read_text())
    assert numeric_terminal['passed'] and numeric_terminal['cases_passed']==9
    tilings=[]
    for name in ['Gmm2NativeMsdDown_control_v6','Gmm2SparseMsdDown_candidate_v6']:
        header=next((FROZEN/('kernel_meta_'+name)/'kernel_meta').glob('*_tiling_data.h'))
        match=re.search(r'const uint8_t __ascendc_arr_##tiling_data\[(\d+)\]\s*=\s*\{([^}]+)\}',header.read_text())
        assert match
        blob=bytes(int(x.strip()) for x in match[2].split(',') if x.strip())
        assert len(blob)==int(match[1])==2032 and hashlib.sha256(blob).hexdigest()=='5a596a726706e39d34be71b61a6ed563325c1d482953937050a7b8373e9a263c'
        tilings.append(blob)
    assert tilings[0]==tilings[1]
    expected = json.loads((NUMERIC / 'weights_source_manifest.json').read_text())
    cpu_weight, cpu_scale, cpu_assist, manifest = load_weights()
    assert manifest['processed'] == expected['processed'] and manifest['raw_tensor_records'] == expected['raw_tensor_records']
    import torch
    import torch_npu
    torch.npu.config.allow_internal_format = True
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    lib = ctypes.CDLL(str(ROOT / 'libm1_event_perf_bridge.so'))
    p, u64, u32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32
    lib.m1_probe_mapping.argtypes=[];lib.m1_probe_mapping.restype=ctypes.c_int
    lib.m1_probe_open.argtypes=[ctypes.c_char_p,u64,ctypes.POINTER(p)];lib.m1_probe_open.restype=ctypes.c_int
    lib.m1_probe_close.argtypes=[ctypes.POINTER(p)];lib.m1_probe_close.restype=ctypes.c_int
    lib.m1_probe_read.argtypes=[u64,p,u64];lib.m1_probe_read.restype=ctypes.c_int
    lib.m1_perf_bind.argtypes=[p,u64,u64,u64,u64,u64,u64];lib.m1_perf_bind.restype=ctypes.c_int
    lib.m1_perf_submit.argtypes=[p,u32];lib.m1_perf_submit.restype=ctypes.c_int
    lib.m1_perf_finish.argtypes=[p,p,p];lib.m1_perf_finish.restype=ctypes.c_int
    lib.m1_perf_counters.argtypes=[p,p];lib.m1_perf_counters.restype=ctypes.c_int
    lib.m1_perf_snapshot.argtypes=[p,p,p,p,p];lib.m1_perf_snapshot.restype=ctypes.c_int
    checked('mapping',lib.m1_probe_mapping())
    owner=p();stream=torch.npu.current_stream();done=[]
    terminal={'passed':False,'physical':6,'logical':0,'M1_capacity6_only':True,'synthetic_activation':True,
              'warmup_rounds_per_case':WARMUP_ROUNDS,'measure_rounds_per_case':MEASURE_ROUNDS,
              'fullnet_or_K5_claim':False,'container_PID':os.getpid(),
              'pre_event_raw_smoke_required':True,'pre_event_raw_smoke_completed':False,
              'actual_sdk_submit_accepted':0,'actual_sdk_numeric_completed':0,
              'actual_raw_numeric_completed':0,'actual_candidate_numeric_completed':0,
              'measured_event_samples_completed':0}
    try:
        weight=torch_npu.npu_format_cast(cpu_weight.to('npu:0'),29).view(torch.int32).contiguous()
        scale=cpu_scale.to('npu:0');assist=cpu_assist.to('npu:0');torch.npu.synchronize()
        assert torch_npu.get_npu_format(weight)==29 and list(weight.shape)==[48,2304,640]
        checked('open',lib.m1_probe_open(str(FROZEN).encode(),stream.npu_stream,ctypes.byref(owner)))
        def read(gm,size):
            h=ctypes.create_string_buffer(size);checked('D2H outside event',lib.m1_probe_read(gm,h,size));return h.raw
        def full_readonly_seal():
            h=hashlib.sha256();chunk=2304*5120//2
            for e in range(48):h.update(read(weight.data_ptr()+e*chunk,chunk))
            assert h.hexdigest()==expected['processed']['expected_physical_NZ']['sha256']
            for name,tensor in [('scale',scale),('assist',assist)]:
                assert hashlib.sha256(read(tensor.data_ptr(),expected['processed'][name]['bytes'])).hexdigest()==expected['processed'][name]['sha256']
        full_readonly_seal()
        for index,(name,cpu_counts) in enumerate(cases()):
            terminal['current_case']=name
            terminal['phase']='bind_outside_events'
            cpu_x,cpu_token=activation(index);rows=int(cpu_counts.sum())
            x=torch.from_numpy(cpu_x).to('npu:0');token=torch.from_numpy(cpu_token).to('npu:0');counts=torch.from_numpy(cpu_counts).to('npu:0')
            torch.npu.synchronize();x_ptr=x.data_ptr()
            checked('bind outside event',lib.m1_perf_bind(owner,x_ptr,weight.data_ptr(),assist.data_ptr(),scale.data_ptr(),counts.data_ptr(),token.data_ptr()))
            ref=np.fromfile(NUMERIC/f'{name}_sdk_native.bf16.bin',dtype=np.uint16).reshape(6,5120)[:rows].copy()
            old=json.loads((NUMERIC/f'{name}_result.json').read_text());assert array_sha(ref)==old['SDK_output']['BF16_bytes_sha256']
            observed=[];measurements=[];expected_frame=oracle(cpu_counts)
            terminal['partial_case_result']={'case':name,'numeric_checks':observed,'measurements':measurements}
            def snapshot():
                args=np.zeros(9,dtype=np.uint64);plan=np.zeros(2,dtype=np.uint64)
                addresses=np.zeros(8,dtype=np.uint64);descriptors=np.zeros(20,dtype=np.uint64)
                checked('snapshot outside events',lib.m1_perf_snapshot(owner,args.ctypes.data,plan.ctypes.data,addresses.ctypes.data,descriptors.ctypes.data))
                expected_args=np.array([addresses[7],*addresses[:4],counts.data_ptr(),token.data_ptr(),addresses[4],addresses[6]],dtype=np.uint64)
                assert np.array_equal(args,expected_args) and addresses[4]!=addresses[5]
                assert int(addresses[6])%512==0 and int(addresses[7])!=0
                assert plan.tolist()==[counts.data_ptr(),int(addresses[6])+16*1024*1024]
                expected_descriptors=np.array([word for gm in [x_ptr,weight.data_ptr(),assist.data_ptr(),scale.data_ptr(),int(addresses[5])] for word in [24,4294967296,4294967295,gm]],dtype=np.uint64)
                assert np.array_equal(descriptors,expected_descriptors)
                return {'GMM_HostArgs_72B':args.tolist(),'planner_HostArgs_16B':plan.tolist(),
                        'descriptor_addresses_y_workspace_ffts':addresses.tolist(),
                        'five_descriptor_32B_words':descriptors.tolist()}
            bound_snapshot=snapshot();descriptor_check_count=0
            def restore():
                x.copy_(torch.from_numpy(cpu_x));assert x.data_ptr()==x_ptr
            def submit(method):
                if method=='SDK':
                    result=torch_npu.npu_grouped_matmul(x=[x],weight=[weight],scale=[scale],bias=[assist],per_token_scale=[token],split_item=2,group_list_type=1,group_type=0,group_list=counts,output_dtype=torch.bfloat16)[0]
                    terminal['actual_sdk_submit_accepted']+=1
                    return result
                checked('submit',lib.m1_perf_submit(owner,1 if method=='CANDIDATE' else 0));return owner
            def finish(method,lifetime):
                if method=='SDK':bits=lifetime.cpu().view(torch.uint16).numpy().copy()
                else:
                    bits=np.empty((6,5120),dtype=np.uint16);frame=np.zeros(32,dtype=np.uint32)
                    checked('finish after stop',lib.m1_perf_finish(owner,bits.ctypes.data,frame.ctypes.data))
                    assert (bits[rows:]==0xa5a5).all()
                    if method=='CANDIDATE':assert np.array_equal(frame,expected_frame)
                comparison=compare_bf16(ref,bits[:rows],1);assert comparison['passed'],comparison
                observed.append({'method':method,'max_ULP':comparison['max_ULP'],'phase':terminal['phase']})
                terminal[{'SDK':'actual_sdk_numeric_completed','RAW_NATIVE':'actual_raw_numeric_completed','CANDIDATE':'actual_candidate_numeric_completed'}[method]]+=1
            def light_seal_outside_event():
                nonlocal descriptor_check_count
                assert x.data_ptr()==x_ptr
                assert hashlib.sha256(read(x_ptr,cpu_x.nbytes)).hexdigest()==old['x_input_lifecycle'][1]['full_capacity_sha256']
                assert read(counts.data_ptr(),384)==cpu_counts.tobytes() and read(token.data_ptr(),24)==cpu_token.tobytes()
                assert snapshot()==bound_snapshot;descriptor_check_count+=1
                # Full E48 seal below is per case, outside all timed samples.
            if index==0:
                # One actual raw smoke before any Event or candidate submission.
                # A failure ends this process and prevents every event sample.
                terminal['phase']='pre_event_raw_numeric_smoke'
                restore();stream.synchronize()
                lifetime=submit('RAW_NATIVE');stream.synchronize()
                finish('RAW_NATIVE',lifetime);light_seal_outside_event();full_readonly_seal()
                smoke={'case':name,'raw_API_accepted':1,'raw_numeric_completed':1,'max_ULP':observed[-1]['max_ULP'],
                       'canary_unused_rows_counts_token_fresh_x_and_full_E48_seals_passed':True,
                       'HostArgs_and_descriptors':bound_snapshot,'Event_objects_created':0,
                       'included_in_event_samples':False,'passed':True}
                (ROOT/'pre_event_raw_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
                terminal['pre_event_raw_smoke_completed']=True
                emit({'pre_event_raw_smoke_passed':True,'max_ULP':smoke['max_ULP']})
            for round_index in range(WARMUP_ROUNDS+MEASURE_ROUNDS):
                for method in interleaved_order(round_index):
                    terminal['phase']='event_warmup' if round_index<WARMUP_ROUNDS else 'event_measured'
                    terminal['current_method']=method
                    record=measure_one(torch,stream,method,restore,submit,finish,light_seal_outside_event)
                    if round_index>=WARMUP_ROUNDS:
                        measurements.append({'round':round_index-WARMUP_ROUNDS,**record})
                        terminal['measured_event_samples_completed']+=1
            full_readonly_seal()
            result={'case':name,'R':rows,'active_experts':cpu_counts.nonzero()[0].tolist(),'measurements':measurements,
                    'numeric_checks':observed,'same_GM_fresh_x':True,'planner_included':True,'full_E48_seal_passed':True,
                    'bound_HostArgs_and_descriptors':bound_snapshot,'descriptor_after_call_checks':descriptor_check_count,
                    'per_method_submit_and_sync_completed':{'SDK':WARMUP_ROUNDS+MEASURE_ROUNDS,'RAW_NATIVE':WARMUP_ROUNDS+MEASURE_ROUNDS,'CANDIDATE':WARMUP_ROUNDS+MEASURE_ROUNDS,'PLANNER':WARMUP_ROUNDS+MEASURE_ROUNDS},'passed':True}
            (ROOT/f'{name}_event_result.json').write_text(json.dumps(result,indent=2)+'\n');done.append(name);emit({'case_passed':name})
        terminal['passed']=True
        terminal['phase']='all_cases_completed'
        terminal.pop('partial_case_result',None)
    except BaseException as e:
        terminal.update({'failure':str(e),'traceback':traceback.format_exc()});emit({'terminal_failure':str(e),'stop_remaining':True})
    finally:
        accepted=np.zeros(3,dtype=np.uint64)
        if owner.value:checked('accepted counters outside events',lib.m1_perf_counters(owner,accepted.ctypes.data))
        terminal['actual_raw_native_candidate_planner_submit_accepted']=accepted.tolist()
        close=lib.m1_probe_close(ctypes.byref(owner)) if owner.value else 0
        terminal.update({'cleanup_code':close,'owner_retained':bool(owner.value),'cases_completed':done})
        terminal['passed'] &= close==0 and not owner.value
        (ROOT/'runtime_terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');emit({'terminal':True,**terminal})
    return 0 if terminal['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
