"""CPU-only v2 bridge/compiled ABI gate and v1 regression, no device calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/work/gmm_lane_sparse_msd_event_perf_p6_20261011_v2')
SOURCE=ROOT/'source'
FROZEN=Path('/work/gmm_lane_sparse_msd_build_20261011_v3_repro')
NUMERIC=Path('/work/gmm_lane_sparse_msd_m1_numeric_p6_20261011_v5')
V1=Path('/work/gmm_lane_sparse_msd_event_perf_cpu_20261011_v1')
V2=Path('/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011')
V3=Path('/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011')
SDK=Path('/usr/local/Ascend/cann-9.1.0')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def execute(command,name):
    with (ROOT/(name+'.log')).open('wb') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    return {'command':command,'exit_code':result.returncode,'log_sha256':digest(ROOT/(name+'.log')),
            'CPU_only':True,'NPU_calls':0}

def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')==os.environ.get('ASCEND_VISIBLE_DEVICES')==''
    assert not (ROOT/'build_receipt.json').exists()
    ref=json.loads((SOURCE/'v5_actual_args_reference.json').read_text())
    actual_path=NUMERIC/'run_receipt.json'
    assert digest(actual_path)==ref['source_sha256']
    actual=json.loads(actual_path.read_text())
    assert actual['exit_code']==0 and actual['terminal']['passed']
    assert [e for e in actual['events'] if e.get('raw_launch_pointers')]==ref['raw_launch_records']
    assert len(ref['raw_launch_records'])==27
    includes=['-I',str(SOURCE),'-I',str(FROZEN/'source'),'-I',str(V2/'source/runtime_loader_v2'),
              '-I',str(V3/'source'),'-I',str(SDK/'include'),
              '-I',str(SDK/'aarch64-linux/pkg_inc/runtime'),'-I',str(SDK/'aarch64-linux/pkg_inc')]
    command=['g++','-std=c++17','-O2','-shared','-fPIC',str(SOURCE/'perf_bridge.cpp'),*includes,
             '-L',str(FROZEN),'-lgmm2_sparse_msd_probe','-L',str(V2/'build'),'-lgmm2_sparse_msd_loader_v2',
             '-L',str(V3/'build'),'-lgmm2_launch_args_v3','-L',str(SDK/'lib64'),'-lascendcl','-lruntime',
             '-Wl,--no-undefined','-Wl,-rpath,'+':'.join(map(str,[FROZEN,V2/'build',V3/'build'])),
             '-o',str(ROOT/'libm1_event_perf_bridge.so')]
    bridge=execute(command,'compile');assert bridge['exit_code']==0
    bridge.update({'source_sha256':digest(SOURCE/'perf_bridge.cpp'),
                   'shared_sha256':digest(ROOT/'libm1_event_perf_bridge.so'),
                   'shared_bytes':(ROOT/'libm1_event_perf_bridge.so').stat().st_size})
    # --gc-sections discards unused SDK entry points; all reached ACL operations
    # and launch APIs are intercepted in the CPU fixture. No SDK runtime links.
    fixture_command=['g++','-std=c++17','-O2','-ffunction-sections','-fdata-sections',
                     str(SOURCE/'verify_perf_bind_cpu.cpp'),str(FROZEN/'source/host_adapter.cpp'),
                     *includes,'-Wl,--gc-sections','-Wl,--no-undefined','-o',str(ROOT/'verify_perf_bind_cpu')]
    fixture_build=execute(fixture_command,'abi_fixture_build');assert fixture_build['exit_code']==0
    fixture_run=execute([str(ROOT/'verify_perf_bind_cpu')],'abi_fixture_run');assert fixture_run['exit_code']==0
    fixture_json=json.loads((ROOT/'abi_fixture_run.log').read_text().splitlines()[-1])
    assert fixture_json['passed'] and fixture_json['output_GM_rejected_examples']==27
    regression_command=fixture_command.copy()
    regression_command.insert(1,'-DPERF_BRIDGE_SOURCE="'+str(V1/'source/perf_bridge.cpp')+'"')
    regression_command[-1]=str(ROOT/'verify_v1_bind_regression_cpu')
    regression_build=execute(regression_command,'v1_regression_build');assert regression_build['exit_code']==0
    regression_run=execute([str(ROOT/'verify_v1_bind_regression_cpu')],'v1_regression_run')
    assert regression_run['exit_code']==1
    regression_text=(ROOT/'v1_regression_run.log').read_text()
    assert 'field mismatch: index=7 offset=56' in regression_text
    contract={'CPU_only':True,'NPU_calls':0,'SDK_runtime_linked_in_fixture':False,
              'independent_v5_actual_source_sha256':ref['source_sha256'],'v5_actual_records':27,
              'compiled_v2_fixture':fixture_json,'fixture_build':fixture_build,'fixture_run':fixture_run,
              'compiled_v1_regression_rejected':True,'regression_build':regression_build,'regression_run':regression_run,
              'wrong_output_GM_field_offset':56,'passed':True}
    (ROOT/'CPU_ABI_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (ROOT/'build_receipt.json').write_text(json.dumps(bridge,indent=2)+'\n')
    (ROOT/'frozen_dependencies.json').write_bytes((NUMERIC/'frozen_dependencies.json').read_bytes())
    sources={str(p.relative_to(SOURCE)):digest(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    (ROOT/'source_freeze.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps({'bridge':bridge,'CPU_ABI_contract_passed':True,'v1_regression_rejected':True}))

if __name__=='__main__':main()
