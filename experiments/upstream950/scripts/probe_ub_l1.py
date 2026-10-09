"""Compile-only gate for direct AIV UB->L1; never launch unsupported primitives."""
import hashlib
import json
import subprocess
from pathlib import Path

root=Path('/work/build/ub_l1')
root.mkdir(parents=True,exist_ok=True)
sdk=Path('/usr/local/Ascend/cann-9.1.0/tools/tikcpp/tikcfw')
sources={}
for relative in ['impl/dav_c220/kernel_operator_data_copy_impl.h',
                 'impl/dav_c220/kernel_operator_scm_data_copy_impl.h',
                 'impl/dav_3510/kernel_operator_data_copy_impl.h']:
    path=sdk/relative
    content=path.read_text()
    lines=content.splitlines()
    needles=['DataCopyUB2L1Impl','CopyUbufToCbuf','software-emulated','ubAddr >= 0','ubAddr != -1']
    starts=[i for i,s in enumerate(lines) if any(n in s for n in needles)]
    excerpts=[]
    for start in starts:
        excerpts.append({'line':start+1,'text':'\n'.join(lines[max(0,start-3):start+70])})
    sources[relative]={'sha256':hashlib.sha256(content.encode()).hexdigest(),
                       'path':str(path),'excerpts':excerpts}
code='''#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void up950_ub_l1_instruction_probe() {
    CopyUbufToCbuf((__cbuf__ half*)0, (__ubuf__ half*)0, 1, 16, 0, 0);
}
'''
source=root/'direct_instruction.cpp'
source.write_text(code)
results=[]
control='''#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void up950_gm_ub_control(GM_ADDR input, GM_ADDR output) {
    GlobalTensor<half> source, destination;
    source.SetGlobalBuffer((__gm__ half*)input, 256);
    destination.SetGlobalBuffer((__gm__ half*)output, 256);
    TPipe pipe;
    TBuf<TPosition::VECIN> buffer;
    pipe.InitBuffer(buffer, 512);
    LocalTensor<half> tile = buffer.Get<half>();
    DataCopy(tile, source, 256);
    PipeBarrier<PIPE_ALL>();
    DataCopy(destination, tile, 256);
}
'''
control_source=root/'gm_ub_control.cpp';control_source.write_text(control)
for label,arch,input_source in [('control','dav-c220-vec',control_source),
                               ('direct','dav-c220-vec',source),('direct','dav-c220-cube',source)]:
    command=['/usr/local/Ascend/cann-9.1.0/bin/bisheng','-xcce','-std=c++17','-O2','-c',
             '--cce-aicore-only','--cce-aicore-arch='+arch,
             '-I'+str(sdk),'-I'+str(sdk/'interface'),'-I'+str(sdk/'impl'),
             str(input_source),'-o',str(root/(label+'_'+arch+'.o'))]
    process=subprocess.run(command,text=True,capture_output=True)
    log=process.stdout+process.stderr
    (root/(label+'_'+arch+'.log')).write_text(log)
    results.append({'label':label,'arch':arch,'command':command,'exit_code':process.returncode,'log':log})
    print('COMPILE',label,arch,process.returncode,log[-2500:],flush=True)
report={'device_target':'Ascend910_9382 / DAV_2201','compile_only':True,
        'instruction_source_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'sources':sources,'compiles':results,
        'note':'Public TSCM DataCopy must be inspected for KFC/GM emulation; successful compilation alone is not hardware support.'}
Path('/work/results/ub_l1_capability.json').write_text(json.dumps(report,indent=2))
