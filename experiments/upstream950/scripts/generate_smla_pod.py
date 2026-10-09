"""Use CANN's own tiling generator, then prove its host serialization layout."""
import importlib.util
import json
import re
import shlex
import subprocess
from pathlib import Path

root = Path('/work/build/smla')
tree = root/'csrc'
script = tree/'cmake/scripts/utest/gen_tiling_data_stub.py'
spec = importlib.util.spec_from_file_location('cann_tiling_generator',script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import tbe.tikcpp.get_op_tiling as sdk_tiling
results = []
for folder,op in [('up950_sparse_flash_mla','Up950SparseFlashMla'),
                  ('up950_base_sparse_flash_mla','Up950BaseSparseFlashMla')]:
    directory = tree/'attention'/folder
    host = directory/'op_host'/(folder+'_tiling.h')
    kernel = directory/'op_kernel'
    generated = module.Process._get_tiling_source(host)
    # Keep the official POD declarations/padding; runtime uses the regular
    # Ascend GET_TILING_DATA_WITH_STRUCT, not the CPU memcpy stub helpers.
    generated = re.sub(r'inline void Init\w+\([^}]+}\s*','',generated)
    generated = generated[:generated.index('#undef GET_TILING_DATA')]
    generated = generated.replace('#include <cstring>\n','').replace('#include <securec.h>\n','')
    generated = generated.replace('#include <kernel_tiling/kernel_tiling.h>\n','')
    pod = kernel/(folder+'_pod_tiling.h')
    # No automatic private tiling registration was emitted by the compiler.
    # Emit the SDK's ELF size section so opParaSize includes these 224 bytes.
    previous_flag = sdk_tiling.global_var_storage.get_variable('ascendc_tiling_no_register')
    sdk_tiling.global_var_storage.set_variable('ascendc_tiling_no_register',True)
    sdk_copy_macros = sdk_tiling.gen_dynamic_shape_v2(op,'')
    sdk_tiling.global_var_storage.set_variable('ascendc_tiling_no_register',previous_flag)
    # On the initial inspection pass our fallback exposes/registers the
    # structure. On the real compile pass CANN injects its generated classes
    # and macros first; avoid redefining those classes.
    pod.write_text('#pragma once\n#if !defined(__CCE_AICORE__) || !defined(GET_TILING_DATA_WITH_STRUCT)\n'
                   +generated+'\n#ifdef __CCE_AICORE__\n'+sdk_copy_macros+'\n#endif\n#endif\n')
    cpp = kernel/(folder+'.cpp')
    text = cpp.read_text()
    include = '#include "'+pod.name+'"\n'
    if include not in text:
        cpp.write_text(include+text)
    # Each scalar receives a distinct nonzero sentinel. Compare all fields,
    # nested structure sizes and raw host SaveToBuffer against generated POD.
    source = host.read_text()
    blocks = re.findall(r'BEGIN_TILING_DATA_DEF\((\w+)\)(.*?)END_TILING_DATA_DEF',source,re.S)
    code = '#include <cassert>\n#include <cstdio>\n#include <vector>\n'
    code += '#include "'+str(pod)+'"\n#include "'+str(host)+'"\nint main(){\n'
    values = {}
    for name,body in blocks:
        fields = re.findall(r'TILING_DATA_FIELD_DEF\((\w+),\s*(\w+)\)',body)
        if not fields:
            continue
        variable = 'value_'+name
        values[name] = variable
        code += 'optiling::'+name+' '+variable+';\n'
        for number,(dtype,field) in enumerate(fields):
            value = (str(number+1)+'.25f' if dtype == 'float'
                     else str(0x1234567800+number)+'ULL' if dtype in ['uint64_t','int64_t']
                     else str(300+number))
            code += variable+'.set_'+field+'(static_cast<'+dtype+'>('+value+'));\n'
        code += '{ std::vector<uint8_t> bytes('+variable+'.GetDataSize());\n'
        code += variable+'.SaveToBuffer(bytes.data(),bytes.size());\n'
        code += 'assert(bytes.size()==sizeof(::'+name+'));\n'
        code += 'const auto& plain=*reinterpret_cast<const ::'+name+'*>(bytes.data());\n'
        for dtype,field in fields:
            code += 'assert(plain.'+field+'=='+variable+'.get_'+field+'());\n'
        code += '}\n'
    name,body = blocks[-1]
    code += 'optiling::'+name+' combined;\n'
    nested = re.findall(r'TILING_DATA_FIELD_DEF_STRUCT\((\w+),\s*(\w+)\)',body)
    for dtype,field in nested:
        nested_body = next(b for n,b in blocks if n == dtype)
        for _,scalar in re.findall(r'TILING_DATA_FIELD_DEF\((\w+),\s*(\w+)\)',nested_body):
            code += 'combined.'+field+'.set_'+scalar+'('+values[dtype]+'.get_'+scalar+'());\n'
    code += 'std::vector<uint8_t> bytes(combined.GetDataSize());\n'
    code += 'combined.SaveToBuffer(bytes.data(),bytes.size());\n'
    code += 'assert(bytes.size()==sizeof(::'+name+'));\n'
    code += 'const auto& plain=*reinterpret_cast<const ::'+name+'*>(bytes.data());\n'
    for dtype,member in nested:
        nested_body = next(b for n,b in blocks if n == dtype)
        for _,field in re.findall(r'TILING_DATA_FIELD_DEF\((\w+),\s*(\w+)\)',nested_body):
            code += 'assert(plain.'+member+'.'+field+'=='+values[dtype]+'.get_'+field+'());\n'
    code += 'printf("validated %zu serialized bytes\\n",bytes.size());\n}\n'
    check = root/(folder+'_layout.cpp')
    check.write_text(code)
    binary = check.with_suffix('')
    cann = Path('/usr/local/Ascend/cann-9.1.0')
    command = ['c++','-std=c++17','-D_GLIBCXX_USE_CXX11_ABI=0',str(check),'-o',str(binary),
               '-include'+str(tree/'common/include/cann_compat.h'),
               '-I'+str(tree/'common/include'),'-I'+str(cann/'include'),
               '-I'+str(cann/'pkg_inc'),'-I'+str(cann/'pkg_inc/base'),
               '-I'+str(cann/'include/op_common'),
               '-I'+str(cann/'include/exe_graph'),'-I'+str(cann/'include/ascendc/highlevel_api'),
               '-L'+str(cann/'lib64'),'-lregister','-ltiling_api','-lplatform','-lascendalog',
               '-lgraph','-lgraph_base','-lexe_graph','-lops_base']
    subprocess.run(command,check=True)
    output = subprocess.check_output([str(binary)],text=True)
    results.append({'op':op,'pod':str(pod),'host_serialization_layout':'PASS','output':output.strip()})
    print(json.dumps(results[-1]),flush=True)
(root/'tiling_layout_validation.json').write_text(json.dumps(results,indent=2))
