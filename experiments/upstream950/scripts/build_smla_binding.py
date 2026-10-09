"""Compile a separate Torch library against the installed adapter and private ops."""
import hashlib
import json
import os
import re
from pathlib import Path

import torch
import torch_npu
from torch.utils.cpp_extension import load

root = Path('/work/build/smla')
tree = root/'csrc'
binding = root/'binding'
binding.mkdir(exist_ok=True)
original = Path('/vllm-workspace/vllm-ascend/csrc/torch_binding.cpp').read_text()
begin = original.index('"npu_sparse_flash_mla(Tensor q,')
end = original.index(';',begin)
schema = ''.join(json.loads('"'+value+'"') for value in
                 re.findall(r'"((?:\\.|[^"\\])*)"',original[begin:end]))
assert schema.startswith('npu_sparse_flash_mla(') and 'metadata' in schema
prefix = '''#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "ops.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/up950_sparse_flash_mla/up950_sparse_flash_mla_torch_adpt.h"
#include "attention/up950_base_sparse_flash_mla/up950_base_sparse_flash_mla_torch_adpt.h"
thread_local char g_hashBuf[kHashBufSize] = {};
thread_local int g_hashOffset = 0;
'''
code = prefix+'\nTORCH_LIBRARY(up950_native, m) {\n'
code += 'm.def('+json.dumps(schema.replace('npu_sparse_flash_mla(', 'smla(',1))+');\n'
code += 'm.def('+json.dumps(schema.replace('npu_sparse_flash_mla(', 'base_smla(',1))+');\n}\n'
code += '''TORCH_LIBRARY_IMPL(up950_native, PrivateUse1, m) {
m.impl("smla", TORCH_FN(up950_prefetch_adapter::npu_up950_sparse_flash_mla));
m.impl("base_smla", TORCH_FN(up950_base_adapter::npu_up950_base_sparse_flash_mla));
}
'''
cpp = binding/'binding.cpp'
cpp.write_text(code)
os.environ['MAX_JOBS'] = '2'
os.environ['TORCH_EXTENSIONS_DIR'] = '/work/cache/torch_extensions'
npu = Path(torch_npu.__file__).resolve().parent
cann = Path('/usr/local/Ascend/ascend-toolkit/latest')
library = load(name='up950_smla_binding',
               sources=[str(cpp),str(tree/'aclnn_torch_adapter/NPUBridge.cpp'),
                        str(tree/'aclnn_torch_adapter/NPUStorageImpl.cpp')],
               extra_include_paths=[str(tree),str(tree/'aclnn_torch_adapter'),
                                    str(npu/'include'),str(npu/'include/third_party/acl/inc'),
                                    str(cann/'include')],
               extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu',
                              '-L'+str(cann/'lib64'),'-lascendcl'],
               extra_cflags=['-O2'],is_python_module=False,verbose=True)
result = {'library':str(library),'schema':schema,
          'binding_sha256':hashlib.sha256(cpp.read_bytes()).hexdigest(),
          'torch':torch.__version__,'torch_npu':torch_npu.__version__}
(root/'binding_manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
