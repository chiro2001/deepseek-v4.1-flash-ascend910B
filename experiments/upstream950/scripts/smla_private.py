"""Load private CANN/Torch ops while leaving the installed native op available."""
import json
import os
from pathlib import Path

root = Path('/work/build/smla')
vendor = root/'opp/vendors/up950_transformer'
assert vendor.exists(), 'Build/install private SMLA ops first'
assert (root/'kernel_parameter_validation.json').exists(), 'Validate kernel tiling parameter sizes first'
previous = os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
paths = [str(vendor)] + [p for p in previous.split(':') if p and p != str(vendor)]
os.environ['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(paths)

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
manifest = json.loads((root/'binding_manifest.json').read_text())
torch.ops.load_library(manifest['library'])


def fake(q, **kwargs):
    output = torch.empty_like(q)
    if not kwargs.get('return_softmax_lse',False):
        return output, torch.empty(0,dtype=torch.float32,device=q.device)
    kv = kwargs.get('ori_kv')
    if kv is None:
        kv = kwargs['cmp_kv']
    nkv = kv.shape[1 if kwargs.get('layout_kv','BSND') == 'TND' else 2]
    shape = ((q.shape[0],nkv,q.shape[1],q.shape[2]//nkv)
             if kwargs.get('layout_q','BSND') == 'BSND'
             else (nkv,q.shape[0],q.shape[1]//nkv))
    return output, torch.empty(shape,dtype=torch.float32,device=q.device)


torch.library.register_fake('up950_native::smla')(fake)
torch.library.register_fake('up950_native::base_smla')(fake)
native = torch.ops._C_ascend.npu_sparse_flash_mla
control = torch.ops.up950_native.base_smla
prefetch = torch.ops.up950_native.smla
