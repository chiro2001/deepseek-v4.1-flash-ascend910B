"""A real routed candidate replay for msprof op Default/MemoryDetail."""
import torch
import torch_npu

from clamped_swiglu import clamped_swiglu


torch.npu.set_device(0)
torch.manual_seed(20261009)
x=torch.randn((2,512),dtype=torch.bfloat16,device='npu')
for _ in range(30):
    output=clamped_swiglu(x,7.0,mutate_input=True)
torch.npu.synchronize()
print('Routed probe completed: shape',tuple(x.shape),'output',tuple(output.shape),flush=True)
