"""Exact selected GMM1+activation shape for msprof op counter collection."""
import torch
import torch_npu

from gmm1_activation import gmm1_activation


torch.npu.set_device(0);torch.manual_seed(20261010)
x=torch.randn((2,5120),device='npu',dtype=torch.bfloat16)
w=(torch.randn((8,512,5120),device='npu')/5120**.5).to(torch.bfloat16)
counts=torch.tensor([0,1,0,0,0,0,1,0],device='npu',dtype=torch.int64)
for _ in range(30):
    raw,output=gmm1_activation(x,w,counts,7.,bn=16,bk=512)
torch.npu.synchronize()
print('GMM_ACT_REPLAY_COMPLETE',tuple(x.shape),tuple(w.shape),tuple(output.shape),flush=True)
