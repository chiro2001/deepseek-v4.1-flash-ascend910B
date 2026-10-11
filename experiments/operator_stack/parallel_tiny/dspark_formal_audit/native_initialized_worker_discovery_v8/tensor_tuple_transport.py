"""Explicit Tensor input/output bounded transport, no torch import on import."""
from dataclasses import dataclass


FIELDS = ('values','positions','valid','epoch','copy_count','row_count')


@dataclass(frozen=True)
class SnapshotSpec:
    max_rows: int
    width: int
    position_width: int = 1

    def check(self):
        if min(self.max_rows,self.width,self.position_width) <= 0:
            raise ValueError('Need fixed positive bounded snapshot dimensions')


def allocate_buffers(torch, spec, *, device, dtype):
    spec.check()
    return (torch.zeros((spec.max_rows,spec.width),device=device,dtype=dtype),
            torch.full((spec.max_rows,spec.position_width),-1,device=device,dtype=torch.int64),
            torch.zeros((spec.max_rows,),device=device,dtype=torch.bool),
            torch.zeros((1,),device=device,dtype=torch.int64),
            torch.zeros((1,),device=device,dtype=torch.int64),
            torch.zeros((1,),device=device,dtype=torch.int64))


def snapshot_tuple(torch, carrier, values, positions, valid, epoch, buffers):
    if len(buffers) != len(FIELDS):
        raise ValueError('Explicit snapshot Tensor tuple ABI changed')
    out_values,out_positions,out_valid,out_epoch,out_count,out_rows = buffers
    rows = values.shape[0]
    if (values.ndim != 2 or rows > out_values.shape[0] or values.shape[1] != out_values.shape[1]
            or positions.shape != (rows,out_positions.shape[1]) or valid.shape != (rows,)):
        raise ValueError('Bounded snapshot extent/positions/valid mismatch')
    if epoch.shape != (1,) or epoch.dtype != torch.int64:
        raise ValueError('Need explicit device int64[1] epoch Tensor')
    if (values.dtype != out_values.dtype or positions.dtype != torch.int64 or valid.dtype != torch.bool
            or any(tensor.device != out_values.device for tensor in (values,positions,valid,epoch,*buffers))):
        raise ValueError('Actual snapshot dtype/device differs from the preallocated tuple')
    out_values.zero_();out_values[:rows].copy_(values)
    out_positions.fill_(-1);out_positions[:rows].copy_(positions)
    out_valid.zero_();out_valid[:rows].copy_(valid)
    out_rows.fill_(rows);out_count.add_(1);out_epoch.copy_(epoch)
    # Every mutated Tensor remains an explicit graph output. Original carrier
    # and buffer addresses must also be checked in the real compiled SDK gate.
    return (carrier,*buffers)


def reset_after_prior_fence(buffers):
    # The caller performs a real device fence before reset; no pretend flag.
    values,positions,valid,epoch,count,rows = buffers
    values.zero_();positions.fill_(-1);valid.zero_();epoch.zero_();count.zero_();rows.zero_()
