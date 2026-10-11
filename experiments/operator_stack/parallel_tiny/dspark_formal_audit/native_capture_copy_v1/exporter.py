"""Native ATen bounded-copy/epoch buffers; no custom-op/None/event transport.

Importing this file is CPU-only. make_exporter_class imports no SDK itself.
All allocation occurs before capture. The returned carrier is unchanged; only
owned observer buffers are mutated. AOT destination aliases need a real gate.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PointSpec:
    name: str
    role: str
    layer: int
    max_rows: int
    width: int
    position_width: int = 1

    def validate(self):
        if self.role not in ('main','draft','producer','cache-write','cache-read','verifier'):
            raise ValueError('Unknown actual consumer role')
        if self.max_rows<=0 or self.width<=0 or self.position_width<=0:
            raise ValueError('Observer must have a fixed nonempty bound')


def make_exporter_class(torch):
    class NativeBoundedCopy(torch.nn.Module):
        def __init__(self,spec,*,device,dtype):
            super().__init__();spec.validate();self.spec=spec
            self.register_buffer('values',torch.zeros((spec.max_rows,spec.width),device=device,dtype=dtype))
            self.register_buffer('positions',torch.full((spec.max_rows,spec.position_width),-1,device=device,dtype=torch.int64))
            self.register_buffer('valid',torch.zeros((spec.max_rows,),device=device,dtype=torch.bool))
            self.register_buffer('epoch',torch.zeros((1,),device=device,dtype=torch.int64))
            self.register_buffer('copy_count',torch.zeros((1,),device=device,dtype=torch.int64))
            self.register_buffer('row_count',torch.zeros((1,),device=device,dtype=torch.int64))

        def forward(self,carrier,values,positions,valid,epoch):
            rows=values.shape[0]
            if rows>self.spec.max_rows or values.shape[1]!=self.spec.width:
                raise ValueError('Bounded copy value extent mismatch')
            if positions.shape!=(rows,self.spec.position_width) or valid.shape!=(rows,):
                raise ValueError('Actual positions/valid-mask layout mismatch')
            if epoch.shape!=(1,) or epoch.dtype!=torch.int64:
                raise ValueError('Device epoch must be an explicit int64[1] input')
            # Standard ATen mutations are visible to AOT; no opaque None op.
            # Marker is written last, after the payload and device write count.
            self.values.zero_();self.values[:rows].copy_(values)
            self.positions.fill_(-1);self.positions[:rows].copy_(positions)
            self.valid.zero_();self.valid[:rows].copy_(valid)
            self.row_count.fill_(rows);self.copy_count.add_(1)
            self.epoch.copy_(epoch)
            return carrier

        def reset_outside_capture(self):
            self.values.zero_();self.positions.fill_(-1);self.valid.zero_()
            self.epoch.zero_();self.copy_count.zero_();self.row_count.zero_()

        def tensor_tuple(self):
            return (self.values,self.positions,self.valid,self.epoch,self.copy_count,self.row_count)

    return NativeBoundedCopy


def assert_export(export,*,epoch,expected_writes,values,positions,valid):
    """CPU scalar/list checks after a real completion fence; no device claim."""
    if export['epoch']!=epoch:raise ValueError('Stale or missing device epoch')
    if export['copy_count']!=expected_writes:raise ValueError('Missing/duplicate actual device GM writes')
    if export['rows']!=len(values):raise ValueError('Actual device copy extent mismatch')
    if export['values'][:len(values)]!=values or export['positions'][:len(values)]!=positions:
        raise ValueError('Actual bounded consumer payload mismatch')
    if export['valid'][:len(valid)]!=valid:raise ValueError('Actual validity mask mismatch')
    return {'structure_passed':True,'device_audit_passed':False,'safe_formal_receipt':False}


def hook_carrier_policy(qualified_function):
    if qualified_function.endswith(('.token_dispatch','._compute_routing')):
        return 'return the original result structure; thread the selected Tensor carrier through native observer'
    if qualified_function.endswith(('._store_standard_swa_kv','.dsa_kv_compress_scatter')):
        return 'observe live updates BEFORE native cache write; prove post-write values again at actual cache reader'
    if qualified_function.endswith(('._attention','._forward_attention')):
        return 'observe bounded actual cache rows BEFORE attention; return original q carrier to native reader'
    if qualified_function.endswith(('.update_from_output','._bookkeeping_sync')):
        return 'CPU correspondence after device fence; never claim this callback as device coverage'
    return 'observe live Tensor at the real consumer point; return the unchanged Tensor carrier'
