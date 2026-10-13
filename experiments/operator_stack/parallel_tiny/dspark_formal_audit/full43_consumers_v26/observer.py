"""Tensor-only observer buffers and explicit graph-output envelope."""
from dataclasses import dataclass

try:
    import torch as _observer_torch
except Exception:
    _observer_torch = None


def _observer_compiling():
    try:
        return _observer_torch.compiler.is_compiling()
    except Exception:
        return True


@dataclass(frozen=True)
class PointSpec:
    name: str
    role: str
    global_layer: int
    max_rows: int
    width: int
    dtype: object
    position_width: int = 1


def make_observer_class(torch):
    class Observer(torch.nn.Module):
        def __init__(self,spec,*,device,epoch):
            super().__init__()
            if min(spec.max_rows,spec.width,spec.position_width) <= 0:raise ValueError('Empty observer bound')
            self.spec = spec;self.audit_epoch_input = epoch
            for name,tensor in [('values',torch.zeros((spec.max_rows,spec.width),device=device,dtype=spec.dtype)),
                ('positions',torch.full((spec.max_rows,spec.position_width),-1,device=device,dtype=torch.int64)),
                ('valid',torch.zeros((spec.max_rows,),device=device,dtype=torch.bool)),
                ('epoch',torch.zeros((1,),device=device,dtype=torch.int64)),
                ('copy_count',torch.zeros((1,),device=device,dtype=torch.int64)),
                ('row_count',torch.zeros((1,),device=device,dtype=torch.int64))]:self.register_buffer(name,tensor)

        def forward(self,carrier,payload,positions,valid):
            rows = payload.shape[0];flat = payload.reshape(rows,-1)
            try:
                if not _observer_compiling():
                    self.python_calls = getattr(self,'python_calls',0)+1
            except Exception:pass
            if rows > self.spec.max_rows or flat.shape[1] != self.spec.width:
                try:
                    if not _observer_compiling():self.python_last_skip = 'rows_or_width'
                except Exception:pass
                return carrier  # Original dynamic/prefill carrier; no coverage inferred.
            if positions.shape != (rows,self.spec.position_width) or valid.shape != (rows,):
                try:
                    if not _observer_compiling():self.python_last_skip = 'positions_or_valid_shape'
                except Exception:pass
                return carrier
            if flat.dtype != self.values.dtype or positions.dtype != torch.int64 or valid.dtype != torch.bool:
                try:
                    if not _observer_compiling():self.python_last_skip = 'dtype'
                except Exception:pass
                return carrier
            self.values.zero_();self.values[:rows].copy_(flat)
            self.positions.fill_(-1);self.positions[:rows].copy_(positions)
            self.valid.zero_();self.valid[:rows].copy_(valid)
            self.row_count.fill_(rows);self.copy_count.add_(1);self.epoch.copy_(self.audit_epoch_input)
            try:
                if not _observer_compiling():
                    self.python_writes = getattr(self,'python_writes',0)+1
                    self.python_last_skip = None
            except Exception:pass
            return carrier

        def tensor_tuple(self):
            return (self.values,self.positions,self.valid,self.epoch,self.copy_count,self.row_count)

        def reset_after_fence(self):
            self.values.zero_();self.positions.fill_(-1);self.valid.zero_()
            self.epoch.zero_();self.copy_count.zero_();self.row_count.zero_()

    return Observer


def make_graph_envelope_class(torch):
    class GraphEnvelope(torch.nn.Module):
        """Compiled entry returns native result AND every owned mutated buffer."""
        def __init__(self,native,observers,extra_tensor_outputs=()):
            super().__init__();self.native = native
            self.observers = torch.nn.ModuleList(observers);self.extra_tensor_outputs = tuple(extra_tensor_outputs)

        def forward(self,*args,**kwargs):
            result = self.native(*args,**kwargs)
            outputs = tuple(tensor for observer in self.observers for tensor in observer.tensor_tuple())
            return result,(*outputs,*self.extra_tensor_outputs)

    return GraphEnvelope


def native_result_from_envelope(output,expected_buffers):
    if not isinstance(output,tuple) or len(output) != 2 or not isinstance(output[1],tuple):
        raise ValueError('Private graph dropped explicit observer Tensor outputs')
    if len(output[1]) != len(expected_buffers):raise ValueError('Private graph observer output arity changed')
    return output[0],output[1]
