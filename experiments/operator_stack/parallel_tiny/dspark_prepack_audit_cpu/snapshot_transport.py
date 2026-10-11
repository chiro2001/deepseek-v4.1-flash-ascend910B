"""Fixed bounded INT8 snapshot tuple, no torch import or native modification."""

FIELDS=('raw_values','positions','valid','epoch','copy_count','row_count')


def allocate(torch, capacity, width, *, device):
    return (torch.empty((capacity,width),dtype=torch.int8,device=device),
        torch.empty((capacity,1),dtype=torch.int64,device=device),
        torch.empty((capacity,),dtype=torch.bool,device=device),
        torch.zeros((1,),dtype=torch.int64,device=device),
        torch.zeros((1,),dtype=torch.int64,device=device),
        torch.zeros((1,),dtype=torch.int64,device=device))


def snapshot_before_native_tuple(torch, x, positions, valid, epoch, buffers):
    if len(buffers)!=6:
        raise ValueError('Every fixed snapshot output tensor must be retained')
    raw,p,v,e,count,rows=buffers
    if (x.dtype!=torch.int8 or raw.dtype!=torch.int8 or raw.shape!=x.shape or
            positions.shape!=(x.shape[0],1) or positions.dtype!=torch.int64 or
            valid.shape!=(x.shape[0],) or valid.dtype!=torch.bool or
            epoch.shape!=(1,) or epoch.dtype!=torch.int64 or
            p.shape!=positions.shape or p.dtype!=torch.int64 or
            v.shape!=valid.shape or v.dtype!=torch.bool or
            any(t.shape!=(1,) or t.dtype!=torch.int64 for t in (e,count,rows)) or
            any(t.device!=x.device for t in (positions,valid,epoch,*buffers))):
        raise ValueError('Original raw INT8 or fixed snapshot tuple metadata changed')
    # Fixed raw snapshot is copied BEFORE the opaque original consumer. No
    # cast, zeroing, clamping, NaN fix, valid-row sanitization or math rewrite.
    raw.copy_(x);p.copy_(positions);v.copy_(valid)
    count.add_(1);rows.fill_(x.shape[0]);e.copy_(epoch)
    return (x,*buffers)


def validate_buffer_owners_eager(x,buffers):
    # Outside capture only. No buffer may be another field's alias or a view
    # of native input storage. The pure Tensor graph does not query pointers.
    pointers=[t.untyped_storage().data_ptr() for t in (x,*buffers)]
    if len(set(pointers))!=len(pointers):
        raise ValueError('Snapshot fields must retain distinct original buffer owners')
