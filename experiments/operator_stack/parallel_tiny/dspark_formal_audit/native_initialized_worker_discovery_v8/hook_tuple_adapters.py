"""Minimal explicit tuple adapters for a future private consumer audit graph.

No install/patch or SDK import. Callers must retain all observer Tensor outputs
through their private outer graph and prove the real device epochs/pointers.
"""
if __package__:
    from .tensor_tuple_transport import snapshot_tuple
else:
    from tensor_tuple_transport import snapshot_tuple


def decoder_output_tuple(torch, native_result, positions, valid, epoch, buffers):
    if not isinstance(native_result,tuple) or len(native_result)!=2:
        raise ValueError('Actual native decoder (hidden,mix) Tensor ABI changed')
    hidden,mix=native_result
    if not isinstance(hidden,torch.Tensor) or not isinstance(mix,torch.Tensor):
        raise ValueError('Decoder carrier/mix must be actual native Tensors')
    rows=hidden.shape[0]
    observed=snapshot_tuple(torch,hidden,hidden.reshape(rows,-1),positions.reshape(rows,1),valid,epoch,buffers)
    return (observed[0],mix,*observed[1:])


def activation_before_mutating_consumer_tuple(torch, x, positions, valid, epoch, buffers):
    # Actual MSD pre-process may overwrite INT8 x with packed INT4 via raw
    # CANN GM writes without incrementing Tensor._version. Copy NOW, before
    # the original consumer/pack call; a later reference/version is no proof.
    rows=x.shape[0]
    return snapshot_tuple(torch,x,x.reshape(rows,-1),positions.reshape(rows,1),valid,epoch,buffers)


def cache_write_input_tuple(torch, updates, coordinates, epoch, buffers):
    if updates is None or coordinates.ndim!=2 or coordinates.shape[1]!=2:
        raise ValueError('Observe actual updates and formatted block+offset before native scatter')
    rows=coordinates.shape[0]
    valid=(coordinates[:,0]>=0)&(coordinates[:,1]>=0)
    return snapshot_tuple(torch,updates,updates.reshape(rows,-1),coordinates.to(torch.int64),valid,epoch,buffers)


def cache_reader_input_tuple(torch, q, cache, coordinates, epoch, buffers):
    if (cache.ndim<3 or min(cache.shape[:2])<=0 or coordinates.ndim!=2 or coordinates.shape[1]!=2
            or coordinates.device!=cache.device):
        raise ValueError('Need actual bounded PA cache rows and same-device block+offset coordinates')
    valid=((coordinates[:,0]>=0)&(coordinates[:,1]>=0)
           &(coordinates[:,0]<cache.shape[0])&(coordinates[:,1]<cache.shape[1]))
    blocks=coordinates[:,0].long().clamp(0,cache.shape[0]-1)*valid.long()
    offsets=coordinates[:,1].long().clamp(0,cache.shape[1]-1)*valid.long()
    values=cache[blocks,offsets].reshape(coordinates.shape[0],-1)
    return snapshot_tuple(torch,q,values,coordinates.to(torch.int64),valid,epoch,buffers)


def accepted_cache_input_mask(torch, rows, accepted_drafts, *, device):
    # Last actual + accepted drafts are committed inputs. Bonus is next-round input.
    return torch.arange(rows,device=device)<accepted_drafts+1
