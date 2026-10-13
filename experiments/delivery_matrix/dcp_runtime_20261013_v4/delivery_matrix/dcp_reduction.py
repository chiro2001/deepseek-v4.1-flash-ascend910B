"""Bounded rank-ordered gather/sum; no CPU reads of device tensor contents."""
import math

MAX_GATHER_BYTES = 128 * 1024 * 1024
MAX_CHUNK_ROWS = 64
ALIGNMENT_BYTES = 512


def plan_chunks(shape, element_bytes, world_size, max_gather_bytes=MAX_GATHER_BYTES,
                max_chunk_rows=MAX_CHUNK_ROWS):
    if (not shape or any(type(size) is not int or size < 0 for size in shape)
            or type(element_bytes) is not int or element_bytes <= 0
            or ALIGNMENT_BYTES % element_bytes or type(world_size) is not int or world_size < 1
            or type(max_gather_bytes) is not int or max_gather_bytes < 1
            or type(max_chunk_rows) is not int or max_chunk_rows < 1):
        raise ValueError("Invalid bounded DCP gather geometry/budget")
    if not math.prod(shape):
        return []
    row_elements = math.prod(shape[1:])
    alignment_elements = ALIGNMENT_BYTES // element_bytes
    rows = min(shape[0], max_chunk_rows, max_gather_bytes // (world_size * row_elements * element_bytes))
    def padded(count):
        return (count + alignment_elements - 1) // alignment_elements * alignment_elements
    while rows > 0 and world_size * padded(rows * row_elements) * element_bytes > max_gather_bytes:
        rows -= 1
    if rows < 1:
        raise ValueError("Even one aligned row exceeds the DCP gather staging budget")
    result = []
    for start in range(0, shape[0], rows):
        stop = min(start + rows, shape[0])
        elements = (stop - start) * row_elements
        result.append((start, stop, elements, padded(elements)))
    return result


def ordered_allreduce(tensor, group, max_gather_bytes=MAX_GATHER_BYTES, max_chunk_rows=MAX_CHUNK_ROWS):
    import torch
    world = int(group.world_size)
    if world <= 1 or tensor.numel() == 0:
        return tensor
    chunks = plan_chunks(tuple(tensor.shape), tensor.element_size(), world, max_gather_bytes, max_chunk_rows)
    output = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device)
    for start, stop, count, padded_count in chunks:
        # Contiguous, explicitly aligned input/output avoids strided HCCL views.
        source = tensor[start:stop].contiguous().reshape(-1)
        padded = torch.zeros(padded_count, dtype=tensor.dtype, device=tensor.device)
        padded[:count].copy_(source)
        gathered = torch.empty(world * padded_count, dtype=tensor.dtype, device=tensor.device)
        torch.distributed.all_gather_into_tensor(gathered, padded, group=group.device_group)
        lanes = gathered.view(world, padded_count)
        summed = lanes[0, :count]
        for rank in range(1, world):
            summed = summed + lanes[rank, :count]
        output[start:stop].copy_(summed.view(stop - start, *tensor.shape[1:]))
        # Release references before allocating the next chunk. The cap bounds
        # gather staging, not the unavoidable full output or one local sum.
        del source, padded, gathered, lanes, summed
    return output
