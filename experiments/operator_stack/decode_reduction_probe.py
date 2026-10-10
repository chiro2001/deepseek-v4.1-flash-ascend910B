"""Experimental FP32 reduction of one-token hidden vectors; strict math audit."""
import torch

ORIGINALS = {}
ENABLED = False
RECORDS = []
CALLS = 0
EXPECTED_PER_STEP = 80


def install(worker):
    from vllm.distributed import get_tp_group, get_ep_group
    groups = (get_tp_group(), get_ep_group())
    assert all(g.world_size == 8 for g in groups)
    for group in groups:
        cls = type(group)
        if cls in ORIGINALS:
            continue
        original = cls.all_reduce
        ORIGINALS[cls] = original

        def bind(original):
            def reduced(self, value):
                global CALLS
                selected = (self.world_size == 8 and value.dtype == torch.bfloat16 and
                            value.numel() == 5120 and value.shape[-1] == 5120)
                if not selected:
                    return original(self, value)
                save = ENABLED and len(RECORDS) < EXPECTED_PER_STEP
                local = value.clone() if save else None
                output = original(self, value.float())
                result = output.to(value.dtype)
                if ENABLED:
                    CALLS += 1
                if save:
                    RECORDS.append((self, original, local, output.clone(), result.clone()))
                return result
            return reduced

        cls.all_reduce = bind(original)
    return {'classes': len(ORIGINALS), 'dtype': 'BF16 -> native FP32 allreduce -> BF16',
            'shape_guard': '8 ranks and 5120 elements, last dimension 5120',
            'scope': 'Experimental one-token hidden-state reductions only; no global determinism flag'}


def begin(worker):
    global ENABLED, CALLS
    assert ORIGINALS and not ENABLED
    torch.npu.synchronize()
    RECORDS.clear()
    CALLS = 0
    ENABLED = True
    return {'capture': 'first 80 decode hidden-state reductions; count full request'}


def finish(worker, decode_steps):
    global ENABLED
    ENABLED = False
    torch.npu.synchronize()
    assert len(RECORDS) == EXPECTED_PER_STEP and CALLS == EXPECTED_PER_STEP * decode_steps, (len(RECORDS), CALLS, decode_steps)
    return {'first_decode_records': len(RECORDS), 'request_selected_calls': CALLS,
            'decode_steps': decode_steps}


@torch.inference_mode()
def audit(worker):
    from vllm.distributed import get_tensor_model_parallel_rank
    assert not ENABLED and len(RECORDS) == EXPECTED_PER_STEP
    # Keep real local vectors on this host for inexpensive subsequent isolated
    # collective experiments; no weights or >=1MB artifacts travel over SSH.
    import os
    from pathlib import Path
    path = Path(os.environ['VLLM_CACHE_ROOT']).parent
    rank = get_tensor_model_parallel_rank()
    vectors = torch.stack([record[2].cpu() for record in RECORDS])
    vector_file = path / f'reduction_local_vectors_rank{rank}.pt'
    torch.save({'vectors': vectors, 'groups': [record[0].unique_name for record in RECORDS],
                'scope': 'first real decode rank-local BF16 hidden reductions'}, vector_file)
    rows = []
    for ordinal, (group, original, local, fp32, repaired) in enumerate(RECORDS):
        # BF16 values are exactly representable in FP32; all-gather then CPU
        # FP64 sum establishes an independent high-precision reference.
        gathered = group.all_gather(local.float(), dim=0).cpu().double().reshape(8, 5120)
        assert torch.isfinite(gathered).all()
        reference = gathered.sum(dim=0).reshape(local.shape).to(torch.bfloat16)
        actual = repaired.cpu()
        native = original(group, local.clone()).cpu()
        equal = torch.equal(reference, actual)
        rows.append({'ordinal': ordinal, 'group': group.unique_name,
                     'repaired_bitwise_equal_fp64_reference': equal,
                     'repaired_differing_elements': int((reference != actual).sum()),
                     'repaired_max_abs': float((reference.float() - actual.float()).abs().max()),
                     'fp32_max_abs_vs_fp64': float((gathered.sum(dim=0).reshape(local.shape) - fp32.cpu().double()).abs().max()),
                     'native_bf16_differing_elements': int((reference != native).sum()),
                     'native_bf16_max_abs': float((reference.float() - native.float()).abs().max())})
    return {'rank': rank, 'records': rows, 'vectors_file': str(vector_file),
            'passed': all(r['repaired_bitwise_equal_fp64_reference'] for r in rows),
            'scope': 'First decode 80 real rank-local hidden reductions; exact BF16 result of FP64 rank sum'}
