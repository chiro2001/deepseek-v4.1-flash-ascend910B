"""Experimental FP32 reduction of one-token hidden vectors; strict math audit."""
import torch

ORIGINALS = {}
ENABLED = False
RECORDS = []
CALLS = 0


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
                save = ENABLED and len(RECORDS) < 81
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
    return {'capture': 'first 81 decode hidden-state reductions; count full request'}


def finish(worker, decode_steps):
    global ENABLED
    ENABLED = False
    torch.npu.synchronize()
    assert len(RECORDS) == 81 and CALLS == 81 * decode_steps, (len(RECORDS), CALLS, decode_steps)
    return {'first_decode_records': len(RECORDS), 'request_selected_calls': CALLS,
            'decode_steps': decode_steps}


@torch.inference_mode()
def audit(worker):
    from vllm.distributed import get_tensor_model_parallel_rank
    assert not ENABLED and len(RECORDS) == 81
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
    return {'rank': get_tensor_model_parallel_rank(), 'records': rows,
            'passed': all(r['repaired_bitwise_equal_fp64_reference'] for r in rows),
            'scope': 'First decode 81 real rank-local hidden reductions; exact BF16 result of FP64 rank sum'}
