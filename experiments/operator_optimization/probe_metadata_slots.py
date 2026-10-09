"""Exact V4.1 slot parity tests, including padding and physical group ownership."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from metadata_kernels import prepare_slots


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20261010)
    rows = []
    for n in [1, 2, 129, 2048]:
        for compressed, ratio in [(False, 1), (True, 1), (True, 2)]:
            for block_size in [1, 64, 128]:
                for skip in [False, True]:
                    for with_positions in [False, True]:
                        # Include negative PAD, slot-completion disagreement,
                        # high physical page IDs and changing odd/even positions.
                        raw = torch.randint(-3, 10000, (n,), device='npu', dtype=torch.int64)
                        raw[0] = -1
                        if n > 1: raw[1] = 2**32 + 127
                        positions = torch.arange(n, device='npu', dtype=torch.int64) + 127 if with_positions else None
                        actual_reqs = 0 if skip else 1
                        actual_tokens = max(0, n - 1)
                        query = torch.tensor([0, n, n], device='npu', dtype=torch.int32)
                        common = SimpleNamespace(slot_mapping=raw, query_start_loc=query)
                        builders = [SimpleNamespace(_slot_mapping_2d=torch.full((n + 3, 2), -777, device='npu', dtype=torch.int32)) for _ in range(2)]
                        active = raw
                        if compressed and ratio != 1:
                            valid = (raw >= 0) & ((raw + 1) % ratio == 0)
                            active = torch.where(valid, raw // ratio, -1)
                        valid = active >= 0
                        if compressed and ratio == 2:
                            if skip: valid.zero_()
                            else:
                                valid &= torch.arange(n, device='npu') < query[actual_reqs].clamp_max(actual_tokens)
                                if positions is not None: valid &= positions.remainder(2) == 1
                        physical = active.clamp_min(0)
                        expected = torch.stack([torch.where(valid, physical // block_size, -1),
                                                torch.where(valid, physical % block_size, -1)], dim=-1).int()
                        for builder in builders:
                            actual = prepare_slots(builder, common, positions, n, actual_reqs, actual_tokens,
                                                   compressed, ratio, block_size, skip)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            assert torch.all(builder._slot_mapping_2d[n:] == -777)
                        assert builders[0]._slot_mapping_2d.data_ptr() != builders[1]._slot_mapping_2d.data_ptr()
                        rows.append(dict(n=n, compressed=compressed, ratio=ratio, block_size=block_size,
                                         skip=skip, positions=with_positions, exact=True))
    # Reuse the same buffer while advancing across block and C2 boundaries.
    builder = SimpleNamespace(_slot_mapping_2d=torch.full((4, 2), -777, device='npu', dtype=torch.int32))
    for slot in [-1, 0, 1, 126, 127, 128, 129, 255, 256, 257]:
        for pos in [126, 127, 128, 129]:
            for actual_reqs in [0, 1]:
                raw = torch.tensor([slot], device='npu', dtype=torch.int64)
                positions = torch.tensor([pos], device='npu', dtype=torch.int64)
                common = SimpleNamespace(slot_mapping=raw, query_start_loc=torch.tensor([0, 1], device='npu', dtype=torch.int32))
                actual = prepare_slots(builder, common, positions, 1, actual_reqs, 1, True, 2, 64, False)
                valid = slot >= 0 and slot % 2 == 1 and pos % 2 == 1 and actual_reqs == 1
                value = [slot // 2 // 64, slot // 2 % 64] if valid else [-1, -1]
                expected = torch.tensor([value], device='npu', dtype=torch.int32)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                rows.append(dict(n=1, slot=slot, position=pos, actual_reqs=actual_reqs, exact=True))
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    result = {'cases': len(rows), 'all_bit_equal': True, 'tail_unchanged': True,
              'different_groups_keep_different_buffers': True, 'results': rows}
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('METADATA_SLOTS_PARITY', json.dumps({k:v for k,v in result.items() if k != 'results'}), flush=True)


if __name__ == '__main__': main()
