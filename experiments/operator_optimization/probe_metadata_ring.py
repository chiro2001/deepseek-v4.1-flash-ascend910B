"""Exact compressor ownership, completion, source-position and RoPE checks."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from metadata_kernels import prepare_ring


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); torch.npu.set_device(0); torch.manual_seed(20261010)
    rows = []
    for nr in [1, 3]:
        for n in [1, 5, 129]:
            for base in [0, 127, 128]:
                for skip in [False, True]:
                    for actual_reqs in [0, nr]:
                        for has_rope in [False, True]:
                            query = torch.linspace(0, n, nr + 1, device='npu').int()
                            seq_lens = torch.tensor([base + 1] * nr, device='npu', dtype=torch.int32)
                            positions = torch.arange(n, device='npu', dtype=torch.int64) + base
                            blocks = torch.randint(1, 10000, (nr, 3), device='npu', dtype=torch.int32)
                            actual_tokens = max(0, n - 1) if nr > 1 else n
                            common = SimpleNamespace(query_start_loc=query, block_table_tensor=blocks)
                            dim = 64
                            full_cos = torch.randn((512, 1, 1, dim), device='npu') if has_rope else None
                            full_sin = torch.randn_like(full_cos) if has_rope else None
                            builder = SimpleNamespace(
                                _c2_ring_metadata=torch.full((5 * (nr + 2),), -777, device='npu', dtype=torch.int32),
                                _c2_complete_mask=torch.zeros((n + 3,), device='npu', dtype=torch.bool),
                                _c2_source_positions=torch.full((n + 3,), -777, device='npu', dtype=torch.int64),
                                _c2_source_cos=torch.full((n + 3, 1, 1, dim), -777., device='npu'),
                                _c2_source_sin=torch.full((n + 3, 1, 1, dim), -777., device='npu'))
                            starts = query[:-1]; ends = query[1:]
                            used = (ends.clamp_max(actual_tokens) - starts).clamp_min(0)
                            used = torch.where(torch.arange(nr, device='npu') < actual_reqs, used, 0)
                            if skip: used.zero_()
                            expected_ring = torch.stack([(seq_lens - (ends - starts)).clamp_min(0), used,
                                                         starts, starts, torch.where(used > 0, blocks[:, 0], 0)])
                            complete = (positions % 2 == 1) & (torch.arange(n, device='npu') < query[actual_reqs].clamp_max(actual_tokens))
                            if skip: complete.zero_()
                            source = torch.where(complete, positions - 1, torch.zeros_like(positions))
                            prepare_ring(builder, common, positions, seq_lens, nr, actual_reqs, actual_tokens,
                                         n, skip, full_cos, full_sin)
                            torch.testing.assert_close(builder._c2_ring_metadata[:5 * nr].view(5, nr), expected_ring, rtol=0, atol=0)
                            torch.testing.assert_close(builder._c2_complete_mask[:n], complete, rtol=0, atol=0)
                            torch.testing.assert_close(builder._c2_source_positions[:n], source, rtol=0, atol=0)
                            if has_rope:
                                torch.testing.assert_close(builder._c2_source_cos[:n], full_cos.index_select(0, source), rtol=0, atol=0)
                                torch.testing.assert_close(builder._c2_source_sin[:n], full_sin.index_select(0, source), rtol=0, atol=0)
                            assert torch.all(builder._c2_ring_metadata[5 * nr:] == -777)
                            assert torch.all(builder._c2_source_positions[n:] == -777)
                            assert torch.all(builder._c2_source_cos[n:] == -777)
                            rows.append(dict(nr=nr, n=n, base=base, skip=skip, actual_reqs=actual_reqs, has_rope=has_rope, exact=True))
    result = {'cases': len(rows), 'all_bit_equal': True, 'tail_unchanged': True, 'results': rows}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print('METADATA_RING_PARITY', json.dumps({k:v for k,v in result.items() if k != 'results'}), flush=True)


if __name__ == '__main__': main()
