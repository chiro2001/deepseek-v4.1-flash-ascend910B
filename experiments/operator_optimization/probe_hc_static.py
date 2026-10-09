"""Screen static HF32 and K/output tile variants against the proven HC path."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

from hc_vector import hc_vector
from hc_static import hc_static, round_hf32
from microbench import capture, paired


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    torch.manual_seed(20261010)
    x = torch.randn((1, 4, 5120), device='npu', dtype=torch.bfloat16)
    weights = [torch.randn((24, 20480), device='npu') * .1 / 20480**.5 for _ in range(8)]
    rounded = [round_hf32(w) for w in weights]
    scale = torch.tensor([1., .8, 1.2], device='npu')
    base = torch.randn((24,), device='npu') * .1
    mix = torch.rand((1, 4), device='npu')
    configurations = [(1, 4096, 1024), (1, 2048, 1024), (1, 8192, 1024),
                      (2, 4096, 1024), (2, 2048, 1024), (4, 2048, 1024),
                      (1, 4096, 512), (1, 4096, 2048), (1, 4096, 8192)]
    results = []
    baseline = capture(lambda i: hc_vector(x, weights[i % 8], scale, base), 24)
    for parts, bk, by in configurations:
        row = {'parts': parts, 'bk': bk, 'by': by}
        try:
            worst = [0.] * 4
            for size in [0., .001, 1., 100.]:
                for pmix in [None, mix]:
                    sample = x * size
                    ref = hc_vector(sample, weights[0], scale, base, pmix)
                    actual = hc_static(sample, rounded[0], scale, base, pmix,
                                       parts=parts, bk=bk, by=by)
                    for j, (a, b) in enumerate(zip(actual, ref)):
                        tol = 4e-3 if j == 0 else 1e-4
                        torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)
                        worst[j] = max(worst[j], float((a.float() - b.float()).abs().max()))
            bank = capture(lambda i: hc_static(x, rounded[i % 8], scale, base,
                                               parts=parts, bk=bk, by=by), 24)
            row.update(paired({'baseline': baseline, 'static': bank}))
            row['speedup'] = row['medians_us']['baseline'] / row['medians_us']['static']
            row.update(status='passed_screen', max_abs_vs_proven_hc=worst)
            del bank
        except Exception as exc:
            row.update(status='rejected', error=str(exc))
        results.append(row)
        target.write_text(json.dumps({'results': results, 'end_to_end': False,
                                      'reference': 'proven HC; native model audit required'}, indent=2) + '\n')
        print('CONFIG', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
