"""Screen genuine tiny wo_a/GMM/linear shapes with ND layout and cache controls."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

from selected_gemv import gemv
from microbench import capture, paired
from probe_wo_a import transpose_batch


SHAPES = {'wo_a': (8, 4096, 512), 'gmm1': (2, 5120, 512),
          'gmm2': (2, 256, 5120), 'q_b': (1, 512, 32768),
          'wo_b': (1, 4096, 5120), 'shared1': (1, 5120, 512),
          'shared2': (1, 256, 5120)}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--shape', choices=list(SHAPES), required=True)
    p.add_argument('--kind', choices=['cube', 'vector', 'all'], default='all')
    p.add_argument('--rotations', type=int, default=1)
    p.add_argument('--tiles', help='Explicit bn:bk pairs for a single kind, e.g. 128:512,256:256')
    args = p.parse_args()
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0); torch.manual_seed(20261010)
    slots, k, n = SHAPES[args.shape]
    grouped = args.shape.startswith('gmm')
    experts = 8 if grouped or args.shape == 'wo_a' else 1
    x = torch.randn((slots, k), device='npu', dtype=torch.bfloat16)
    weights = [(torch.randn((experts, k, n), device='npu') / k**.5).to(torch.bfloat16)
               for _ in range(args.rotations)]
    groups = torch.tensor([0, 1, 0, 0, 0, 0, 1, 0], device='npu', dtype=torch.int64) if grouped else None

    def native(sample, w):
        if grouped:
            return torch_npu.npu_grouped_matmul([sample], [w], group_list=groups,
                      split_item=2, group_type=0, group_list_type=1)[0]
        if args.shape == 'wo_a':
            return transpose_batch(sample.unsqueeze(0), w).squeeze(0)
        return torch.mm(sample, w[0])

    tasks = max(24, args.rotations)
    reference_bank = capture(lambda i: native(x, weights[i % args.rotations]), tasks)
    configurations = []
    if args.kind in ['cube', 'all']:
        configurations += [('cube', False, bn, bk) for bn, bk in
                           [(32, 128), (64, 128), (64, 256), (128, 256), (128, 512)]]
    if args.kind in ['vector', 'all']:
        configurations += [('vector', True, bn, bk) for bn, bk in
                           [(1, 4096), (2, 2048), (4, 1024), (8, 512)]]
        configurations += [('vector', False, bn, bk) for bn, bk in
                           [(16, 256), (32, 128)]]
    # This compiler asserts when the Cube K-loop has only one iteration.
    # Preserve the initial failure, then screen supported multi-iteration tiles.
    configurations = [c for c in configurations if c[0] != 'cube' or c[3] < k]
    if args.tiles:
        assert args.kind != 'all'
        configurations = [(args.kind, args.kind == 'vector', *map(int, token.split(':')))
                          for token in args.tiles.split(',')]
    nk_weights = None
    result = {'shape': args.shape, 'x_shape': list(x.shape), 'weight_shape': list(weights[0].shape),
              'rotations': args.rotations, 'weight_bytes': sum(w.numel()*w.element_size() for w in weights),
              'physical_format': [torch_npu.get_npu_format(w) for w in weights], 'results': [],
              'precision_gate': 'all elements rtol=1/64, atol=1/64; FP64 error recorded',
              'end_to_end': False}
    golden_w = weights[0].cpu().double()
    for kind, nk, bn, bk in configurations:
        row = {'kind': kind, 'nk_layout': nk, 'bn': bn, 'bk': bk}
        try:
            if nk and nk_weights is None:
                nk_weights = [w.transpose(1, 2).contiguous() for w in weights]
            chosen = nk_weights if nk else weights
            worst = 0.; golden_errors = []
            for factor in [0., .001, 1., 100.]:
                sample = x * factor
                ref = native(sample, weights[0])
                actual = gemv(sample, chosen[0], counts=groups, kind=kind, bn=bn, bk=bk, nk_layout=nk)
                torch.testing.assert_close(actual, ref, rtol=1/64, atol=1/64)
                worst = max(worst, float((actual-ref).abs().max()))
                wc = golden_w[[1, 6]] if grouped else golden_w
                golden = torch.bmm(sample.cpu().double().unsqueeze(1), wc).squeeze(1)
                golden_errors.append({'scale': factor,
                                      'native_max_abs': float((ref.cpu().double()-golden).abs().max()),
                                      'candidate_max_abs': float((actual.cpu().double()-golden).abs().max())})
            bank = capture(lambda i: gemv(x, chosen[i % args.rotations], counts=groups,
                            kind=kind, bn=bn, bk=bk, nk_layout=nk), tasks)
            row.update(paired({'native': reference_bank, 'candidate': bank}, repeats=30))
            row['speedup'] = row['medians_us']['native']/row['medians_us']['candidate']
            row.update(status='passed_screen', max_abs_vs_native=worst, fp64=golden_errors)
            del bank
        except Exception as exc:
            row.update(status='rejected', error=str(exc))
        result['results'].append(row)
        target.write_text(json.dumps(result, indent=2)+'\n')
        compact = {k: (v[-600:] if k == 'error' else v) for k, v in row.items()
                   if k not in ['samples_us', 'fp64']}
        print('CONFIG', json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
