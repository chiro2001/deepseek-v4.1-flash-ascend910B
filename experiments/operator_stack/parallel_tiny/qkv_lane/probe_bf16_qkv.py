"""Projection-only BF16 probe; never report this as a model or tiny result."""
import argparse
import json
import os
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
import torch_npu


def ordered(tensor):
    bits = tensor.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -bits - 32768, bits)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '6'
    assert os.environ['QKV_LANE_PHYSICAL_CHIP'] == '6'
    assert os.environ['HCCL_DETERMINISTIC'] == 'strict'
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261011)
    torch.npu.set_device(0)
    # The native Ascend unquantized_gemm implementation delegates to F.linear.
    # These are synthetic weights, not projections exported from a live model.
    weights = [(torch.randn((512, 5120))*0.001).to(torch.bfloat16).npu() for _ in range(2)]
    packed_nd = torch.cat(weights, dim=0).contiguous()
    packed_nz = torch_npu.npu_format_cast(packed_nd, 29)
    actual_formats = {'2':int(torch_npu.get_npu_format(packed_nd)), '29':int(torch_npu.get_npu_format(packed_nz))}
    assert torch.equal(torch_npu.npu_format_cast(packed_nz, 2), packed_nd)
    reports = []
    for seed in range(24):
        generator = torch.Generator().manual_seed(seed+20261011)
        scale = (0., 1e-5, 1e-3, 1., 8., 32.)[seed%6]
        x = (torch.randn((1, 5120), generator=generator)*scale).to(torch.bfloat16).npu()
        expected = [F.linear(x, w) for w in weights]
        for layout, packed in ((2, packed_nd), (29, packed_nz)):
            merged = F.linear(x, packed)
            for kind, a, b in zip(('q_a','kv'), (merged[:, :512],merged[:,512:]), expected):
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                reports.append({'seed':seed, 'scale':scale, 'layout':layout, 'actual_layout':actual_formats[str(layout)], 'projection':kind,
                                'bitwise_equal':bool(torch.equal(a.view(torch.int16),b.view(torch.int16))),
                                'max_bf16_ulp':int((ordered(a)-ordered(b)).abs().max().item()),
                                'max_abs':float((a.float()-b.float()).abs().max().item())})
    precision = {str(layout):{'passed':max(r['max_bf16_ulp'] for r in reports if r['layout']==layout)<=1,
                             'all_bitwise_equal':all(r['bitwise_equal'] for r in reports if r['layout']==layout),
                             'max_bf16_ulp':max(r['max_bf16_ulp'] for r in reports if r['layout']==layout)}
                 for layout in (2,29)}
    (args.output/'precision.json').write_text(json.dumps({'scope':'Projection-only synthetic BF16 probe',
                                                        'summary':precision,'rows':reports},indent=2)+'\n')
    print('PROBE_PRECISION',json.dumps(precision),flush=True)
    x = (torch.randn((1,5120))*0.001).to(torch.bfloat16).npu()
    functions = {'native_a':lambda:(F.linear(x,weights[0]),F.linear(x,weights[1])),
                 'native_b':lambda:(F.linear(x,weights[0]),F.linear(x,weights[1]))}
    for layout, packed in ((2,packed_nd),(29,packed_nz)):
        if precision[str(layout)]['passed']:
            functions[f'merged_{layout}'] = lambda w=packed:F.linear(x,w)
    banks = {}
    for name, fn in functions.items():
        for _ in range(8):
            fn()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            refs = fn()
        banks[name] = (graph, refs)

    def time_us(name):
        graph, refs = banks[name]
        for _ in range(32):
            graph.replay()
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(1024):
            graph.replay()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end)*1000/1024

    pairs = []
    arms = list(banks)
    for pair in range(16):
        order = arms if pair%2==0 else list(reversed(arms))
        values = {name:time_us(name) for name in order}
        pairs.append({'pair':pair,'projection_us':values,
                      'speedup':{name:values['native_a']/values[name] for name in arms if name.startswith('merged_')}})
    result = {'scope':'BF16 projection-only probe; not tiny/full model; no RoPE/cache/compressor/MoE',
              'weights':'synthetic random BF16', 'hardware':'a3-21 Ascend910C physical6',
              'input_shape':[1,5120], 'q_a_shape':[1,512], 'kv_shape':[1,512],
              'native_layout':int(torch_npu.get_npu_format(weights[0])),
              'requested_merged_layouts':[2,29], 'actual_merged_layouts':actual_formats,
              'strict':os.environ['HCCL_DETERMINISTIC'],
              'same_process':True,'precision':precision,'pairs':pairs,
              'median_projection_us':{name:statistics.median(p['projection_us'][name] for p in pairs) for name in arms},
              'pairs_faster':{name:sum(p['speedup'][name]>1 for p in pairs) for name in arms if name.startswith('merged_')}}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PROBE_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)


if __name__ == '__main__':
    main()
