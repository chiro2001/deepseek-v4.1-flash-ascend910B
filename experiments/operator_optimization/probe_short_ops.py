"""Audit every tiny expert pair and screen route/HC post block choices."""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

from microbench import capture, paired
from short_ops import route_init, route_combine, hc_post


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    args = p.parse_args(); target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0); enable_custom_op(); torch.manual_seed(20261010)
    x = torch.randn((1, 5120), device='npu', dtype=torch.bfloat16)
    ids = torch.tensor([[6, 1]], device='npu', dtype=torch.int32)
    probs = torch.tensor([[.3, .7]], device='npu', dtype=torch.bfloat16)
    down = torch.randn((2, 5120), device='npu', dtype=torch.bfloat16)

    def native_init(sample, experts):
        return torch_npu.npu_moe_init_routing_v2(sample, experts, scale=None,
            active_num=2, expert_num=8, expert_tokens_num_type=1,
            expert_tokens_num_flag=True, active_expert_range=[0, 8], quant_mode=-1,
            x_dtype=None)

    def native_combine(sample, reverse):
        return torch_npu.npu_moe_token_unpermute(sample, reverse, probs)

    results = {'route_pairs': 0, 'route_init': [], 'route_combine': [], 'hc_post': []}
    for a, b in itertools.product(range(8), repeat=2):
        pair = torch.tensor([[a, b]], device='npu', dtype=torch.int32)
        expected = native_init(x, pair); actual = route_init(x, pair)
        for index in range(3):
            torch.testing.assert_close(actual[index], expected[index].to(actual[index].dtype), rtol=0, atol=0)
        reference = native_combine(down, expected[1])
        output = route_combine(down, actual[1], probs)
        torch.testing.assert_close(output, reference, rtol=1/64, atol=1/64)
        results['route_pairs'] += 1
    expected = native_init(x, ids)
    native_banks = {'init': capture(lambda _: native_init(x, ids)),
                    'combine': capture(lambda _: native_combine(down, expected[1]))}
    for block in [256, 512, 1024, 2048]:
        for kind, call in [('route_init', lambda _: route_init(x, ids, block)),
                           ('route_combine', lambda _: route_combine(down, expected[1], probs, block))]:
            bank = capture(call)
            row = {'block': block, **paired({'native': native_banks['init' if kind == 'route_init' else 'combine'], 'candidate': bank})}
            results[kind].append(row)
            print(kind, json.dumps(row), flush=True)
            del bank
    residual = torch.randn((1, 1, 4, 5120), device='npu', dtype=torch.bfloat16)
    post = torch.randn((1, 1, 4), device='npu')
    comb = torch.randn((1, 1, 4, 4), device='npu')
    hidden = x.reshape(1, 1, 5120)
    native_post = lambda _: torch.ops._C_ascend.npu_hc_post(hidden, residual, post, comb)
    baseline = capture(native_post)
    for fma, block in itertools.product([False, True], [256, 512, 1024, 2048]):
        row = {'fma': fma, 'block': block}
        try:
            worst = 0.; equal = []
            for factor in [0., .001, 1., 100.]:
                reference = torch.ops._C_ascend.npu_hc_post(hidden*factor, residual*factor, post, comb)
                output = hc_post(hidden*factor, residual*factor, post, comb, block, fma)
                torch.testing.assert_close(output, reference, rtol=1/64, atol=1/64)
                worst = max(worst, float((output-reference).abs().max()))
                equal.append(bool(torch.equal(output, reference)))
            bank = capture(lambda _: hc_post(hidden, residual, post, comb, block, fma))
            row.update(paired({'native': baseline, 'candidate': bank}))
            row.update(status='passed_screen', max_abs_vs_native=worst, bit_equal=equal)
            del bank
        except Exception as exc:
            row.update(status='rejected', error=str(exc))
        results['hc_post'].append(row)
        print('HC_POST', json.dumps(row), flush=True)
        target.write_text(json.dumps(results, indent=2)+'\n')
    target.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
