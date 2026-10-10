"""Isolated collectives on saved formal inputs; never claim model E2E latency."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics

import torch
import torch.multiprocessing as mp
import torch_npu
import triton
import triton.language as tl


@triton.jit
def ordered_sum(values, output, BLOCK: tl.constexpr, COMPENSATED: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    error = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(8):
        value = tl.load(values + rank * 5120 + indices, indices < 5120, 0).to(tl.float32)
        updated = total + value
        if COMPENSATED:
            correction = tl.where(tl.abs(total) >= tl.abs(value),
                                  (total - updated) + value, (value - updated) + total)
            error = error + correction
        total = updated
    if COMPENSATED:
        total = total + error
    tl.store(output + indices, total, indices < 5120)


def worker(rank, args, port):
    import torch.distributed as dist
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=8, timeout=timedelta(seconds=180))
    source = args.input / f'reduction_local_vectors_rank{rank}.pt'
    saved = torch.load(source, weights_only=True, map_location='cpu')
    cpu = saved['vectors']
    assert cpu.dtype == torch.bfloat16 and cpu.numel() == 80 * 5120
    cpu = cpu.reshape(80, 5120)
    assert torch.isfinite(cpu).all()
    vectors = cpu.to(f'npu:{rank}')
    # One untimed large gather creates all independent FP64 references.
    reference_buffer = torch.empty((8 * 80, 5120), dtype=torch.bfloat16, device=vectors.device)
    dist.all_gather_into_tensor(reference_buffer, vectors)
    reference = reference_buffer.cpu().double().reshape(8, 80, 5120).sum(dim=0).to(torch.bfloat16)
    gathered = torch.empty((8, 5120), dtype=torch.bfloat16, device=vectors.device)
    output = torch.empty(5120, dtype=torch.bfloat16, device=vectors.device)
    methods = [('native_bf16', None, None), ('native_fp32', None, None)]
    methods += [(f'ordered_{compensated}_{block}', compensated, block)
                for compensated in (False, True) for block in (256, 512, 1024)]
    receipts = []
    for name, compensated, block in methods:
        def operation(x):
            if name == 'native_bf16':
                result = x.clone()
                dist.all_reduce(result)
                return result
            if name == 'native_fp32':
                result = x.float()
                dist.all_reduce(result)
                return result.to(torch.bfloat16)
            dist.all_gather_into_tensor(gathered.view(-1), x)
            ordered_sum[(triton.cdiv(5120, block),)](
                gathered, output, block, compensated, enable_fp_fusion=False)
            return output

        audit = []
        for index in range(80):
            observed = [operation(vectors[index]).clone() for _ in range(3)]
            torch.npu.synchronize()
            actual = [x.cpu() for x in observed]
            audit.append({'input': index, 'repeat_equal': all(torch.equal(actual[0], x) for x in actual),
                          'fp64_reference_equal': all(torch.equal(reference[index], x) for x in actual),
                          'max_abs_vs_reference': max(float((reference[index].float() - x.float()).abs().max()) for x in actual),
                          'differing_elements': max(int((reference[index] != x).sum()) for x in actual)})
        (args.output / f'rank{rank}_{name}_audit.json').write_text(json.dumps({
            'rank': rank, 'method': name, 'audit': audit,
            'scope': 'Math and repeatability only; timing has not run'}, indent=2) + '\n')
        # Actual inputs stay stable; timing uses same-process graph replays,
        # alternated across methods by later paired invocations if necessary.
        for _ in range(3): operation(vectors[0])
        torch.npu.synchronize()
        dist.barrier()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output = operation(vectors[0])
        torch.npu.synchronize()
        graph_observed = []
        for _ in range(3):
            graph.replay()
            torch.npu.synchronize()
            graph_observed.append(graph_output.cpu().clone())
        graph_equal = all(torch.equal(graph_observed[0], x) for x in graph_observed)
        graph_reference_equal = all(torch.equal(reference[0], x) for x in graph_observed)
        (args.output / f'rank{rank}_{name}_graph_audit.json').write_text(json.dumps({
            'rank': rank, 'method': name, 'repeat_equal': graph_equal,
            'fp64_reference_equal': graph_reference_equal,
            'scope': 'Three graph replays on the first saved attention reduction vector'}, indent=2) + '\n')
        for _ in range(10): graph.replay()
        torch.npu.synchronize()
        timings = []
        for _ in range(8):
            dist.barrier()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(100): graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / 100)
        receipts.append({'method': name, 'audit': audit,
                         'passed': all(r['repeat_equal'] and r['fp64_reference_equal'] for r in audit) and graph_equal and graph_reference_equal,
                         'graph_repeat_equal': graph_equal,
                         'graph_fp64_reference_equal': graph_reference_equal,
                         'event_us_per_collective': timings,
                         'event_median_us': statistics.median(timings)})
        # Keep the captured output alive until all replays finish.
        assert graph_output.numel() == 5120
        print('FORMAL_ISOLATED_REDUCTION', rank, name, receipts[-1]['passed'], flush=True)
    (args.output / f'rank{rank}.json').write_text(json.dumps({
        'rank': rank, 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'methods': receipts, 'performance_claim': None,
        'scope': 'Independent FP64 reference on 80 saved formal rank-local vectors; '
                 'isolated event timing, fixed method order, no model E2E claim'}, indent=2) + '\n')
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--physical-chips', required=True)
    args = parser.parse_args()
    assert args.physical_chips == os.environ['STACK_PHYSICAL_CHIPS'] == '8,9,10,11,12,13,14,15'
    assert len(list(args.input.glob('reduction_local_vectors_rank*.pt'))) == 8
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=8, join=True)
    results = [json.loads((args.output / f'rank{rank}.json').read_text()) for rank in range(8)]
    (args.output / 'result.json').write_text(json.dumps({'ranks': results,
        'scope': 'Isolated diagnostic/collective optimization only', 'performance_claim': None}, indent=2) + '\n')


if __name__ == '__main__':
    main()
