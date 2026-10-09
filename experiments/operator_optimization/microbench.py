"""Paired graph timing, with replay overhead amortized over multiple tasks."""
import statistics

import torch


def capture(call, count=20):
    for i in range(count):
        call(i)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [call(i) for i in range(count)]
    # Keep outputs and their storage alive for the graph lifetime.
    return graph, outputs, count


def elapsed(bank, repeats=50):
    graph, _, count = bank
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (count * repeats)


def paired(banks, pairs=7, repeats=50):
    samples = {name: [] for name in banks}
    for pair in range(pairs):
        order = list(banks)
        if pair % 2:
            order.reverse()
        for name in order:
            samples[name].append(elapsed(banks[name], repeats))
    return {"samples_us": samples,
            "medians_us": {name: statistics.median(values) for name, values in samples.items()},
            "profiler": "OFF", "same_process": True}
