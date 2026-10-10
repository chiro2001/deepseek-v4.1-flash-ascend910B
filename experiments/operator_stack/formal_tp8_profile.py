"""Bounded, rank-separated formal-model profiling; no dummy-model setup."""
import os
from pathlib import Path

METRICS = ('PipeUtilization', 'ArithmeticUtilization', 'Memory', 'MemoryL0',
           'MemoryUB', 'L2Cache', 'ResourceConflictRatio')


def configure(worker, metric, output, warmup=5, active=10):
    worker = getattr(worker, 'worker', worker)
    assert os.getenv('STACK_REAL_WEIGHTS') == '1'
    assert os.getenv('STACK_REAL_AUDIT') == '0', 'Profile without route/clone audit'
    assert metric in METRICS and warmup >= 0 and active > 0
    import torch_npu
    from vllm.config import ProfilerConfig
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper
    rank = get_tensor_model_parallel_rank()
    chips = os.environ['STACK_PHYSICAL_CHIPS'].split(',')
    assert len(chips) == 8 and 0 <= rank < 8
    name = f'formal_tp8_rank{rank}_chip{chips[rank]}'
    target = str(Path(output) / name)
    config = ProfilerConfig(profiler='torch', torch_profiler_dir=target,
                           torch_profiler_with_stack=False, torch_profiler_record_shapes=True)

    class CounterProfiler(TorchNPUProfilerWrapper):
        @staticmethod
        def _create_profiler(config, trace_name):
            return torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1),
                record_shapes=True, with_stack=False, profile_memory=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=getattr(torch_npu.profiler.AiCMetrics, metric),
                    export_type=torch_npu.profiler.ExportType.Text, data_simplification=False),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(target, worker_name=name))

    worker.profiler_config = config
    worker.profiler = CounterProfiler(config, name)
    return {'rank': rank, 'expected_physical_chip': int(chips[rank]),
            'metric': metric, 'output': target, 'warmup': warmup, 'active': active}


def advance(worker):
    worker = getattr(worker, 'worker', worker)
    worker.profiler.profiler.step()
