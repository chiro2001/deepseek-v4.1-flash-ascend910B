"""Bounded, rank-separated formal-model profiling; no dummy-model setup."""
import os
from pathlib import Path

METRICS = ('PipeUtilization', 'ArithmeticUtilization', 'Memory', 'MemoryL0',
           'MemoryUB', 'L2Cache', 'ResourceConflictRatio')


def configure(worker, metric, output, warmup=5, active=10, worker_driven=False):
    worker = getattr(worker, 'worker', worker)
    assert os.getenv('STACK_REAL_WEIGHTS') == '1'
    assert os.getenv('STACK_REAL_AUDIT') == '0', 'Profile without route/clone audit'
    assert metric in METRICS and warmup >= 0 and active > 0
    import torch_npu
    from vllm.config import ProfilerConfig
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper
    from profile_step_clock import ProfileStepClock
    rank = get_tensor_model_parallel_rank()
    chips = os.environ['STACK_PHYSICAL_CHIPS'].split(',')
    assert len(chips) == 8 and 0 <= rank < 8
    name = f'formal_tp8_rank{rank}_chip{chips[rank]}'
    target = str(Path(output) / name)
    config = ProfilerConfig(profiler='torch', torch_profiler_dir=target,
                           torch_profiler_with_stack=False, torch_profiler_record_shapes=True)

    class CounterProfiler(TorchNPUProfilerWrapper):
        def _start(self):
            self.step_clock = ProfileStepClock(warmup, active, worker_driven)
            super()._start()

        def _profiler_step(self):
            # NPUWorker calls this before every real execute_model. The
            # vendor wrapper returns True without advancing an NPU schedule.
            # Advance only here for asynchronous scheduling; frontend polls
            # may be more frequent than model steps or drain several tokens.
            if self.step_clock.worker_step():
                self.profiler.step()
            return True

        def _stop(self):
            if not worker_driven:
                # End outside RECORD even when the manual loop supplies
                # exactly warmup+active advances, as older collectors do.
                while not self.step_clock.receipt()['complete']:
                    self.step_clock.manual_step()
                    self.profiler.step()
            super()._stop()

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
            'metric': metric, 'output': target, 'warmup': warmup, 'active': active,
            'schedule_driver': 'worker.execute_model' if worker_driven else 'manual RPC',
            'required_schedule_steps': warmup + active + 1}


def advance(worker):
    worker = getattr(worker, 'worker', worker)
    worker.profiler.step_clock.manual_step()
    worker.profiler.profiler.step()


def step_receipt(worker):
    worker = getattr(worker, 'worker', worker)
    from vllm.distributed import get_tensor_model_parallel_rank
    return dict(worker.profiler.step_clock.receipt(), rank=get_tensor_model_parallel_rank())
