"""CPU integration check of the installed WorkerProfiler and our schedule clock."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.update(STACK_REAL_WEIGHTS='1', STACK_REAL_AUDIT='0',
                  STACK_PHYSICAL_CHIPS='8,9,10,11,12,13,14,15')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch_npu
    import torch
    assert not torch.npu.is_initialized(), 'CPU clock test must not initialize a device'
    import vllm.distributed
    import formal_tp8_profile as profile

    class RecordingProfiler:
        def __init__(self):
            self.starts = self.steps = self.stops = 0

        def start(self):
            self.starts += 1

        def step(self):
            self.steps += 1

        def stop(self):
            self.stops += 1

    original_factory = torch_npu.profiler.profile
    original_rank = vllm.distributed.get_tensor_model_parallel_rank
    torch_npu.profiler.profile = lambda **kwargs: RecordingProfiler()
    vllm.distributed.get_tensor_model_parallel_rank = lambda: 0
    results = []
    try:
        for worker_driven in (False, True):
            worker = SimpleNamespace()
            profile.configure(worker, 'PipeUtilization', '/tmp/unused-profile-step-test', 5, 10, worker_driven)
            wrapper = worker.profiler
            wrapper.start()
            for _ in range(10000):
                # Polling the frontend does not call WorkerProfiler.step.
                assert wrapper.step_clock.worker_calls == 0
            for _ in range(50 if worker_driven else 15):
                wrapper.step()
                if not worker_driven:
                    profile.advance(worker)
            if worker_driven:
                try:
                    profile.advance(worker)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Frontend advanced the asynchronous clock')
            wrapper.stop()
            receipt = profile.step_receipt(worker)
            assert receipt['complete'] and receipt['schedule_steps'] == 16
            assert wrapper.profiler.starts == wrapper.profiler.stops == 1
            assert wrapper.profiler.steps == 16
            assert receipt['manual_calls'] == (0 if worker_driven else 16)
            assert receipt['worker_calls'] == (50 if worker_driven else 15)
            results.append(receipt)
    finally:
        torch_npu.profiler.profile = original_factory
        vllm.distributed.get_tensor_model_parallel_rank = original_rank
    result = {'passed': True, 'cases': results, 'device_initialized': False,
              'scope': 'Installed WorkerProfiler control flow and bounded schedule; fake recording backend, no NPU collection claim'}
    assert not torch.npu.is_initialized()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('WORKER_PROFILE_STEP_DRIVER_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
