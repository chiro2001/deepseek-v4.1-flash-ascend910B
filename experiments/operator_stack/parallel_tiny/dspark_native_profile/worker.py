"""New-job native DSpark profiler; no device/consumer/performance admission."""
import functools
import json
import os
from pathlib import Path
import torch
from parallel_tiny.dspark_formal_prep.worker import FormalDSparkWorker
from .phase import is_real_decode


class FormalDSparkProfileWorker(FormalDSparkWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        from vllm.distributed import get_tensor_model_parallel_rank
        self._dspark_profile_state = {'rank': get_tensor_model_parallel_rank(),
            'armed': False, 'started': False, 'completed': False, 'actual_decode_steps': 0,
            'warmup_steps': 5, 'active_steps': 10, 'profiler': None, 'request_core_id': None}
        runner = self.model_runner
        execute, sample = runner.execute_model, runner.sample_tokens
        state = self._dspark_profile_state
        def receipt():
            return {k: v for k, v in state.items() if k != 'profiler'} | {
                'prefill_capture_profile_excluded': True, 'extra_observer_copies': 0,
                'scope': 'intrusive CPU/NPU native DSpark timeline only; not a quote',
                'full_consumer_audit_passed': False, 'performance_receipt_granted': False}
        self._dspark_profile_receipt = receipt
        @functools.wraps(execute)
        def observed_execute(scheduler, *args, **kwargs):
            if state['armed'] and not state['started'] and is_real_decode(scheduler, runner.requests):
                import torch_npu
                req_id = next(iter(scheduler.num_scheduled_tokens))
                state['request_core_id'] = req_id
                output = Path(os.environ['STACK_DSPARK_FORMAL_OUTPUT']) / 'profile'
                profile = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=10, repeat=1),
                    record_shapes=True, with_stack=False, profile_memory=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        export_type=torch_npu.profiler.ExportType.Text, data_simplification=False),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(output), worker_name='dspark_rank'+str(state['rank'])))
                profile.start()
                state['profiler'] = profile
                state['started'] = True
            return execute(scheduler, *args, **kwargs)
        @functools.wraps(sample)
        def observed_sample(*args, **kwargs):
            counter = runner._formal_dspark_counter
            pending_decode = (counter.pending is not None and
                              any(row['phase'] == 'decode' for row in counter.pending.values()))
            output = sample(*args, **kwargs)
            if state['started'] and not state['completed'] and pending_decode:
                state['actual_decode_steps'] += 1
                state['profiler'].step()
                if state['actual_decode_steps'] == 15:
                    state['profiler'].stop()
                    state['profiler'] = None
                    state['completed'] = True
                    path = Path(os.environ['STACK_DSPARK_FORMAL_OUTPUT']) / ('profile_rank'+str(state['rank'])+'.json')
                    path.write_text(json.dumps(receipt(), indent=2)+'\n')
            return output
        runner.execute_model = observed_execute
        runner.sample_tokens = observed_sample
        return result


def arm(worker):
    state = worker._dspark_profile_state
    if state['armed'] or state['started']:
        raise ValueError('Profiler already armed')
    if worker.model_runner._formal_dspark_counter.tag is None:
        raise ValueError('Real counter window must precede profile arming')
    state['armed'] = True
    return worker._dspark_profile_receipt()


def finish(worker):
    state = worker._dspark_profile_state
    if not state['completed'] or state['actual_decode_steps'] != 15:
        raise ValueError('Native profile did not observe five warmup and ten active real decode steps')
    return worker._dspark_profile_receipt()
