"""Private standard NPUWorker with source-guarded observer lifecycle only."""
import json
import os
from pathlib import Path
import types

from vllm_ascend.worker.worker import NPUWorker
import torch
import torch_npu

from .lifecycle import runner_load_with_callbacks
from .session import AuditSession


class Full43AuditWorker(NPUWorker):
    def init_device(self):
        result = super().init_device()
        if os.environ.get('V10_AUDIT_ENABLED','1') != '1':return result
        if not torch_npu.npu.is_initialized():raise ValueError('Standard parent did not return initialized NPU context')
        output = Path(os.environ['V10_AUDIT_OUTPUT'])/f'rank{self.rank}'
        plan = json.loads(Path(__file__).with_name('consumer_plan.json').read_text())
        self._v10_session = AuditSession(torch,torch_npu,self.model_runner,output,plan)
        self._v10_load_patch = runner_load_with_callbacks(self.model_runner,self._v10_session)
        return result

    def load_model(self,*args,**kwargs):
        result = super().load_model(*args,**kwargs)
        if not hasattr(self,'_v10_session'):return result
        session = self._v10_session;session.after_normal_load()
        runner = self.model_runner
        original_execute = runner.execute_model
        def execute(owner,scheduler_output,*actual,**keywords):
            session.begin_execute(scheduler_output)
            return original_execute(scheduler_output,*actual,**keywords)
        runner.execute_model = types.MethodType(execute,runner)
        original_native_sample = runner._sample
        def native_sample(owner,logits,spec_decode_metadata):
            session.actual_spec_decode_round = spec_decode_metadata is not None
            return original_native_sample(logits,spec_decode_metadata)
        runner._sample = types.MethodType(native_sample,runner)
        if hasattr(runner,'sample_tokens'):
            original_sample = runner.sample_tokens
            def sample(owner,*actual,**keywords):
                value = original_sample(*actual,**keywords)
                session.end_sample()
                return value
            runner.sample_tokens = types.MethodType(sample,runner)
        return result
