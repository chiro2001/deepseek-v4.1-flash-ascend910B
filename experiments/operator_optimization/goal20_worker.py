"""Install next-round hooks after platform/model loading and prior to compile."""
import os

import torch
from operator_worker import OperatorOptWorker
import goal20_patches as patches


class Goal20Worker(OperatorOptWorker):
    def load_model(self, *args, **kwargs):
        if os.getenv('GOAL20_DEBUG_TIMEOUT_MS'):
            timeout=int(os.environ['GOAL20_DEBUG_TIMEOUT_MS'])
            assert timeout>0
            torch.npu.set_op_timeout_ms(timeout)
            print('DIAGNOSTIC_OP_TIMEOUT_MS',timeout,flush=True)
        if os.getenv('GOAL20_SERIAL_DUMMY')=='1':
            from vllm.model_executor.model_loader import weight_utils
            original=weight_utils.initialize_single_dummy_weight
            calls=0
            def initialize(param,*a,**kw):
                nonlocal calls
                result=original(param,*a,**kw)
                if param.device.type=='npu':torch.npu.synchronize()
                calls+=1
                return result
            weight_utils.initialize_single_dummy_weight=initialize
            try:result=super().load_model(*args,**kwargs)
            finally:weight_utils.initialize_single_dummy_weight=original
            print('SERIAL_DUMMY_INITIALIZATION',calls,flush=True)
        else:
            result = super().load_model(*args, **kwargs)
        patches.install(self.model_runner.model)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        print('GOAL20_EFFECTIVE', patches.save_bank(self, patches.ARM), flush=True)
        return result
