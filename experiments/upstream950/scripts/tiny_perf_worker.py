"""Install model-specific patches after the plugin loads its standard model."""
import os
import json
from vllm_ascend.worker.worker import NPUWorker
import runtime_patches as patches


class TinyPerfWorker(NPUWorker):
    def load_model(self,*args,**kwargs):
        patches.install_router()
        result=super().load_model(*args,**kwargs)
        if os.getenv('TINY_PERF_RANDOM_VALIDATION')=='1':
            print('RANDOMIZED_BEFORE_COMPILE',patches.randomize_validation_weights(self),flush=True)
        if os.getenv('TINY_PERF_HC_ENABLE')=='1' or os.getenv('TINY_PERF_ARM') in ['hc','both']:
            patches.install_hc(self.model_runner.model)
        patches.ARM=os.getenv('TINY_PERF_ARM','native')
        return result

    def compile_or_warm_up_model(self,*args,**kwargs):
        result=super().compile_or_warm_up_model(*args,**kwargs)
        arm=patches.ARM
        state={'arm':arm,'router_calls':len(patches.ROUTER_REFS.get(arm,[])),
               'hc_calls':len(patches.HC_REFS.get(arm,[])),
               'visible_devices':os.getenv('ASCEND_RT_VISIBLE_DEVICES')}
        if arm in ['router','both']:assert state['router_calls']==40,state
        if arm in ['hc','both']:assert state['hc_calls']==80,state
        print('TINY_PERF_EFFECTIVE',json.dumps(state),flush=True)
        return result
