"""New-job native formal worker; no existing runner/ARM/source modifications."""
import os
import json
from pathlib import Path
import torch
from vllm_ascend.worker.worker import NPUWorker


class FormalDSparkWorker(NPUWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        cfg=self.vllm_config.model_config.hf_text_config
        spec=self.vllm_config.speculative_config
        assert cfg.hidden_size==5120 and cfg.num_hidden_layers==40
        assert cfg.n_routed_experts==384 and cfg.num_experts_per_tok==6
        assert list(cfg.engram_layer_ids)==[1,14]
        assert cfg.num_nextn_predict_layers==3 and list(cfg.dspark_target_layer_ids)==[37,38,39]
        assert spec is not None and spec.method=='dspark' and spec.num_speculative_tokens>=cfg.dspark_block_size
        assert self.vllm_config.load_config.load_format!='dummy'
        assert not self.vllm_config.scheduler_config.async_scheduling
        assert os.environ['HCCL_DETERMINISTIC']=='strict'
        assert os.environ['STACK_PHYSICAL_CHIPS']=='8,9,10,11,12,13,14,15'
        assert os.getenv('STACK_TP8_ARM')=='tp8base'
        assert os.getenv('STACK_FP32_DECODE_REDUCTION')=='0'
        # Run after the standard Ascend platform/worker import chain has
        # installed its supported factory, before any target weight loading.
        from .sdk_gate import inspect_native_factory
        from vllm.distributed import get_tensor_model_parallel_rank
        rank=get_tensor_model_parallel_rank()
        sdk=inspect_native_factory()
        sdk['scope']='actual new-worker SDK after standard platform initialization'
        sdk['rank']=rank
        (Path(os.environ['STACK_DSPARK_FORMAL_OUTPUT'])/f'sdk_rank{rank}.json').write_text(json.dumps(sdk,indent=2)+'\n')
        if not sdk['compatible']:
            raise ValueError('Native SDK factory does not accept required bias_vl; no kwargs filtering or vision-bias bypass is allowed')
        audit=os.environ['STACK_REAL_AUDIT']=='1'
        if audit:
            import route_capture_patch
            route_capture_patch.install()
        result=super().load_model(*args,**kwargs)
        runner=self.model_runner
        from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
        target=runner.model
        backbones=[module for module in target.modules() if isinstance(module,DeepseekV41Model)]
        if len(backbones)!=1:
            raise ValueError('Expected one actual native full40 backbone before graph wrapping')
        # Warm-up replaces these top-level objects with ACLGraphWrapper. Keep
        # the loaded native modules now; their cache buffers are populated later.
        runner._formal_dspark_native_target=target
        runner._formal_dspark_native_backbone=backbones[0]
        runner._formal_dspark_native_draft_model=runner.drafter.model
        runner._formal_dspark_native_draft_backbone=runner.drafter.model.model
        import formal_dspark_counters
        receipt=formal_dspark_counters.install_runner(runner,
            rank,audit=audit)
        print('FORMAL_DSPARK_COUNTERS_INSTALLED',receipt,flush=True)
        return result
