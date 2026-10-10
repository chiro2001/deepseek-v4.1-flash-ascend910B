"""Formal checkpoint worker: no dummy initialization or weight randomization."""
import os
import json

import torch
from vllm_ascend.worker.worker import NPUWorker
import runtime_patches as base
import tp8_patches


class RealTP8StackWorker(NPUWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        assert os.getenv('STACK_REAL_WEIGHTS') == '1'
        assert os.getenv('TINY_PERF_RANDOM_VALIDATION') != '1'
        config = self.vllm_config.model_config.hf_text_config
        assert config.hidden_size == 5120 and config.num_hidden_layers == 40
        assert config.n_routed_experts == 384 and config.num_experts_per_tok == 6
        assert self.vllm_config.load_config.load_format != 'dummy'
        print('REAL_WEIGHT_LOADER', {'load_format': self.vllm_config.load_config.load_format,
                                    'quantization': self.vllm_config.model_config.quantization}, flush=True)
        if os.getenv('STACK_REAL_AUDIT') == '1':
            import route_capture_patch
            route_capture_patch.install()
        if os.getenv('STACK_FP32_DECODE_REDUCTION') == '1':
            import decode_reduction_probe
            print('REAL_TP8_DECODE_REDUCTION',decode_reduction_probe.install(self),flush=True)
        result = super().load_model(*args, **kwargs)
        # Router/GMM candidates remain native: their TP1 contract is 8/top2.
        base.install_hc(self.model_runner.model)
        tp8_patches.install(self.model_runner.model)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        if os.getenv('STACK_METADATA_MANY_SLOTS_ENABLED')=='1':
            from tp8_slot_batches import REGISTRY
            # V1's initialized KVCacheConfig is authoritative. The legacy
            # cache_config.num_gpu_blocks may be unset on this vendor build.
            REGISTRY.bind_pool(self.model_runner.kv_cache_config.num_blocks)
            print('REAL_SLOT_POOL_BOUND',REGISTRY.model_pool_blocks,flush=True)
        result = super().compile_or_warm_up_model(*args, **kwargs)
        print('REAL_TP8_EFFECTIVE', tp8_patches.save(self, os.getenv('STACK_TP8_ARM', 'tp8base')), flush=True)
        return result


class ServingRealTP8StackWorker(RealTP8StackWorker):
    """API worker for an arm selected by the separate formal-model audit."""

    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        assert os.getenv('STACK_SERVE_ARM') in ('tp8base', 'tp8core', 'tp8act', 'tp8stack', 'tp8meta', 'tp8metastack')
        assert os.getenv('STACK_REAL_AUDIT', '0') == '0', 'Serve without route/clone audit'
        assert self.vllm_config.speculative_config is None, 'This adapter is validated for A=1'
        # Establish the same native bank used by the comparison, then compile
        # the selected candidate from those captured layouts outside requests.
        os.environ['STACK_REAL_WEIGHTS'] = '1'
        os.environ['STACK_REAL_AUDIT'] = '0'
        os.environ['STACK_TP8_ARM'] = 'tp8base'
        return super().load_model(*args, **kwargs)

    def compile_or_warm_up_model(self, *args, **kwargs):
        config = self.vllm_config
        assert config.parallel_config.tensor_parallel_size == 8
        assert config.scheduler_config.max_num_seqs == 1
        assert list(config.compilation_config.cudagraph_capture_sizes) == [1]
        result = super().compile_or_warm_up_model(*args, **kwargs)
        arm = os.environ['STACK_SERVE_ARM']
        receipt = tp8_patches.save(self, arm) if arm == 'tp8base' else tp8_patches.create(self, arm)
        tp8_patches.switch(self, arm)
        print('REAL_TP8_SERVE_ARM', json.dumps(receipt), flush=True)
        return result
