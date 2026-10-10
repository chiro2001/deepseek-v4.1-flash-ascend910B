"""Formal checkpoint worker: no dummy initialization or weight randomization."""
import os

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
        result = super().load_model(*args, **kwargs)
        # Router/GMM candidates remain native: their TP1 contract is 8/top2.
        base.install_hc(self.model_runner.model)
        tp8_patches.install(self.model_runner.model)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        print('REAL_TP8_EFFECTIVE', tp8_patches.save(self, os.getenv('STACK_TP8_ARM', 'tp8base')), flush=True)
        return result
