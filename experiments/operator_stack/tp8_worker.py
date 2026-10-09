"""Full TP8 load, preserving native distributed MoE for incompatible shapes."""
import os

import torch
from vllm_ascend.worker.worker import NPUWorker
import runtime_patches as base
import tp8_patches


class TP8StackWorker(NPUWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        # Platform patches run during init_device and replace the capturer.
        # Install after those patches, before load_model binds callbacks.
        if os.getenv('TINY_PERF_RANDOM_VALIDATION') == '1':
            import route_capture_patch
            route_capture_patch.install()
        base.install_router()
        from vllm.model_executor.model_loader import weight_utils
        original = weight_utils.initialize_single_dummy_weight
        def serial(param, *a, **kw):
            result = original(param, *a, **kw)
            if param.device.type == 'npu': torch.npu.synchronize()
            return result
        weight_utils.initialize_single_dummy_weight = serial
        try: result = super().load_model(*args, **kwargs)
        finally: weight_utils.initialize_single_dummy_weight = original
        if os.getenv('TINY_PERF_RANDOM_VALIDATION') == '1':
            base.randomize_validation_weights(self)
            import activation_patches, math
            activation_patches.randomize_expert_validation(self)
            for name, param in self.model_runner.model.named_parameters():
                if param.ndim >= 2 and name.endswith('weight') and any(t in name for t in ['.wq_a.', '.wq_b.', '.wkv.', '.wo_a.', '.wo_b.', '.wk.']):
                    param.copy_(torch.randn_like(param) / math.sqrt(param.shape[-1]))
            for module in self.model_runner.model.modules():
                if getattr(module, 'precast_fp32_weight', False) and hasattr(module, 'weight_fp32'):
                    module.weight_fp32.copy_(module.weight.float())
        base.install_hc(self.model_runner.model)
        tp8_patches.install(self.model_runner.model)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        print('TP8_STACK_EFFECTIVE', tp8_patches.save(self, os.getenv('STACK_TP8_ARM', 'tp8base')), flush=True)
        return result
