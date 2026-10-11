"""Native worker with only the isolated BF16 QKV adapter installed."""
import os

import torch
from vllm_ascend.worker.worker import NPUWorker
import qkv_patch


class QKVWorker(NPUWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        assert os.getenv('QKV_LANE_PHYSICAL_CHIP') == '2'
        # Device isolation has logical0; the already-working privileged tiny
        # container retains host IDs and must select physical2 explicitly.
        namespace = os.getenv('QKV_LANE_RUNTIME_NAMESPACE', 'isolated')
        assert (namespace, os.getenv('ASCEND_RT_VISIBLE_DEVICES')) in (
            ('isolated', '0'), ('host', '2'))
        assert os.getenv('HCCL_DETERMINISTIC') == 'strict'
        cfg = self.vllm_config
        text = cfg.model_config.hf_text_config
        assert (text.num_hidden_layers, text.hidden_size, text.n_routed_experts,
                text.num_experts_per_tok, text.q_lora_rank) == (40, 5120, 8, 2, 512)
        assert cfg.parallel_config.tensor_parallel_size == 1
        assert cfg.load_config.load_format == 'dummy'
        # Serialize the known tiny dummy-loader initialization; do not modify
        # the process-global implementation after this load completes.
        from vllm.model_executor.model_loader import weight_utils
        original = weight_utils.initialize_single_dummy_weight
        calls = 0

        def initialize(param, *a, **kw):
            nonlocal calls
            result = original(param, *a, **kw)
            if param.device.type == 'npu':
                torch.npu.synchronize()
            calls += 1
            return result

        weight_utils.initialize_single_dummy_weight = initialize
        try:
            result = super().load_model(*args, **kwargs)
        finally:
            weight_utils.initialize_single_dummy_weight = original
        qkv_patch.install(self.model_runner.model)
        print('QKV_LANE_LOADED', {'dummy_initialization_calls': calls, **qkv_patch.status()}, flush=True)
        return result
