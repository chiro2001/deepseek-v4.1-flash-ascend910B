"""Production model worker for A/A control, without operator bank patches."""
import torch
from vllm_ascend.worker.worker import NPUWorker


class FormalControlWorker(NPUWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        config = self.vllm_config.model_config.hf_text_config
        assert config.hidden_size == 5120 and config.num_hidden_layers == 40
        assert config.n_routed_experts == 384 and config.num_experts_per_tok == 6
        assert self.vllm_config.load_config.load_format != 'dummy'
        import route_capture_patch
        route_capture_patch.install()
        return super().load_model(*args, **kwargs)
