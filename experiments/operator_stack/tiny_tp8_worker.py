"""Native diagnostic worker; formal deployment continues to require 40 layers."""
import hashlib
import json
import os
from pathlib import Path

import torch
from vllm_ascend.worker.worker import NPUWorker


class TinyTP8Worker(NPUWorker):
    @torch.no_grad()
    def load_model(self, *args, **kwargs):
        assert os.getenv('STACK_TINY_PROFILE') == '1'
        assert os.getenv('HCCL_DETERMINISTIC') == 'strict'
        cfg = self.vllm_config
        text = cfg.model_config.hf_text_config
        model = Path(cfg.model_config.model)
        manifest = json.loads((model / 'tiny_manifest.json').read_text())
        assert manifest['diagnostic_only'] and manifest['copied_tensors_byte_exact']
        assert hashlib.sha256((model / 'config.json').read_bytes()).hexdigest() == manifest['config_sha256']
        assert (text.num_hidden_layers, text.hidden_size, text.n_routed_experts,
                text.num_experts_per_tok) == (8, 5120, 384, 6)
        assert cfg.parallel_config.tensor_parallel_size == 8
        assert cfg.load_config.load_format == 'auto'
        assert cfg.speculative_config is None
        from vllm_ascend.core import deepseek_v41 as cache
        if not hasattr(cache, 'TINY_CACHE_PLAN_SOURCE_SHA256'):
            from tiny_cache_plan import install
            install(cache)
        # Engram hashes seed their multipliers from the original layer ID.
        # Remapping layer14 to layer4 must not change which real rows are read.
        from vllm_ascend.models.deepseek_v41 import engram_hash
        original = engram_hash.compute_hash_multipliers
        reverse = {value: int(key) for key, value in manifest['layer_map'].items()}

        def source_multipliers(layer_ids, max_ngram_size, tokenizer_vocab_size):
            assert tuple(layer_ids) == (1, 4), layer_ids
            return original(tuple(reverse[i] for i in layer_ids), max_ngram_size, tokenizer_vocab_size)

        engram_hash.compute_hash_multipliers = source_multipliers
        if os.getenv('STACK_REAL_AUDIT') == '1':
            import route_capture_patch
            route_capture_patch.install()
        return super().load_model(*args, **kwargs)


def topology_receipt(worker):
    worker = getattr(worker, 'worker', worker)
    from vllm.distributed import get_tensor_model_parallel_rank
    cfg = worker.vllm_config
    text = cfg.model_config.hf_text_config
    methods, schemes = {}, {}
    for name, module in worker.model_runner.model.named_modules():
        method = getattr(module, 'quant_method', None)
        if method is not None:
            kind = type(method).__name__
            methods.setdefault(kind, []).append(name)
            scheme = getattr(method, 'quant_method', method)
            schemes.setdefault(type(scheme).__name__, []).append(name)
    assert len(schemes.get('AscendW4A8DynamicFusedMoEMethod', [])) == 8, schemes
    groups = worker.model_runner.kv_cache_config.kv_cache_groups
    from vllm_ascend.core import deepseek_v41 as cache
    assert getattr(cache, 'TINY_CACHE_PLAN_SOURCE_SHA256', None)
    return {'rank': get_tensor_model_parallel_rank(), 'layers': text.num_hidden_layers,
            'hidden_size': text.hidden_size, 'global_experts': text.n_routed_experts,
            'topk': text.num_experts_per_tok, 'tp': cfg.parallel_config.tensor_parallel_size,
            'local_experts': text.n_routed_experts // cfg.parallel_config.tensor_parallel_size,
            'compression_ratios': list(text.compress_ratios),
            'engram_layer_ids': list(text.engram_layer_ids),
            'engram_hash_seed_source_layers': [1, 14],
            'cache_groups': len(groups),
            'cache_specs': [type(group.kv_cache_spec).__name__ for group in groups],
            'cache_planner_original_sha256': cache.TINY_CACHE_PLAN_SOURCE_SHA256,
            'quant_methods': {k: {'count': len(v), 'examples': v[:2]} for k, v in methods.items()},
            'quant_schemes': {k: {'count': len(v), 'examples': v[:2]} for k, v in schemes.items()},
            'async_scheduling': bool(cfg.scheduler_config.async_scheduling)}
