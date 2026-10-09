"""Overlay the verified Indexer post kernel on the current complete TP1 line."""
import math
import os

import torch
import goal20_patches as goal
import indexer_patches as indexer

INDEX_REFS = {}


@torch.inference_mode()
def install(model):
    if os.getenv('TINY_PERF_RANDOM_VALIDATION') == '1':
        count = 0
        for name, param in model.named_parameters():
            if (param.ndim >= 2 and name.endswith('weight') and
                    any(tag in name for tag in ['.wq_a.', '.wq_b.', '.wkv.', '.wo_a.', '.wo_b.', '.wk.'])):
                param.copy_(torch.randn_like(param) / math.sqrt(param.shape[-1])); count += 1
        assert count >= 200, count
        for module in model.modules():
            if getattr(module, 'precast_fp32_weight', False) and hasattr(module, 'weight_fp32'):
                module.weight_fp32.copy_(module.weight.float())
        print('STACK_RANDOM_ATTENTION', count, flush=True)
    indexer.install(model)
    goal.CONFIGS['stacked'] = set(goal.CONFIGS['mdfull']) | {'metadata_all', 'blockmap'}
    old_save, old_create, old_switch, old_warm, old_audit = (
        goal.save_bank, goal.create_bank, goal.switch_bank, goal.warm, goal.audit)

    def save(worker, name):
        result = old_save(worker, name)
        role = 'fused' if name == 'stacked' else 'baseline'
        INDEX_REFS[name] = indexer.REFS[role]
        result['indexer'] = indexer.coverage(role)
        return result

    def create(worker, name):
        role = 'fused' if name == 'stacked' else 'baseline'
        indexer.set_arm(role); indexer.reset(role)
        return old_create(worker, name)

    def switch(worker, name):
        role = 'fused' if name == 'stacked' else 'baseline'
        indexer.set_arm(role); indexer.REFS[role] = INDEX_REFS[name]
        return old_switch(worker, name)

    def warm(worker):
        result = old_warm(worker)
        indexer.prepare('fused', 'baseline'); torch.npu.synchronize()
        return result

    @torch.inference_mode()
    def audit(worker, name):
        result = old_audit(worker, name)
        role = 'fused' if name == 'stacked' else 'baseline'
        indexer.REFS[role] = INDEX_REFS[name]
        result['indexer'] = indexer.audit(worker, role)
        return result

    goal.save_bank, goal.create_bank, goal.switch_bank, goal.warm, goal.audit = save, create, switch, warm, audit
    print('STACK_INSTALL', {'arms': ['mdfull', 'stacked'], 'indexer_sources': 4}, flush=True)
