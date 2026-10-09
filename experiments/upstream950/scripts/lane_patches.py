"""Generic same-process graph banks for independently verified candidates."""
import importlib
import math
import os

import torch
import runtime_patches as baseline

CANDIDATE_MODULES = {'sparse_gmm': 'sparse_gmm_patches', 'smla': 'smla_patches',
                     'indexer': 'indexer_patches','epilogue':'epilogue_patches',
                     'projection':'projection_patches'}
BANKS = {}


def candidate():
    return importlib.import_module(CANDIDATE_MODULES[os.environ['UP950_CANDIDATE']])


def install(model):
    candidate().install(model)


@torch.inference_mode()
def randomize_mlp(worker):
    torch.manual_seed(20261010)
    counts = {'routed': 0, 'shared': 0, 'attention': 0}
    for name, param in baseline.actual(worker).model_runner.model.named_parameters():
        if param.dtype not in [torch.bfloat16,torch.float32]:
            continue
        routed = '.mlp.' in name and name.endswith(('w13_weight', 'w2_weight'))
        shared = '.shared_experts.' in name and name.endswith('weight') and param.ndim == 2
        attention = (os.environ['UP950_CANDIDATE'] in ['smla','indexer','epilogue','projection'] and param.ndim >= 2
                     and name.endswith('weight') and
                     any(tag in name for tag in ['.wq_a.','.wq_b.','.wkv.','.wo_a.','.wo_b.','.wk.']))
        if routed or shared or attention:
            param.copy_(torch.randn_like(param) / math.sqrt(param.shape[-1]))
            counts['routed' if routed else 'shared' if shared else 'attention'] += 1
    assert counts['routed'] == counts['shared'] == 80, counts
    if os.environ['UP950_CANDIDATE'] in ['smla','indexer','epilogue','projection']:
        assert counts['attention'] >= 200,counts
    for module in baseline.actual(worker).model_runner.model.modules():
        if getattr(module,'precast_fp32_weight',False) and hasattr(module,'weight_fp32'):
            module.weight_fp32.copy_(module.weight.float())
    baseline.ND_WEIGHTS.clear()
    torch.npu.synchronize()
    return counts


def save_bank(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    wrappers = list(ag._acl_graph_wrappers)
    entries = [(wrapper, wrapper.concrete_aclgraph_entries) for wrapper in wrappers]
    assert sum(len(value) for _, value in entries) == 1
    coverage = candidate().coverage(name)
    assert len(baseline.ROUTER_REFS['both']) == 40
    assert len(baseline.HC_REFS['both']) == 80
    BANKS[name] = (entries, ag._graph_params,
                   baseline.ROUTER_REFS['both'], baseline.HC_REFS['both'])
    return {'arm': name, **coverage, 'hc_calls': 80, 'router_calls': 40}


def create_bank(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize()
    module = candidate()
    module.prepare(name, next(iter(BANKS)))
    torch.npu.synchronize()
    module.set_arm(name)
    module.reset(name)
    baseline.ROUTER_REFS['both'] = []
    baseline.HC_REFS['both'] = []
    for wrapper in ag._acl_graph_wrappers:
        wrapper.concrete_aclgraph_entries = {}
    ag._graph_params = None
    ag.set_graph_params([1])
    baseline.actual(worker).model_runner.capture_model()
    return save_bank(worker, name)


def switch_bank(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize()
    entries, params, router_refs, hc_refs = BANKS[name]
    for wrapper, value in entries:
        wrapper.concrete_aclgraph_entries = value
    ag._graph_params = params
    baseline.ROUTER_REFS['both'] = router_refs
    baseline.HC_REFS['both'] = hc_refs
    candidate().set_arm(name)
    return {'arm': name}


@torch.inference_mode()
def audit(worker, name):
    return candidate().audit(worker, name)
