"""TP8 graph banks with honest coverage and explicit TP1 MoE fallback."""
import os

import torch
import runtime_patches as base
import goal20_patches as goal
import activation_patches as activation
import indexer_patches as indexer
import overlap_patches as overlap
import tp8_activation as generic_activation

BANKS = {}


def switch_mode(name):
    assert name in ['tp8base', 'tp8core', 'tp8act', 'tp8stack']
    goal.ARM = name
    base.ARM = 'native' if name == 'tp8base' else 'both'
    activation.ARM = 'baseline' if name == 'tp8base' else 'overlap'
    indexer.set_arm('fused' if name == 'tp8stack' else 'baseline')
    overlap.set_enabled(name != 'tp8base')


@torch.inference_mode()
def install(model):
    goal.CONFIGS['tp8base'] = set()
    for name in ['tp8core', 'tp8act', 'tp8stack']:
        goal.CONFIGS[name] = {'hcstatic', 'hcpost', 'metadata_all', 'blockmap'}
    for name in ['tp8act','tp8stack']:
        goal.CONFIGS[name].add('activation_generic')
    goal.install(model)
    activation.install(model)
    generic_activation.install(model)
    overlap.initialize(model)
    import metadata_patches, blockmap_patches
    # These installers inherit only the four compatible core features.
    goal.CONFIGS['gmmact'] = set(goal.CONFIGS['tp8core'])
    metadata_patches.install(model); blockmap_patches.install(model)
    indexer.install(model)
    switch_mode(os.getenv('STACK_TP8_ARM', 'tp8base'))


def save(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    from vllm.distributed import get_tensor_model_parallel_rank
    entries = [(wrapper, wrapper.concrete_aclgraph_entries) for wrapper in ag._acl_graph_wrappers]
    assert sum(len(value) for _, value in entries) == 1
    role = 'fused' if name == 'tp8stack' else 'baseline'
    refs = activation.REFS.get(activation.ARM, {'routed': [], 'shared': []})
    hc_refs = base.HC_REFS.get(base.ARM, [])
    router_refs = base.ROUTER_REFS.get(base.ARM, [])
    assert len(hc_refs) == 80 and len(router_refs) == 40, (name, len(hc_refs), len(router_refs))
    index_refs = indexer.REFS[role]
    generic_refs=generic_activation.REFS.get(name,{'routed':[],'shared':[]})
    BANKS[name] = (entries, ag._graph_params, hc_refs, router_refs, refs, index_refs,
                   goal.REFS.get(name, {}),generic_refs)
    return {'rank': get_tensor_model_parallel_rank(), 'arm': name, 'hc_calls': len(hc_refs),
            'router_calls': len(router_refs), 'indexer': indexer.coverage(role),
            'activation_eligible_calls': {kind: len(rows) for kind, rows in refs.items()},
            'activation_selected_calls': {kind: sum(row[3] for row in rows) for kind, rows in refs.items()},
            'goal_selected_calls': {kind: sum(row[2] for row in rows) for kind, rows in goal.REFS.get(name, {}).items()},
            'generic_activation':{kind:{'calls':len(rows),'selected':sum(r[3] for r in rows),
                                        'shapes':sorted({tuple(r[0].shape) for r in rows})} for kind,rows in generic_refs.items()},
            'tp1_route_and_selected_gmm_enabled': False}


def create(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    worker = base.actual(worker); torch.npu.synchronize()
    indexer.prepare('fused', 'baseline')
    # Compile each real row/width contract outside the graph capture region.
    from clamped_swiglu import clamped_swiglu
    native_generic=BANKS['tp8base'][-1]
    for kind,rows in native_generic.items():
        for x,params,_,_ in rows:
            if not x.numel():continue
            if kind=='routed':clamped_swiglu(x.clone(),params[0],mutate_input=True)
            else:clamped_swiglu(x,params[0],alpha=params[1],beta=params[2],staged_rounding=True)
    torch.npu.synchronize()
    switch_mode(name)
    indexer.reset(indexer.ARM)
    base.HC_REFS[base.ARM] = []; base.ROUTER_REFS[base.ARM] = []
    activation.REFS[activation.ARM] = {'routed': [], 'shared': []}; goal.REFS[name] = {}
    generic_activation.REFS[name]={'routed':[],'shared':[]}
    for wrapper in ag._acl_graph_wrappers: wrapper.concrete_aclgraph_entries = {}
    ag._graph_params = None; ag.set_graph_params([1])
    worker.model_runner.capture_model()
    return save(worker, name)


def switch(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize(); switch_mode(name)
    entries, params, hc, router, act, idx, refs, generic_refs = BANKS[name]
    for wrapper, value in entries: wrapper.concrete_aclgraph_entries = value
    ag._graph_params = params
    base.HC_REFS[base.ARM] = hc; base.ROUTER_REFS[base.ARM] = router
    activation.REFS[activation.ARM] = act; indexer.REFS[indexer.ARM] = idx; goal.REFS[name] = refs
    generic_activation.REFS[name]=generic_refs
    return {'arm': name}


@torch.inference_mode()
def audit(worker, name):
    # Run the native references with the original capture's tensors restored.
    switch(worker, name)
    result = {'indexer': indexer.audit(worker, indexer.ARM),
              'hc': base.audit_hc(worker, base.ARM), 'router': base.audit_router(worker, base.ARM)}
    refs = activation.REFS[activation.ARM]
    if refs['routed'] and refs['shared']:
        result['activation'] = activation.audit(worker, activation.ARM)
    result['coverage'] = save(worker, name)
    result['generic_activation']=generic_activation.audit(name)
    return result
