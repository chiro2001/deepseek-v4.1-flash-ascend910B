"""TP8 graph banks with honest coverage and explicit TP1 MoE fallback."""
import os

import torch
import runtime_patches as base
import goal20_patches as goal
import activation_patches as activation
import indexer_patches as indexer
import overlap_patches as overlap
import tp8_activation as generic_activation
import formal_w4a8_prefix as prefix

BANKS = {}


def switch_mode(name):
    assert name in ['tp8base', 'tp8core', 'tp8act', 'tp8stack', 'tp8meta', 'tp8metastack', 'tp8prefix', 'tp8prefixroute', 'tp8prefixup', 'tp8prefixuproute', 'tp8hostmeta']
    prefix.set_arm(name)
    goal.ARM = name
    base.ARM = 'native' if name == 'tp8base' or os.getenv('STACK_REAL_WEIGHTS') == '1' else 'both'
    activation.ARM = 'baseline' if name == 'tp8base' else 'overlap'
    indexer.set_arm('fused' if name in ('tp8stack','tp8metastack','tp8prefix','tp8prefixroute','tp8prefixup','tp8prefixuproute','tp8hostmeta') else 'baseline')
    overlap.set_enabled(name != 'tp8base')


@torch.inference_mode()
def install(model):
    goal.CONFIGS['tp8base'] = set()
    for name in ['tp8core', 'tp8act', 'tp8stack', 'tp8meta', 'tp8metastack', 'tp8prefix', 'tp8prefixroute', 'tp8prefixup', 'tp8prefixuproute', 'tp8hostmeta']:
        goal.CONFIGS[name] = {'hcstatic', 'hcpost', 'metadata_all', 'blockmap'}
        if os.getenv('STACK_REAL_WEIGHTS') == '1':
            # Formal HC static failed the original numerical gate. Keep native
            # HC until a corrected candidate passes independent and model audit.
            goal.CONFIGS[name] -= {'hcstatic', 'hcpost'}
    for name in ['tp8act','tp8stack','tp8metastack','tp8prefix','tp8prefixroute','tp8prefixup','tp8prefixuproute','tp8hostmeta']:
        goal.CONFIGS[name].add('activation_generic')
    goal.CONFIGS['tp8hostmeta'].add('metadata_spec')
    goal.CONFIGS['tp8meta'].add('metadata_manyslots')
    for name in ('tp8metastack','tp8prefix','tp8prefixroute','tp8prefixup','tp8prefixuproute','tp8hostmeta'):
        goal.CONFIGS[name].add('metadata_manyslots')
    goal.install(model)
    activation.install(model)
    generic_activation.install(model)
    overlap.initialize(model)
    import metadata_patches, blockmap_patches
    # These installers inherit only the four compatible core features.
    goal.CONFIGS['gmmact'] = set(goal.CONFIGS['tp8core'])
    from tp8_slot_batches import REGISTRY
    slot_prepare=REGISTRY.prepare if os.getenv('STACK_METADATA_MANY_SLOTS_ENABLED')=='1' else None
    metadata_patches.install(model, slot_prepare=slot_prepare); blockmap_patches.install(model)
    indexer.install(model)
    prefix.install()
    switch_mode(os.getenv('STACK_TP8_ARM', 'tp8base'))


def save(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    from vllm.distributed import get_tensor_model_parallel_rank
    entries = [(wrapper, wrapper.concrete_aclgraph_entries, wrapper.graph_pool)
               for wrapper in ag._acl_graph_wrappers]
    assert sum(len(value) for _, value, _ in entries) == 1
    role = 'fused' if name in ('tp8stack','tp8metastack','tp8prefix','tp8prefixroute','tp8prefixup','tp8prefixuproute','tp8hostmeta') else 'baseline'
    refs = activation.REFS.get(activation.ARM, {'routed': [], 'shared': []})
    hc_refs = base.HC_REFS.get(base.ARM, [])
    router_refs = base.ROUTER_REFS.get(base.ARM, [])
    expected_router = 0 if os.getenv('STACK_REAL_WEIGHTS') == '1' else 40
    assert len(hc_refs) == 80 and len(router_refs) == expected_router, (name, len(hc_refs), len(router_refs))
    index_refs = indexer.REFS[role]
    generic_refs=generic_activation.REFS.get(name,{'routed':[],'shared':[]})
    BANKS[name] = (entries, ag._graph_params, hc_refs, router_refs, refs, index_refs,
                   goal.REFS.get(name, {}),generic_refs)
    from tp8_slot_batches import REGISTRY
    return {'rank': get_tensor_model_parallel_rank(), 'arm': name, 'hc_calls': len(hc_refs),
            'metadata_slot_batches': REGISTRY.status(),
            'w4a8_prefix': prefix.status(name),
            'metadata_build': __import__('metadata_patches').stats(worker),
            'router_calls': len(router_refs), 'indexer': indexer.coverage(role),
            'activation_eligible_calls': {kind: len(rows) for kind, rows in refs.items()},
            'activation_selected_calls': {kind: sum(row[3] for row in rows) for kind, rows in refs.items()},
            'goal_selected_calls': {kind: sum(row[2] for row in rows) for kind, rows in goal.REFS.get(name, {}).items()},
            'generic_activation':{kind:{'calls':len(rows),'selected':sum(r[3] for r in rows),
                                        'shapes':sorted({tuple(r[0].shape) for r in rows})} for kind,rows in generic_refs.items()},
            'graph_pools':[repr(pool) for _, value, pool in entries if value],
            'isolated_candidate_pools_requested':os.getenv('STACK_TP8_ISOLATED_POOLS') == '1',
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
    # Precompile compatible HC kernels outside graph capture, on actual
    # captured layouts. Static weights have already been cached at install.
    if os.getenv('STACK_REAL_WEIGHTS') != '1':
        for x, fn, scale, bias, mix, _ in BANKS['tp8base'][2]:
            goal.compute_hc(x, fn, scale, bias, mix)
    from short_ops import hc_post
    if goal.enabled('hcpost'):
        for values, _, _ in BANKS['tp8base'][6].get('hcpost', []):
            hc_post(*values, block=512, fma=False)
    torch.npu.synchronize()
    indexer.reset(indexer.ARM)
    prefix.reset(name)
    base.HC_REFS[base.ARM] = []; base.ROUTER_REFS[base.ARM] = []
    activation.REFS[activation.ARM] = {'routed': [], 'shared': []}; goal.REFS[name] = {}
    generic_activation.REFS[name]={'routed':[],'shared':[]}
    pool = torch.npu.graph_pool_handle() if os.getenv('STACK_TP8_ISOLATED_POOLS') == '1' else None
    for wrapper in ag._acl_graph_wrappers:
        wrapper.concrete_aclgraph_entries = {}
        if pool is not None: wrapper.graph_pool = pool
    ag._graph_params = None; ag.set_graph_params([1])
    worker.model_runner.capture_model()
    return save(worker, name)


def switch(worker, name):
    from vllm_ascend.compilation import acl_graph as ag
    torch.npu.synchronize(); switch_mode(name)
    entries, params, hc, router, act, idx, refs, generic_refs = BANKS[name]
    for wrapper, value, pool in entries:
        wrapper.concrete_aclgraph_entries = value
        wrapper.graph_pool = pool
    ag._graph_params = params
    base.HC_REFS[base.ARM] = hc; base.ROUTER_REFS[base.ARM] = router
    activation.REFS[activation.ARM] = act; indexer.REFS[indexer.ARM] = idx; goal.REFS[name] = refs
    generic_activation.REFS[name]=generic_refs
    return {'arm': name}


@torch.inference_mode()
def audit(worker, name):
    # Run the native references with the original capture's tensors restored.
    switch(worker, name)
    result = {'indexer': indexer.audit(worker, indexer.ARM)}
    if name in ('tp8meta','tp8metastack','tp8prefix','tp8prefixroute','tp8prefixup','tp8prefixuproute','tp8hostmeta'):
        from tp8_slot_batches import REGISTRY
        result['metadata_slots'] = REGISTRY.audit()
        assert REGISTRY.counts['fused_launches']>0, 'Slot batching candidate has no actual coverage'
    if os.getenv('STACK_REAL_WEIGHTS') == '1':
        result['hc'] = {'candidate_calls': 0, 'path': 'native; formal static candidate failed original numerical gate'}
        result['router'] = {'candidate_calls': 0, 'path': 'native 384-expert top6; TP1 specializations disabled'}
    else:
        result['hc'] = base.audit_hc(worker, base.ARM)
        result['router'] = base.audit_router(worker, base.ARM)
    refs = activation.REFS[activation.ARM]
    if refs['routed'] and refs['shared']:
        result['activation'] = activation.audit(worker, activation.ARM)
    result['coverage'] = save(worker, name)
    result['generic_activation']=generic_activation.audit(name)
    result['w4a8_prefix']=prefix.audit(name)
    if name=='tp8hostmeta':
        result['metadata_build']=__import__('metadata_patches').stats(worker)
        assert result['metadata_build']['spec_static_build_calls'].get(name,0)>0, 'Metadata spec candidate has no actual coverage'
    return result
