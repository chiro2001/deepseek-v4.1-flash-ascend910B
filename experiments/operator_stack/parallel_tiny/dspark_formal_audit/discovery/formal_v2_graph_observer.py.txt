"""Observe actual stock draft graph submissions in a NEW private job only.

No SDK file edits, device copies, graph nodes, or capture-time observations.
Successful replay submission is distinct from device consumer/value coverage.
"""
import hashlib
import inspect
from pathlib import Path

STOCK_PROPOSER_SHA256='d5003b772a1982f032f59a65d9ad4970d598311898c487a50791ee2337bdaae4'
ACL_WRAPPER_SHA256='c8ab2fe6aaae034b03e9e34b9c07fd445046564acd5d45e2c24b5ea5232c5a74'
NPU_GRAPHS_SHA256='eb11cac181087766e8489cf3a1da0c0fa150933b7e74454dba4a47db6114164e'


def file_receipt(fn,expected):
    fn=inspect.unwrap(fn)
    path=Path(inspect.getfile(fn))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=expected:raise ValueError(f'Stock graph observation source mismatch: {path} {sha}')
    return {'path':str(path),'sha256':sha,'function':fn.__qualname__,
            'function_sha256':hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()}


def qualifies(row):
    return (row.get('runtime_mode')=='FULL' and row.get('wrapper_runtime_mode')=='FULL'
            and row.get('is_draft_model') is True and row.get('capturing') is False
            and row.get('existing_graph_entry') is True and row.get('replay_returned') is True
            and row.get('input_addresses_match') is True)


class ActualDraftGraphObserver:
    def __init__(self,counter):
        self.counter=counter
        self.scope_depth=0
        self.events=[]
        self.expected=None
        self.sources={}

    def begin_proposal(self):
        if self.scope_depth:raise RuntimeError('Nested actual DSpark proposal is outside this receipt')
        self.events=[]
        self.scope_depth=1

    def finish_proposal(self):
        self.scope_depth=0
        self.expected=None
        return {'actual_graph_submission':any(qualifies(row) for row in self.events),
                'events':list(self.events),'sources':self.sources,
                'device_consumer_proof':False,'scope':'actual existing NPUGraph replay submission; no layer/cache device coverage'}

    def cancel_proposal(self):
        self.scope_depth=0
        self.expected=None

    def active(self):
        return self.scope_depth==1 and self.counter.tag is not None


def install(counter,runner,draft_original):
    # Runtime imports occur only after LLM standard platform initialization and
    # native weight loading in the new snapshot's worker.
    import torch
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation import acl_graph as ag
    observer=ActualDraftGraphObserver(counter)
    wrapper_class=ag.ACLGraphWrapper
    original_call=wrapper_class.__call__
    graph_class=torch.npu.NPUGraph
    original_replay=graph_class.replay
    if getattr(original_call,'_formal_dspark_actual_replay_observer',False):
        raise ValueError('Use a fresh private process; stock graph observer already installed')
    observer.sources={
        'stock_proposer':file_receipt(draft_original,STOCK_PROPOSER_SHA256),
        'acl_wrapper':file_receipt(original_call,ACL_WRAPPER_SHA256),
        'npu_graph_replay':file_receipt(original_replay,NPU_GRAPHS_SHA256)}

    def observed_call(wrapper,*args,**kwargs):
        if not observer.active() or wrapper is not runner.drafter.model:
            return original_call(wrapper,*args,**kwargs)
        context=get_forward_context()
        entry=wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
        graph=getattr(entry,'aclgraph',None) if entry is not None else None
        addresses=[value.data_ptr() for value in args if isinstance(value,torch.Tensor)]
        captured=list(entry.input_addresses) if entry is not None and entry.input_addresses is not None else None
        row={'runtime_mode':getattr(context.cudagraph_runtime_mode,'name',str(context.cudagraph_runtime_mode)),
             'wrapper_runtime_mode':getattr(wrapper.runtime_mode,'name',str(wrapper.runtime_mode)),
             'capturing':bool(getattr(context,'capturing',False)),
             'is_draft_model':bool(ag._EXTRA_CTX.is_draft_model),
             'descriptor':repr(context.batch_descriptor),'wrapper_id':id(wrapper),
             'graph_id':id(graph) if graph is not None else None,
             'existing_graph_entry':graph is not None,'replay_returned':False,
             'input_addresses_actual':addresses,'input_addresses_captured':captured,
             'input_addresses_match':captured is not None and addresses==captured,
             'address_scope':'stock wrapper positional Tensor inputs only; not hidden cache/value proof'}
        if len(observer.events)>=4:raise ValueError('Unexpected unbounded draft wrapper calls in one proposal')
        observer.events.append(row)
        previous=observer.expected
        observer.expected=(graph,row)
        try:
            return original_call(wrapper,*args,**kwargs)
        finally:
            observer.expected=previous

    def observed_replay(graph,*args,**kwargs):
        tracked=observer.expected
        if not observer.active() or tracked is None or tracked[0] is not graph:
            return original_replay(graph,*args,**kwargs)
        context=get_forward_context()
        if getattr(context,'capturing',False):
            raise ValueError('Actual replay observer must never run inside capture')
        result=original_replay(graph,*args,**kwargs)
        tracked[1]['replay_returned']=True
        tracked[1]['submission_stream']=int(torch.npu.current_stream().npu_stream)
        return result

    observed_call._formal_dspark_actual_replay_observer=True
    wrapper_class.__call__=observed_call
    graph_class.replay=observed_replay
    return observer
