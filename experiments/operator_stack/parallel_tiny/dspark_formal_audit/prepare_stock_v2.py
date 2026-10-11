"""Generate stock-compatible v2 COPIES; never modify frozen baseline modules."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    output=Path(__file__).parent/'stock_compat_v2'
    output.mkdir(exist_ok=True)
    counter=(root/'formal_dspark_counters.py').read_text()
    bench=(root/'bench_formal_dspark.py').read_text()
    assert hashlib.sha256(counter.encode()).hexdigest()=='49fec8997feaa8dec6ec8cf069f6f3998f02cafc61160835895ae7015b67b7e3'
    assert hashlib.sha256(bench.encode()).hexdigest()=='aee6039f070ae73b5051c9a8cf39697e57463b4349dee001635cac0f8c4cd16b'
    old="""    graph_query = inspect.unwrap(draft_original).__globals__.get('_dspark_use_graph_rt')
    if not callable(graph_query):
        raise ValueError('Actual draft runtime graph selector is unavailable')
    counter.runtime_graph_selector = graph_query
"""
    new="""    from parallel_tiny.dspark_formal_audit.stock_graph_observer import install as install_graph_observer
    graph_observer=install_graph_observer(counter,runner,draft_original)
    counter.actual_draft_graph_observer=graph_observer
"""
    assert counter.count(old)==1
    counter=counter.replace(old,new)
    old="""        result = draft_original(*args, **kwargs)
        if counter.tag is not None:
            k = kwargs.get('num_speculative_tokens', args[0] if args else None)
            counter.draft_finished(k=k, output_shape=result.shape,
                                   runtime_graph=graph_query(drafter), method=drafter.method)
        return result
"""
    new="""        active=counter.tag is not None
        if active:graph_observer.begin_proposal()
        try:
            result = draft_original(*args, **kwargs)
        except BaseException:
            if active:graph_observer.cancel_proposal()
            raise
        if active:
            evidence=graph_observer.finish_proposal()
            k = kwargs.get('num_speculative_tokens', args[0] if args else None)
            counter.draft_finished(k=k, output_shape=result.shape,
                runtime_graph=evidence['actual_graph_submission'],method=drafter.method)
            for row in counter.pending.values():
                row['draft_calls'][-1]['actual_execution_evidence']=evidence
        return result
"""
    assert counter.count(old)==1
    counter=counter.replace(old,new)
    old="            'draft_runtime_graph_selected': bool(counter.runtime_graph_selector(drafter)),"
    new="""            'draft_graph_configured':bool(drafter.use_cuda_graph),
            'actual_graph_submission_proof':None,
            'graph_observer_sources':counter.actual_draft_graph_observer.sources,"""
    assert counter.count(old)==1
    counter=counter.replace(old,new)
    # Timing-independent A/A equality must not compare process-specific graph
    # ids, addresses or execution timestamps. Preserve the actual evidence in
    # raw receipts, compare only proposal semantics in the existing strict gate.
    counter=counter.replace("r['rejection_sampler_called'], r['draft_calls']) for r in rounds]",
        "r['rejection_sampler_called'], [{k:c[k] for k in ('k','output_shape','runtime_graph','method')} for c in r['draft_calls']]) for r in rounds]")
    old="""        if draft_graph and any(not p['runtime_graph'] for p in proposals):
            raise ValueError('Draft graph requested but actual DSpark proposal used eager')
"""
    new="""        steady=[c for row in rounds for c in row['draft_calls']]
        if any(not row['draft_calls'] for row in rounds):
            raise ValueError('A real decode round did not run its DSpark proposal')
        if draft_graph and any(not p['runtime_graph'] for p in steady):
            raise ValueError('Draft graph requested but actual steady DSpark proposal used eager')
"""
    assert counter.count(old)==1
    counter=counter.replace(old,new)
    bench=bench.replace("bank['draft_runtime_graph_selected']","bank['draft_graph_configured']")
    bench=bench.replace("'Draft graph ON was not selected in the actual runtime'",
                        "'Draft graph is not configured; actual replay is checked for every real proposal'")
    old="return [(x['scheduled_draft_token_ids'],x['token_ids'],x['draft_calls']) for x in r['rounds']]"
    new="return [(x['scheduled_draft_token_ids'],x['token_ids'],[{k:c[k] for k in ('k','output_shape','runtime_graph','method')} for c in x['draft_calls']]) for x in r['rounds']]"
    assert bench.count(old)==1
    bench=bench.replace(old,new)
    old="'parallel_tiny/dspark_formal_prep/sdk_gate.py']"
    new="'parallel_tiny/dspark_formal_prep/sdk_gate.py','parallel_tiny/dspark_formal_audit/stock_graph_observer.py']"
    assert bench.count(old)==1
    bench=bench.replace(old,new)
    (output/'formal_dspark_counters.py').write_text(counter)
    (output/'bench_formal_dspark.py').write_text(bench)
    result={'scope':'new stock-compatible snapshot copies only; no NPU executed',
        'replaces_only_in_new_snapshot':['stack/formal_dspark_counters.py','stack/bench_formal_dspark.py'],
        'also_copy':'stack/parallel_tiny/dspark_formal_audit/stock_graph_observer.py',
        'formal_model_validated':False,'device_audit_passed':False,'safe_formal_receipt':False,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.py')}}
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    freeze=json.loads((root/'parallel_tiny/dspark_formal_prep/freeze.json').read_text())
    freeze['files']['formal_dspark_counters.py']=result['files']['formal_dspark_counters.py']
    freeze['files']['bench_formal_dspark.py']=result['files']['bench_formal_dspark.py']
    observer=Path(__file__).parent/'stock_graph_observer.py'
    freeze['files']['parallel_tiny/dspark_formal_audit/stock_graph_observer.py']=hashlib.sha256(observer.read_bytes()).hexdigest()
    freeze['phase']='CPU-only-stock-compatible-v2; fresh NPU native AA pending'
    freeze['runtime_graph_observation']='actual existing draft entry NPUGraph.replay return plus forward context; configured bool is separate'
    freeze['initial_draft_policy']='record actual NONE/eager/FULL mode; FULL_DECODE_ONLY only requires steady decode proposals to replay'
    (output/'prep_freeze_for_new_snapshot.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
