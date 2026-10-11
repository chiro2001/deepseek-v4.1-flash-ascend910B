"""Freeze a new stock snapshot with zero-token cleanup and exact core IDs."""
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    parent = Path(__file__).parent
    source = parent / 'stock_compat_v2'
    output = parent / 'stock_compat_v3'
    output.mkdir(exist_ok=True)
    expected = {
        'formal_dspark_counters.py': 'bdc95a286e85a22a51bb9fed155fd7f0bf010010768ff43db7b034ff11fe0cc0',
        'bench_formal_dspark.py': 'f24656d67684728b3c28bb4e9db782bc11ea76aac405d3d9de4e38b7102dcfd8',
    }
    texts = {}
    for name, digest in expected.items():
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        texts[name] = data.decode()
    counter = texts['formal_dspark_counters.py']
    counter = replace_once(counter, '        self.max_tokens = None\n',
                           '        self.max_tokens = None\n        self.zero_token_executions = []\n')
    counter = replace_once(counter, '        self.max_tokens = max_tokens\n',
                           '        self.max_tokens = max_tokens\n        self.zero_token_executions = []\n')
    counter = replace_once(counter,
        '        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))\n',
        '''        if not scheduler.num_scheduled_tokens:
            # The stock runner updates/frees request state and returns EMPTY
            # before any target, sampler or draft execution for this ABI.
            # This commonly occurs after the final request has completed.
            if (type(scheduler.total_num_scheduled_tokens) is not int or
                    scheduler.total_num_scheduled_tokens != 0 or new or cached.req_ids or
                    scheduler.scheduled_spec_decode_tokens or
                    (getattr(scheduler, 'num_invalid_spec_tokens', None) or {}) or
                    (getattr(scheduler, 'scheduled_encoder_inputs', None) or {})):
                raise ValueError('Inconsistent zero-token cleanup scheduler')
            finished = sorted(scheduler.finished_req_ids)
            if any(type(req_id) is not str for req_id in finished):
                raise ValueError('Unknown cleanup request ID ABI')
            self.zero_token_executions.append({'finished_req_ids': finished,
                'total_num_scheduled_tokens': 0, 'execute_start_ns': self.now(),
                'counted_as_decode_round': False})
            return False
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
''')
    counter = replace_once(counter, '        self.pending = pending\n',
                           '        self.pending = pending\n        return True\n')
    counter = replace_once(counter,
        "                  'actual_device_consumer_proof': False}\n        self.tag = None\n",
        "                  'actual_device_consumer_proof': False,\n"
        "                  'zero_token_executions': copy.deepcopy(self.zero_token_executions)}\n        self.tag = None\n")
    counter = replace_once(counter,
        'def summarize(receipts, final_tokens, client_end_ns, client_clock, *, draft_graph):\n',
        'def summarize(receipts, final_tokens, client_end_ns, client_clock, *, draft_graph, expected_core_request_id=None):\n')
    counter = replace_once(counter,
        "        request = next(iter(receipt['requests'].values()))\n",
        '''        if expected_core_request_id is not None:
            if (type(expected_core_request_id) is not str or
                    set(receipt['requests']) != {expected_core_request_id}):
                raise ValueError('Worker counters do not match actual core request ID')
            for cleanup in receipt.get('zero_token_executions', []):
                if not set(cleanup['finished_req_ids']).issubset({expected_core_request_id}):
                    raise ValueError('Cleanup counters include a different core request ID')
                if cleanup.get('execute_end_ns') is None:
                    raise ValueError('Zero-token cleanup did not return stock empty output')
        request = next(iter(receipt['requests'].values()))
''')
    counter = replace_once(counter,
        "                   rank_counters=rows, actual_device_consumer_proof=False,\n",
        "                   rank_counters=rows, actual_device_consumer_proof=False,\n"
        "                   core_request_id=expected_core_request_id,\n"
        "                   exact_core_request_id_bound=expected_core_request_id is not None,\n")
    counter = replace_once(counter,
        "        counter.execute_begin(scheduler_output, runner.requests)\n        return originals['execute_model'](scheduler_output, *args, **kwargs)\n",
        '''        active = counter.tag is not None
        counted = counter.execute_begin(scheduler_output, runner.requests)
        result = originals['execute_model'](scheduler_output, *args, **kwargs)
        if active and not counted:
            # A zero-token return must not have a pending sample or output
            # tokens. Check only the stock CPU ABI, never a device tensor.
            if (result is None or runner.execute_model_state is not None or
                    getattr(result, 'req_ids', None) != [] or
                    getattr(result, 'sampled_token_ids', None) != []):
                raise ValueError('Zero-token cleanup did not return stock empty output')
            counter.zero_token_executions[-1]['execute_end_ns'] = counter.now()
        return result
''')

    bench = texts['bench_formal_dspark.py']
    bench = replace_once(bench, '\ndef run_native(args, result, checkpoint, sources):\n',
        '''
def mapped_generation(llm, prompt, sampling):
    """Record the real external -> core mapping from the existing engine ABI."""
    engine = llm.llm_engine
    original = engine.add_request
    identities = []
    def add_request(request_id, *args, **kwargs):
        core_id = original(request_id, *args, **kwargs)
        if (type(core_id) is not str or
                engine.output_processor.external_req_ids.get(request_id) != [core_id]):
            raise ValueError('Actual engine external/core request mapping is unavailable')
        identities.append((request_id, core_id))
        return core_id
    engine.add_request = add_request
    try:
        outputs = llm.generate([{'prompt_token_ids': prompt}], sampling, use_tqdm=False)
    finally:
        engine.add_request = original
    if (len(outputs) != 1 or len(identities) != 1 or
            outputs[0].request_id != identities[0][0]):
        raise ValueError('Single-request output does not match the actual engine mapping')
    return outputs[0], identities[0][1]


def run_native(args, result, checkpoint, sources):
''')
    bench = replace_once(bench,
        "        output=llm.generate([{'prompt_token_ids':prompts[seed]}],sampling,use_tqdm=False)[0]\n",
        "        output,core_request_id=mapped_generation(llm,prompts[seed],sampling)\n")
    bench = replace_once(bench,
        "                                       draft_graph=bool(args.draft_graph))\n",
        "                                       draft_graph=bool(args.draft_graph),\n"
        "                                       expected_core_request_id=core_request_id)\n")
    bench = replace_once(bench,
        "             'request_id':output.request_id,'client_end_ns':ended}\n",
        "             'request_id':output.request_id,'core_request_id':core_request_id,'client_end_ns':ended}\n")
    for name, value in (('formal_dspark_counters.py', counter), ('bench_formal_dspark.py', bench)):
        (output / name).write_text(value)
    freeze = json.loads((source / 'prep_freeze_for_new_snapshot.json').read_text())
    freeze['phase'] = 'stock-compatible-v3-zero-token-cleanup-and-exact-core-IDs; native AA pending'
    freeze['cpu_counter_tests'] = 26
    for name in expected:
        freeze['files'][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    freeze['previous_snapshot_files'] = expected
    freeze['counter_scope'] = 'actual target/verifier rounds; zero-token stock cleanup recorded separately'
    freeze['request_identity'] = 'actual LLMEngine.add_request return and external_req_ids; exact eight-rank core binding'
    (output / 'prep_freeze_for_new_snapshot.json').write_text(json.dumps(freeze, indent=2) + '\n')
    receipt = {'scope': 'new stock snapshot CPU source correction; no NPU proof',
               'source_v2_files': expected, 'files': {name: freeze['files'][name] for name in expected},
               'native_aa_passed': False, 'device_audit_passed': False,
               'performance_receipt_granted': False}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
