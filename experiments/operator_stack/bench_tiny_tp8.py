"""Reduced real-weight TP8 diagnostic, with formal-prefix and native A/A gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-chips', required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--async-scheduling', action='store_true')
    parser.add_argument('--hccl-deterministic', choices=['strict'], default='strict')
    args = parser.parse_args()
    from verify_tiny_cache_plan import validate
    cache_preflight = validate()
    assert not (args.audit and args.profile)
    assert not (args.async_scheduling and args.profile)
    assert os.getenv('STACK_TINY_PROFILE') == '1'
    assert os.getenv('HCCL_DETERMINISTIC') == args.hccl_deterministic == 'strict'
    assert args.pairs >= 3
    assert args.physical_chips == os.environ['ASCEND_RT_VISIBLE_DEVICES']
    assert len(set(args.physical_chips.split(','))) == 8
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      VLLM_ALLOW_INSECURE_SERIALIZATION='1', STACK_REAL_WEIGHTS='1',
                      STACK_REAL_AUDIT='1' if args.audit else '0')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cache_preflight.json').write_text(json.dumps(cache_preflight, indent=2) + '\n')
    os.environ['VLLM_CACHE_ROOT'] = str(out / 'cache')
    manifest = json.loads((args.model / 'tiny_manifest.json').read_text())
    assert manifest['diagnostic_only'] and manifest['prefix_layers_exact'] == [0, 1, 2, 3]
    assert (args.reference_root / 'run.exit').read_text().strip() == '0'
    reference_result = json.loads((args.reference_root / 'result.json').read_text())
    assert reference_result['audit'] and reference_result['hccl_deterministic_env'] == 'strict'
    assert reference_result['checkpoint_config_sha256'] == manifest['source_config_sha256']
    prompts = json.loads((args.reference_root / 'prompts.json').read_text())
    assert len(prompts) == 3 and all(len(row) == 2048 for row in prompts)
    golden = {}
    if args.audit:
        for row in json.loads((args.reference_root / 'requests.json').read_text()):
            if row['arm'] == 'tp8base' and row['tag'].startswith('pair-'):
                golden[row['seed']] = [[layer for layer in token[:4]] for token in row['routes'][:2048]]
        assert set(golden) == {0, 1, 2}
    from vllm import LLM, SamplingParams
    import numpy as np
    from tiny_tp8_worker import topology_receipt
    started = time.monotonic()
    llm = LLM(model=str(args.model), tokenizer=str(args.model), load_format='auto', dtype='bfloat16',
              safetensors_load_strategy='lazy',
              model_loader_extra_config={'enable_multithread_load': True, 'num_threads': 128},
              worker_cls='tiny_tp8_worker.TinyTP8Worker', tensor_parallel_size=8,
              distributed_executor_backend='mp', enable_expert_parallel=True, seed=0,
              trust_remote_code=True, async_scheduling=args.async_scheduling,
              max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=2048,
              gpu_memory_utilization=.70, kv_cache_memory_bytes=4 * 1024**3,
              block_size=128, enable_prefix_caching=False, limit_mm_per_prompt={'image': 4},
              enable_return_routed_experts=args.audit,
              profiler_config={'profiler': 'torch', 'torch_profiler_dir': str(out / 'prof'),
                               'torch_profiler_with_stack': False} if args.profile else None,
              compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1]},
              additional_config={'enable_engram': True, 'engram_storage': 'int8',
                  'enable_cpu_binding': False,
                  'ascend_compilation_config': {'enable_npugraph_ex': True, 'enable_static_kernel': True},
                  'multistream_dsv4_dsa_overlap': False})
    startup_s = time.monotonic() - started
    topology = llm.collective_rpc(topology_receipt)
    assert len(topology) == 8 and {r['rank'] for r in topology} == set(range(8))
    (out / 'topology.json').write_text(json.dumps(topology, indent=2) + '\n')
    engine, counter, records = llm.llm_engine, 0, []
    prefix_checks, comparisons = [], []

    def request(seed, tag, metric=None):
        nonlocal counter
        counter += 1
        count = 47 + seed % 2 if args.audit else 48
        engine.add_request(f'tiny8-{counter}', {'prompt_token_ids': prompts[seed]},
                           SamplingParams(temperature=0, max_tokens=count, ignore_eos=True,
                                          detokenize=False, logprobs=5 if args.audit else None))
        times, arrivals, observed, final, profiling = [], [], 0, None, False
        while engine.has_unfinished_requests():
            if metric and len(times) == 9:
                from formal_tp8_profile import configure
                receipt = llm.collective_rpc(configure, args=(metric, str(out / 'prof' / metric), 5, 10))
                (out / f'profile_receipt_{metric}.json').write_text(json.dumps(receipt, indent=2) + '\n')
                llm.start_profile()
                profiling = True
            start = time.perf_counter()
            outputs = engine.step()
            times.append((time.perf_counter() - start) * 1000)
            if profiling:
                from formal_tp8_profile import advance
                llm.collective_rpc(advance)
                if len(times) == 24:
                    llm.stop_profile()
                    profiling = False
            for value in outputs:
                if value.outputs:
                    total = len(value.outputs[0].token_ids)
                    if total > observed:
                        arrivals.append((total, time.perf_counter()))
                        observed = total
                if value.finished:
                    final = value
        assert not profiling and final is not None
        output = final.outputs[0]
        assert len(output.token_ids) == count
        ms = statistics.median(times[9:])
        if args.async_scheduling:
            warm = next(r for r in arrivals if r[0] >= 9)
            assert arrivals[-1][0] - warm[0] >= 20
            ms = (arrivals[-1][1] - warm[1]) * 1000 / (count - warm[0])
        record = {'seed': seed, 'tag': tag, 'token_ids': output.token_ids,
                  'decode_ms': ms, 'profile_metric': metric}
        if args.audit:
            routes = np.asarray(output.routed_experts)
            assert routes.shape == (2048 + count - 1, 8, 6), routes.shape
            assert np.issubdtype(routes.dtype, np.integer)
            assert np.all((routes >= 0) & (routes < 384))
            assert np.all(np.diff(np.sort(routes, axis=-1), axis=-1) > 0)
            actual_prefix = routes[:2048, :4]
            expected_prefix = np.asarray(golden[seed])
            equal = np.array_equal(actual_prefix, expected_prefix)
            check = {'seed': seed, 'tag': tag, 'prefill_tokens': 2048, 'layers': 4,
                     'prefix_routes_equal': bool(equal),
                     'differing_token_layers': int(np.any(actual_prefix != expected_prefix, axis=-1).sum())}
            prefix_checks.append(check)
            (out / 'formal_prefix_comparisons.json').write_text(json.dumps(prefix_checks, indent=2) + '\n')
            assert equal, ('Formal-prefix route mismatch', check)
            record['routes'] = routes.tolist()
            record['logprobs'] = [{str(k): v.logprob for k, v in row.items()} for row in output.logprobs]
        records.append(record)
        (out / 'requests.json').write_text(json.dumps(records) + '\n')
        print('TINY_TP8_REQUEST', json.dumps({k: v for k, v in record.items()
                                          if k not in ('routes', 'logprobs', 'token_ids')}), flush=True)
        return record

    request(0, 'warmup')
    for pair in range(args.pairs):
        seed = pair % 3
        reference = request(seed, f'pair-{pair}-A')
        actual = request(seed, f'pair-{pair}-B')
        check = {'pair': pair, 'tokens_equal': reference['token_ids'] == actual['token_ids']}
        if args.audit:
            from formal_comparison import compare_requests
            check.update(compare_requests(reference, actual, f'tiny-pair-{pair}', layers=8))
        assert check.get('passed', check['tokens_equal']), check
        comparisons.append(check)
        (out / 'comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    ms = statistics.median(r['decode_ms'] for r in records if r['tag'].startswith('pair-'))
    result = {'diagnostic_only': True, 'formal_weights': False, 'source_formal_tensors_byte_exact': True,
              'formal_quality_claim': False, 'tensor_parallel_size': 8, 'layers': 8,
              'physical_chips': args.physical_chips, 'startup_seconds': startup_s,
              'ms_per_step': ms, 'A': 1, 'tokens_per_second': 1000 / ms,
              'audit': args.audit, 'profiler_during_timing': 'OFF', 'async_scheduling': args.async_scheduling,
              'timing_scope': 'token-arrival elapsed/decoded tokens' if args.async_scheduling else 'engine.step median',
              'native_pairs': comparisons, 'formal_prefix_comparisons': prefix_checks,
              'checkpoint_config_sha256': manifest['config_sha256'],
              'source_config_sha256': manifest['source_config_sha256'],
              'reference_result_sha256': hashlib.sha256((args.reference_root / 'result.json').read_bytes()).hexdigest()}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('TINY_TP8_COMPLETE', json.dumps(result), flush=True)
    if args.profile:
        from formal_tp8_profile import METRICS
        for metric in METRICS:
            request(0, 'profile-' + metric, metric)


if __name__ == '__main__':
    main()
