"""Same-namespace serial HTTP client. Default mode performs no HTTP requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import urllib.parse
import urllib.request
import uuid

from .quality_cpu import quality
from .runtime_guards import (COUNTER_SHA256, clock_scope, false_claims,
    monotonic_ns, token_hash, token_list)


def validate_workload(workload):
    if (workload.get('cpu_mock') is not False or
            workload.get('prompt_kind') != 'real_text_tasks' or
            not workload.get('tokenizer_provenance') or not workload.get('corpus_provenance')):
        raise ValueError('Actual acceptance requires frozen real-text/tokenizer provenance')
    if len(workload.get('warmup', [])) < 2 or len(workload.get('measured', [])) < 8:
        raise ValueError('At least two warmup and eight measured prompts are required')
    names = set()
    hashes = set()
    for phase in ('warmup', 'measured'):
        for row in workload[phase]:
            prompt = token_list(row['token_ids'], size=2048)
            if token_hash(prompt) != row['prompt_sha256'] or not row['task'] or not row['name']:
                raise ValueError('Prompt hash, length, or actual task provenance mismatch')
            if row['name'] in names:
                raise ValueError('Prompt names must be distinct across warmup/measured')
            names.add(row['name'])
            if phase == 'measured':
                hashes.add(row['prompt_sha256'])
    if len(hashes) < 8:
        raise ValueError('Eight independent measured real-text prompts are required')
    return workload


def make_completion(model, prompt, request_id, seed):
    return {'model': model, 'prompt': token_list(prompt, size=2048), 'request_id': request_id,
        'n': 1, 'stream': False, 'echo': False, 'max_tokens': 256, 'min_tokens': 256,
        'ignore_eos': True, 'add_special_tokens': False, 'return_token_ids': True,
        'temperature': 0.0, 'seed': seed, 'use_beam_search': False}


def aggregate(rows):
    if len(rows) < 8 or any(row['phase'] != 'measured' for row in rows):
        raise ValueError('Aggregate must contain at least eight measured windows and no warmup')
    if [row['index'] for row in rows] != list(range(len(rows))):
        raise ValueError('Measured windows are not a complete serial sequence')
    metrics = [row['finish']['metrics'] for row in rows]
    for m in metrics:
        if (m['decoded_tokens'] != 255 or m['emitted_tokens_including_prefill_seed'] != 256 or
                m['prefill_seed_tokens'] != 1 or type(m['rounds']) is not int or m['rounds'] <= 0 or
                not math.isclose(m['ms_per_step'], 1000 * m['A'] / m['tokens_per_second'], rel_tol=1e-12)):
            raise ValueError('Per-request actual-round/seed/triple definition changed')
    decoded = sum(m['decoded_tokens'] for m in metrics)
    rounds = sum(m['rounds'] for m in metrics)
    elapsed = sum(m['decode_elapsed_s'] for m in metrics)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Measured decode elapsed time is invalid')
    result = {'samples': len(rows), 'concurrency': 1, 'prompt_tokens': 2048,
        'emitted_output_tokens_per_request': 256, 'decoded_tokens_per_request': 255,
        'total_actual_rounds': rounds, 'total_decoded_tokens': decoded,
        'decode_elapsed_s_sum': elapsed, 'A': decoded / rounds,
        'ms_per_step': 1000 * elapsed / rounds, 'tokens_per_second': decoded / elapsed,
        'aggregation': 'sum measured decode elapsed / sum actual rounds; warmup excluded',
        'ms_per_step_median': statistics.median(m['ms_per_step'] for m in metrics),
        'per_stream_tokens_per_second_median': statistics.median(m['tokens_per_second'] for m in metrics),
        'clean_count': sum(row['quality']['clean'] for row in rows),
        'repetition_count': sum(row['quality']['repeat_loop'] or row['quality']['repeated_token_window'] for row in rows),
        'acceptance_length_review_required': any(m['A'] > 3.5 for m in metrics)}
    result['clean_rate'] = result['clean_count'] / result['samples']
    if not math.isclose(result['ms_per_step'], 1000 * result['A'] / result['tokens_per_second'], rel_tol=1e-12):
        raise ValueError('Measured aggregate triple is inconsistent')
    return result


def read_http_response(response, *, now=monotonic_ns):
    # This ordering is the acceptance clock contract. No JSON parsing, output
    # write, or counter retrieval RPC may precede this immediate read-end stamp.
    raw = response.read()
    client_end_ns = now()
    return raw, client_end_ns


def _json_request(url, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload, separators=(',', ':')).encode()
    request = urllib.request.Request(url, data=data,
        headers={'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def verify_identity(identity, selection, expected_model, client_clock):
    if (identity['served_model'] != expected_model or identity['service_id'] != selection['service_id'] or
            selection.get('service_started') is not True or selection['served_model'] != expected_model or
            selection['model_kwargs']['speculative_config']['method'] != 'dspark' or
            identity['counter_sha256'] != COUNTER_SHA256 or identity['tensor_parallel_size'] != 8 or
            identity['clock_scope'] != client_clock or identity.get('cpu_mock') is not False or
            identity.get('audit') is not False or identity.get('profiler') != 'OFF' or
            identity.get('failed') is not False or identity['completed'] != {'warmup': 0, 'measured': 0}):
        raise ValueError('Private service ownership/source/clock/idle identity mismatch')
    descriptions = selection['worker_descriptions']
    if len(descriptions) != 8 or {row['rank'] for row in descriptions} != set(range(8)):
        raise ValueError('Service selection does not identify all actual worker ranks')
    for row in descriptions:
        if row['clock_scope'] != client_clock or row['counter_sha256'] != COUNTER_SHA256 or row['audit']:
            raise ValueError('Client cannot subtract another worker boot/time namespace clock')


def run_actual(args, workload):
    parsed = urllib.parse.urlsplit(args.base_url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or
            parsed.path or parsed.query or parsed.fragment):
        raise ValueError('Run the HTTP acceptance client locally in the worker boot/time namespace')
    selection = json.loads(args.service_selection.read_text())
    client_clock = clock_scope()
    identity = _json_request(args.base_url + '/private/dspark/identity')
    verify_identity(identity, selection, args.model, client_clock)
    if parsed.port != selection['port']:
        raise ValueError('Client base URL port differs from the owned service selection')
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'acceptance.json'
    if report_path.exists():
        raise FileExistsError('Do not overwrite an earlier actual acceptance report')
    rows = {'warmup': [], 'measured': []}
    report = {'cpu_preparation_only': False, 'cpu_mock': False,
        'service_id': identity['service_id'], 'clock_scope': client_clock,
        'serial_requests': True, 'warmup': rows['warmup'], 'measured': rows['measured'],
        **false_claims()}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        for phase in ('warmup', 'measured'):
            for index, prompt in enumerate(workload[phase]):
                request_id = 'http_' + uuid.uuid4().hex
                window_id = phase + '_' + str(index) + '_' + uuid.uuid4().hex
                envelope = {'window_id': window_id, 'phase': phase, 'index': index,
                    'client_clock': client_clock,
                    'completion': make_completion(args.model, prompt['token_ids'], request_id, index)}
                data = json.dumps(envelope, separators=(',', ':')).encode()
                request = urllib.request.Request(args.base_url + '/private/dspark/completions', data=data,
                    headers={'Content-Type': 'application/json', 'X-Request-Id': request_id})
                with urllib.request.urlopen(request, timeout=600) as response:
                    raw, ended = read_http_response(response)
                # Files are preserved even if subsequent verification fails.
                raw_path = args.output / (phase + '_' + str(index) + '_http.json')
                raw_path.write_bytes(raw)
                output = json.loads(raw)
                binding = output['bridge']
                completion = output['completion']
                choice = completion['choices'][0]
                tokens = token_list(choice['token_ids'], size=256)
                if (clock_scope() != client_clock or binding['clock_scope'] != client_clock or
                        binding['cpu_mock'] is not False or binding['service_id'] != identity['service_id'] or
                        binding['window_id'] != window_id or binding['phase'] != phase or binding['index'] != index or
                        completion['model'] != args.model or len(completion['choices']) != 1 or
                        choice['finish_reason'] != 'length' or choice['index'] != 0 or
                        token_hash(token_list(choice['prompt_token_ids'], size=2048)) != prompt['prompt_sha256'] or
                        completion['usage'] != {'prompt_tokens': 2048, 'completion_tokens': 256, 'total_tokens': 2304}):
                    raise ValueError('Actual HTTP response identity/token budget is not auditable')
                finish_payload = {'window_id': window_id, 'service_id': identity['service_id'],
                    'client_clock': client_clock, 'client_end_ns': ended, 'client_read_completed': True,
                    'http_body_sha256': hashlib.sha256(raw).hexdigest(), 'final_tokens_sha256': token_hash(tokens)}
                finish = _json_request(args.base_url + '/private/dspark/finish', finish_payload)
                m = finish['metrics']
                if (finish['cpu_mock'] is not False or finish['window_id'] != window_id or
                        finish['service_id'] != identity['service_id'] or m['client_end_ns'] != ended or
                        m['clock_scope'] != client_clock or m['core_request_id'] != binding['core_request_id'] or
                        finish['mapping']['engine_external_request_id'] != binding['engine_external_request_id'] or
                        finish['mapping']['http_request_id'] != request_id or
                        m['decoded_tokens'] != len(tokens) - 1 or not m['exact_core_request_id_bound']):
                    raise ValueError('Finish RPC does not summarize this exact actual HTTP output')
                row = {'phase': phase, 'index': index, 'prompt_name': prompt['name'],
                    'task': prompt['task'], 'prompt_sha256': prompt['prompt_sha256'],
                    'client_end_ns': ended, 'raw_http_sha256': hashlib.sha256(raw).hexdigest(),
                    'raw_http_path': str(raw_path), 'finish': finish,
                    'quality': quality(choice['text'], tokens)}
                rows[phase].append(row)
                report_path.write_text(json.dumps(report, indent=2) + '\n')
        report['aggregate'] = aggregate(rows['measured'])
        report['actual_api_performance_observed'] = True
        report['actual_api_quality_observed'] = True
        report['actual_client_quality_passed'] = report['aggregate']['clean_rate'] == 1.0
        report['performance_target_observed'] = report['aggregate']['ms_per_step'] <= 20
        report['manual_quality_review_completed'] = False
        report['formal_receipt_granted'] = report['performance_receipt_granted'] = False
    except BaseException as exc:
        report['failed'] = True
        report['failure_type'] = type(exc).__name__
        report['failure'] = str(exc)
        report.update(false_claims())
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        raise
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--service-selection', required=True, type=Path)
    parser.add_argument('--prompts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    workload = validate_workload(json.loads(args.prompts.read_text()))
    if args.execute:
        report = run_actual(args, workload)
        print(json.dumps({'aggregate': report['aggregate'], **{k: report[k] for k in false_claims()}}, indent=2))
    else:
        print(json.dumps({'cpu_preparation_only': True, 'http_requests': 0,
            'warmup_count': len(workload['warmup']), 'measured_count': len(workload['measured']),
            'decoded_tokens_per_request': 255, 'output_tokens_per_request': 256,
            'clock_scope': clock_scope(), 'client_must_run_in_worker_boot_and_time_namespace': True,
            **false_claims()}, indent=2))


if __name__ == '__main__':
    main()
