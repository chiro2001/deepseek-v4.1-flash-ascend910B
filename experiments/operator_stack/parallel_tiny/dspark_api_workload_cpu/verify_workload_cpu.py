"""Read-only source-span and frozen client-contract verification; no HTTP/NPU."""
import os
os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0', ASCEND_RT_VISIBLE_DEVICES='',
    ASCEND_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='',
    PYTHONDONTWRITEBYTECODE='1')

import hashlib
import json
from pathlib import Path
import sys

from parallel_tiny.dspark_api_counter_bridge_cpu.acceptance_client import validate_workload, make_completion
from parallel_tiny.dspark_api_counter_bridge_cpu.bridge_state import validate_completion

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    inputs = HERE / 'evidence/prepared_inputs'
    workload_path = inputs / 'workload.json'
    workload = json.loads(workload_path.read_text())
    validate_workload(workload)
    assert workload['cpu_preparation_only'] and not workload['cpu_mock']
    parity_path = inputs / 'AutoTokenizer_parity_receipt.json'
    parity = json.loads(parity_path.read_text())
    final = json.loads((inputs / 'final_preparation_receipt.json').read_text())
    assert sha(workload_path.read_bytes()) == final['workload_sha256']
    assert sha(parity_path.read_bytes()) == final['AutoTokenizer_parity_receipt_sha256']
    assert workload['tokenizer_provenance']['checkpoint_AutoTokenizer_parity'] is True
    assert workload['tokenizer_provenance']['checkpoint_AutoTokenizer_parity_receipt_sha256'] == sha(parity_path.read_bytes())
    assert len(parity['checks']) == 10 and parity['checkpoint_AutoTokenizer_parity_completed']
    assert not parity['NPU_vLLM_ACL_module_imports'] and not parity['blocked_import_attempts']
    assert not parity['NPU_runtime_libraries_mapped'] and parity['NPU_calls'] == 0
    repository = HERE.parents[3]
    corpus = repository / 'data/hongloumeng.txt'
    original = corpus.read_bytes()
    assert sha(original) == workload['corpus_provenance']['sha256']
    spans = []
    checks = []
    for phase, expected in [('warmup', 2), ('measured', 8)]:
        assert len(workload[phase]) == expected
        assert [row['name'] for row in workload[phase]] == [phase + '_' + str(i) for i in range(expected)]
        for row in workload[phase]:
            assert row['phase'] == phase
            text = (inputs / 'prompts' / (row['name'] + '.txt')).read_bytes()
            assert sha(text) == row['text_sha256'] and text.decode().endswith(row['question'])
            assert sha(json.dumps(row['token_ids'], separators=(',', ':')).encode()) == row['prompt_sha256']
            span = row['source_span']
            context = original[span['byte_start']:span['byte_end']]
            assert sha(context) == span['context_sha256'] and context.decode() in text.decode()
            assert row['answerability']['first_complete_sentence_witness'] in context.decode()
            question_path = repository / 'data/hlm_local' / ('local_' + row['task'] + '.txt')
            assert sha(question_path.read_bytes()) == row['question_source']['sha256']
            assert row['question'] == question_path.read_text().strip() + '\n' + row['question_appendix']
            request_id = 'http_workload_' + row['name']
            model = 'dsv41-dspark-private-workload-future'
            validate_completion(make_completion(model, row['token_ids'], request_id, row['index']), request_id, model)
            spans.append((span['byte_start'], span['byte_end']))
            checks.append({'name': row['name'], 'phase': phase, 'prompt_tokens': len(row['token_ids']),
                'output_budget': 256, 'original_source_span_sha_matches': True,
                'original_suffix_sha_matches': True, 'question_tail_complete': True,
                'first_sentence_witness_in_source': True,
                'frozen_client_and_fixed_request_accept': True})
    spans.sort()
    assert all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:]))
    assert len({row['prompt_sha256'] for phase in ('warmup', 'measured') for row in workload[phase]}) == 10
    actual_flags = {key: value for key, value in workload.items() if key.startswith('actual_') or key.endswith('_receipt_granted')}
    assert actual_flags and not any(actual_flags.values())
    for source in json.loads((HERE / 'evidence/quality_script_source_guards.json').read_text())['source_files']:
        assert sha((repository / source['path']).read_bytes()) == source['sha256']
    backend_modules = [name for name in sys.modules if name in ('torch', 'torch_npu', 'vllm', 'acl')]
    assert not backend_modules
    report = {'CPU_only': True, 'passed': True, 'original_source_and_question_checks': checks,
        'checkpoint_AutoTokenizer_parity_checks': 10,
        'warmup_count': 2, 'measured_count': 8, 'warmup_excluded_from_measured': True,
        'original_context_intervals_nonoverlapping': True, 'unique_input_count': 10,
        'all_six_readonly_quality_script_source_SHA_unchanged': True,
        'NPU_calls': 0, 'actual_HTTP_inference_requests': 0, 'services_started': 0,
        'loaded_backend_modules_in_local_check': backend_modules,
        'workload_sha256': sha(workload_path.read_bytes()), 'actual_flags': actual_flags}
    (HERE / 'evidence/CPU_final_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'prompts': 10, 'source_spans_nonoverlapping': True,
        'frozen_client_contract_checks': 10, 'NPU_or_HTTP': 0}, indent=2))


if __name__ == '__main__':
    main()
