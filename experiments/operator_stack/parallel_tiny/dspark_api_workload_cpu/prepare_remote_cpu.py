"""Real checkpoint tokenizer + bounded existing corpus excerpts; CPU only.

No model weight, vLLM, torch, backend, socket, HTTP or service is used. Large
tokenizer/corpus files remain remote. All actual-inference admission flags stay
false; cpu_mock=False means the tokenizer/input data are genuine, not invented.
"""
import os

CPU_ENV = {'TORCH_DEVICE_BACKEND_AUTOLOAD': '0', 'ASCEND_RT_VISIBLE_DEVICES': '',
    'ASCEND_VISIBLE_DEVICES': '', 'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '',
    'USE_TORCH': '0', 'USE_TF': '0', 'USE_FLAX': '0', 'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
    'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1'}
os.environ.update(CPU_ENV)

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

CONFIG_SHA = '40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424'
CORPUS_SHA = 'a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578'
TASKS = ('quote', 'qa', 'continue', 'extract')
CLAIMS = {'actual_model_inference_executed': False, 'actual_api_performance_observed': False,
    'actual_api_quality_observed': False, 'actual_client_quality_passed': False,
    'actual_device_consumer_proof': False, 'performance_receipt_granted': False,
    'formal_receipt_granted': False}
NAMES = ('贾宝玉', '林黛玉', '薛宝钗', '王熙凤', '贾母', '贾政', '贾琏', '袭人',
    '晴雯', '香菱', '史湘云', '甄士隐', '贾雨村', '刘姥姥', '凤姐', '宝玉', '黛玉',
    '宝钗', '贾赦', '平儿', '探春', '迎春', '惜春', '秦可卿', '李纨', '赵姨娘')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_hash(ids):
    return sha(json.dumps(ids, separators=(',', ':')).encode())


def receipt(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'resolved_path': str(path.resolve()),
        'bytes': len(data), 'sha256': sha(data), 'source_file_transferred': False}


def reject_backends():
    loaded = [name for name in sys.modules if name == 'torch' or
        name.startswith(('torch.', 'torch_npu', 'vllm', 'tensorflow', 'jax'))]
    if loaded:
        raise ValueError('A forbidden model/device backend module was imported: ' + repr(loaded))
    if any(os.environ.get(key) != value for key, value in CPU_ENV.items()):
        raise ValueError('CPU-only environment guard changed')
    return loaded


def first_sentence(text):
    found = re.search(r'.+?[。！？]', text, flags=re.S)
    return found.group(0).strip() if found else None


def paragraph_start(body, offset):
    index = body.find('\n', offset)
    if index < 0:
        raise ValueError('No later source paragraph')
    index += 1
    while index < len(body) and body[index].isspace():
        index += 1
    return index


def calibrate(tokenizer, body, prefix, suffix, start):
    # Encode the whole textual prompt, preserving character/UTF-8 boundaries.
    # No placeholder padding or source duplication is used to hit the budget.
    lo, hi = 1, min(8192, len(body) - start)
    while lo <= hi:
        mid = (lo + hi) // 2
        count = len(tokenizer.encode(prefix + body[start:start + mid] + suffix,
            add_special_tokens=False).ids)
        if count < 2048:
            lo = mid + 1
        elif count > 2048:
            hi = mid - 1
        else:
            return mid
    for length in range(max(1, hi - 64), min(len(body) - start, lo + 64) + 1):
        count = len(tokenizer.encode(prefix + body[start:start + length] + suffix,
            add_special_tokens=False).ids)
        if count == 2048:
            return length
    return None


def prepare(model, corpus, suffix_dir, output):
    reject_backends()
    model, corpus, suffix_dir, output = map(Path, (model, corpus, suffix_dir, output))
    if (output / 'workload.json').exists():
        raise FileExistsError('Do not overwrite an earlier CPU-prepared workload')
    config_receipt = receipt(model / 'config.json')
    if config_receipt['sha256'] != CONFIG_SHA or receipt(corpus)['sha256'] != CORPUS_SHA:
        raise ValueError('Formal checkpoint config or existing formal corpus source changed')
    config = json.loads((model / 'config.json').read_text())
    text_config = config.get('text_config', config)
    if (text_config['num_hidden_layers'] != 40 or text_config['hidden_size'] != 5120 or
            text_config['n_routed_experts'] != 384 or text_config['num_experts_per_tok'] != 6 or
            text_config['dspark_block_size'] != 5 or not config.get('vision_config')):
        raise ValueError('Source config is not the formal full40/DSpark/vision checkpoint')
    tokenizer_config = json.loads((model / 'tokenizer_config.json').read_text())
    if (tokenizer_config['tokenizer_class'] != 'PreTrainedTokenizerFast' or
            tokenizer_config.get('add_bos_token') or tokenizer_config.get('add_eos_token')):
        raise ValueError('Unexpected tokenizer config; recheck raw Completion semantics')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    reject_backends()
    raw = corpus.read_bytes()
    whole = raw.decode('utf-8')
    marker = '========正文========'
    mark_at = whole.find(marker)
    body_at = mark_at + len(marker) if mark_at >= 0 else 0
    body = whole[body_at:]
    question_paths = {task: suffix_dir / ('local_' + task + '.txt') for task in TASKS}
    questions = {task: p.read_text().strip() for task, p in question_paths.items()}
    corpus_meta = receipt(corpus)
    corpus_meta['body_marker'] = marker
    corpus_meta['body_marker_present'] = mark_at >= 0
    corpus_meta['body_start_character'] = body_at
    corpus_meta['body_characters'] = len(body)
    corpus_meta['suffixes'] = {task: receipt(path) for task, path in question_paths.items()}
    files = [receipt(model / name) for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json')]
    workload = {'schema': 'dspark_2k256_CPU_real_tokenizer_v1', 'cpu_preparation_only': True,
        'cpu_mock': False, 'mock_tokenizer_used': False, 'mock_corpus_used': False,
        'CPU_preparation_does_not_grant_API_or_model_admission': True,
        'prompt_kind': 'real_text_tasks', 'output_budget': 256,
        'tokenizer_provenance': {'model': str(model), 'resolved_model': str(model.resolve()),
            'files': files, 'tokenizer_class_from_config': tokenizer_config['tokenizer_class'],
            'CPU_encoder': 'tokenizers.Tokenizer.from_file; add_special_tokens=False',
            'checkpoint_AutoTokenizer_parity': False,
            'checkpoint_AutoTokenizer_parity_status': 'unperformed; installed AutoTokenizer imports torch and was rejected by the strict CPU module guard',
            'transformers_version': importlib.metadata.version('transformers'),
            'tokenizers_version': importlib.metadata.version('tokenizers'),
            'vocab_size_with_added_tokens': tokenizer.get_vocab_size(with_added_tokens=True)},
        'corpus_provenance': corpus_meta, 'warmup': [], 'measured': [], **CLAIMS}
    audit = []
    spans = []
    output.mkdir(parents=True, exist_ok=True)
    (output / 'prompts').mkdir(exist_ok=True)
    for ordinal in range(10):
        phase = 'warmup' if ordinal < 2 else 'measured'
        index = ordinal if phase == 'warmup' else ordinal - 2
        name = phase + '_' + str(index)
        task = TASKS[ordinal % len(TASKS)]
        prefix = f'[CPU-prepared-request-{ordinal:02d}] 请阅读下面的小说节选，并只根据片段完成末尾任务。\n\n【节选】\n'
        appendix = '请将回答控制在约150字以内。'
        question = questions[task] + '\n' + appendix
        suffix = '\n\n【任务】\n' + question
        selected = None
        attempted_starts = set()
        for attempt in range(80):
            start = paragraph_start(body, ordinal * 12000 + attempt * 157)
            if start in attempted_starts:
                continue
            attempted_starts.add(start)
            sentence = first_sentence(body[start:start + 1200])
            if not sentence or not 12 <= len(sentence) <= 150:
                continue
            length = calibrate(tokenizer, body, prefix, suffix, start)
            if length is None:
                continue
            context = body[start:start + length]
            names = [person for person in NAMES if person in context]
            if task == 'qa' and not names:
                continue
            if any(not (start + length <= a or b <= start) for a, b in spans):
                continue
            selected = (start, length, context, sentence, names)
            break
        if selected is None:
            raise ValueError('Exact non-overlapping real-text slice could not be prepared: ' + name)
        start, length, context, sentence, names = selected
        full = prefix + context + suffix
        ids = tokenizer.encode(full, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        reencoded = tokenizer.encode(decoded, add_special_tokens=False).ids
        if (len(ids) != 2048 or ids != reencoded or decoded != full or
                not decoded.endswith(question) or '\ufffd' in decoded or
                max(ids) >= text_config['vocab_size']):
            raise ValueError('Actual tokenizer length/roundtrip/tail/parity check failed: ' + name)
        char_start, char_end = body_at + start, body_at + start + length
        byte_start = len(whole[:char_start].encode('utf-8'))
        byte_end = byte_start + len(context.encode('utf-8'))
        if raw[byte_start:byte_end].decode('utf-8') != context:
            raise ValueError('Excerpt does not match the original corpus byte interval')
        path = output / 'prompts' / (name + '.txt')
        path.write_text(full)
        excerpt_sha = sha(context.encode())
        text_sha = sha(full.encode())
        source_span = {'character_start': char_start, 'character_end': char_end,
            'byte_start': byte_start, 'byte_end': byte_end, 'context_sha256': excerpt_sha}
        row = {'name': name, 'phase': phase, 'index': index, 'task': task,
            'token_ids': ids, 'prompt_sha256': token_hash(ids), 'prompt_tokens': 2048,
            'output_budget': 256, 'text_path': str(path), 'text_sha256': text_sha,
            'question': question, 'question_source': corpus_meta['suffixes'][task],
            'question_appendix': appendix, 'source_span': source_span,
            'answerability': {'requires_external_book_content': False,
                'first_complete_sentence_witness': sentence,
                'named_characters_present_in_excerpt': names,
                'scope': 'task refers only to the exact visible excerpt; no model answer or semantic accuracy has been tested'}}
        workload[phase].append(row)
        audit.append({k: v for k, v in row.items() if k != 'token_ids'})
        spans.append((start, start + length))
    measured_hashes = {r['prompt_sha256'] for r in workload['measured']}
    if len(measured_hashes) != 8 or len({r['prompt_sha256'] for p in ('warmup', 'measured') for r in workload[p]}) != 10:
        raise ValueError('Prepared warmup/measurement inputs are not independent')
    reject_backends()
    (output / 'workload.json').write_text(json.dumps(workload, indent=2, ensure_ascii=False) + '\n')
    (output / 'prompt_audit.json').write_text(json.dumps({'CPU_only': True,
        'real_tokenizer_and_source_used': True, 'mock_tokenizer_used': False,
        'warmup_names': [r['name'] for r in workload['warmup']],
        'measured_names': [r['name'] for r in workload['measured']],
        'warmup_excluded_from_measured': True, 'source_context_intervals_nonoverlapping': True,
        'all_prompt_tokens': 2048, 'all_output_budgets': 256, 'prompts': audit, **CLAIMS},
        indent=2, ensure_ascii=False) + '\n')
    receipt_json = {'CPU_only': True, 'real_tokenizer_preparation_completed': True,
        'model_weight_files_opened': 0, 'services_started': 0, 'NPU_calls': 0,
        'actual_HTTP_inference_requests': 0, 'device_backend_imports': reject_backends(),
        'backend_environment': {k: os.environ[k] for k in CPU_ENV},
        'tokenizer_source': files, 'corpus_source': corpus_meta,
        'warmup_count': 2, 'measured_count': 8, 'unique_prompt_count': 10,
        'source_excerpt_bytes_in_total': sum(end - start for start, end in
            [(r['source_span']['byte_start'], r['source_span']['byte_end']) for phase in ('warmup', 'measured') for r in workload[phase]]),
        'full_corpus_exported': False, 'full_tokenizer_exported': False,
        'full_original_data_remains_remote': True, 'checkpoint_AutoTokenizer_parity_checks': 0,
        'checkpoint_AutoTokenizer_parity_status': 'unperformed; strict guard rejected CPU torch import',
        'encode_decode_reencode_checks': 10,
        'raw_text_roundtrip_checks': 20, 'question_tail_checks': 10,
        'workload_sha256': sha((output / 'workload.json').read_bytes()),
        'prompt_audit_sha256': sha((output / 'prompt_audit.json').read_bytes()),
        'generated_text_files': {str(p.relative_to(output)): receipt(p)
            for p in sorted(output.rglob('*')) if p.is_file()}, **CLAIMS}
    for p in output.rglob('*'):
        if p.is_file() and p.stat().st_size >= 1_000_000:
            raise ValueError('Generated transfer file reached 1MB: ' + str(p))
    (output / 'preparation_receipt.json').write_text(json.dumps(receipt_json, indent=2, ensure_ascii=False) + '\n')
    return {'remote_output': str(output), 'CPU_only': True, 'warmup_count': 2,
        'measured_count': 8, 'prompt_tokens': 2048, 'output_budget': 256,
        'workload_bytes': (output / 'workload.json').stat().st_size,
        'workload_sha256': receipt_json['workload_sha256'],
        'question_tail_checks': 10, 'checkpoint_AutoTokenizer_parity_checks': 0,
        'encode_decode_reencode_checks': 10,
        'backend_modules_loaded': [], 'NPU_calls': 0, 'HTTP_inference_requests': 0, **CLAIMS}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--suffix-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model, args.corpus, args.suffix_dir, args.output), indent=2))


if __name__ == '__main__':
    main()
