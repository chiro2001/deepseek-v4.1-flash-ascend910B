"""Prepare real-text 2048-token Completion inputs using the actual tokenizer.

CPU only. This helper is deferred: no prompt set or tokenizer success is claimed
by the prepared bridge artifacts. Source corpus/tokenizer files stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .runtime_guards import token_hash


def prepare(model, corpus, suffix_dir, output, measured=8):
    if measured < 8:
        raise ValueError('At least eight measured prompts are required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Do not replace an earlier frozen prompt set')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    corpus = Path(corpus)
    body = corpus.read_text()
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    tasks = ['quote', 'qa', 'continue', 'extract']
    suffixes = {task: Path(suffix_dir) / ('local_' + task + '.txt') for task in tasks}
    source_files = [Path(model) / name for name in ('config.json', 'tokenizer.json',
        'tokenizer_config.json', 'special_tokens_map.json') if (Path(model) / name).exists()]
    if not (Path(model) / 'tokenizer.json').is_file():
        raise ValueError('Actual tokenizer.json provenance is required')
    workload = {'cpu_mock': False, 'prompt_kind': 'real_text_tasks',
        'tokenizer_provenance': {'model': str(Path(model).resolve()),
            'class': type(tokenizer).__module__ + '.' + type(tokenizer).__name__,
            'files': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}},
        'corpus_provenance': {'path': str(corpus.resolve()),
            'sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
            'suffixes': {task: {'path': str(p.resolve()),
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for task, p in suffixes.items()}},
        'warmup': [], 'measured': []}
    head = tokenizer.encode('请阅读以下小说节选，并完成节选之后的任务。\n\n', add_special_tokens=False)
    count = measured + 2
    if len(body_ids) < count * 2048:
        raise ValueError('Corpus is too short for ten independent real-text slices')
    for i in range(count):
        phase = 'warmup' if i < 2 else 'measured'
        index = i if i < 2 else i - 2
        task = tasks[i % len(tasks)]
        tail = tokenizer.encode('\n\n' + suffixes[task].read_text().strip(), add_special_tokens=False)
        budget = 2048 - len(head) - len(tail)
        if budget < 1600:
            raise ValueError('Task text leaves too little real context')
        start = i * 2048
        ids = [*head, *body_ids[start:start + budget], *tail]
        if len(ids) != 2048:
            raise ValueError('Exact real-text prompt budget could not be prepared')
        workload[phase].append({'name': phase + '_' + str(index), 'task': task,
            'corpus_token_offset': start, 'token_ids': ids,
            'prompt_sha256': token_hash(ids)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(workload, indent=2, ensure_ascii=False) + '\n')
    return workload


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--suffix-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--measured', type=int, default=8)
    args = parser.parse_args()
    workload = prepare(args.model, args.corpus, args.suffix_dir, args.output, args.measured)
    print(json.dumps({'prepared_on_cpu': True, 'http_requests': 0, 'npu_calls': 0,
        'warmup': len(workload['warmup']), 'measured': len(workload['measured'])}))


if __name__ == '__main__':
    main()
