"""Independent checkpoint AutoTokenizer parity; CPU torch is allowed.

Device backend autoload/visibility remain disabled. An import finder rejects
NPU SDK, vLLM and ACL imports before they happen. No tensor/model/device/HTTP
operation is performed. Only tokenizer/config and ten bounded text inputs read.
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
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys

FORBIDDEN = ('torch_npu', 'vllm', 'vllm_ascend', 'acl', 'torchair')
CLAIMS = {'actual_model_inference_executed': False, 'actual_api_performance_observed': False,
    'actual_api_quality_observed': False, 'actual_client_quality_passed': False,
    'actual_device_consumer_proof': False, 'performance_receipt_granted': False,
    'formal_receipt_granted': False}


class NoNPUImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.blocked = []

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in FORBIDDEN):
            self.blocked.append(fullname)
            raise ImportError('CPU parity forbids importing NPU/vLLM/ACL: ' + fullname)
        return None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_hash(ids):
    return sha(json.dumps(ids, separators=(',', ':')).encode())


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    guard = NoNPUImports()
    sys.meta_path.insert(0, guard)
    workload_path = args.output / 'workload.json'
    initial_sha = sha(workload_path.read_bytes())
    workload = json.loads(workload_path.read_text())
    model = Path(workload['tokenizer_provenance']['model'])
    for file in workload['tokenizer_provenance']['files']:
        if sha(Path(file['path']).read_bytes()) != file['sha256']:
            raise ValueError('The actual checkpoint tokenizer/config source changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=False,
        local_files_only=True)
    checks = []
    for phase in ('warmup', 'measured'):
        for row in workload[phase]:
            raw = Path(row['text_path']).read_bytes()
            if sha(raw) != row['text_sha256']:
                raise ValueError('Prepared prompt text changed')
            text = raw.decode('utf-8')
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            if (ids != row['token_ids'] or len(ids) != 2048 or decoded != text or
                    not decoded.endswith(row['question']) or token_hash(ids) != row['prompt_sha256']):
                raise ValueError('Actual checkpoint AutoTokenizer parity failed: ' + row['name'])
            checks.append({'name': row['name'], 'phase': phase, 'prompt_tokens': len(ids),
                'token_ids_equal_to_Rust_JSON_backend': True, 'full_text_decode_equal': True,
                'question_tail_equal': True, 'prompt_sha256': token_hash(ids)})
    loaded = [name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in FORBIDDEN)]
    maps = Path('/proc/self/maps').read_text()
    npu_libraries = sorted({line.split()[-1] for line in maps.splitlines()
        if any(label in line.lower() for label in ('libtorch_npu', 'libascendcl', 'libhccl', 'libopapi'))})
    if loaded or guard.blocked or npu_libraries:
        raise ValueError('NPU import/library appeared in the CPU parity process')
    if any(os.environ.get(key) != value for key, value in CPU_ENV.items()):
        raise ValueError('The CPU backend/visibility guard changed')
    receipt = {'CPU_only': True, 'CPU_torch_import_authorized': True,
        'CPU_torch_imported': 'torch' in sys.modules,
        'CPU_torch_version_if_imported': getattr(sys.modules.get('torch'), '__version__', None),
        'mock_tokenizer_used': False, 'checkpoint_AutoTokenizer_parity_completed': True,
        'tokenizer_class': type(tokenizer).__module__ + '.' + type(tokenizer).__name__,
        'transformers_version': importlib.metadata.version('transformers'),
        'actual_model_weights_loaded': False, 'NPU_calls': 0,
        'actual_HTTP_inference_requests': 0, 'services_started': 0,
        'NPU_vLLM_ACL_module_imports': loaded, 'blocked_import_attempts': guard.blocked,
        'NPU_runtime_libraries_mapped': npu_libraries, 'backend_environment': CPU_ENV,
        'initial_Rust_prepared_workload_sha256': initial_sha,
        'warmup_count': 2, 'measured_count': 8, 'checks': checks, **CLAIMS}
    receipt_path = args.output / 'AutoTokenizer_parity_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    workload['tokenizer_provenance']['checkpoint_AutoTokenizer_parity'] = True
    workload['tokenizer_provenance']['checkpoint_AutoTokenizer_parity_status'] = 'passed on CPU for all ten prepared prompts; actual model/API admission remains false'
    workload['tokenizer_provenance']['checkpoint_AutoTokenizer_parity_receipt_sha256'] = sha(receipt_path.read_bytes())
    workload_path.write_text(json.dumps(workload, indent=2, ensure_ascii=False) + '\n')
    final = {'CPU_only': True, 'workload_sha256': sha(workload_path.read_bytes()),
        'workload_bytes': workload_path.stat().st_size,
        'AutoTokenizer_parity_receipt_sha256': sha(receipt_path.read_bytes()),
        'warmup_count': 2, 'measured_count': 8, 'prompt_tokens': 2048, 'output_budget': 256,
        'all_ten_checkpoint_AutoTokenizer_encoding_and_decoding_checks_passed': True,
        'CPU_torch_imported': receipt['CPU_torch_imported'],
        'NPU_vLLM_ACL_imports': [], 'NPU_calls': 0, 'HTTP_inference_requests': 0,
        'full_tokenizer_or_corpus_transferred': False, **CLAIMS}
    (args.output / 'final_preparation_receipt.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
