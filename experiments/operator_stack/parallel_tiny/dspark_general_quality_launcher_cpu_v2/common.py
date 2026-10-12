"""Strict file provenance and checkpoint metadata helpers; stdlib only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys

from parallel_tiny.dspark_general_quality_api_cpu.contract import (
    COUNTER_SHA, async_arm, counter_sha, false_claims, fingerprint, validate_candidate,
    verify_source_files,
)

SAFE_ID = re.compile(r'^[A-Za-z0-9_-]{1,96}$')
CONFIG_SHA = '40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424'
TOKENIZER_SHA = 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b'
TOKENIZER_CONFIG_SHA = '6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_pinned(path, expected):
    if not re.fullmatch(r'[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError('Pinned manifest/source changed: ' + str(path))
    return json.loads(Path(path).read_text())


def source_map(manifest):
    raw = manifest.get('source_sha256', manifest.get('files', manifest.get('text_files')))
    if not isinstance(raw, dict) or not raw:
        raise ValueError('An explicit frozen source manifest is required')
    result = {}
    for name, item in raw.items():
        value = item.get('sha256') if isinstance(item, dict) else item
        relative = Path(name)
        if (not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) or
                relative.is_absolute() or not relative.parts or '..' in relative.parts):
            raise ValueError('Invalid source manifest path/SHA: ' + str(name))
        result[str(relative)] = value
    return result


def verify_map(root, mapping):
    root = Path(root).resolve(strict=True)
    for name, expected in mapping.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError('Frozen candidate source changed: ' + name)
    return mapping


def verify_sdk_manifest(manifest):
    references = {}
    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                path = Path(value['path'])
                if not path.is_absolute() or sha(path) != value['sha256']:
                    raise ValueError('Actual SDK/library source changed: ' + str(path))
                if 'bytes' in value and path.stat().st_size != value['bytes']:
                    raise ValueError('Actual SDK/library source size changed: ' + str(path))
                references[str(path)] = value['sha256']
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(manifest)
    if not references:
        raise ValueError('SDK manifest contains no actual pinned SDK/library files')
    return {'actual_source_files_readonly_verified': references, 'SDK_modules_imported': False,
            'NPU_calls': 0, **false_claims()}


def reject_fixture(candidate):
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('CPU_fixture_only', 'CPU_mock_candidate_identity_scores_and_ASGI') and item:
                    raise ValueError('CPU fixtures cannot become actual candidate contracts')
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(candidate)
    if str(candidate.get('schema', '')).startswith('CPU_fixture'):
        raise ValueError('CPU fixture schema is forbidden for actual candidate export')
    if any(candidate.get('actual_flags', {}).values()):
        raise ValueError('A source contract cannot inherit actual quality/performance/formal grants')


def guarded_python_source(candidate, suffix):
    root = Path(candidate['source_root']).resolve(strict=True)
    matches = [name for name in candidate['source_sha256'] if name == suffix or name.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError('The actual source is not uniquely frozen: ' + suffix)
    return root / matches[0]


def load_stdlib_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mtp_vision_header_metadata(model, index):
    """Read only the original indexed MTP/vision headers, never tensor payloads."""
    model = Path(model)
    selected = {name: shard for name, shard in index.items()
                if name.startswith(('vision.', 'aligner.', 'image_', 'mtp.'))}
    if not any(name.startswith('vision.') for name in selected):
        raise ValueError('The full checkpoint has no indexed original vision weights')
    by_shard = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)
    headers = {}
    for shard, names in sorted(by_shard.items()):
        path = model / shard
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < length <= 100_000_000 or length + 8 > path.stat().st_size:
                raise ValueError('Invalid original MTP/vision safetensors header: ' + shard)
            header = json.loads(stream.read(length))
        for name in names:
            if name not in header:
                raise ValueError('Indexed MTP/vision tensor is absent from the actual shard header: ' + name)
            tensor = header[name]
            offsets = tensor.get('data_offsets', [])
            shape = tensor.get('shape', [])
            if (len(offsets) != 2 or not all(type(item) is int for item in offsets) or
                    not 0 <= offsets[0] <= offsets[1] <= path.stat().st_size - 8 - length or
                    not isinstance(shape, list) or not all(type(item) is int and item > 0 for item in shape) or
                    not tensor.get('dtype')):
                raise ValueError('Original MTP/vision tensor metadata is invalid: ' + name)
            headers[name] = {'shard': shard, 'dtype': tensor['dtype'], 'shape': shape, 'data_offsets': offsets}
    return headers


def checkpoint_metadata(candidate):
    """Run the unchanged CPU header/symlink checker from this frozen source."""
    verify_source_files(candidate)
    model = Path(candidate['model_kwargs']['model'])
    required = {'config.json': CONFIG_SHA, 'tokenizer.json': TOKENIZER_SHA,
                'tokenizer_config.json': TOKENIZER_CONFIG_SHA}
    files = {}
    for name, expected in required.items():
        path = model / name
        actual = sha(path)
        if actual != expected:
            raise ValueError('The original checkpoint/tokenizer metadata changed: ' + name)
        files[name] = {'path': str(path), 'resolved': str(path.resolve(strict=True)),
                       'sha256': actual, 'bytes': path.stat().st_size}
    formal_path = guarded_python_source(candidate, 'formal_model_contract.py')
    contract_path = guarded_python_source(candidate, 'parallel_tiny/dspark_formal_prep/contract.py')
    formal = load_stdlib_source(formal_path, '_quality_frozen_formal_model_contract')
    contract = load_stdlib_source(contract_path, '_quality_frozen_dspark_contract')
    previous = sys.modules.get('formal_model_contract')
    sys.modules['formal_model_contract'] = formal
    try:
        inspection = contract.inspect_dspark_checkpoint(str(model), 5)
    finally:
        if previous is None:
            sys.modules.pop('formal_model_contract', None)
        else:
            sys.modules['formal_model_contract'] = previous
    for name in ('quant_model_weights.safetensors.index.json', 'quant_model_description.json'):
        path = model / name
        files[name] = {'path': str(path), 'resolved': str(path.resolve(strict=True)),
                       'sha256': sha(path), 'bytes': path.stat().st_size}
    index = json.loads((model / 'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    vision = {name: shard for name, shard in index.items()
              if name.startswith(('vision.', 'aligner.', 'image_'))}
    if not vision or not any(name.startswith('vision.') for name in vision):
        raise ValueError('The full checkpoint has no indexed original vision weights')
    headers = mtp_vision_header_metadata(model, index)
    return {'metadata_files': files, 'unchanged_original_checkpoint_checker': inspection,
            'vision_indexed_tensor_count': len(vision), 'vision_weight_map_fingerprint': fingerprint(vision),
            'indexed_MTP_and_vision_tensor_headers_verified': len(headers),
            'MTP_and_vision_tensor_header_fingerprint': fingerprint(headers),
            'shard_content_hashed': False, 'weight_tensor_contents_loaded': False,
            'actual_NPU_load_verified': False, **false_claims()}


def verify_basis(candidate):
    basis = candidate.get('startup_env_basis', {})
    if (basis.get('qualified_function') != 'parallel_tiny.dspark_formal_prep.contract.startup_env' or
            basis.get('audit') is not False or type(basis.get('draft_graph')) is not bool):
        raise ValueError('The original candidate startup_env basis must be explicit')
    path = Path(basis['source_path']).resolve(strict=True)
    if path != guarded_python_source(candidate, 'parallel_tiny/dspark_formal_prep/contract.py').resolve():
        raise ValueError('Startup environment basis belongs to another source tree')
    if sha(path) != basis['source_sha256']:
        raise ValueError('Original startup environment function source changed')
    original = load_stdlib_source(path, '_quality_frozen_environment_contract')
    expected = original.startup_env(candidate['startup_env']['STACK_DSPARK_FORMAL_OUTPUT'],
                                    audit=False, draft_graph=basis['draft_graph'])
    if any(candidate['startup_env'].get(key) != value for key, value in expected.items()):
        raise ValueError('Candidate startup switches are not its frozen original basis')
    spec = candidate['model_kwargs']['speculative_config']
    if spec.get('enforce_eager') is not (not basis['draft_graph']):
        raise ValueError('Candidate draft graph model config/environment disagree')
    return basis


def validate_real_candidate(candidate):
    reject_fixture(candidate)
    validate_candidate(candidate)
    verify_source_files(candidate)
    verify_basis(candidate)
    paths = candidate.get('pythonpath')
    root = Path(candidate['source_root']).resolve(strict=True)
    if not isinstance(paths, list) or not paths:
        raise ValueError('The original complete candidate PYTHONPATH must be provided')
    for value in paths:
        path = Path(value).resolve(strict=True)
        if not path.is_dir() or not path.is_relative_to(root):
            raise ValueError('Candidate PYTHONPATH escapes its frozen source tree')
    worker_module = candidate['model_kwargs']['worker_cls'].rsplit('.', 1)[0]
    suffix = worker_module.replace('.', '/') + '.py'
    guarded_python_source(candidate, suffix)
    if candidate['startup_env'].get('STACK_SRC_ROOT') != str(root):
        raise ValueError('Complete original startup environment must bind STACK_SRC_ROOT')
    visible = '8,9,10,11,12,13,14,15'
    if any(candidate['startup_env'].get(key) != visible for key in
           ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'STACK_PHYSICAL_CHIPS')):
        raise ValueError('Complete startup environment must bind the exact TP8 physical chips')
    return candidate


def relocate(value, old_root, new_root):
    """Relocate source paths only; model/SDK absolute owners remain unchanged."""
    old, new = str(old_root), str(new_root)
    if isinstance(value, str):
        return ':'.join(new + item[len(old):] if item == old or item.startswith(old + '/') else item
                        for item in value.split(':'))
    if isinstance(value, dict):
        return {key: relocate(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, old_root, new_root) for item in value]
    return copy.deepcopy(value)
