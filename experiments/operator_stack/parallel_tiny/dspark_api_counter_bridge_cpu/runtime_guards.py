"""Strict source/clock guards; importing this module performs no runtime work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

COUNTER_SHA256 = 'da845a5891557f21e80eac06d96aec1955263d1ae93afbc4d84e86dd9585dbec'
FORMAL_WORKER_SHA256 = '00811b9b83e69f6fd359e2577c2536875ce46e5effb21938f25b90cb5426ac5c'
HERE = Path(__file__).resolve().parent
SOURCE_GUARDS = HERE / 'evidence' / 'installed_source_guards.json'


def monotonic_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def clock_scope():
    return {'clock': 'CLOCK_MONOTONIC',
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'time_namespace': os.readlink('/proc/self/ns/time')}


def token_list(value, *, size=None):
    if type(value) not in (list, tuple) or any(type(t) is not int or t < 0 for t in value):
        raise ValueError('Only existing CPU token IDs are accepted')
    result = list(value)
    if size is not None and len(result) != size:
        raise ValueError(f'Expected exactly {size} tokens')
    return result


def token_hash(value):
    return hashlib.sha256(json.dumps(token_list(value), separators=(',', ':')).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_installed_sources(path=SOURCE_GUARDS):
    guards = json.loads(Path(path).read_text())
    if not guards['source_only'] or guards['device_calls'] != 0:
        raise ValueError('Source guard artifact is not source-only')
    for item in guards['files']:
        if any(node.get('missing') for node in item['fragments'].values()):
            raise ValueError('An actual ABI fragment has not been discovered')
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError(f"Installed source changed: {item['path']}")
    return {'source_guards_sha256': sha256_file(path), 'guarded_files': len(guards['files'])}


def load_frozen_counter(path=None):
    path = Path(path or os.environ['DSPARK_API_FROZEN_COUNTER']).resolve()
    if sha256_file(path) != COUNTER_SHA256:
        raise ValueError('Exact stock_compat_v4 counter source is required')
    existing = sys.modules.get('formal_dspark_counters')
    if existing is not None:
        if Path(existing.__file__).resolve() != path or sha256_file(existing.__file__) != COUNTER_SHA256:
            raise ValueError('The formal_dspark_counters import is already bound to another source')
        return existing
    spec = importlib.util.spec_from_file_location('formal_dspark_counters', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def false_claims():
    return {'actual_api_performance_observed': False, 'actual_api_quality_observed': False,
        'actual_client_quality_passed': False, 'actual_device_consumer_proof': False,
        'performance_receipt_granted': False, 'formal_receipt_granted': False}
