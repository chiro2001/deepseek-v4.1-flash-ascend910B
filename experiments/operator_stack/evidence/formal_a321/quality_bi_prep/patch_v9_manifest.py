"""Re-pin the v9 source manifest to the guarded bi_axis_compat shim."""
import hashlib
import json
import sys
from pathlib import Path

root = Path('/work/src_dspark_native_mls_bi_v1')
manifest_path = Path('/work/q2_candidate_v9/source_manifest_v1.json')

manifest = json.loads(manifest_path.read_text())
current = hashlib.sha256((root / 'stack/bi_axis_compat.py').read_bytes()).hexdigest()
manifest['source_sha256']['stack/bi_axis_compat.py'] = current
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({
    'compat_sha256': current,
    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'entries': len(manifest['source_sha256']),
}))
