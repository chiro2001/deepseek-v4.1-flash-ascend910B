"""Re-pin the v9 seed's embedded source map to the guarded shim."""
import hashlib
import json
from pathlib import Path

root = Path('/work/src_dspark_native_mls_bi_v1')
seed_path = Path('/work/q2_candidate_v9/candidate_seed_v1.json')
seed = json.loads(seed_path.read_text())
current = hashlib.sha256((root / 'stack/bi_axis_compat.py').read_bytes()).hexdigest()
seed['source_sha256']['stack/bi_axis_compat.py'] = current
seed_path.write_text(json.dumps(seed, indent=2) + '\n')
print(json.dumps({'seed_sha256': hashlib.sha256(seed_path.read_bytes()).hexdigest(),
                  'compat_prefix': current[:16]}))
