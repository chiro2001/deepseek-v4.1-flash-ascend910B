"""Verify the exact source and completed formal reduction validation."""
import hashlib
import json
from pathlib import Path


def verify(root, checkpoint_sha256, physical_chips):
    root = Path(root)
    receipt = json.loads((root / 'validated_reduction_receipt.json').read_text())
    source = Path(__file__).with_name('decode_reduction_probe.py')
    assert receipt['passed'] and receipt['probe_source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt['checkpoint_config_sha256'] == checkpoint_sha256
    assert receipt['physical_chips'] == physical_chips
    for name, sha in receipt['files_sha256'].items():
        assert Path(name).name == name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    result = json.loads((root / 'result.json').read_text())
    assert result['passed'] and result['aa_passed'] and result['reduction_math_audit_passed']
    assert result['experimental_fp32_decode_reduction'] and result['eager']
    assert len(result['comparisons']) >= 3 and all(r['passed'] for r in result['comparisons'])
    audit = json.loads((root / 'decode_reduction_math_audit.json').read_text())
    assert {r['rank'] for r in audit} == set(range(8)) and len(audit) == 8
    assert all(r['passed'] and len(r['records']) == 80 and
               all(x['repaired_bitwise_equal_fp64_reference'] and x['repaired_max_abs'] == 0
                   for x in r['records']) for r in audit)
    return dict(receipt, evidence_root=str(root))
