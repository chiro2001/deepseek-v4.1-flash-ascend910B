"""CPU checkpoint/mmap tests; no Torch/Ascend or full-model proof."""
import importlib.util
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("shared_engram_cpu", Path(__file__).with_name("shared_engram.py"))
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
KEY, SCALE = "model.layers.3.engram.weight", "model.layers.3.engram.scale"


def checkpoint(root, mutate=None):
    codes = np.arange(64 * 256, dtype=np.int16).astype(np.int8).reshape(64, 256)
    scales = np.arange(1, 64 * 8 + 1, dtype=np.float32).reshape(64, 8)
    header = {KEY: {"dtype": "I8", "shape": [64, 256], "data_offsets": [0, codes.nbytes]},
              SCALE: {"dtype": "F32", "shape": [64, 8],
                      "data_offsets": [codes.nbytes, codes.nbytes + scales.nbytes]}}
    if mutate: mutate(header)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path = root / "tables.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + codes.tobytes() + scales.tobytes())
    (root / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {KEY: path.name, SCALE: path.name}}))
    return codes, scales


class SharedEngramTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="shared-engram-", dir=os.environ["TMPDIR"])
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_readonly_shared_row_views_match_both_checkpoint_planes(self):
        expected_codes, expected_scales = checkpoint(self.root)
        codes, scales = shared.map_checkpoint_rows(self.root, KEY, 64, 256, 8, 16)
        for mapping, dtype, expected in ((codes, np.int8, expected_codes), (scales, np.float32, expected_scales)):
            buffer = mapping.buffer()
            values = np.frombuffer(buffer, dtype=dtype).reshape(mapping.shape)
            np.testing.assert_array_equal(values, expected[8:16])
            self.assertFalse(values.flags.writeable)
            self.assertTrue(mapping.receipt["MAP_SHARED"])
            self.assertFalse(mapping.receipt["ACL_host_registered"])
            with self.assertRaises(ValueError): values[0, 0] = 7
            del values
            buffer.release()
            mapping.close()

    def test_p_d_independent_mappings_bind_same_inode_and_byte_range(self):
        checkpoint(self.root)
        p = shared.map_checkpoint_rows(self.root, KEY, 64, 256, 16, 24)
        d = shared.map_checkpoint_rows(self.root, KEY, 64, 256, 16, 24)
        for left, right in zip(p, d):
            self.assertIsNot(left.mapping, right.mapping)
            for field in ("device", "inode", "view_begin", "view_bytes", "header_sha256"):
                self.assertEqual(left.receipt[field], right.receipt[field])
            left.close(); right.close()

    def test_mapping_cannot_close_while_cpu_consumer_view_is_alive(self):
        checkpoint(self.root)
        codes, scales = shared.map_checkpoint_rows(self.root, KEY, 64, 256, 0, 8)
        buffer = codes.buffer()
        with self.assertRaises(BufferError): codes.close()
        buffer.release()
        codes.close(); scales.close()

    def test_bf16_source_cannot_masquerade_as_int8_mmap(self):
        checkpoint(self.root, lambda header: header[KEY].update(dtype="BF16"))
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            shared.map_checkpoint_rows(self.root, KEY, 64, 256, 0, 8)

    def test_wrong_scale_range_is_rejected_without_private_fallback(self):
        checkpoint(self.root, lambda header: header[SCALE].update(data_offsets=[16384, 16388]))
        with self.assertRaisesRegex(ValueError, "coverage|range/size"):
            shared.map_checkpoint_rows(self.root, KEY, 64, 256, 0, 8)

    def test_bounded_header_and_shard_negative_controls(self):
        codes, _ = checkpoint(self.root)
        with self.assertRaises(ValueError):
            shared.SharedTensorRows(self.root / "tables.safetensors", KEY, codes.shape, "I8", 0, 65)
        (self.root / "oversized.safetensors").write_bytes(struct.pack("<Q", shared.MAX_HEADER_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "header length"):
            shared.SharedTensorRows(self.root / "oversized.safetensors", KEY, codes.shape, "I8", 0, 8)

    def test_scale_alias_into_codes_is_not_a_valid_checkpoint(self):
        checkpoint(self.root, lambda header: header[SCALE].update(data_offsets=[1024, 3072]))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            shared.map_checkpoint_rows(self.root, KEY, 64, 256, 0, 8)


if __name__ == "__main__":
    unittest.main()
