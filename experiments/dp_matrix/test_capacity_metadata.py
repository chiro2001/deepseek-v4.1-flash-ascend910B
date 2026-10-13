"""CPU-only cache backing-storage metadata accounting with duck tensors."""
from __future__ import annotations

import unittest

from experiments.dp_matrix.capacity_metadata import backing_storages


class FakeStorage:
    def __init__(self, pointer, size):
        self.pointer = pointer
        self.size = size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class FakeTensor:
    def __init__(self, device, storage):
        self.device = device
        self._storage = storage

    def untyped_storage(self):
        return self._storage

    def _forbidden_value_read(self, *_args, **_kwargs):
        raise AssertionError("capacity metadata must not read tensor values")

    cpu = _forbidden_value_read
    item = _forbidden_value_read
    numpy = _forbidden_value_read
    tolist = _forbidden_value_read


class CapacityMetadataTests(unittest.TestCase):
    def test_recurses_through_v41_indexer_mapping_tuple_and_scales(self):
        shared_full = FakeStorage(0x1000, 4096)
        full_a = FakeTensor("npu:0", shared_full)
        full_alias = FakeTensor("npu:0", shared_full)
        indexer_keys = FakeTensor("npu:0", FakeStorage(0x2000, 1024))
        indexer_scales = FakeTensor("npu:0", FakeStorage(0x3000, 128))
        caches = {
            "full_attention": [full_a, (None, full_alias)],
            "indexer": {
                "keys": (indexer_keys,),
                "scales": (indexer_scales, None),
            },
        }

        rows = backing_storages(caches)

        self.assertEqual(
            {(row["device"], row["data_ptr"], row["size_bytes"]) for row in rows},
            {
                ("npu:0", 0x1000, 4096),
                ("npu:0", 0x2000, 1024),
                ("npu:0", 0x3000, 128),
            },
        )
        self.assertEqual(len(rows), 3)

    def test_deduplicates_by_device_and_pointer(self):
        storage = FakeStorage(77, 512)
        rows = backing_storages([
            FakeTensor("npu:0", storage),
            {"alias": (FakeTensor("npu:0", storage),)},
        ])
        self.assertEqual(rows, [{"device": "npu:0", "data_ptr": 77, "size_bytes": 512}])

    def test_same_pointer_on_different_devices_is_distinct_storage(self):
        rows = backing_storages([
            FakeTensor("npu:0", FakeStorage(99, 256)),
            FakeTensor("npu:1", FakeStorage(99, 256)),
        ])
        self.assertEqual({row["device"] for row in rows}, {"npu:0", "npu:1"})
        self.assertEqual(len(rows), 2)

    def test_same_device_and_pointer_with_conflicting_sizes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting sizes"):
            backing_storages([
                FakeTensor("npu:0", FakeStorage(55, 256)),
                FakeTensor("npu:0", FakeStorage(55, 512)),
            ])

    def test_none_empty_and_unknown_objects_fail_closed(self):
        for caches in (None, [], (), {}, [None, (None, {})]):
            with self.subTest(caches=caches), self.assertRaises(ValueError):
                backing_storages(caches)

        with self.assertRaisesRegex(ValueError, "Unknown cache backing object"):
            backing_storages({"keys": object()})

    def test_boolean_or_zero_storage_pointer_is_rejected(self):
        for pointer in (True, False, 0, -1, None):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                backing_storages(FakeTensor("npu:0", FakeStorage(pointer, 256)))

    def test_boolean_or_nonpositive_storage_size_is_rejected(self):
        for size in (True, False, 0, -1, None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                backing_storages(FakeTensor("npu:0", FakeStorage(100, size)))

    def test_tensor_value_conversion_apis_are_never_called(self):
        tensor = FakeTensor("npu:0", FakeStorage(123, 456))
        rows = backing_storages({"main": tensor, "nested": ((None,),)})
        self.assertEqual(rows, [{"device": "npu:0", "data_ptr": 123, "size_bytes": 456}])


if __name__ == "__main__":
    unittest.main()
