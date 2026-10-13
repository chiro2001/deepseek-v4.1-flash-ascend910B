import importlib.util
import mmap
import os
from pathlib import Path
import sys
import unittest
import hashlib

spec = importlib.util.spec_from_file_location("kv_content_pool_cpu", Path(__file__).with_name("kv_content_pool.py"))
pool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pool
spec.loader.exec_module(pool)


class ContentPoolTests(unittest.TestCase):
    def test_eight_rank_aliases_store_exactly_one_payload(self):
        with pool.KVContentPool(128) as arena:
            refs = [arena.put((rank, "group0", "page0"), b"same actual KV bytes") for rank in range(8)]
            self.assertEqual(len({ref["content_id"] for ref in refs}), 1)
            self.assertEqual(arena.stats()["allocated_span_bytes"], 64)
            self.assertEqual(arena.stats()["logical_aliases"], 8)
            self.assertEqual(arena.stats()["unique_contents"], 1)

    def test_hash_collision_cannot_merge_different_payloads(self):
        with pool.KVContentPool(128, digest=lambda _: "synthetic-collision") as arena:
            first, second = arena.put("a", b"AAAA"), arena.put("b", b"BBBB")
            self.assertNotEqual(first["content_id"], second["content_id"])
            with arena.acquire("a") as value: self.assertEqual(value, b"AAAA")
            with arena.acquire("b") as value: self.assertEqual(value, b"BBBB")

    def test_pool_full_rolls_back_alias_remap(self):
        with pool.KVContentPool(64) as arena:
            arena.put("alias", b"original")
            with self.assertRaises(pool.PoolFull): arena.put("alias", b"replacement")
            with arena.acquire("alias") as value: self.assertEqual(value, b"original")
            self.assertEqual(arena.stats()["unique_contents"], 1)

    def test_read_lease_pins_old_payload_across_alias_remap(self):
        with pool.KVContentPool(128) as arena:
            arena.put("alias", b"original")
            lease = arena.acquire("alias")
            arena.put("alias", b"replacement")
            self.assertEqual(lease.view, b"original")
            self.assertTrue(lease.view.readonly)
            self.assertEqual(arena.stats()["unique_contents"], 2)
            with self.assertRaises(RuntimeError): arena.close()
            lease.close(); lease.close()
            self.assertEqual(arena.stats()["unique_contents"], 1)
            with arena.acquire("alias") as value: self.assertEqual(value, b"replacement")

    def test_free_spans_coalesce_and_payload_budget_is_reusable(self):
        with pool.KVContentPool(192) as arena:
            for key in range(3): arena.put(key, bytes([key]) * 64)
            arena.drop(0); arena.drop(1)
            arena.put("wide", b"x" * 128)
            self.assertEqual(arena.stats()["allocated_span_bytes"], 192)
            arena.drop("wide"); arena.drop(2)
            self.assertEqual(arena.stats()["free_span_bytes"], 192)

    def test_peer_fd_is_same_inode_and_cannot_map_writable(self):
        with pool.KVContentPool(128) as arena:
            ref = arena.put("a", b"payload")
            fd = arena.readonly_fd()
            try:
                self.assertEqual(os.fstat(fd).st_ino, arena.stats()["inode"])
                with self.assertRaises(PermissionError):
                    mmap.mmap(fd, 128, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
                mapped = mmap.mmap(fd, 128, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
                self.assertEqual(mapped[ref["offset"]:ref["offset"] + ref["length"]], b"payload")
                mapped.close()
            finally:
                os.close(fd)

    def test_empty_payload_and_bad_geometry_are_rejected(self):
        with self.assertRaises(ValueError): pool.KVContentPool(True)
        with self.assertRaises(ValueError): pool.KVContentPool(128, alignment=3)
        with pool.KVContentPool(128) as arena:
            with self.assertRaises(ValueError): arena.put("empty", b"")
            self.assertEqual(arena.stats()["allocated_span_bytes"], 0)

    def test_mutating_dma_source_cannot_commit_a_wrong_digest(self):
        source = bytearray(b"AAAA")
        changed = False
        def digest(value):
            nonlocal changed
            actual = hashlib.sha256(value).hexdigest()
            if not changed:
                source[0] = ord("B")
                changed = True
            return actual
        with pool.KVContentPool(128, digest=digest) as arena:
            with self.assertRaisesRegex(ValueError, "changed"):
                arena.put("unstable", source)
            self.assertEqual(arena.stats()["logical_aliases"], 0)
            self.assertEqual(arena.stats()["free_span_bytes"], 128)


if __name__ == "__main__":
    unittest.main()
