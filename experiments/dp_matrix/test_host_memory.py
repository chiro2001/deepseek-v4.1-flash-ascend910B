"""NUMA policy contracts with mocks only; no actual memory-policy changes."""
import unittest
from unittest.mock import patch
from experiments.dp_matrix.host_memory import ENV, node_list, verify_interleave


class HostMemoryTests(unittest.TestCase):
    def test_explicit_bounded_node_list(self):
        self.assertEqual(node_list("0,1,2,3,7"), [0, 1, 2, 3, 7])
        for bad in (None, True, "all", "0-7", "6", "0,0", "1,0", "0,1024", "0,01", "0,1;true"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                node_list(bad)

    def test_default_does_not_claim_or_query_policy(self):
        with patch.dict("os.environ", {}, clear=True), patch("experiments.dp_matrix.host_memory.current_policy") as query:
            self.assertFalse(verify_interleave()["policy_verified"])
            query.assert_not_called()

    def test_exact_interleave_required(self):
        with patch.dict("os.environ", {ENV: "0,1"}), patch(
                "experiments.dp_matrix.host_memory.current_policy", return_value={"mode": 3, "nodes": [0, 1]}):
            self.assertTrue(verify_interleave()["policy_verified"])
        for row in ({"mode": 0, "nodes": []}, {"mode": 2, "nodes": [0, 1]},
                    {"mode": 3, "nodes": [0, 1, 6]}):
            with self.subTest(row=row), patch("experiments.dp_matrix.host_memory.current_policy", return_value=row):
                with self.assertRaises(ValueError):
                    verify_interleave("0,1")
