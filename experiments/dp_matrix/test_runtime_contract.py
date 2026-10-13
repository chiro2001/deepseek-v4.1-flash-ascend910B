"""CPU-only tests for new DP geometry gates; no native/runtime initialization."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.dp_matrix.runtime_contract import expected_tp_size, validate_worker_config
from experiments.dp_matrix.topology import CHIPS


def _parallel(layout: str, **changes):
    dimensions = {
        "dp2-tp4": (2, 4, 1, 0),
        "dp4-tp2": (4, 2, 1, 0),
        "dp4-tp2-dcp2": (4, 2, 2, 0),
    }[layout]
    dp, tp, dcp, dp_rank = dimensions
    values = {
        "data_parallel_size": dp,
        "data_parallel_size_local": dp,
        "tensor_parallel_size": tp,
        "decode_context_parallel_size": dcp,
        "data_parallel_rank": dp_rank,
        "enable_expert_parallel": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _environment(layout: str, **changes):
    values = {
        "DP_MATRIX_LAYOUT": layout,
        "STACK_PHYSICAL_CHIPS": ",".join(map(str, CHIPS)),
    }
    values.update(changes)
    return values


class RuntimeContractTests(unittest.TestCase):
    def test_accepts_only_exact_requested_worker_geometries(self):
        for layout in ("dp2-tp4", "dp4-tp2", "dp4-tp2-dcp2"):
            with self.subTest(layout=layout):
                result = validate_worker_config(
                    SimpleNamespace(parallel_config=_parallel(layout)),
                    _environment(layout),
                )
                self.assertEqual(result.name, layout)

    def test_expected_tp_size_uses_explicit_layout(self):
        for layout, expected in (("dp2-tp4", 4), ("dp4-tp2", 2), ("dp4-tp2-dcp2", 2)):
            with patch.dict(os.environ, {"DP_MATRIX_LAYOUT": layout}, clear=False):
                self.assertEqual(expected_tp_size(), expected)

        with patch.dict(os.environ, {"DP_MATRIX_LAYOUT": "old-tp8"}, clear=False):
            with self.assertRaisesRegex(ValueError, "Unknown requested layout"):
                expected_tp_size()

    def test_rejects_old_tp8_tag_even_when_new_dimensions_match(self):
        for tag in ("tp8base", "", "legacy"):
            with self.subTest(tag=tag), self.assertRaisesRegex(
                ValueError, "must not inherit the TP8 arm tag"
            ):
                validate_worker_config(
                    SimpleNamespace(parallel_config=_parallel("dp2-tp4")),
                    _environment("dp2-tp4", STACK_TP8_ARM=tag),
                )

    def test_rejects_each_wrong_geometry_field(self):
        cases = (
            ("dp2-tp4", "data_parallel_size", 4),
            ("dp2-tp4", "data_parallel_size_local", 1),
            ("dp2-tp4", "tensor_parallel_size", 8),
            ("dp4-tp2", "decode_context_parallel_size", 2),
            ("dp4-tp2-dcp2", "decode_context_parallel_size", 1),
            ("dp4-tp2", "data_parallel_rank", 4),
        )
        for layout, key, value in cases:
            message = "Foreign data-parallel rank" if key == "data_parallel_rank" else "Actual worker topology differs"
            with self.subTest(layout=layout, key=key), self.assertRaisesRegex(ValueError, message):
                validate_worker_config(
                    SimpleNamespace(parallel_config=_parallel(layout, **{key: value})),
                    _environment(layout),
                )

    def test_rejects_bool_and_float_integer_fields(self):
        fields = (
            "data_parallel_size",
            "data_parallel_size_local",
            "tensor_parallel_size",
            "decode_context_parallel_size",
            "data_parallel_rank",
        )
        for field in fields:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "integer"):
                validate_worker_config(
                    SimpleNamespace(parallel_config=_parallel("dp4-tp2", **{field: True})),
                    _environment("dp4-tp2"),
                )

        with self.assertRaisesRegex(ValueError, "integer"):
            validate_worker_config(
                SimpleNamespace(parallel_config=_parallel("dp4-tp2", tensor_parallel_size=2.0)),
                _environment("dp4-tp2"),
            )

    def test_requires_real_boolean_expert_parallel(self):
        for value in (False, None, 1, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "Cross-DP native expert parallelism is required"
            ):
                validate_worker_config(
                    SimpleNamespace(parallel_config=_parallel("dp2-tp4", enable_expert_parallel=value)),
                    _environment("dp2-tp4"),
                )

    def test_rejects_noncanonical_or_foreign_chip_placement(self):
        for chips in (
            "0,1,2,3,4,5,6,7",
            "8,9,10,11,12,13,14,14",
            "9,8,10,11,12,13,14,15",
            "8,9,10,11,12,13,14",
        ):
            with self.subTest(chips=chips), self.assertRaisesRegex(
                ValueError, "Explicit eight-chip DP placement"
            ):
                validate_worker_config(
                    SimpleNamespace(parallel_config=_parallel("dp2-tp4")),
                    _environment("dp2-tp4", STACK_PHYSICAL_CHIPS=chips),
                )

    def test_module_import_is_cpu_only(self):
        code = """
import sys
import experiments.dp_matrix.runtime_contract
import experiments.dp_matrix.native_rpc
forbidden = ("torch", "torch_npu", "vllm", "httpx", "requests", "urllib.request")
loaded = [name for name in sys.modules if any(name == item or name.startswith(item + ".") for item in forbidden)]
if loaded:
    raise SystemExit("forbidden modules imported: " + repr(loaded))
"""
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
