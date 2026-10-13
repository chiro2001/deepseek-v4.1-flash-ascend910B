"""CPU-only CLI and physical-identity binding checks for HTTP clients."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.dp_matrix import benchmark, canary, code8_replay, code8_score
from experiments.dp_matrix.topology import (
    DEFAULT_CHIP_LIST, LOW_CHIPS, Topology, topology_from_args,
)


TASK_TMP_ROOT = Path(os.environ.get("TMPDIR") or
                     (Path.home() / "tmp" / "dsv41-dp-matrix-client-cli-tests"))
TASK_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class ClientPhysicalChipsCliTests(unittest.TestCase):
    @staticmethod
    def capture(module, entrypoint, arguments):
        captured = []
        with patch.object(module, entrypoint,
                          side_effect=lambda args: captured.append(args)), \
                patch.object(sys, "argv", ["dp-client", *arguments]):
            module.main()
        if len(captured) != 1:
            raise AssertionError("CLI did not reach its mocked runner exactly once")
        return captured[0]

    @staticmethod
    def explicit_arguments(module_name, physical_chips=None):
        common = ["--model", "dsv41-dp-cli", "--layout", "dp2-tp4",
                  "--sid", "a" * 32, "--fingerprint", "b" * 64]
        if module_name == "benchmark":
            args = ["--base-url", "http://127.0.0.1:18960", *common,
                    "--manifest", "/inputs/manifest.json", "--manifest-sha256", "c" * 64,
                    "--output", "/runs/benchmark", "--mode", "serial", "--input-tokens", "2048"]
            module, entrypoint = benchmark, "run"
        elif module_name == "canary":
            args = ["--base-url", "http://127.0.0.1:18960", *common,
                    "--manifest", "/inputs/manifest.json", "--manifest-sha256", "c" * 64,
                    "--output", "/runs/canary"]
            module, entrypoint = canary, "run"
        elif module_name == "code8_replay":
            args = ["--base-url", "http://127.0.0.1:18960", *common,
                    "--manifest", "/inputs/manifest.json", "--requests", "/inputs/requests",
                    "--source-root", "/frozen/sources", "--output", "/runs/replay"]
            module, entrypoint = code8_replay, "replay"
        elif module_name == "code8_score":
            args = ["--manifest", "/inputs/manifest.json", "--requests", "/inputs/requests",
                    "--responses", "/runs/replay", "--workloads", "/frozen/workloads.py",
                    "--output", "/runs/score", *common, "--scope", "ordinary_quality"]
            module, entrypoint = code8_score, "score"
        else:
            raise AssertionError("unknown client module")
        if physical_chips is not None:
            args.extend(["--physical-chips", physical_chips])
        return module, entrypoint, args

    @staticmethod
    def identity(args, chips):
        topology = Topology.named(args.layout, chips=chips)
        rows = []
        for index, row in enumerate(topology.workers()):
            tp_start = row["dp_rank"] * topology.tp
            dcp_start = tp_start + (row["tp_rank"] // topology.dcp) * topology.dcp
            rows.append({
                **row, "dp": topology.dp, "tp": topology.tp, "dcp": topology.dcp,
                "ep_world_size": 8, "enable_expert_parallel": True,
                "worker_pid": 200 + index,
                "tp_group_ranks": list(range(tp_start, tp_start + topology.tp)),
                "dcp_group_ranks": list(range(dcp_start, dcp_start + topology.dcp)),
                "ep_group_ranks": list(range(8)),
            })
        return {
            "served_model": args.model, "service_instance_id": args.sid,
            "runtime_fingerprint": args.fingerprint, "layout": args.layout,
            "engine_initialized": True, "ASGI_startup_completed": True,
            "rank_descriptions": rows,
        }

    def test_each_http_client_cli_accepts_explicit_zero_through_seven(self):
        selected = ",".join(map(str, LOW_CHIPS))
        for name in ("benchmark", "canary", "code8_replay", "code8_score"):
            with self.subTest(client=name):
                module, entrypoint, arguments = self.explicit_arguments(name, selected)
                args = self.capture(module, entrypoint, arguments)
                self.assertEqual(args.physical_chips, selected)
                self.assertEqual(topology_from_args(args).chips, LOW_CHIPS)

    def test_http_client_cli_defaults_to_legacy_group_and_explicit_group_changes_topology(self):
        for name in ("benchmark", "canary", "code8_replay", "code8_score"):
            with self.subTest(client=name):
                module, entrypoint, arguments = self.explicit_arguments(name)
                args = self.capture(module, entrypoint, arguments)
                self.assertEqual(args.physical_chips, DEFAULT_CHIP_LIST)
                self.assertEqual(topology_from_args(args).chips, tuple(range(8, 16)))

    def test_invalid_client_chip_group_fails_before_runner(self):
        name = "benchmark"
        module, entrypoint, arguments = self.explicit_arguments(name, "0,1,2,3,4,5,6,8")
        with patch.object(module, "http_json", side_effect=AssertionError("HTTP before placement gate")), \
                patch.object(sys, "argv", ["dp-client", *arguments]):
            with self.assertRaises(ValueError):
                module.main()

    def test_canary_and_replay_reject_cross_group_identity_before_generation(self):
        selected = ",".join(map(str, LOW_CHIPS))
        with tempfile.TemporaryDirectory(prefix="client-cli-", dir=TASK_TMP_ROOT) as directory:
            root = Path(directory)
            canary_module, _entrypoint, canary_argv = self.explicit_arguments("canary", selected)
            canary_args = self.capture(canary_module, "run", canary_argv)
            canary_args.output = str(root / "canary")
            legacy_identity = self.identity(canary_args, tuple(range(8, 16)))
            canary_calls = []
            with patch.object(canary, "load_frozen_dihuo_inputs", return_value={"samples": []}), \
                    patch.object(canary, "http_json",
                                 side_effect=lambda base, path, **kwargs:
                                 (canary_calls.append(path), legacy_identity)[1]):
                with self.assertRaisesRegex(ValueError, "Unexpected worker physical_chip"):
                    canary.run(canary_args)
            self.assertEqual(canary_calls, ["/dp/identity"])

            replay_module, _entrypoint, replay_argv = self.explicit_arguments("code8_replay", selected)
            replay_args = self.capture(replay_module, "replay", replay_argv)
            replay_args.output = str(root / "replay")
            replay_calls = []
            with patch.object(code8_replay, "load_inputs", return_value={"samples": []}), \
                    patch.object(code8_replay, "load_frozen_client", return_value=object()), \
                    patch.object(code8_replay, "http_json",
                                 side_effect=lambda base, path, *args:
                                 (replay_calls.append(path), legacy_identity)[1]):
                with self.assertRaisesRegex(ValueError, "Unexpected worker physical_chip"):
                    code8_replay.replay(replay_args)
            self.assertEqual(replay_calls, ["/dp/identity"])


if __name__ == "__main__":
    unittest.main()
