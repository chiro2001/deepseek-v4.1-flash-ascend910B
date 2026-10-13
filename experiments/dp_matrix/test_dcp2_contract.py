"""CPU-only tests for the narrow native DP4/TP2/DCP2 Engram admission.

These tests deliberately avoid importing vLLM, torch, NPU/native modules,
starting a service, or touching HTTP/SSH. A passing contract is not a quality
or capacity receipt.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from dp_matrix.dcp2_contract import (  # noqa: E402
    derive_engram,
    validate_engram_config,
    validate_engram_groups,
)


FROZEN_ENGRAM = (
    PROJECT_ROOT
    / "experiments/delivery_matrix/dcp_runtime_20261013_v6/"
    "vllm_ascend/models/deepseek_v41/engram_hbm.py"
)
EXPECTED_ENVIRONMENT = {
    "DP_MATRIX_LAYOUT": "dp4-tp2-dcp2",
    "V41_DCP": "1",
    "V41_ENGRAM_REUSE_EP_GROUP": "1",
    "V41_ENGRAM_DEVICE_INDEX": "0",
    "V41_DCP_REPLICATE_INDEXER": "1",
}


def _parallel(dp_rank=0, **changes):
    values = {
        "data_parallel_size": 4,
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "data_parallel_rank": dp_rank,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _environment(**changes):
    values = dict(EXPECTED_ENVIRONMENT)
    values.update(changes)
    return values


class _Group:
    def __init__(self, world_size, ranks, rank_in_group):
        self.world_size = world_size
        self.ranks = ranks
        self.rank_in_group = rank_in_group


def _groups(dp_rank, tp_rank):
    global_rank = 2 * dp_rank + tp_rank
    return (
        _Group(8, list(range(8)), global_rank),
        _Group(2, [2 * dp_rank, 2 * dp_rank + 1], tp_rank),
        [list(range(8))],
    )


class DCP2ContractTests(unittest.TestCase):
    def test_each_native_dp_rank_accepts_its_exact_tp_and_ep_groups(self):
        for dp_rank in range(4):
            with self.subTest(dp_rank=dp_rank):
                self.assertEqual(
                    validate_engram_config(_parallel(dp_rank), _environment()),
                    dp_rank,
                )
                for tp_rank in range(2):
                    ep, tp, node_groups = _groups(dp_rank, tp_rank)
                    with self.subTest(tp_rank=tp_rank):
                        self.assertIsNone(
                            validate_engram_groups(ep, tp, node_groups, _parallel(dp_rank))
                        )

    def test_requires_exact_native_dp4_tp2_dcp2_ep8_configuration(self):
        mismatches = (
            ("data_parallel_size", 2),
            ("tensor_parallel_size", 4),
            ("decode_context_parallel_size", 1),
            ("pipeline_parallel_size", 2),
            ("prefill_context_parallel_size", 2),
        )
        for key, value in mismatches:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                ValueError, "DCP2 Engram configuration differs"
            ):
                validate_engram_config(_parallel(**{key: value}), _environment())
        for value in (False, 1):
            with self.subTest(key="enable_expert_parallel", value=value):
                with self.assertRaisesRegex(ValueError, "cross-DP eight-owner EP group"):
                    validate_engram_config(
                        _parallel(enable_expert_parallel=value), _environment()
                    )

        # The retired TP8=DCP8 exception must not be admitted by the new gate.
        retired = _parallel(
            data_parallel_size=1,
            tensor_parallel_size=8,
            decode_context_parallel_size=8,
        )
        with self.assertRaisesRegex(ValueError, "DCP2 Engram configuration differs"):
            validate_engram_config(retired, _environment())

    def test_rejects_boolean_non_integer_and_foreign_dp_ranks(self):
        for key, value in (
            ("data_parallel_size", True),
            ("tensor_parallel_size", True),
            ("decode_context_parallel_size", True),
            ("pipeline_parallel_size", True),
            ("prefill_context_parallel_size", True),
            ("data_parallel_rank", False),
            ("data_parallel_rank", True),
            ("data_parallel_rank", 1.0),
            ("data_parallel_rank", "1"),
            ("data_parallel_rank", -1),
            ("data_parallel_rank", 4),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_engram_config(_parallel(**{key: value}), _environment())

    def test_every_required_environment_setting_is_exact_and_string_valued(self):
        for key, expected in EXPECTED_ENVIRONMENT.items():
            wrong_values = (None, "", "0" if expected != "0" else "1", "true", True)
            for wrong in wrong_values:
                environment = _environment()
                if wrong is None:
                    environment.pop(key)
                else:
                    environment[key] = wrong
                with self.subTest(key=key, wrong=wrong), self.assertRaisesRegex(
                    ValueError, "DCP2 Engram environment differs"
                ):
                    validate_engram_config(_parallel(), environment)

    def test_rejects_multiple_nodes_and_foreign_or_malformed_node_rank_lists(self):
        ep, tp, _ = _groups(2, 1)
        invalid_node_groups = (
            [[0, 1, 2, 3], [4, 5, 6, 7]],  # multi-node split
            [[0, 1, 2, 3, 4, 5, 6, 8]],  # foreign rank
            [[0, 1, 2, 3, 4, 5, 7, 6]],  # wrong order
            [[0, 1, 2, 3, 4, 5, 6]],  # missing rank
            [[0, 1, 2, 3, 4, 5, 6, 7], [8]],
            [(0, 1, 2, 3, 4, 5, 6, 7)],  # wrong container type
        )
        for node_groups in invalid_node_groups:
            with self.subTest(node_groups=node_groups), self.assertRaisesRegex(
                ValueError, "one node, EP8|strict global integer ranks"
            ):
                validate_engram_groups(ep, tp, node_groups, _parallel(2))

    def test_rejects_wrong_ep_tp_world_sizes_and_rank_membership(self):
        valid_ep, valid_tp, node_groups = _groups(1, 0)
        cases = (
            (_Group(7, list(range(7)), 2), valid_tp),
            (_Group(8, [0, 1, 2, 3, 4, 5, 6, 8], 2), valid_tp),
            (_Group(8, list(reversed(range(8))), 2), valid_tp),
            (valid_ep, _Group(4, [2, 3, 4, 5], 0)),
            (valid_ep, _Group(2, [0, 1], 0)),  # foreign TP subgroup for DP1
            (valid_ep, _Group(2, [2, 3], 2)),  # invalid TP requester rank
            (_Group(8, [0, True, 2, 3, 4, 5, 6, 7], 2), valid_tp),
        )
        for ep, tp in cases:
            with self.subTest(ep_ranks=ep.ranks, tp_ranks=tp.ranks), self.assertRaises(
                ValueError
            ):
                validate_engram_groups(ep, tp, node_groups, _parallel(1))

    def test_rejects_inconsistent_actual_group_requester_ranks(self):
        for dp_rank, tp_rank in ((0, 0), (1, 1), (3, 0)):
            ep, tp, node_groups = _groups(dp_rank, tp_rank)
            for group, field, value in (
                (ep, "rank_in_group", True),
                (tp, "rank_in_group", True),
                (ep, "rank_in_group", (2 * dp_rank + tp_rank + 1) % 8),
                (tp, "rank_in_group", 2),
            ):
                altered = _Group(group.world_size, group.ranks, group.rank_in_group)
                setattr(altered, field, value)
                bad_ep, bad_tp = (altered, tp) if group is ep else (ep, altered)
                with self.subTest(dp_rank=dp_rank, tp_rank=tp_rank, field=field, value=value):
                    with self.assertRaises(ValueError):
                        validate_engram_groups(
                            bad_ep, bad_tp, node_groups, _parallel(dp_rank)
                        )

    def test_derivation_changes_only_engram_admission_and_preserves_lookup_bytes(self):
        original = FROZEN_ENGRAM.read_text()
        derived = derive_engram(original)
        self.assertIn(
            "from dp_matrix.dcp2_contract import validate_engram_config, validate_engram_groups",
            derived,
        )
        self.assertIn("and int(parallel.data_parallel_size) == 4", derived)
        self.assertNotIn(
            'and dcp == int(parallel.tensor_parallel_size) == 8)', derived
        )

        # Undoing exactly the two pinned admission edits must recover every
        # original byte, including real metadata all-gather and TP-local output
        # broadcasts. This is a positive and mutation-sensitive lookup control.
        old_local = (
            '        local_reused_dcp = (os.environ.get("V41_DCP") == "1"\n'
            '                            and dcp == int(parallel.tensor_parallel_size) == 8)\n'
        )
        new_local = (
            "        from dp_matrix.dcp2_contract import validate_engram_config, validate_engram_groups\n"
            "        validate_engram_config(parallel, os.environ)\n"
            "        local_reused_dcp = (dcp == int(parallel.tensor_parallel_size) == 2\n"
            "                            and int(parallel.data_parallel_size) == 4)\n"
        )
        old_group = (
            "        if dcp != 1 and (len(node_groups) != 1 or ep.world_size != tp.world_size or tp.world_size != 8):\n"
            '            raise ValueError("Engram DCP8 requires one node and exactly the same eight TP/EP ranks")\n'
        )
        new_group = "        validate_engram_groups(ep, tp, node_groups, parallel)\n"
        self.assertEqual(derived.count(new_local), 1)
        self.assertEqual(derived.count(new_group), 1)
        restored = derived.replace(new_local, old_local, 1).replace(new_group, old_group, 1)
        self.assertEqual(restored, original)

        # The DP>1 path remains ineligible for metadata-equivalence shortcuts;
        # metadata collectives and the requester TP-group broadcast are intact.
        self.assertIn("q.equiv_ranks = bool(dp == 1 and single_node)", derived)
        self.assertIn("dist.all_gather(gathered, metadata, group=q.cpu_group)", derived)
        self.assertIn("dist.all_gather_into_tensor(gathered_device, metadata_device, group=group)", derived)
        self.assertIn("dist.broadcast(result, src=q.tp_source, group=q.tp_group)", derived)
        self.assertIn("dist.broadcast(combined, src=q.tp_source, group=q.tp_group)", derived)

    def test_derivation_rejects_missing_or_duplicated_frozen_gate_anchors(self):
        original = FROZEN_ENGRAM.read_text()
        anchors = (
            '        local_reused_dcp = (os.environ.get("V41_DCP") == "1"\n'
            '                            and dcp == int(parallel.tensor_parallel_size) == 8)\n',
            "        if dcp != 1 and (len(node_groups) != 1 or ep.world_size != tp.world_size or tp.world_size != 8):\n"
            '            raise ValueError("Engram DCP8 requires one node and exactly the same eight TP/EP ranks")\n',
        )
        for anchor in anchors:
            missing = original.replace(anchor, "", 1)
            duplicated = original.replace(anchor, anchor + anchor, 1)
            with self.subTest(kind="missing", anchor=anchor[:30]):
                with self.assertRaisesRegex(ValueError, "exact frozen DCP8 Engram admission source"):
                    derive_engram(missing)
            with self.subTest(kind="duplicated", anchor=anchor[:30]):
                with self.assertRaisesRegex(ValueError, "exact frozen DCP8 Engram admission source"):
                    derive_engram(duplicated)


if __name__ == "__main__":
    unittest.main()
