"""No torch, native import, HTTP or NPU initialization."""
import copy
import unittest

from experiments.dp_matrix.topology import CHIPS, Topology, validate_identity


class TopologyTests(unittest.TestCase):
    def observed(self, topology):
        workers = topology.workers()
        ep_group_ranks = list(range(8))
        observed = []
        for i, row in enumerate(workers):
            tp_start = row["dp_rank"] * topology.tp
            dcp_start = tp_start + (row["tp_rank"] // topology.dcp) * topology.dcp
            observed.append({
                **row,
                "dp": topology.dp,
                "tp": topology.tp,
                "dcp": topology.dcp,
                "ep_world_size": 8,
                "enable_expert_parallel": True,
                "worker_pid": 100 + i,
                "tp_group_ranks": list(range(tp_start, tp_start + topology.tp)),
                "dcp_group_ranks": list(range(dcp_start, dcp_start + topology.dcp)),
                "ep_group_ranks": ep_group_ranks.copy(),
            })
        return observed

    def test_requested_layouts(self):
        for name in ("dp2-tp4", "dp4-tp2", "dp4-tp2-dcp2"):
            with self.subTest(name=name):
                topology = Topology.named(name)
                plan = topology.plan()
                self.assertEqual(plan["chip_count"], 8)
                self.assertEqual(len(plan["expected_workers"]), 8)
                self.assertEqual(plan["physical_chips"], list(CHIPS))
                self.assertFalse(plan["service_started"])
                self.assertFalse(plan["placement_observed"])
                topology.validate_workers(self.observed(topology))

    def test_dcp_is_inside_tp_not_additional_workers(self):
        topology = Topology(4, 2, 2)
        self.assertEqual(topology.plan()["expected_tp_groups"],
                         [[8, 9], [10, 11], [12, 13], [14, 15]])
        self.assertEqual(topology.plan()["expected_dcp_groups"],
                         topology.plan()["expected_tp_groups"])
        self.assertEqual([r["dcp_rank"] for r in topology.workers()], [0, 1] * 4)

    def test_repeated_tp_rank_not_a_duplicate_across_dp(self):
        topology = Topology(4, 2)
        topology.validate_workers(self.observed(topology))

    def test_invalid_dimensions_and_scope(self):
        for dims in ((1, 8, 1), (2, 4, 2), (4, 2, 4), (4, True, 1), (2.0, 4, 1)):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                Topology(*dims)
        with self.assertRaises(ValueError):
            Topology(2, 4, chips=tuple(range(8)))
        with self.assertRaises(ValueError):
            Topology.named("tp8")

    def test_bad_actual_descriptions_fail_closed(self):
        topology = Topology(4, 2, 2)
        rows = self.observed(topology)
        cases = [rows[:2], rows + [rows[0]], [rows[0]] * 8]
        for key, value in (("tp", 8), ("dp", 1), ("dcp", 8), ("ep_world_size", 2),
                           ("dp_rank", True), ("physical_chip", 0), ("worker_pid", 0),
                           ("enable_expert_parallel", False), ("ep_rank", 0)):
            changed = copy.deepcopy(rows)
            changed[-1][key] = value
            cases.append(changed)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                topology.validate_workers(case)

    def test_actual_communication_group_rank_lists_are_required_and_exact(self):
        topology = Topology(4, 2, 2)
        valid = self.observed(topology)
        topology.validate_workers(valid)

        cases = []

        missing_tp = copy.deepcopy(valid)
        missing_tp[0].pop("tp_group_ranks")
        cases.append(("missing TP group", missing_tp))

        missing_dcp = copy.deepcopy(valid)
        missing_dcp[0].pop("dcp_group_ranks")
        cases.append(("missing DCP group", missing_dcp))

        missing_ep = copy.deepcopy(valid)
        missing_ep[0].pop("ep_group_ranks")
        cases.append(("missing EP group", missing_ep))

        wrong_tp = copy.deepcopy(valid)
        wrong_tp[0]["tp_group_ranks"] = [2, 3]
        cases.append(("foreign TP members", wrong_tp))

        wrong_dcp = copy.deepcopy(valid)
        wrong_dcp[0]["dcp_group_ranks"] = [0]
        cases.append(("truncated DCP members", wrong_dcp))

        duplicate_tp = copy.deepcopy(valid)
        duplicate_tp[0]["tp_group_ranks"] = [0, 0]
        cases.append(("duplicate TP member", duplicate_tp))

        reordered_ep = copy.deepcopy(valid)
        reordered_ep[0]["ep_group_ranks"] = list(reversed(reordered_ep[0]["ep_group_ranks"]))
        cases.append(("wrong EP order", reordered_ep))

        boolean_dcp = copy.deepcopy(valid)
        boolean_dcp[0]["dcp_group_ranks"] = [False, 1]
        cases.append(("boolean DCP member", boolean_dcp))

        wrong_type = copy.deepcopy(valid)
        wrong_type[0]["ep_group_ranks"] = tuple(wrong_type[0]["ep_group_ranks"])
        cases.append(("non-list EP group", wrong_type))

        for name, case in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                topology.validate_workers(case)

    def test_new_identity_not_old_receipt_or_plan(self):
        topology = Topology(2, 4)
        sid, fingerprint = "a" * 32, "b" * 64
        identity = {"served_model": "new-model", "service_instance_id": sid,
                    "runtime_fingerprint": fingerprint, "layout": topology.name,
                    "engine_initialized": True, "ASGI_startup_completed": True,
                    "rank_descriptions": self.observed(topology)}
        kwargs = dict(served_model="new-model", sid=sid, fingerprint=fingerprint)
        validate_identity(identity, topology, **kwargs)
        for key, value in (("served_model", "old-model"), ("service_instance_id", "c" * 32),
                           ("runtime_fingerprint", "d" * 64), ("layout", "dp4-tp2"),
                           ("engine_initialized", False), ("ASGI_startup_completed", 1)):
            changed = {**identity, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_identity(changed, topology, **kwargs)
        with self.assertRaises(ValueError):
            validate_identity(identity, topology, **{**kwargs, "sid": None})


if __name__ == "__main__":
    unittest.main()
