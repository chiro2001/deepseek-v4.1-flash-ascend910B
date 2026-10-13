"""CPU-only placement and identity contracts for the three requested layouts.

DCP partitions existing TP ranks: it does not multiply the eight-chip budget.
These are expected placements, not proof that a service or process exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


CHIPS = tuple(range(8, 16))
LOW_CHIPS = tuple(range(8))
CHIP_GROUPS = (LOW_CHIPS, CHIPS)
DEFAULT_CHIP_LIST = ",".join(map(str, CHIPS))
LAYOUTS = {
    "dp2-tp4": (2, 4, 1),
    "dp4-tp2": (4, 2, 1),
    "dp4-tp2-dcp2": (4, 2, 2),
}


def integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def parse_chips(value):
    """Only the two explicitly authorized contiguous eight-chip groups."""
    if isinstance(value, str):
        if value not in (",".join(map(str, group)) for group in CHIP_GROUPS):
            raise ValueError("Physical chips must be exactly 0..7 or 8..15, in order")
        value = tuple(int(part) for part in value.split(","))
    elif isinstance(value, (list, tuple)):
        value = tuple(value)
    else:
        raise ValueError("An explicit eight-chip group is required")
    if value not in CHIP_GROUPS or any(type(chip) is not int for chip in value):
        raise ValueError("Physical chips must be exactly 0..7 or 8..15, in order")
    return value


@dataclass(frozen=True)
class Topology:
    dp: int
    tp: int
    dcp: int = 1
    chips: tuple[int, ...] = CHIPS

    def __post_init__(self):
        for name in ("dp", "tp", "dcp"):
            integer(getattr(self, name), name, minimum=1)
        if (self.dp, self.tp, self.dcp) not in LAYOUTS.values():
            raise ValueError("Only the three requested eight-chip layouts are admitted")
        if self.dp * self.tp != 8 or self.tp % self.dcp:
            raise ValueError("DP*TP must equal eight; DCP must divide TP")
        if parse_chips(self.chips) != self.chips:
            raise ValueError("Topology requires a canonical physical-chip tuple")

    @classmethod
    def named(cls, name, *, chips=CHIPS):
        if name not in LAYOUTS:
            raise ValueError("Unknown requested layout")
        return cls(*LAYOUTS[name], chips=parse_chips(chips))

    @property
    def name(self):
        return next(name for name, dims in LAYOUTS.items()
                    if dims == (self.dp, self.tp, self.dcp))

    def workers(self):
        return [
            {"dp_rank": dp_rank, "tp_rank": tp_rank,
             "ep_rank": dp_rank * self.tp + tp_rank,
             "dcp_rank": tp_rank % self.dcp,
             "physical_chip": self.chips[dp_rank * self.tp + tp_rank]}
            for dp_rank in range(self.dp) for tp_rank in range(self.tp)
        ]

    def plan(self):
        workers = self.workers()
        return {
            "schema": "eight_chip_dp_topology_v1", "layout": self.name,
            "dp": self.dp, "tp": self.tp, "dcp": self.dcp,
            "chip_count": 8, "physical_chips": list(self.chips),
            "enable_expert_parallel": True, "ep_world_size": 8,
            "expected_workers": workers,
            "expected_tp_groups": [
                [row["physical_chip"] for row in workers if row["dp_rank"] == dp]
                for dp in range(self.dp)
            ],
            "expected_dcp_groups": [
                list(self.chips[start:start + self.dcp])
                for start in range(0, 8, self.dcp)
            ],
            "placement_observed": False, "service_started": False,
            "actual_quality_passed": False,
        }

    def validate_workers(self, observed):
        """Require every native DP/TP rank, without deduplicating repeated TP ranks.

        A separate host PID/NPU/cgroup check must establish physical placement;
        config declarations alone do not establish it.
        """
        if not isinstance(observed, list) or len(observed) != 8:
            raise ValueError("Exactly eight actual worker descriptions are required")
        expected = {(row["dp_rank"], row["tp_rank"]): row for row in self.workers()}
        seen = set()
        pids = set()
        for row in observed:
            if not isinstance(row, dict):
                raise ValueError("Worker description must be an object")
            pair = (integer(row.get("dp_rank"), "dp_rank"),
                    integer(row.get("tp_rank"), "tp_rank"))
            if pair not in expected or pair in seen:
                raise ValueError("Missing, duplicate or foreign DP/TP rank")
            seen.add(pair)
            for key, value in expected[pair].items():
                if integer(row.get(key), key) != value:
                    raise ValueError(f"Unexpected worker {key}")
            for key, value in (("dp", self.dp), ("tp", self.tp),
                               ("dcp", self.dcp), ("ep_world_size", 8)):
                if integer(row.get(key), key, minimum=1) != value:
                    raise ValueError(f"Unexpected worker {key}")
            if row.get("enable_expert_parallel") is not True:
                raise ValueError("Native cross-DP expert parallelism must remain enabled")
            # Actual site parallel_state layout: ExternalDP x DP x PP x PCP x TP.
            # Here PP=PCP=ExternalDP=1. Members use global ranks 0..7, not
            # physical chip IDs or per-engine local ranks.
            global_rank = pair[0] * self.tp + pair[1]
            tp_start = pair[0] * self.tp
            dcp_start = tp_start + (pair[1] // self.dcp) * self.dcp
            for key, members in (
                ("tp_group_ranks", list(range(tp_start, tp_start + self.tp))),
                ("dcp_group_ranks", list(range(dcp_start, dcp_start + self.dcp))),
                ("ep_group_ranks", list(range(8))),
            ):
                actual = row.get(key)
                if (not isinstance(actual, list) or any(type(r) is not int for r in actual)
                        or actual != members):
                    raise ValueError("Unexpected actual communication group: " + key)
            if row["ep_group_ranks"][row["ep_rank"]] != global_rank:
                raise ValueError("Actual global membership disagrees with the DP/TP rank")
            pid = integer(row.get("worker_pid"), "worker_pid", minimum=1)
            if pid in pids:
                raise ValueError("Each native worker must have a distinct actual PID")
            pids.add(pid)
        return observed


def validate_identity(identity, topology, *, served_model, sid, fingerprint):
    """Bind a new run; never infer readiness from a plan or a previous SID."""
    if not isinstance(identity, dict):
        raise ValueError("Actual identity must be an object")
    if not isinstance(sid, str) or re.fullmatch(r"[0-9a-f]{32}", sid) is None:
        raise ValueError("An explicit actual SID is required")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("An explicit runtime fingerprint is required")
    for key, value in (("served_model", served_model), ("service_instance_id", sid),
                       ("runtime_fingerprint", fingerprint), ("layout", topology.name)):
        if identity.get(key) != value:
            raise ValueError(f"Actual service {key} differs from the pinned run")
    if (identity.get("engine_initialized") is not True
            or identity.get("ASGI_startup_completed") is not True):
        raise ValueError("Native engine and ASGI must both be ready")
    topology.validate_workers(identity.get("rank_descriptions"))
    return identity


def topology_from_args(args):
    return Topology.named(args.layout, chips=getattr(args, "physical_chips", CHIPS))
