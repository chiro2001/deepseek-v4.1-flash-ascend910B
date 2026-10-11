"""CPU negative controls for actual-value comparator; no device export claims."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types


class Tensor:
    def __init__(self, value, shape=(1,), dtype="bf16"):
        self.value, self.shape, self.dtype = value, shape, dtype

    def view(self, dtype):
        assert dtype == "u8"
        return Tensor(struct.pack("d", self.value), (8,), dtype)

    def numel(self):
        count = 1
        for size in self.shape:
            count *= size
        return count


def main():
    torch = types.ModuleType("torch")
    torch.uint8 = "u8"
    torch.equal = lambda a, b: a.shape == b.shape and a.value == b.value
    sys.modules["torch"] = torch
    base = Path(__file__).resolve().parent
    path = base / "graph_precision_audit.py"
    spec = importlib.util.spec_from_file_location("audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    roles = [f"target.{i}" for i in range(40)] + [f"draft.{i}" for i in range(3)]
    data = {"query_positions": [159, 160, 161, 162, 163], "parameter_seal_passed": True,
            "steps": [{"is_decode": True, "proposed_ids": Tensor(7, (1, 5), "i64"),
                       "routes": {name: {"weights": Tensor(.5, (5, 2), "f32"),
                                         "ids": Tensor(1, (5, 2), "i32")} for name in roles},
                       "attention": {f"mtp.{i}.self_attn.attn": Tensor(.1, (5, 64, 512)) for i in range(3)}}],
            "cache": {f"draft.{i}": {"logical_rows": 164, "values": Tensor(.2, (164, 1, 512))}
                      for i in range(3)}}
    audit.STATE["parameter_seal"] = {"real_runtime_seal_required": True}
    checks = []

    def setup():
        audit.STATE["requests"] = {"a": copy.deepcopy(data), "b": copy.deepcopy(data)}
        return audit.STATE["requests"]["b"]

    setup()
    assert audit.compare(None, "a", "b")["passed"]
    checks.append("matching 43-layer routes, three attention outputs and three logical caches pass")
    mutations = {
        "proposal mismatch": lambda b: setattr(b["steps"][0]["proposed_ids"], "value", 8),
        "route id mismatch": lambda b: setattr(b["steps"][0]["routes"]["draft.2"]["ids"], "value", 2),
        "route weight mismatch": lambda b: setattr(b["steps"][0]["routes"]["target.39"]["weights"], "value", .6),
        "attention output mismatch": lambda b: setattr(b["steps"][0]["attention"]["mtp.1.self_attn.attn"], "value", .2),
        "cache value mismatch": lambda b: setattr(b["cache"]["draft.0"]["values"], "value", .3),
        "cache dtype mismatch": lambda b: setattr(b["cache"]["draft.0"]["values"], "dtype", "f32"),
        "missing router consumer": lambda b: b["steps"][0]["routes"].pop("target.39"),
        "missing attention consumer": lambda b: b["steps"][0]["attention"].pop("mtp.2.self_attn.attn"),
        "missing cache consumer": lambda b: b["cache"].pop("draft.2"),
        "query position mismatch": lambda b: b["query_positions"].__setitem__(0, 0),
        "unsealed parameters": lambda b: b.__setitem__("parameter_seal_passed", False),
    }
    for label, mutate in mutations.items():
        b = setup()
        mutate(b)
        try:
            audit.compare(None, "a", "b")
        except AssertionError:
            checks.append(label + " rejected")
        else:
            raise AssertionError(label + " was falsely admitted")
    assert not audit._bitwise(Tensor(0.0), Tensor(-0.0)), "Numeric equality must not claim bitwise equality"
    checks.append("negative zero bit mismatch rejected")
    result = {"passed": True, "cpu_mock_only_no_npu": True, "checks": checks,
              "actual_device_exporter_or_precision_proof": False,
              "audit_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (base / "evidence/graph_precision_cpu_gate.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
