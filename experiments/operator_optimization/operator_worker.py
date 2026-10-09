"""Load the proven HC/router baseline, then install activation hooks."""
import json
import os

from tiny_perf_worker import TinyPerfWorker
import activation_patches as activation


class OperatorOptWorker(TinyPerfWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if os.getenv("TINY_PERF_RANDOM_VALIDATION") == "1":
            print("RANDOMIZED_MLP_BEFORE_COMPILE", activation.randomize_expert_validation(self), flush=True)
        activation.install(self.model_runner.model)
        if os.getenv("OPT_TEST_OVERLAP") == "1":
            import overlap_patches
            overlap_patches.initialize(self.model_runner.model)
            print("OVERLAP_INITIAL", overlap_patches.set_enabled(activation.ARM == "overlap"), flush=True)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        state = {"arm": activation.ARM,
                 "routed_calls": len(activation.REFS.get(activation.ARM, {}).get("routed", [])),
                 "shared_calls": len(activation.REFS.get(activation.ARM, {}).get("shared", [])),
                 "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES")}
        assert state["routed_calls"] == state["shared_calls"] == 40, state
        refs = activation.REFS[activation.ARM]
        state["selected"] = {kind: sum(item[3] for item in values) for kind, values in refs.items()}
        assert state["selected"]["routed"] == (40 if activation.ARM in ["routed", "fused", "overlap"] else 0), state
        assert state["selected"]["shared"] == (40 if activation.ARM in ["shared", "fused", "overlap"] else 0), state
        print("OPERATOR_OPT_EFFECTIVE", json.dumps(state), flush=True)
        return result
