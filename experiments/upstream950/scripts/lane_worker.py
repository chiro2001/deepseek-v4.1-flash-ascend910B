"""Install one candidate after the established HC/router baseline loads."""
import json
import os

from tiny_perf_worker import TinyPerfWorker
import lane_patches as lane


class Upstream950Worker(TinyPerfWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if os.getenv('TINY_PERF_RANDOM_VALIDATION') == '1':
            print('UP950_RANDOMIZED_MLP', lane.randomize_mlp(self), flush=True)
        lane.install(self.model_runner.model)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        state = lane.candidate().coverage(lane.candidate().ARM)
        print('UP950_EFFECTIVE', json.dumps(state), flush=True)
        return result
