"""Native tiny worker with private ID witness instrumentation only."""
from vllm_ascend.worker.worker import NPUWorker
import ids_patches


class IdsWorker(NPUWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        ids_patches.set_arm("off")
        print("IDS_TINY_TOPOLOGY", ids_patches.initialize(self), flush=True)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        print("IDS_TINY_CAPTURE", ids_patches.coverage("off"), flush=True)
        return result
