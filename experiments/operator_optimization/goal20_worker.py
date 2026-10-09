"""Install next-round hooks after platform/model loading and prior to compile."""
from operator_worker import OperatorOptWorker
import goal20_patches as patches


class Goal20Worker(OperatorOptWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        patches.install(self.model_runner.model)
        return result

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = super().compile_or_warm_up_model(*args, **kwargs)
        print('GOAL20_EFFECTIVE', patches.save_bank(self, patches.ARM), flush=True)
        return result
