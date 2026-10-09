"""Install the overlay after the proven line has initialized and before capture."""
from goal20_worker import Goal20Worker
import stack_patches


class StackWorker(Goal20Worker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        stack_patches.install(self.model_runner.model)
        return result
