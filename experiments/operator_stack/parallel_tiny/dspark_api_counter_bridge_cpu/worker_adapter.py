"""Deferred private worker class; only new worker processes import this file."""
import inspect

from parallel_tiny.dspark_formal_prep.worker import FormalDSparkWorker

from .runtime_guards import FORMAL_WORKER_SHA256, load_frozen_counter, sha256_file
from .worker_rpc import ApiCounterWorkerRpcMixin


class ApiCounterDSparkWorker(ApiCounterWorkerRpcMixin, FormalDSparkWorker):
    def load_model(self, *args, **kwargs):
        if sha256_file(inspect.getfile(FormalDSparkWorker)) != FORMAL_WORKER_SHA256:
            raise ValueError('The inherited formal worker source changed')
        # Pin the generic import used by the inherited worker before it loads.
        load_frozen_counter()
        return super().load_model(*args, **kwargs)
