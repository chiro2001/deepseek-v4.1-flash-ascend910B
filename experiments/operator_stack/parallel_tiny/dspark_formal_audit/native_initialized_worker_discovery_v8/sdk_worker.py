"""Resolved only by the standard WorkerWrapper in a new authorized SDK job."""
from vllm_ascend.worker.worker import NPUWorker
from .worker_hook import make_worker_class

InitializedSdkGateWorker = make_worker_class(NPUWorker)
