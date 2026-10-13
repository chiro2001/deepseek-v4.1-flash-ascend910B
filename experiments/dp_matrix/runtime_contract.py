"""New geometry gate; never reuse the TP8 arm tag as a DP admission."""
import os

from .topology import Topology, integer


def expected_tp_size():
    return Topology.named(os.environ.get("DP_MATRIX_LAYOUT")).tp


def validate_worker_config(config, environment):
    topology = Topology.named(environment.get("DP_MATRIX_LAYOUT"),
                              chips=environment.get("STACK_PHYSICAL_CHIPS"))
    parallel = config.parallel_config
    for key, value in (("data_parallel_size", topology.dp),
                       ("data_parallel_size_local", topology.dp),
                       ("tensor_parallel_size", topology.tp),
                       ("decode_context_parallel_size", topology.dcp)):
        if integer(getattr(parallel, key, None), key, minimum=1) != value:
            raise ValueError("Actual worker topology differs: " + key)
    if parallel.enable_expert_parallel is not True:
        raise ValueError("Cross-DP native expert parallelism is required")
    rank = integer(parallel.data_parallel_rank, "data_parallel_rank")
    if rank >= topology.dp:
        raise ValueError("Foreign data-parallel rank")
    if environment.get("STACK_TP8_ARM") is not None:
        raise ValueError("Explicit eight-chip DP placement must not inherit the TP8 arm tag")
    chip_list = ",".join(map(str, topology.chips))
    for key in ("STACK_EXPECTED_CHIPS", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        if key in environment and environment[key] != chip_list:
            raise ValueError("Actual worker visible-device setting differs from explicit placement: " + key)
    return topology
