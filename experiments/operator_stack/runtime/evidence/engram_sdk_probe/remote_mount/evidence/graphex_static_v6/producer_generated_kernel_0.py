
import threading
from collections import OrderedDict

import torch
from torch._dynamo.testing import rand_strided

import torch_npu
from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph, AclGraphCacheInfo
from npugraph_ex._acl_concrete_graph.acl_graph_cache_utils import SerializableGraphModule
from npugraph_ex.ops._tagged_event import _npu_create_tagged_event

assert_size_stride = torch._C._dynamo.guards.assert_size_stride

from torch._dynamo.guards import _get_closure_vars
globals().update(_get_closure_vars())
globals().update({"nan": float("nan")})

def forward(*args, node_info=[], is_capturing: bool = False):
    arg0_1, arg1_1, arg2_1 = args

    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    matmul = torch.ops.aten.matmul.default(arg0_1, t);  arg0_1 = t = None
    as_strided_default = torch.ops.aten.as_strided.default(arg2_1, [64], [1], 0);  arg2_1 = None
    clone_default = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
    as_strided_default_1 = torch.ops.aten.as_strided.default(clone_default, [1, 64], [64, 1], 0);  clone_default = None
    publish_default = torch.ops.engram_sdk_probe.publish.default(matmul, as_strided_default_1, 281472013991248);  matmul = publish_default = None
    return (as_strided_default_1,)


compile_configs = {}
compile_configs["force_eager"] = True
compile_configs["reuse_graph_pool_in_same_fx"] = True
compile_configs["clone_output"] = False
compile_configs["capture_limit"] = "64"
compile_configs["clone_input"] = False
compile_configs["deadlock_check"] = False
compile_configs["capture_error_mode"] = "global"
compile_configs["force_recapture"] = False
compile_configs["static_kernel_compile"] = True
compile_configs["_vllm_aclnn_static_kernel_sym_range"] = "[1]"
compile_configs["_vllm_aclnn_static_kernel_sym_index"] = "0"
compile_configs["remove_noop_ops"] = True
compile_configs["pattern_fusion_pass"] = True
compile_configs["frozen_parameter"] = False
compile_configs["sym_input_index"] = "[]"

acl_graph = AclGraph(fx_forward=forward, config=compile_configs)

aclgraph_cache_info = AclGraphCacheInfo(
    pool=None,
    stream=torch.npu.Stream(),
    capture_error_mode="global",
    num_warmup_iters=0,
    fx_graph_name="graph_1",
    user_inputs_mapping=OrderedDict({'arg0_1': 0, 'arg1_1': 1, 'arg2_1': 2}),
    unupdated_sym_input_index=[],
    updated_ops_param={},
    ops_update_rulers={},
    mutated_user_inputs=[],
    tagged_event_names=[],
    parameter_user_inputs=[],
    user_stream_label=set(),
    user_stream_info={},
    userinput_ref_with_output={}
)
acl_graph.load(aclgraph_cache_info)


_is_first_run = True
def kernel(*args, **kwargs):

    global _is_first_run
    if _is_first_run:
        _is_first_run = False

        assert_size_stride(args[0], (1, 64), (64, 1))
        assert_size_stride(args[1], (64, 64), (64, 1))
        assert_size_stride(args[2], (1, 64), (64, 1))
    return acl_graph(*args, **kwargs)

def main():
    arg0_1 = rand_strided((1, 64),(64, 1),device ='npu:0',dtype =torch.bfloat16)
    arg1_1 = rand_strided((64, 64),(64, 1),device ='npu:0',dtype =torch.bfloat16)
    arg2_1 = rand_strided((1, 64),(64, 1),device ='npu:0',dtype =torch.bfloat16)
    return kernel(arg0_1, arg1_1, arg2_1)
