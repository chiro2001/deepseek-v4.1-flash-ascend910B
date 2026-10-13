
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1 = args

    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    matmul = torch.ops.aten.matmul.default(arg0_1, t);  arg0_1 = t = None
    fill = torch.ops.aten.fill.Scalar(arg4_1, -1);  arg4_1 = None
    slice_scatter_1 = torch.ops.aten.slice_scatter.default(fill, arg3_1, 0, 0, 6);  fill = None
    zero_1 = torch.ops.aten.zero.default(arg7_1);  arg7_1 = None
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(zero_1, arg5_1, 0, 0, 6);  zero_1 = None
    fill_1 = torch.ops.aten.fill.Scalar(arg10_1, 6);  arg10_1 = None
    add = torch.ops.aten.add.Tensor(arg9_1, 1);  arg9_1 = None
    copy_3 = torch.ops.aten.copy.default(arg8_1, arg6_1);  arg8_1 = None
    index_copy = torch.ops.aten.index_copy.default(arg11_1, 0, arg12_1, matmul);  arg11_1 = None
    index_select = torch.ops.aten.index_select.default(index_copy, 0, arg12_1);  arg12_1 = None
    fill_2 = torch.ops.aten.fill.Scalar(arg14_1, -1);  arg14_1 = None
    slice_scatter_4 = torch.ops.aten.slice_scatter.default(fill_2, arg3_1, 0, 0, 6);  fill_2 = arg3_1 = None
    zero_3 = torch.ops.aten.zero.default(arg15_1);  arg15_1 = None
    slice_scatter_5 = torch.ops.aten.slice_scatter.default(zero_3, arg5_1, 0, 0, 6);  zero_3 = arg5_1 = None
    fill_3 = torch.ops.aten.fill.Scalar(arg18_1, 6);  arg18_1 = None
    add_1 = torch.ops.aten.add.Tensor(arg17_1, 1);  arg17_1 = None
    copy_7 = torch.ops.aten.copy.default(arg16_1, arg6_1);  arg16_1 = arg6_1 = None
    mul = torch.ops.aten.mul.Tensor(matmul, 2)
    return (matmul, slice_scatter_1, slice_scatter_2, copy_3, add, fill_1, index_copy, index_select, slice_scatter_4, slice_scatter_5, copy_7, add_1, fill_3, mul)


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
    user_inputs_mapping=OrderedDict({'arg0_1': 0, 'arg1_1': 1, 'arg2_1': 2, 'arg3_1': 3, 'arg4_1': 4, 'arg5_1': 5, 'arg6_1': 6, 'arg7_1': 7, 'arg8_1': 8, 'arg9_1': 9, 'arg10_1': 10, 'arg11_1': 11, 'arg12_1': 12, 'arg13_1': 13, 'arg14_1': 14, 'arg15_1': 15, 'arg16_1': 16, 'arg17_1': 17, 'arg18_1': 18}),
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

        assert_size_stride(args[0], (6, 64), (64, 1))
        assert_size_stride(args[1], (64, 64), (64, 1))
        assert_size_stride(args[2], (6, 64), (64, 1))
        assert_size_stride(args[3], (6, 1), (1, 1))
        assert_size_stride(args[4], (6, 1), (1, 1))
        assert_size_stride(args[5], (6,), (1,))
        assert_size_stride(args[6], (1,), (1,))
        assert_size_stride(args[7], (6,), (1,))
        assert_size_stride(args[8], (1,), (1,))
        assert_size_stride(args[9], (1,), (1,))
        assert_size_stride(args[10], (1,), (1,))
        assert_size_stride(args[11], (16, 64), (64, 1))
        assert_size_stride(args[12], (6,), (1,))
        assert_size_stride(args[13], (6, 64), (64, 1))
        assert_size_stride(args[14], (6, 1), (1, 1))
        assert_size_stride(args[15], (6,), (1,))
        assert_size_stride(args[16], (1,), (1,))
        assert_size_stride(args[17], (1,), (1,))
        assert_size_stride(args[18], (1,), (1,))
    return acl_graph(*args, **kwargs)

def main():
    arg0_1 = rand_strided((6, 64),(64, 1),device ='npu:5',dtype =torch.bfloat16)
    arg1_1 = rand_strided((64, 64),(64, 1),device ='npu:5',dtype =torch.bfloat16)
    arg2_1 = rand_strided((6, 64),(64, 1),device ='npu:5',dtype =torch.bfloat16)
    arg3_1 = rand_strided((6, 1),(1, 1),device ='npu:5',dtype =torch.int64)
    arg4_1 = rand_strided((6, 1),(1, 1),device ='npu:5',dtype =torch.int64)
    arg5_1 = rand_strided((6,),(1,),device ='npu:5',dtype =torch.bool)
    arg6_1 = rand_strided((1,),(1,),device ='npu:5',dtype =torch.int64)
    arg7_1 = rand_strided((6,),(1,),device ='npu:5',dtype =torch.bool)
    arg8_1 = rand_strided((1,),(1,),device ='npu:5',dtype =torch.int64)
    arg9_1 = rand_strided((1,),(1,),device ='npu:5',dtype =torch.int64)
    arg10_1 = rand_strided((1,),(1,),device ='npu:5',dtype =torch.int64)
    arg11_1 = rand_strided((16, 64),(64, 1),device ='npu:5',dtype =torch.bfloat16)
    arg12_1 = rand_strided((6,),(1,),device ='npu:5',dtype =torch.int64)
    arg13_1 = rand_strided((6, 64),(64, 1),device ='npu:5',dtype =torch.bfloat16)
    arg14_1 = rand_strided((6, 1),(1, 1),device ='npu:5',dtype =torch.int64)
    arg15_1 = rand_strided((6,),(1,),device ='npu:5',dtype =torch.bool)
    arg16_1 = rand_strided((1,),(1,),device ='npu:5',dtype =torch.int64)
    arg17_1 = rand_strided((1,),(1,),device ='npu:5',dtype =torch.int64)
    arg18_1 = rand_strided((1,),(1,),device ='npu:5',dtype =torch.int64)
    return kernel(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1)
