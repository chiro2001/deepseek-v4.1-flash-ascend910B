


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1):
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    matmul = torch.ops.aten.matmul.default(arg0_1, t);  arg0_1 = t = None
    zero = torch.ops.aten.zero.default(arg2_1)
    slice_2 = torch.ops.aten.slice.Tensor(zero, 0, 0, 6)
    copy = torch.ops.aten.copy.default(slice_2, matmul);  slice_2 = None
    slice_scatter = torch.ops.aten.slice_scatter.default(zero, copy, 0, 0, 6);  zero = copy = None
    fill = torch.ops.aten.fill.Scalar(arg4_1, -1)
    slice_5 = torch.ops.aten.slice.Tensor(fill, 0, 0, 6)
    copy_1 = torch.ops.aten.copy.default(slice_5, arg3_1);  slice_5 = None
    slice_scatter_1 = torch.ops.aten.slice_scatter.default(fill, copy_1, 0, 0, 6);  fill = copy_1 = None
    zero_1 = torch.ops.aten.zero.default(arg7_1)
    slice_8 = torch.ops.aten.slice.Tensor(zero_1, 0, 0, 6)
    copy_2 = torch.ops.aten.copy.default(slice_8, arg5_1);  slice_8 = None
    slice_scatter_2 = torch.ops.aten.slice_scatter.default(zero_1, copy_2, 0, 0, 6);  zero_1 = copy_2 = None
    fill_1 = torch.ops.aten.fill.Scalar(arg10_1, 6)
    add = torch.ops.aten.add.Tensor(arg9_1, 1)
    copy_3 = torch.ops.aten.copy.default(arg8_1, arg6_1)
    index_copy = torch.ops.aten.index_copy.default(arg11_1, 0, arg12_1, matmul)
    index_select = torch.ops.aten.index_select.default(index_copy, 0, arg12_1);  arg12_1 = None
    zero_2 = torch.ops.aten.zero.default(arg13_1)
    slice_11 = torch.ops.aten.slice.Tensor(zero_2, 0, 0, 6)
    copy_4 = torch.ops.aten.copy.default(slice_11, index_select);  slice_11 = index_select = None
    slice_scatter_3 = torch.ops.aten.slice_scatter.default(zero_2, copy_4, 0, 0, 6);  zero_2 = copy_4 = None
    fill_2 = torch.ops.aten.fill.Scalar(arg14_1, -1)
    slice_14 = torch.ops.aten.slice.Tensor(fill_2, 0, 0, 6)
    copy_5 = torch.ops.aten.copy.default(slice_14, arg3_1);  slice_14 = arg3_1 = None
    slice_scatter_4 = torch.ops.aten.slice_scatter.default(fill_2, copy_5, 0, 0, 6);  fill_2 = copy_5 = None
    zero_3 = torch.ops.aten.zero.default(arg15_1)
    slice_17 = torch.ops.aten.slice.Tensor(zero_3, 0, 0, 6)
    copy_6 = torch.ops.aten.copy.default(slice_17, arg5_1);  slice_17 = arg5_1 = None
    slice_scatter_5 = torch.ops.aten.slice_scatter.default(zero_3, copy_6, 0, 0, 6);  zero_3 = copy_6 = None
    fill_3 = torch.ops.aten.fill.Scalar(arg18_1, 6)
    add_1 = torch.ops.aten.add.Tensor(arg17_1, 1)
    copy_7 = torch.ops.aten.copy.default(arg16_1, arg6_1);  arg6_1 = None
    mul = torch.ops.aten.mul.Tensor(matmul, 2);  matmul = None
    copy_ = torch.ops.aten.copy_.default(arg2_1, slice_scatter);  arg2_1 = slice_scatter = copy_ = None
    copy__1 = torch.ops.aten.copy_.default(arg4_1, slice_scatter_1);  arg4_1 = slice_scatter_1 = copy__1 = None
    copy__2 = torch.ops.aten.copy_.default(arg7_1, slice_scatter_2);  arg7_1 = slice_scatter_2 = copy__2 = None
    copy__3 = torch.ops.aten.copy_.default(arg8_1, copy_3);  arg8_1 = copy_3 = copy__3 = None
    copy__4 = torch.ops.aten.copy_.default(arg9_1, add);  arg9_1 = add = copy__4 = None
    copy__5 = torch.ops.aten.copy_.default(arg10_1, fill_1);  arg10_1 = fill_1 = copy__5 = None
    copy__6 = torch.ops.aten.copy_.default(arg11_1, index_copy);  arg11_1 = index_copy = copy__6 = None
    copy__7 = torch.ops.aten.copy_.default(arg13_1, slice_scatter_3);  arg13_1 = slice_scatter_3 = copy__7 = None
    copy__8 = torch.ops.aten.copy_.default(arg14_1, slice_scatter_4);  arg14_1 = slice_scatter_4 = copy__8 = None
    copy__9 = torch.ops.aten.copy_.default(arg15_1, slice_scatter_5);  arg15_1 = slice_scatter_5 = copy__9 = None
    copy__10 = torch.ops.aten.copy_.default(arg16_1, copy_7);  arg16_1 = copy_7 = copy__10 = None
    copy__11 = torch.ops.aten.copy_.default(arg17_1, add_1);  arg17_1 = add_1 = copy__11 = None
    copy__12 = torch.ops.aten.copy_.default(arg18_1, fill_3);  arg18_1 = fill_3 = copy__12 = None
    return (mul,)
    