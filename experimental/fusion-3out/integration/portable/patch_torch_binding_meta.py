#!/usr/bin/env python3
"""给 torch_binding_meta.cpp 补上 npu_rms_norm_dynamic_quant_bf16 的 Meta 实现。

背景：既有 _C_ascend 算子的 Meta 实现都注册在这个文件里（`npu_hc_post` / 
`npu_rms_norm_dynamic_quant` 等）。我新增算子时只改了 torch_binding.cpp（真实现 +
schema），漏了这个文件 ⇒ `_dispatch_has_kernel_for_dispatch_key(..., "Meta") = False`
⇒ torch.compile 追踪时该算子无法做 fake 传播。

实测对照（op-hcfuse 容器）：
    npu_hc_post                      Meta=True
    npu_rms_norm_dynamic_quant       Meta=True
    npu_rms_norm_dynamic_quant_bf16  Meta=False   ← 就是这里
"""
import pathlib
import sys

F = pathlib.Path("/vllm-workspace/vllm-ascend/csrc/torch_binding_meta.cpp")
s = F.read_text()

MARK = "npu_rms_norm_dynamic_quant_bf16_meta"
if MARK in s:
    print("已打过，跳过")
    sys.exit(0)

# 1) 在现有 2 输出算子之后插入 3 输出实现
ANCHOR_FN = """    return std::make_tuple(y_out, scale_out);
}

void kv_compress_epilog_meta("""
NEW_FN = """    return std::make_tuple(y_out, scale_out);
}

// [BF16-3OUT] 三输出版：bf16 归一化结果 + int8 量化 + per-token scale
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant_bf16_meta(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& smooth_scale,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    auto options = x.options();
    at::Tensor y_bf16_out = at::empty_like(x, options.dtype(at::kBFloat16));
    at::Tensor y_out = at::empty_like(x, options.dtype(at::kChar));
    c10::SymDimVector scale_out_shape;
    for (size_t i = 0; i < x.dim() - 1; i++) {
        scale_out_shape.push_back(x.sym_size(i));
    }
    at::Tensor scale_out = at::empty_symint(scale_out_shape, options.dtype(at::kFloat));
    return std::make_tuple(y_bf16_out, y_out, scale_out);
}

void kv_compress_epilog_meta("""
assert ANCHOR_FN in s, "函数锚点未找到"
s = s.replace(ANCHOR_FN, NEW_FN, 1)

# 2) 注册
ANCHOR_REG = '    ops.impl("npu_rms_norm_dynamic_quant", &vllm_ascend::meta::npu_rms_norm_dynamic_quant_meta);\n'
NEW_REG = (ANCHOR_REG
           + '    ops.impl("npu_rms_norm_dynamic_quant_bf16",\n'
           + '             &vllm_ascend::meta::npu_rms_norm_dynamic_quant_bf16_meta);\n')
assert ANCHOR_REG in s, "注册锚点未找到"
s = s.replace(ANCHOR_REG, NEW_REG, 1)

F.write_text(s)
print("已插入 Meta 实现 + 注册")
