#!/usr/bin/env python3
"""给 torch_binding.cpp 加 npu_rms_norm_dynamic_quant_bf16 绑定。"""
import pathlib
F = pathlib.Path("/vllm-workspace/vllm-ascend/csrc/torch_binding.cpp")
src = F.read_text()
MARK = "RMS_NORM_DYNAMIC_QUANT_BF16_BIND"
if MARK in src:
    print("already bound"); raise SystemExit(0)

# 1) 在 npu_rms_norm_dynamic_quant_npu 的注册之后插入新算子
anchor = '    ops.impl("npu_rms_norm_dynamic_quant", torch::kPrivateUse1, &vllm_ascend::npu_rms_norm_dynamic_quant_npu);'
assert anchor in src, "reg anchor"
impl = anchor + '''

    // ''' + MARK + '''
    ops.def(
        "npu_rms_norm_dynamic_quant_bf16("
            "Tensor x, "
            "Tensor gamma, "
            "Tensor? smooth_scale=None, "
            "Tensor? beta=None, "
            "float epsilon=1e-6"
        ") -> (Tensor y_bf16, Tensor y_out, Tensor scale_out)"
        );
    ops.impl("npu_rms_norm_dynamic_quant_bf16", torch::kPrivateUse1,
             &vllm_ascend::npu_rms_norm_dynamic_quant_bf16_npu);'''
src = src.replace(anchor, impl, 1)

# 2) 在 npu_rms_norm_dynamic_quant_npu 的定义之后插入实现
fn_anchor = 'std::tuple<at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant_npu('
i = src.find(fn_anchor)
assert i > 0, "impl anchor"
# 找该函数结束（下一个顶层 "}" 后跟空行 + 非缩进）
j = src.find("\n}\n", i)
assert j > 0, "impl end"
new_fn = '''

// ''' + MARK + '''
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant_bf16_npu(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& smooth_scale,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    constexpr int32_t SIZE = 8;
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(gamma.numel() > 0, "Input tensor gamma should not be empty.");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x.size(-1), "gamma dim are not equal to last dim of x shape.");
    TORCH_CHECK(epsilon > 0, "epsilon should be greater than 0.");
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16, "x should be FLOAT16, BFLOAT16.");

    at::Tensor smooth_scale2{nullptr};
    auto options = x.options();
    at::Tensor y_out = at::empty_like(x, options.dtype(at::kChar));
    at::Tensor y2_out = at::empty({1}, options.dtype(at::kChar));
    at::Tensor y_bf16_out = at::empty_like(x, options.dtype(at::kBFloat16));

    c10::SmallVector<int64_t, SIZE> scale_out_shape;
    for (size_t i = 0; i < x.sizes().size() - 1; i++) {
        scale_out_shape.push_back(x.sizes()[i]);
    }
    at::Tensor scale_out = at::empty(scale_out_shape, options.dtype(at::kFloat));
    at::Tensor scale2_out = at::empty_like(scale_out);
    std::array<bool, 2>* output_mask = nullptr;
    int64_t* dst_type = nullptr;

    EXEC_NPU_CMD(aclnnRmsNormDynamicQuantBf16, x, gamma, smooth_scale, smooth_scale2, beta, epsilon,
                 output_mask, dst_type, y_out, y2_out, scale_out, scale2_out, y_bf16_out);
    return {y_bf16_out, y_out, scale_out};
}
'''
src = src[:j+2] + new_fn + src[j+2:]
F.write_text(src)
print("绑定已加：实现 + 注册")
