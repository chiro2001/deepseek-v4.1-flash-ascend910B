import torch
import torch_npu  # noqa
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
names = ["npu_hc_post", "npu_hc_pre_v2", "npu_rms_norm_dynamic_quant",
         "npu_rms_norm_dynamic_quant_bf16", "npu_dynamic_quant", "npu_quant_matmul"]
for n in names:
    op = getattr(torch.ops._C_ascend, n, None)
    if op is None:
        print("  %-32s NOT REGISTERED" % n); continue
    sch = op.default._schema
    nm = sch.name + (("." + sch.overload_name) if sch.overload_name else "")
    out = []
    for key in ("Meta", "CompositeExplicitAutograd", "CPU"):
        try:
            ok = torch._C._dispatch_has_kernel_for_dispatch_key(nm, key)
        except Exception as e:
            ok = "ERR"
        out.append("%s=%s" % (key[:5], ok))
    print("  %-32s %s" % (n, "  ".join(out)))
