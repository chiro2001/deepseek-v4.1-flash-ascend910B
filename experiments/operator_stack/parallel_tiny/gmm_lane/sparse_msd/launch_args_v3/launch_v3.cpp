#include "launch_v3.h"
#include <acl/acl_rt.h>
#include <cstddef>

static_assert(sizeof(void *) == 8 && sizeof(Gmm2PlannerArgsV3) == 16);
static_assert(sizeof(Gmm2MsdArgsV3) == 72 && offsetof(Gmm2MsdArgsV3, workspace_system_base) == 64);

namespace {
bool profile(void *function, void *stream, void *host, size_t bytes, unsigned blocks, unsigned schedule) {
    if (!function || !stream || !host) return false;
    return (bytes == 16 && blocks == 1 && schedule == 0) ||
           (bytes == 72 && blocks == 24 && schedule == 1);
}
aclrtLaunchKernelCfg config(unsigned schedule, aclrtLaunchKernelAttr *attr) {
    *attr = {};
    attr->id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr->value.schemMode = schedule;
    return {attr, 1};
}
}

extern "C" int gmm2_launch_v3_host_args(void *function, void *stream, void *host,
                                       size_t bytes, unsigned blocks, unsigned schedule) {
    if (!profile(function, stream, host, bytes, blocks, schedule)) return GMM2_LAUNCH_V3_INVALID_ARGUMENT;
    aclrtLaunchKernelAttr attr{}; auto cfg = config(schedule, &attr);
    // Explicit host path; do not use aclrtLaunchKernel/V2 (dev/no-copy path).
    return aclrtLaunchKernelWithHostArgs(function, blocks, stream, &cfg, host, bytes, nullptr, 0);
}

extern "C" int gmm2_launch_v3_with_config(void *function, void *stream, void *args_handle,
                                         void *host, size_t bytes, unsigned blocks,
                                         unsigned schedule, unsigned *failed_stage) {
    if (failed_stage) *failed_stage = 0;
    if (!args_handle || !profile(function, stream, host, bytes, blocks, schedule))
        return GMM2_LAUNCH_V3_INVALID_ARGUMENT;
    aclrtParamHandle parameter{};
    int error = aclrtKernelArgsAppend(args_handle, host, bytes, &parameter);
    if (error) { if (failed_stage) *failed_stage = 1; return error; }
    error = aclrtKernelArgsFinalize(args_handle);
    if (error) { if (failed_stage) *failed_stage = 2; return error; }
    aclrtLaunchKernelAttr attr{}; auto cfg = config(schedule, &attr);
    error = aclrtLaunchKernelWithConfig(function, blocks, stream, &cfg, args_handle, nullptr);
    if (error && failed_stage) *failed_stage = 3;
    return error;
}
