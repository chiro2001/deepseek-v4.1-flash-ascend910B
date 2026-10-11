// CPU mock only; no ACL/runtime library is linked or loaded.
#include "launch_v3.h"
#include <acl/acl_rt.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

unsigned checks = 0, fail = 0;
std::vector<unsigned> calls;
std::array<uint8_t, 72> sdk_copied{};
size_t copied_bytes = 0;
void require(bool ok, const char *message) { ++checks; if (!ok) throw std::runtime_error(message); }
void cfg_check(aclrtLaunchKernelCfg *cfg, unsigned blocks) {
    require(cfg && cfg->numAttrs == 1 && cfg->attrs[0].id == ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE &&
            cfg->attrs[0].value.schemMode == (blocks == 1 ? 0 : 1), "Scheduling ABI changed");
}
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, uint32_t blocks, aclrtStream stream,
    aclrtLaunchKernelCfg *cfg, void *host, size_t bytes, aclrtPlaceHolderInfo *holders, size_t count) {
    calls.push_back(4); cfg_check(cfg, blocks);
    require(function && stream && !holders && !count, "Inline pointer placeholders introduced");
    require(bytes == (blocks == 1 ? 16 : 72), "Frozen payload ABI changed");
    std::memcpy(sdk_copied.data(), host, bytes); copied_bytes = bytes;
    return fail == 4 ? 8844 : 0;
}
extern "C" aclError aclrtKernelArgsAppend(aclrtArgsHandle handle, void *param, size_t bytes, aclrtParamHandle *out) {
    calls.push_back(1); require(handle && param && out, "SDK handle/param absent");
    if (fail == 1) return 8811;
    std::memcpy(sdk_copied.data(), param, bytes); copied_bytes = bytes; *out = reinterpret_cast<void *>(9); return 0;
}
extern "C" aclError aclrtKernelArgsFinalize(aclrtArgsHandle handle) {
    calls.push_back(2); require(handle && calls.front() == 1, "Finalize used as a free operation");
    return fail == 2 ? 8822 : 0;
}
extern "C" aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle function, uint32_t blocks, aclrtStream stream,
    aclrtLaunchKernelCfg *cfg, aclrtArgsHandle handle, void *reserve) {
    calls.push_back(3); cfg_check(cfg, blocks);
    require(function && stream && handle && !reserve && calls == std::vector<unsigned>{1, 2, 3}, "SDK construction order changed");
    require(copied_bytes == (blocks == 1 ? 16 : 72), "WithConfig missing SDK-managed payload");
    return fail == 3 ? 8833 : 0;
}
int main() {
    void *function = reinterpret_cast<void *>(1), *stream = reinterpret_cast<void *>(2), *handle = reinterpret_cast<void *>(3);
    Gmm2PlannerArgsV3 planner{0x120000001000, 0x120000002080};
    Gmm2MsdArgsV3 gmm{0x120000000000, 1, 2, 3, 4, 5, 6, 7, 0x120001000000};
    require(gmm2_launch_v3_host_args(function, stream, &planner, 16, 1, 0) == 0 &&
            std::memcmp(sdk_copied.data(), &planner, 16) == 0, "Planner values/order changed");
    const auto saved = sdk_copied;
    planner.frame = 0;
    require(sdk_copied == saved, "Mock did not model independent argument copy");
    calls.clear();
    require(gmm2_launch_v3_host_args(function, stream, &gmm, 72, 24, 1) == 0 &&
            std::memcmp(sdk_copied.data(), &gmm, 72) == 0, "GMM values/order changed");
    for (unsigned error : {0, 1, 2, 3}) {
        calls.clear(); fail = error; unsigned stage = 99;
        int code = gmm2_launch_v3_with_config(function, stream, handle, &gmm, 72, 24, 1, &stage);
        require(code == (error ? 8800 + 11 * error : 0) && stage == error, "Precise SDK error/stage lost");
        require(calls.size() == (error ? error : 3), "Submission continued after SDK error");
    }
    calls.clear(); fail = 4;
    require(gmm2_launch_v3_host_args(function, stream, &planner, 16, 1, 0) == 8844, "Host API error lost");
    calls.clear(); fail = 0;
    require(gmm2_launch_v3_host_args(function, stream, &planner, 16, 24, 1) == GMM2_LAUNCH_V3_INVALID_ARGUMENT && calls.empty(), "Wrong planner geometry reached SDK");
    require(gmm2_launch_v3_with_config(function, stream, nullptr, &gmm, 72, 24, 1, nullptr) == GMM2_LAUNCH_V3_INVALID_ARGUMENT && calls.empty(), "Host bytes treated as SDK handle");
    std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"real_SDK_runtime_calls\":0,\"NPU_launches\":0}" << std::endl;
}
