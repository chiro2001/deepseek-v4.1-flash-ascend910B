// Real frozen host validator + CPU-only launch stubs; no ACL/runtime linkage.
#include "host_adapter.h"
#include "workspace_layout.h"
#include <acl/acl_rt.h>
#include <array>
#include <iostream>
#include <stdexcept>
static unsigned forbidden_launch_calls = 0;
extern "C" aclError aclrtLaunchKernel(aclrtFuncHandle, uint32_t, const void *, size_t, aclrtStream) {
    ++forbidden_launch_calls; return -919999;
}
extern "C" aclError aclrtLaunchKernelV2(aclrtFuncHandle, uint32_t, const void *, size_t,
                                       aclrtLaunchKernelCfg *, aclrtStream) {
    ++forbidden_launch_calls; return -919999;
}
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
Gmm2SparseBuffers profile() {
    Gmm2SparseBuffers b{};
    b.x = {0x100000, 6 * 2304}; b.weight = {0x20000000, 48ull * 2304 * 5120 / 2};
    b.assist = {0x40000000, 48 * 5120 * 4}; b.scale = {0x41000000, 48 * 5120 * 8};
    b.counts = {0x42000000, 384}; b.token_scale = {0x43000000, 24}; b.y = {0x44000000, 6 * 5120 * 2};
    b.x_list = {0x45000000, 32}; b.weight_list = {0x45000020, 32}; b.assist_list = {0x45000040, 32};
    b.scale_list = {0x45000060, 32}; b.y_list = {0x45000080, 32};
    b.ffts_address = 0x50000000; b.row_capacity = 6; b.experts = 48; b.k = 2304; b.n = 5120; b.group_list_type = 1;
    b.workspace.bytes = M1_WORKSPACE_BYTES; return b;
}
int main() {
    constexpr uint64_t observed_v3_workspace = 20618583998592;
    constexpr uint64_t observed_v3_allocation = observed_v3_workspace - 128;
    auto b = profile(); b.workspace.address = observed_v3_workspace;
    require(gmm2_sparse_msd_validate(&b) == GMM2_SPARSE_INVALID_EXTENT, "Real v3 extent failure not reproduced");
    uint64_t aligned{};
    require(m1_workspace_layout(observed_v3_allocation, M1_WORKSPACE_ALLOCATION_BYTES, &aligned), "Observed allocation cannot be planned");
    b.workspace.address = aligned;
    require(gmm2_sparse_msd_validate(&b) == 0, "Aligned real frozen contract rejected");
    unsigned geometries = 0;
    for (uint64_t modulo = 0; modulo < 512; ++modulo) {
        const uint64_t allocation = 0x120000000000ull + modulo;
        uint64_t base{};
        require(m1_workspace_layout(allocation, M1_WORKSPACE_ALLOCATION_BYTES, &base), "Legal backing rejected");
        require(base % 512 == 0 && base - 128 >= allocation, "Front canary/alignment invalid");
        require(base + M1_WORKSPACE_BYTES + 128 <= allocation + M1_WORKSPACE_ALLOCATION_BYTES, "Suffix canary outside owner");
        b.workspace.address = base;
        require(gmm2_sparse_msd_validate(&b) == 0, "Frozen 512B buffer guard rejected planned base");
        std::array<uint64_t, 20> lists{};
        require(gmm2_sparse_msd_pack_lists(&b, lists.data()) == 0, "Frozen pointer-list pack rejected");
        require(lists[3] == b.x.address && lists[7] == b.weight.address && lists[11] == b.assist.address &&
                lists[15] == b.scale.address && lists[19] == b.y.address, "Pointer-list order changed");
        require(gmm2_sparse_msd_frame_offset() == 16777216 && b.workspace.bytes == 117440640, "Effective frame/capacity changed");
        ++geometries;
    }
    require(!m1_workspace_layout(observed_v3_allocation, 117440896, &aligned), "Old short backing accepted");
    require(!m1_workspace_layout(UINT64_MAX - 512, M1_WORKSPACE_ALLOCATION_BYTES, &aligned), "Overflow accepted");
    require(!m1_workspace_layout(0, M1_WORKSPACE_ALLOCATION_BYTES, &aligned), "Zero GM accepted");
    require(forbidden_launch_calls == 0, "A launch stub was called during pure validation");
    std::cout << "{\"CPU_only\":true,\"passed\":true,\"real_SDK_runtime_calls\":0,\"NPU_launches\":0,"
              << "\"v3_failure_reproduced\":true,\"aligned_frozen_validator_passed\":true,\"allocation_modulo_cases\":" << geometries
              << ",\"effective_workspace_bytes\":" << M1_WORKSPACE_BYTES
              << ",\"backing_allocation_bytes\":" << M1_WORKSPACE_ALLOCATION_BYTES
              << ",\"canary_each_bytes\":128,\"workspace_alignment\":512}" << std::endl;
}
