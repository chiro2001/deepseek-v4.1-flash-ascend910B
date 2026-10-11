// Uses mock launch functions only; does not link or load ACL/runtime libraries.
#include "host_adapter.h"
#include <acl/acl_rt.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Launch {
    void *function, *stream;
    uint32_t blocks;
    size_t bytes;
    uint8_t schedule;
    std::array<uint64_t, 9> args{};
};
std::vector<Launch> launches;
int planner_error = 0;
int checks = 0;
void require(bool condition, const char *message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

extern "C" aclError aclrtLaunchKernel(aclrtFuncHandle function, uint32_t blocks,
                                       const void *args, size_t bytes, aclrtStream stream) {
    Launch launch{function, stream, blocks, bytes, 0};
    require(bytes == 16, "Planner argument bytes differ from ELF");
    std::memcpy(launch.args.data(), args, bytes);
    launches.push_back(launch);
    return planner_error;
}
extern "C" aclError aclrtLaunchKernelV2(aclrtFuncHandle function, uint32_t blocks,
                                         const void *args, size_t bytes,
                                         aclrtLaunchKernelCfg *cfg, aclrtStream stream) {
    require(cfg && cfg->numAttrs == 1 &&
            cfg->attrs[0].id == ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE,
            "Native schedule attribute missing");
    require(bytes == 72, "GMM argument bytes differ from ELF");
    Launch launch{function, stream, blocks, bytes, cfg->attrs[0].value.schemMode};
    std::memcpy(launch.args.data(), args, bytes);
    launches.push_back(launch);
    return 0;
}

int main() {
    Gmm2SparseBuffers b{};
    b.x = {0x100000, 6 * 2304};
    b.weight = {0x20000000, 48ULL * 2304 * 5120 / 2};
    b.assist = {0x40000000, 48 * 5120 * 4};
    b.scale = {0x41000000, 48 * 5120 * 8};
    b.counts = {0x42000000, 48 * 8};
    b.token_scale = {0x43000000, 6 * 4};
    b.y = {0x44000000, 6 * 5120 * 2};
    b.x_list = {0x45000000, 32}; b.weight_list = {0x45000020, 32};
    b.assist_list = {0x45000040, 32}; b.scale_list = {0x45000060, 32};
    b.y_list = {0x45000080, 32};
    b.workspace = {0x80000000, gmm2_sparse_msd_workspace_bytes()};
    b.ffts_address = 0x50000000;
    b.row_capacity = 6; b.experts = 48; b.k = 2304; b.n = 5120; b.group_list_type = 1;
    require(gmm2_sparse_msd_validate(&b) == 0, "Valid profile rejected");
    require(b.workspace.bytes == 117440640, "Workspace prefix omitted");
    require(gmm2_sparse_msd_frame_offset() == 16777216, "System reservation differs");
    uint64_t lists[20]{};
    require(gmm2_sparse_msd_pack_lists(&b, lists) == 0, "List packing failed");
    for (size_t i = 0; i < 5; ++i) {
        require(lists[4 * i] == 24 && lists[4 * i + 1] == (1ULL << 32) &&
                lists[4 * i + 2] == 0xffffffff, "Pointer-list descriptor ABI differs");
    }
    require(lists[3] == b.x.address && lists[7] == b.weight.address &&
            lists[11] == b.assist.address && lists[15] == b.scale.address &&
            lists[19] == b.y.address, "Pointer-list data address order differs");
    Gmm2SparseFunctions functions{reinterpret_cast<void *>(1), reinterpret_cast<void *>(2),
                                  reinterpret_cast<void *>(3)};
    void *stream = reinterpret_cast<void *>(4);
    require(gmm2_sparse_msd_launch(&functions, &b, stream, 0) == 0, "Native mock launch failed");
    require(launches.size() == 1 && launches[0].function == functions.native &&
            launches[0].blocks == 24 && launches[0].schedule == 1, "Native launch ABI differs");
    launches.clear();
    require(gmm2_sparse_msd_launch(&functions, &b, stream, 1) == 0, "Sparse mock launch failed");
    require(launches.size() == 2 && launches[0].function == functions.planner &&
            launches[1].function == functions.candidate && launches[0].blocks == 1 &&
            launches[1].blocks == 24 && launches[1].schedule == 1, "Sparse launch ordering differs");
    require(launches[0].stream == stream && launches[1].stream == stream, "Cross-stream planner race");
    require(launches[0].args[0] == b.counts.address &&
            launches[0].args[1] == b.workspace.address + 16777216, "Planner frame points into system area");
    require(launches[1].args == std::array<uint64_t, 9>{b.ffts_address, b.x_list.address,
            b.weight_list.address, b.assist_list.address, b.scale_list.address, b.counts.address,
            b.token_scale.address, b.y_list.address, b.workspace.address}, "GMM argument order differs");
    launches.clear();
    auto bad = b; bad.x.address = b.workspace.address + 16777216;
    require(gmm2_sparse_msd_launch(&functions, &bad, stream, 1) == GMM2_SPARSE_OVERLAPPING_BUFFERS &&
            launches.empty(), "Aliasing not rejected before SDK call");
    bad = b; bad.workspace.bytes -= 1;
    require(gmm2_sparse_msd_validate(&bad) == GMM2_SPARSE_INVALID_EXTENT, "Short workspace accepted");
    bad = b; bad.workspace.address = UINT64_MAX - 511;
    require(gmm2_sparse_msd_validate(&bad) == GMM2_SPARSE_INVALID_EXTENT, "Address overflow accepted");
    bad = b; bad.group_list_type = 0;
    require(gmm2_sparse_msd_validate(&bad) == GMM2_SPARSE_INVALID_PROFILE, "Prefix-sum counts accepted");
    bad = b; bad.counts.bytes = 383;
    require(gmm2_sparse_msd_validate(&bad) == GMM2_SPARSE_INVALID_EXTENT, "Short counts accepted");
    bad = b; bad.ffts_address = 0;
    require(gmm2_sparse_msd_validate(&bad) == GMM2_SPARSE_INVALID_ARGUMENT, "Missing FFTS accepted");
    planner_error = 907;
    require(gmm2_sparse_msd_launch(&functions, &b, stream, 1) == 907 && launches.size() == 1,
            "Candidate launched after failed planner submission");
    std::cout << "{\"passed\":true,\"checks\":" << checks
              << ",\"real_SDK_runtime_calls\":0,\"NPU_launches\":0}" << std::endl;
}
