// Borrow the original Torch SDK context/stream. Own only probe GM and modules.
// No init/set/reset/finalize, SDK installation, timing or model execution here.
#include "loader_v2.h"
#include "launch_v3.h"
#include <acl/acl.h>
#include <runtime/dev.h>
#include <runtime/rt_ffts.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {
constexpr uint64_t SYSTEM = 16ull * 1024 * 1024;
constexpr uint64_t WORKSPACE = 117440640;
constexpr uint64_t Y_BYTES = 6ull * 5120 * 2;
static_assert(ACL_MEM_MALLOC_HUGE_FIRST == 0 && ACL_MEM_MALLOC_NORMAL_ONLY == 2 && ACL_HBM_MEM == 1,
              "Installed public SDK memory policy/attribute ABI");
struct Failure { int code; };
int report(const char *stage, int code) {
    std::cout << "{\"bridge_stage\":\"" << stage << "\",\"code\":" << code << "}" << std::endl;
    if (code) {
        const char *error = aclGetRecentErrMsg();
        std::cerr << "bridge failure: " << stage << " code=" << code << " SDK=" << (error ? error : "") << std::endl;
    }
    return code;
}
void check(const char *stage, int code) { if (report(stage, code)) throw Failure{code}; }
uint64_t addr(void *p) { return reinterpret_cast<uintptr_t>(p); }
struct Probe {
    aclrtContext borrowed_context{};
    aclrtStream borrowed_stream{};
    Gmm2SparseLibraryV2 *library{};
    Gmm2SparseLoaderV2Status loader_status{};
    Gmm2SparseFunctions functions{};
    void *workspace_allocation{}, *y_allocation{};
    std::array<void *, 5> lists{};
    uint64_t ffts{}; uint32_t ffts_bytes{};
    uint64_t native_launches{}, candidate_launches{}, planner_launches{};
    bool synchronized = true;
};
int mapping() {
    for (const char *key : {"ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"}) {
        const char *value = std::getenv(key);
        if (!value || std::string(value) != "6") return report("physical6_env_guard", -910801);
    }
    uint32_t count{}, logical = 0xffffffff, physical = 0xffffffff;
    check("aclrtGetDeviceCount", aclrtGetDeviceCount(&count));
    check("single_visible_guard", count == 1 ? 0 : -910802);
    check("rtGetDeviceIndexByPhyId(6)", rtGetDeviceIndexByPhyId(6, &logical));
    check("rtGetDevicePhyIdByIndex", rtGetDevicePhyIdByIndex(logical, &physical));
    std::cout << "{\"bridge_mapping\":true,\"visible_count\":" << count
              << ",\"logical\":" << logical << ",\"physical\":" << physical << "}" << std::endl;
    check("logical0_physical6_guard", logical == 0 && physical == 6 ? 0 : -910803);
    return 0;
}
void check_canaries(Probe *p) {
    std::array<uint8_t, 512> data{};
    check("D2H:workspace_prefix", aclrtMemcpy(data.data(), 128, p->workspace_allocation, 128, ACL_MEMCPY_DEVICE_TO_HOST));
    check("D2H:workspace_suffix", aclrtMemcpy(data.data() + 128, 128,
          reinterpret_cast<void *>(addr(p->workspace_allocation) + 128 + WORKSPACE), 128, ACL_MEMCPY_DEVICE_TO_HOST));
    check("D2H:y_prefix", aclrtMemcpy(data.data() + 256, 128, p->y_allocation, 128, ACL_MEMCPY_DEVICE_TO_HOST));
    check("D2H:y_suffix", aclrtMemcpy(data.data() + 384, 128,
          reinterpret_cast<void *>(addr(p->y_allocation) + 128 + Y_BYTES), 128, ACL_MEMCPY_DEVICE_TO_HOST));
    bool intact = true;
    for (uint8_t value : data) intact &= value == 0xa5;
    check("workspace_and_y_canary_guard", intact ? 0 : -910807);
}
}

extern "C" int m1_probe_mapping() {
    try { return mapping(); } catch (Failure failure) { return failure.code; }
}

extern "C" int m1_probe_open(const char *root, uint64_t stream, void **owner) {
    if (!root || !stream || !owner || *owner || std::string(root) != "/work/gmm_lane_sparse_msd_build_20261011_v3_repro") return -910804;
    Probe *p = new Probe; *owner = p; p->borrowed_stream = reinterpret_cast<aclrtStream>(stream);
    try {
        check("mapping_guard", mapping());
        check("aclrtGetCurrentContext:borrowed", aclrtGetCurrentContext(&p->borrowed_context));
        check("borrowed_context_nonnull", p->borrowed_context ? 0 : -910805);
        int32_t current = -1;
        check("aclrtGetDevice:borrowed", aclrtGetDevice(&current));
        check("borrowed_logical0_guard", current == 0 ? 0 : -910806);
        std::cout << "{\"borrowed_context\":" << addr(p->borrowed_context)
                  << ",\"borrowed_stream\":" << stream << ",\"context_owner\":\"Torch original SDK consumer\"}" << std::endl;
        const std::string base = root;
        check("loader_v2_open", gmm2_sparse_loader_v2_open((base + "/planner_cpu_rebuild.o").c_str(),
            (base + "/op_build_native/Gmm2NativeMsdDown_control_v6.o").c_str(),
            (base + "/op_build_candidate/Gmm2SparseMsdDown_candidate_v6.o").c_str(), &p->library, &p->loader_status));
        check("loader_v2_functions", gmm2_sparse_loader_v2_functions(p->library, &p->functions));
        std::array<void *, 3> functions{p->functions.planner, p->functions.native, p->functions.candidate};
        std::array<const char *, 3> names{"planner", "native", "candidate"};
        std::array<size_t, 3> aic_expected{0, 5152, 8748}, aiv_expected{368, 8440, 8440};
        for (unsigned i = 0; i < functions.size(); ++i) {
            void *aic{}, *aiv{}; size_t aic_bytes{}, aiv_bytes{};
            check("aclrtGetFunctionAddr", aclrtGetFunctionAddr(functions[i], &aic, &aiv));
            check("aclrtGetFunctionSize", aclrtGetFunctionSize(functions[i], &aic_bytes, &aiv_bytes));
            std::cout << "{\"registered_function\":\"" << names[i] << "\",\"aic_address\":" << addr(aic)
                      << ",\"aiv_address\":" << addr(aiv) << ",\"aic_bytes\":" << aic_bytes
                      << ",\"aiv_bytes\":" << aiv_bytes << "}" << std::endl;
            check("strict_engine_pair_and_size", aiv && (i == 0 ? !aic : !!aic) &&
                  aic_bytes == aic_expected[i] && aiv_bytes == aiv_expected[i] ? 0 : -910808);
        }
        check("rtGetC2cCtrlAddr:current_context", rtGetC2cCtrlAddr(&p->ffts, &p->ffts_bytes));
        check("ffts_nonnull_guard", p->ffts ? 0 : -910809);
        // Installed Torch uses policy0 for large allocations; v2's explicit
        // NORMAL_ONLY failed despite >55GB aggregate HBM free in this context.
        // Keep full workspace capacity and all ownership/canary/ABI guards.
        size_t hbm_free = 0, hbm_total = 0;
        check("aclrtGetMemInfo:HBM_before_workspace", aclrtGetMemInfo(ACL_HBM_MEM, &hbm_free, &hbm_total));
        std::cout << "{\"workspace_allocator_policy\":\"ACL_MEM_MALLOC_HUGE_FIRST\",\"enum_value\":0,\"requested_bytes\":"
                  << WORKSPACE + 256 << ",\"HBM_free_bytes\":" << hbm_free << ",\"HBM_total_bytes\":" << hbm_total << "}" << std::endl;
        check("aclrtMalloc:workspace+canaries:HUGE_FIRST", aclrtMalloc(&p->workspace_allocation, WORKSPACE + 256, ACL_MEM_MALLOC_HUGE_FIRST));
        check("aclrtMalloc:y+canaries", aclrtMalloc(&p->y_allocation, Y_BYTES + 256, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (void *&list : p->lists) check("aclrtMalloc:single_tensor_list32", aclrtMalloc(&list, 32, ACL_MEM_MALLOC_NORMAL_ONLY));
        std::cout << "{\"probe_workspace_base\":" << addr(p->workspace_allocation) + 128
                  << ",\"workspace_bytes\":" << WORKSPACE << ",\"planner_frame_address\":" << addr(p->workspace_allocation) + 128 + SYSTEM
                  << ",\"candidate_native_user_scratch\":" << addr(p->workspace_allocation) + 128 + SYSTEM + 128
                  << ",\"probe_y_address\":" << addr(p->y_allocation) + 128 << ",\"ffts\":" << p->ffts
                  << ",\"ffts_bytes\":" << p->ffts_bytes << "}" << std::endl;
        return 0;
    } catch (Failure failure) { return failure.code; }
}

extern "C" int m1_probe_read(uint64_t gm, void *host, uint64_t bytes) {
    if (!gm || !host || !bytes || bytes > 8 * 1024 * 1024) return -910810;
    return report("D2H:original_parameter_storage", aclrtMemcpy(host, bytes, reinterpret_cast<void *>(gm), bytes, ACL_MEMCPY_DEVICE_TO_HOST));
}

extern "C" int m1_probe_run(void *owner, uint64_t x, uint64_t weight, uint64_t assist,
                           uint64_t scale, uint64_t counts, uint64_t token_scale,
                           uint32_t sparse, uint16_t *output, uint32_t *frame) {
    Probe *p = static_cast<Probe *>(owner);
    if (!p || !output || !frame || sparse > 1 || !p->workspace_allocation || !p->y_allocation) return -910811;
    try {
        aclrtContext current_context{};
        check("aclrtGetCurrentContext:per_case", aclrtGetCurrentContext(&current_context));
        check("same_borrowed_context_guard", current_context == p->borrowed_context ? 0 : -910812);
        Gmm2SparseBuffers b{};
        b.x = {x, 6ull * 2304}; b.weight = {weight, 48ull * 2304 * 5120 / 2};
        b.assist = {assist, 48ull * 5120 * 4}; b.scale = {scale, 48ull * 5120 * 8};
        b.counts = {counts, 384}; b.token_scale = {token_scale, 24};
        b.y = {addr(p->y_allocation) + 128, Y_BYTES};
        b.x_list = {addr(p->lists[0]), 32}; b.weight_list = {addr(p->lists[1]), 32};
        b.assist_list = {addr(p->lists[2]), 32}; b.scale_list = {addr(p->lists[3]), 32}; b.y_list = {addr(p->lists[4]), 32};
        b.workspace = {addr(p->workspace_allocation) + 128, WORKSPACE};
        b.ffts_address = p->ffts; b.row_capacity = 6; b.experts = 48; b.k = 2304; b.n = 5120; b.group_list_type = 1;
        check("frozen_buffer_contract", gmm2_sparse_msd_validate(&b));
        std::array<uint64_t, 20> words{};
        check("frozen_single_tensor_list_pack", gmm2_sparse_msd_pack_lists(&b, words.data()));
        for (unsigned i = 0; i < 5; ++i) check("H2D:single_tensor_list32",
            aclrtMemcpy(p->lists[i], 32, words.data() + 4 * i, 32, ACL_MEMCPY_HOST_TO_DEVICE));
        // Scratch and all guards are reset outside any capture; no timing claim.
        check("memset:workspace_canaries", aclrtMemset(p->workspace_allocation, WORKSPACE + 256, 0xa5, WORKSPACE + 256));
        check("memset:workspace", aclrtMemset(reinterpret_cast<void *>(b.workspace.address), WORKSPACE, 0, WORKSPACE));
        check("memset:y_canaries_and_unused_rows", aclrtMemset(p->y_allocation, Y_BYTES + 256, 0xa5, Y_BYTES + 256));
        Gmm2PlannerArgsV3 planner{counts, b.workspace.address + SYSTEM};
        Gmm2MsdArgsV3 args{p->ffts, b.x_list.address, b.weight_list.address, b.assist_list.address,
                          b.scale_list.address, counts, token_scale, b.y_list.address, b.workspace.address};
        std::cout << "{\"raw_launch_pointers\":true,\"sparse\":" << sparse << ",\"args_API\":\"aclrtLaunchKernelWithHostArgs\",\"GMM_bytes\":72,\"placeholder_count\":0,\"words\":[";
        const uint64_t *payload = reinterpret_cast<const uint64_t *>(&args);
        for (unsigned i = 0; i < 9; ++i) std::cout << (i ? "," : "") << payload[i];
        std::cout << "],\"tensor_GM_addresses\":[" << x << "," << weight << "," << assist << "," << scale << "," << counts << "," << token_scale << "," << b.y.address << "],\"single_tensor_lists\":[";
        for (unsigned i = 0; i < 20; ++i) std::cout << (i ? "," : "") << words[i];
        std::cout << "]}" << std::endl;
        p->synchronized = false;
        if (sparse) {
            ++p->planner_launches;
            check("HostArgs:planner_included", gmm2_launch_v3_host_args(p->functions.planner, p->borrowed_stream, &planner, 16, 1, 0));
            ++p->candidate_launches;
        } else ++p->native_launches;
        check(sparse ? "HostArgs:candidate" : "HostArgs:native",
              gmm2_launch_v3_host_args(sparse ? p->functions.candidate : p->functions.native, p->borrowed_stream, &args, 72, 24, 1));
        check("sync:raw_math", aclrtSynchronizeStreamWithTimeout(p->borrowed_stream, 10000));
        p->synchronized = true;
        check_canaries(p);
        check("D2H:raw_BF16_output", aclrtMemcpy(output, Y_BYTES, reinterpret_cast<void *>(b.y.address), Y_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        if (sparse) check("D2H:planner_frame_after_math", aclrtMemcpy(frame, 128, reinterpret_cast<void *>(planner.frame), 128, ACL_MEMCPY_DEVICE_TO_HOST));
        return 0;
    } catch (Failure failure) { return failure.code; }
}

extern "C" int m1_probe_close(void **owner) {
    if (!owner || !*owner) return 0;
    Probe *p = static_cast<Probe *>(*owner); int first_error = 0;
    auto cleanup = [&](const char *stage, int code) {
        report(stage, code); if (!first_error && code) first_error = code;
    };
    if (!p->synchronized) cleanup("cleanup:borrowed_stream_sync", aclrtSynchronizeStreamWithTimeout(p->borrowed_stream, 10000));
    for (void *&list : p->lists) if (list) { cleanup("cleanup:single_tensor_list", aclrtFree(list)); list = nullptr; }
    if (p->y_allocation) { cleanup("cleanup:y", aclrtFree(p->y_allocation)); p->y_allocation = nullptr; }
    if (p->workspace_allocation) { cleanup("cleanup:workspace", aclrtFree(p->workspace_allocation)); p->workspace_allocation = nullptr; }
    if (p->library) cleanup("cleanup:loader_v2_close", gmm2_sparse_loader_v2_close(&p->library, &p->loader_status));
    std::cout << "{\"bridge_terminal\":true,\"native_launch_attempts\":" << p->native_launches
              << ",\"candidate_launch_attempts\":" << p->candidate_launches
              << ",\"planner_launch_attempts\":" << p->planner_launches
              << ",\"live_binaries\":" << p->loader_status.live_binaries << ",\"blob_bytes\":" << p->loader_status.owned_blob_bytes
              << ",\"owner_retained\":" << (p->library ? "true" : "false")
              << ",\"context_stream_owner\":\"Torch; released at this process exit\",\"explicit_device_reset_calls\":0}" << std::endl;
    if (!p->library) { delete p; *owner = nullptr; }
    return first_error;
}
