// New physical6 window: Data-register frozen objects, HostArgs R0 planner only.
// No GMM math launch, weight data, model, device reset or capture is present.
#include "loader_v2.h"
#include "launch_v3.h"
#include <acl/acl.h>
#include <runtime/dev.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

std::string quote(const std::string &value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << '?';
        else out << c;
    }
    out << '"'; return out.str();
}
struct Failure { std::string stage; int code; };
void checked(const char *stage, int code) {
    std::cout << "{\"stage\":" << quote(stage) << ",\"code\":" << code;
    if (code) {
        const char *message = aclGetRecentErrMsg();
        std::cout << ",\"SDK_recent_error\":" << quote(message ? message : "");
    }
    std::cout << "}" << std::endl;
    if (code) throw Failure{stage, code};
}

std::array<uint32_t, 32> oracle(const std::array<int64_t, 48> &counts) {
    std::array<uint32_t, 32> frame{}; frame[3] = 48;
    for (unsigned i = 0; i < 6; ++i) frame[4 + 4 * i] = 0xffffffff;
    uint32_t rows = 0, slots = 0;
    for (unsigned expert = 0; expert < 48; ++expert) {
        const int64_t count = counts[expert];
        if (count < 0 || count > 6 || rows + count > 6) { frame[2] = 1; return frame; }
        if (!count) continue;
        frame[4 + 4 * slots] = expert;
        frame[5 + 4 * slots] = rows;
        frame[6 + 4 * slots] = static_cast<uint32_t>(count);
        rows += count; ++slots;
    }
    frame[0] = rows; frame[1] = slots; return frame;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const std::string root = argv[1];
    if (root.find("gmm_lane_sparse_msd_build_20261011_v3_repro") == std::string::npos) return 2;
    for (const char *key : {"ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"}) {
        const char *value = std::getenv(key);
        if (!value || std::string(value) != "6") return 2;
    }
    bool initialized = false, registration_passed = false;
    unsigned planner_launches = 0, planner_cases_passed = 0;
    aclrtContext context{}; aclrtStream stream{};
    Gmm2SparseLibraryV2 *library{}; Gmm2SparseLoaderV2Status loader_status{}; Gmm2SparseFunctions functions{};
    void *counts_gm{}, *frame_allocation{}, *counts_host{};
    int terminal_code = 0; std::string terminal_stage;
    std::cout << "{\"container_PID\":" << static_cast<long>(getpid()) << ",\"authorized_physical\":6,\"only_R0\":true}" << std::endl;
    uint32_t count = 0, logical = 0xffffffff, physical = 0xffffffff;
    try {
        checked("aclInit", aclInit(nullptr)); initialized = true;
        checked("aclrtGetDeviceCount", aclrtGetDeviceCount(&count));
        checked("single_visible_device_guard", count == 1 ? 0 : -910501);
        checked("rtGetDeviceIndexByPhyId(6)", rtGetDeviceIndexByPhyId(6, &logical));
        checked("rtGetDevicePhyIdByIndex", rtGetDevicePhyIdByIndex(logical, &physical));
        std::cout << "{\"physical\":" << physical << ",\"logical\":" << logical
                  << ",\"visible_device_count\":" << count << "}" << std::endl;
        checked("physical6_mapping_guard", physical == 6 ? 0 : -910502);
        checked("logical0_guard", logical == 0 ? 0 : -910506);
        checked("aclrtCreateContext", aclrtCreateContext(&context, static_cast<int32_t>(logical)));
        checked("aclrtSetCurrentContext", aclrtSetCurrentContext(context));
        int32_t current = -1;
        checked("aclrtGetDevice", aclrtGetDevice(&current));
        checked("current_device_guard", current == static_cast<int32_t>(logical) ? 0 : -910503);
        const int open_status = gmm2_sparse_loader_v2_open(
            (root + "/planner_cpu_rebuild.o").c_str(),
            (root + "/op_build_native/Gmm2NativeMsdDown_control_v6.o").c_str(),
            (root + "/op_build_candidate/Gmm2SparseMsdDown_candidate_v6.o").c_str(), &library, &loader_status);
        std::cout << "{\"loader_stage\":" << quote(gmm2_sparse_loader_v2_stage_name(loader_status.stage))
                  << ",\"primary_error\":" << loader_status.primary_error << ",\"cleanup_error\":" << loader_status.cleanup_error
                  << ",\"live_binaries\":" << loader_status.live_binaries << ",\"owned_blob_bytes\":" << loader_status.owned_blob_bytes << "}" << std::endl;
        checked("gmm2_sparse_loader_v2_open", open_status);
        checked("gmm2_sparse_loader_v2_functions", gmm2_sparse_loader_v2_functions(library, &functions));
        for (const auto &entry : std::array<std::pair<const char *, void *>, 3>{{
                {"planner", functions.planner}, {"native", functions.native}, {"candidate", functions.candidate}}}) {
            void *aic{}, *aiv{};
            checked((std::string("aclrtGetFunctionAddr:") + entry.first).c_str(),
                    aclrtGetFunctionAddr(entry.second, &aic, &aiv));
            std::cout << "{\"function\":" << quote(entry.first) << ",\"aic_address\":"
                      << reinterpret_cast<uintptr_t>(aic) << ",\"aiv_address\":"
                      << reinterpret_cast<uintptr_t>(aiv) << "}" << std::endl;
            checked("function_engine_pair_guard", aiv && (entry.first == std::string("planner") ? !aic : !!aic) ? 0 : -910504);
            size_t aic_bytes = 0, aiv_bytes = 0;
            checked((std::string("aclrtGetFunctionSize:") + entry.first).c_str(),
                    aclrtGetFunctionSize(entry.second, &aic_bytes, &aiv_bytes));
            std::cout << "{\"function_size\":" << quote(entry.first) << ",\"aic_bytes\":" << aic_bytes
                      << ",\"aiv_bytes\":" << aiv_bytes << "}" << std::endl;
            checked("function_size_pair_guard", aiv_bytes && (entry.first == std::string("planner") ? !aic_bytes : !!aic_bytes) ? 0 : -910507);
        }
        registration_passed = true;
        checked("aclrtCreateStream", aclrtCreateStream(&stream));
        checked("aclrtMalloc:counts384B", aclrtMalloc(&counts_gm, 384, ACL_MEM_MALLOC_NORMAL_ONLY));
        checked("aclrtMalloc:frame+canaries384B", aclrtMalloc(&frame_allocation, 384, ACL_MEM_MALLOC_NORMAL_ONLY));
        checked("aclrtMallocHost:counts384B", aclrtMallocHost(&counts_host, 384));
        std::vector<std::array<int64_t, 48>> cases(1); // R0 only; no other geometry/sample in this window.
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto expected = oracle(cases[index]);
            std::memcpy(counts_host, cases[index].data(), 384);
            checked("aclrtMemcpyAsync:counts", aclrtMemcpyAsync(counts_gm, 384, counts_host, 384,
                                                                ACL_MEMCPY_HOST_TO_DEVICE, stream));
            checked("aclrtMemsetAsync:frame_canaries", aclrtMemsetAsync(frame_allocation, 384, 0xa5, 384, stream));
            Gmm2PlannerArgsV3 args{reinterpret_cast<uintptr_t>(counts_gm),
                                    reinterpret_cast<uintptr_t>(frame_allocation) + 128};
            std::cout << "{\"args_API\":\"aclrtLaunchKernelWithHostArgs\",\"bytes\":16,\"placeholders\":0,"
                      << "\"host_args_address\":" << reinterpret_cast<uintptr_t>(&args)
                      << ",\"counts_GM_address\":" << args.counts << ",\"frame_GM_address\":" << args.frame
                      << ",\"host_source_lifetime\":\"retained through sync and readback\"}" << std::endl;
            checked("gmm2_launch_v3_host_args:planner_R0_only",
                    gmm2_launch_v3_host_args(functions.planner, stream, &args, sizeof(args), 1, 0));
            ++planner_launches;
            checked("aclrtSynchronizeStreamWithTimeout", aclrtSynchronizeStreamWithTimeout(stream, 10000));
            std::array<uint8_t, 384> output{};
            checked("aclrtMemcpy:frame_canaries_D2H", aclrtMemcpy(output.data(), 384, frame_allocation, 384, ACL_MEMCPY_DEVICE_TO_HOST));
            bool canaries = true;
            for (unsigned i = 0; i < 128; ++i) canaries &= output[i] == 0xa5 && output[256 + i] == 0xa5;
            std::array<uint32_t, 32> actual{}; std::memcpy(actual.data(), output.data() + 128, 128);
            const bool equal = actual == expected;
            std::cout << "{\"case\":" << index << ",\"rows\":" << expected[0]
                      << ",\"slots\":" << expected[1] << ",\"frame_byte_exact\":" << (equal ? "true" : "false")
                      << ",\"canaries_intact\":" << (canaries ? "true" : "false") << ",\"frame_words\":[";
            for (size_t i = 0; i < actual.size(); ++i) std::cout << (i ? "," : "") << actual[i];
            std::cout << "]}" << std::endl;
            checked("planner_frame_oracle_guard", equal && canaries ? 0 : -910505);
            ++planner_cases_passed;
        }
    } catch (const Failure &failure) {
        terminal_code = failure.code; terminal_stage = failure.stage;
    }
    auto cleanup = [&](const char *name, int status) {
        std::cout << "{\"cleanup\":" << quote(name) << ",\"code\":" << status << "}" << std::endl;
        if (!terminal_code && status) { terminal_code = status; terminal_stage = name; }
    };
    if (stream) cleanup("aclrtDestroyStream", aclrtDestroyStream(stream));
    if (frame_allocation) cleanup("aclrtFree:frame", aclrtFree(frame_allocation));
    if (counts_gm) cleanup("aclrtFree:counts", aclrtFree(counts_gm));
    if (counts_host) cleanup("aclrtFreeHost:counts", aclrtFreeHost(counts_host));
    if (library) {
        cleanup("gmm2_sparse_loader_v2_close", gmm2_sparse_loader_v2_close(&library, &loader_status));
        std::cout << "{\"owner_retained_after_close\":" << (library ? "true" : "false")
                  << ",\"live_binaries_after_close\":" << loader_status.live_binaries
                  << ",\"blob_bytes_after_close\":" << loader_status.owned_blob_bytes << "}" << std::endl;
    }
    if (context) cleanup("aclrtDestroyContext", aclrtDestroyContext(context));
    if (initialized) cleanup("aclFinalize", aclFinalize());
    std::cout << "{\"terminal\":true,\"passed\":" << (!terminal_code ? "true" : "false")
              << ",\"terminal_stage\":" << quote(terminal_stage) << ",\"terminal_code\":" << terminal_code
              << ",\"registration_passed\":" << (registration_passed ? "true" : "false")
              << ",\"physical\":" << physical << ",\"logical\":" << logical
              << ",\"planner_launches\":" << planner_launches << ",\"planner_cases_passed\":" << planner_cases_passed
              << ",\"GMM_math_launches\":0,\"weight_bytes\":0,\"device_reset_calls\":0}" << std::endl;
    return terminal_code ? 1 : 0;
}
