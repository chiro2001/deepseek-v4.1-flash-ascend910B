// Real frozen ELF inputs, mock SDK functions, and no ACL/runtime library linkage.
#include "loader_v2.h"
#include <acl/acl_rt.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int checks = 0, load_calls = 0, lookup_calls = 0;
int fail_load = -1, fail_lookup = -1, wrong_engine = -1, fail_unload = -1;
bool partial_load_handle = false;
struct Record { const uint8_t *source; std::vector<uint8_t> copied; uint32_t magic; bool live; unsigned index; };
std::vector<std::unique_ptr<Record>> records;
std::vector<unsigned> unload_order;
void require(bool condition, const char *message) {
    ++checks; if (!condition) throw std::runtime_error(message);
}
void live_blobs() {
    for (const auto &r : records) if (r->live) require(std::memcmp(r->source, r->copied.data(), r->copied.size()) == 0,
                                                     "Blob released/moved/modified while SDK handle live");
}
void reset() {
    for (const auto &r : records) require(!r->live, "Reset attempted with live SDK handle");
    records.clear(); unload_order.clear(); load_calls = lookup_calls = 0;
    fail_load = fail_lookup = wrong_engine = fail_unload = -1; partial_load_handle = false;
}

extern "C" aclError aclrtBinaryLoadFromData(const void *data, size_t bytes,
                                            const aclrtBinaryLoadOptions *options, aclrtBinHandle *handle) {
    live_blobs();
    const unsigned index = load_calls++;
    require(options && options->numOpt == 2 && options->options[0].type == ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD &&
            options->options[0].value.isLazyLoad == 0 && options->options[1].type == ACL_RT_BINARY_LOAD_OPT_MAGIC,
            "Data loader options differ from reviewed explicit types");
    const uint32_t magic = index ? ACL_RT_BINARY_MAGIC_ELF_AICORE : ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
    require(options->options[1].value.magic == magic, "Planner/mixed ELF magic changed");
    require(bytes > 64 && bytes < 1000000 && data, "Invalid ELF extent");
    if (static_cast<int>(index) == fail_load && !partial_load_handle) return 8711;
    auto record = std::make_unique<Record>(); record->source = static_cast<const uint8_t *>(data);
    record->copied.assign(record->source, record->source + bytes); record->magic = magic;
    record->live = true; record->index = index; *handle = record.get(); records.push_back(std::move(record));
    return static_cast<int>(index) == fail_load ? 8711 : 0;
}
extern "C" aclError aclrtBinaryGetFunction(aclrtBinHandle handle, const char *name, aclrtFuncHandle *function) {
    live_blobs(); ++lookup_calls;
    auto *r = static_cast<Record *>(handle);
    static const char *names[] = {"gmm2_sparse_row_plan", "Gmm2NativeMsdDown_control_v6", "Gmm2SparseMsdDown_candidate_v6"};
    require(r && r->live && name == std::string(names[r->index]), "Frozen kernel lookup name changed");
    if (static_cast<int>(r->index) == fail_lookup) return 8722;
    *function = handle; return 0;
}
extern "C" aclError aclrtGetFunctionAddr(aclrtFuncHandle function, void **aic, void **aiv) {
    live_blobs(); auto *r = static_cast<Record *>(function);
    *aic = r->index ? reinterpret_cast<void *>(0x1000 + r->index) : nullptr;
    *aiv = reinterpret_cast<void *>(0x2000 + r->index);
    if (static_cast<int>(r->index) == wrong_engine) {
        if (r->index) *aiv = nullptr; else *aic = reinterpret_cast<void *>(0x1000);
    }
    return 0;
}
extern "C" aclError aclrtBinaryUnLoad(aclrtBinHandle handle) {
    live_blobs(); auto *r = static_cast<Record *>(handle);
    require(r && r->live, "Duplicate/invalid unload"); unload_order.push_back(r->index);
    if (static_cast<int>(r->index) == fail_unload) { fail_unload = -1; return 8733; }
    r->live = false; return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const std::string root = argv[1];
    const std::string planner = root + "/planner_cpu_rebuild.o";
    const std::string native = root + "/op_build_native/Gmm2NativeMsdDown_control_v6.o";
    const std::string candidate = root + "/op_build_candidate/Gmm2SparseMsdDown_candidate_v6.o";
    Gmm2SparseLibraryV2 *owner = nullptr; Gmm2SparseLoaderV2Status status{};
    auto open = [&]() { return gmm2_sparse_loader_v2_open(planner.c_str(), native.c_str(), candidate.c_str(), &owner, &status); };
    reset();
    require(open() == 0 && owner && status.live_binaries == 3 && status.stage == 13, "Valid frozen ELF rejected");
    require(status.owned_blob_bytes == 34560 + 19304 + 23488, "Owned blob sizes differ from frozen artifacts");
    Gmm2SparseFunctions functions{};
    require(gmm2_sparse_loader_v2_functions(owner, &functions) == 0 && functions.planner && functions.native && functions.candidate,
            "Function handles unavailable after strict verification");
    auto *original_owner = owner;
    require(open() == GMM2_LOADER_V2_INVALID_ARGUMENT && owner == original_owner && load_calls == 3,
            "Open overwrote a live library owner");
    require(gmm2_sparse_loader_v2_close(&owner, &status) == 0 && !owner && !status.live_binaries && !status.owned_blob_bytes,
            "Successful unload did not release ownership");
    require(unload_order == std::vector<unsigned>{2, 1, 0}, "Close order changed");

    for (int index = 0; index < 3; ++index) {
        reset(); fail_load = index;
        require(open() == 8711 && !owner && status.primary_error == 8711 && status.stage == unsigned(2 + 4 * index),
                "Data load failure lost exact SDK error/stage");
        require(load_calls == index + 1 && !status.live_binaries, "Load continued after failure");
        reset(); fail_lookup = index;
        require(open() == 8722 && !owner && status.primary_error == 8722 && status.stage == unsigned(3 + 4 * index),
                "Lookup failure lost exact SDK error/stage");
        require(!status.live_binaries && lookup_calls == index + 1, "Lookup failure did not rollback handles");
        reset(); wrong_engine = index;
        require(open() == GMM2_SPARSE_WRONG_ENGINE_PAIR && !owner && status.stage == unsigned(4 + 4 * index),
                "Vector/mixed engine mismatch accepted");
    }
    reset(); fail_load = 2; partial_load_handle = true;
    require(open() == 8711 && !owner && unload_order == std::vector<unsigned>{2, 1, 0},
            "Partial SDK output handle leaked after load failure");

    reset(); require(open() == 0, "Second successful open failed"); fail_unload = 2;
    require(gmm2_sparse_loader_v2_close(&owner, &status) == 8733 && owner && status.live_binaries == 1 &&
            status.owned_blob_bytes == 34560 + 19304 + 23488, "Blob dropped after unload failure");
    require(gmm2_sparse_loader_v2_status(owner, &status) == 0 && status.cleanup_error == 8733,
            "Unload error disappeared from retained-owner status");
    require(gmm2_sparse_loader_v2_functions(owner, &functions) == GMM2_LOADER_V2_INVALID_ARGUMENT,
            "Partially closed library returned usable function handles");
    require(gmm2_sparse_loader_v2_close(&owner, &status) == 0 && !owner && !status.owned_blob_bytes,
            "Retained handle cleanup failed");

    reset(); fail_load = 1; fail_unload = 0;
    require(open() == 8711 && owner && status.primary_error == 8711 && status.cleanup_error == 8733 && status.live_binaries == 1,
            "Rollback unload error lost owner/primary error");
    live_blobs();
    require(gmm2_sparse_loader_v2_close(&owner, &status) == 0 && !owner, "Rollback retained owner cleanup failed");

    reset();
    require(gmm2_sparse_loader_v2_open(native.c_str(), native.c_str(), candidate.c_str(), &owner, &status) ==
            GMM2_LOADER_V2_ELF_PROFILE_ERROR && !owner && !load_calls, "Wrong planner ELF profile reached SDK");
    require(gmm2_sparse_loader_v2_open("/missing/file", native.c_str(), candidate.c_str(), &owner, &status) ==
            GMM2_LOADER_V2_READ_ERROR && !owner && !load_calls, "Missing blob reached SDK");
    std::cout << "{\"passed\":true,\"checks\":" << checks
              << ",\"real_SDK_runtime_calls\":0,\"NPU_launches\":0,\"frozen_ELF_bytes_changed\":false}" << std::endl;
}
