#include "loader_v2.h"
#include <acl/acl_rt.h>
#include <array>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace {
constexpr size_t MAX_BLOB_BYTES = 1000000;
const char *NAMES[] = {"gmm2_sparse_row_plan", "Gmm2NativeMsdDown_control_v6", "Gmm2SparseMsdDown_candidate_v6"};
struct Blob {
    std::vector<uint8_t> bytes; // Never moved or modified while SDK handle is live.
    aclrtBinHandle binary{};
    aclrtFuncHandle function{};
};
uint64_t number(const std::vector<uint8_t> &data, size_t offset, size_t bytes) {
    uint64_t value = 0;
    if (offset > data.size() || bytes > data.size() - offset) return UINT64_MAX;
    for (size_t i = 0; i < bytes; ++i) value |= uint64_t(data[offset + i]) << (8 * i);
    return value;
}
struct Section { uint64_t offset{}, bytes{}; };
bool section(const std::vector<uint8_t> &data, const std::string &wanted, Section *result) {
    const uint64_t table = number(data, 40, 8), stride = number(data, 58, 2);
    const uint64_t count = number(data, 60, 2), strings = number(data, 62, 2);
    if (!count || strings >= count || stride < 64 || table > data.size() || count > (data.size() - table) / stride) return false;
    const uint64_t strings_header = table + strings * stride;
    const uint64_t strings_offset = number(data, strings_header + 24, 8), strings_bytes = number(data, strings_header + 32, 8);
    if (strings_offset > data.size() || strings_bytes > data.size() - strings_offset) return false;
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t header = table + i * stride, name = number(data, header, 4);
        if (name >= strings_bytes) continue;
        const char *begin = reinterpret_cast<const char *>(data.data() + strings_offset + name);
        const void *end = std::memchr(begin, 0, strings_bytes - name);
        if (!end || std::string(begin, static_cast<const char *>(end)) != wanted) continue;
        result->offset = number(data, header + 24, 8); result->bytes = number(data, header + 32, 8);
        return result->offset <= data.size() && result->bytes <= data.size() - result->offset;
    }
    return false;
}
bool mix_meta(const std::vector<uint8_t> &data, const std::string &name) {
    Section meta;
    if (!section(data, ".ascend.meta." + name, &meta) || meta.bytes < 16) return false;
    const size_t p = meta.offset;
    // Generated native metadata: K_TYPE_MIX_AIC_MAIN=4, AIC:AIV task ratio1:2.
    return number(data, p, 2) == 1 && number(data, p + 2, 2) == 4 && number(data, p + 4, 4) == 4 &&
           number(data, p + 8, 2) == 3 && number(data, p + 10, 2) == 4 &&
           number(data, p + 12, 2) == 1 && number(data, p + 14, 2) == 2;
}
bool elf_profile(const std::vector<uint8_t> &data, unsigned index) {
    const uint8_t prefix[] = {0x7f, 'E', 'L', 'F', 2, 1};
    if (data.size() < 64 || std::memcmp(data.data(), prefix, sizeof(prefix))) return false;
    // Exact DAV2201 frozen ELF machine and planner/mixed flags, not inference
    // from a caller-provided MAGIC alone.
    if (number(data, 18, 2) != 0x1029 || number(data, 48, 4) != (index ? 0x940000 : 0x930000)) return false;
    Section text, args;
    if (!section(data, ".text", &text) || !text.bytes || !section(data, "__CCE_KernelArgSize", &args)) return false;
    if (index == 0) return args.bytes == 4 && number(data, args.offset, 4) == 16;
    return args.bytes == 8 && number(data, args.offset, 4) == 72 && number(data, args.offset + 4, 4) == 72 &&
           mix_meta(data, std::string(NAMES[index]) + "_mix_aic") &&
           mix_meta(data, std::string(NAMES[index]) + "_mix_aiv");
}
bool read_blob(const char *path, Blob *blob) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto length = file.tellg();
    if (length <= 0 || static_cast<uint64_t>(length) >= MAX_BLOB_BYTES) return false;
    blob->bytes.resize(static_cast<size_t>(length));
    file.seekg(0);
    return bool(file.read(reinterpret_cast<char *>(blob->bytes.data()), length));
}
}

struct Gmm2SparseLibraryV2 {
    std::array<Blob, 3> blobs;
    int primary_error{}, cleanup_error{}; unsigned stage{};
};

namespace {
void report(const Gmm2SparseLibraryV2 *owner, int cleanup_error, Gmm2SparseLoaderV2Status *status) {
    if (!status) return;
    *status = {};
    status->cleanup_error = cleanup_error;
    if (!owner) return;
    status->cleanup_error = owner->cleanup_error;
    status->primary_error = owner->primary_error; status->stage = owner->stage;
    for (const auto &blob : owner->blobs) {
        status->live_binaries += blob.binary != nullptr;
        status->owned_blob_bytes += blob.bytes.size();
    }
}
int unload(Gmm2SparseLibraryV2 *owner) {
    int first_error = 0;
    for (int i = 2; i >= 0; --i) {
        auto &blob = owner->blobs[i];
        if (!blob.binary) continue;
        const int error = aclrtBinaryUnLoad(blob.binary);
        if (error) { if (!first_error) first_error = error; }
        else { blob.binary = nullptr; blob.function = nullptr; }
    }
    return first_error;
}
int load_blob(Gmm2SparseLibraryV2 *owner, unsigned index) {
    auto &blob = owner->blobs[index];
    std::array<aclrtBinaryLoadOption, 2> options{};
    options[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
    options[0].value.isLazyLoad = 0;
    options[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    options[1].value.magic = index ? ACL_RT_BINARY_MAGIC_ELF_AICORE : ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
    aclrtBinaryLoadOptions cfg{options.data(), options.size()};
    owner->stage = 2 + 4 * index;
    int error = aclrtBinaryLoadFromData(blob.bytes.data(), blob.bytes.size(), &cfg, &blob.binary);
    if (error) return error;
    owner->stage = 3 + 4 * index;
    error = aclrtBinaryGetFunction(blob.binary, NAMES[index], &blob.function);
    if (error) return error;
    void *aic{}, *aiv{};
    owner->stage = 4 + 4 * index;
    error = aclrtGetFunctionAddr(blob.function, &aic, &aiv);
    if (error) return error;
    if (!aiv || (index ? !aic : aic != nullptr)) return GMM2_SPARSE_WRONG_ENGINE_PAIR;
    return 0;
}
}

extern "C" const char *gmm2_sparse_loader_v2_stage_name(unsigned stage) {
    static const char *names[] = {"none", "read/profile:planner", "load_data:planner", "get_function:planner", "engine:planner",
        "read/profile:native", "load_data:native", "get_function:native", "engine:native",
        "read/profile:candidate", "load_data:candidate", "get_function:candidate", "engine:candidate", "ready", "closing", "closed"};
    return stage < sizeof(names) / sizeof(names[0]) ? names[stage] : "unknown";
}
extern "C" int gmm2_sparse_loader_v2_open(const char *planner, const char *native, const char *candidate,
                                          Gmm2SparseLibraryV2 **output, Gmm2SparseLoaderV2Status *status) {
    if (status) *status = {};
    if (!planner || !native || !candidate || !output) return GMM2_LOADER_V2_INVALID_ARGUMENT;
    if (*output) return GMM2_LOADER_V2_INVALID_ARGUMENT;
    *output = nullptr;
    auto *owner = new (std::nothrow) Gmm2SparseLibraryV2{};
    if (!owner) return GMM2_LOADER_V2_READ_ERROR;
    const char *paths[] = {planner, native, candidate};
    int error = 0;
    try {
        for (unsigned i = 0; i < 3 && !error; ++i) {
            owner->stage = 1 + 4 * i;
            if (!read_blob(paths[i], &owner->blobs[i])) error = GMM2_LOADER_V2_READ_ERROR;
            else if (!elf_profile(owner->blobs[i].bytes, i)) error = GMM2_LOADER_V2_ELF_PROFILE_ERROR;
        }
        for (unsigned i = 0; i < 3 && !error; ++i) error = load_blob(owner, i);
    } catch (const std::bad_alloc &) { error = GMM2_LOADER_V2_READ_ERROR; }
    if (!error) { owner->stage = 13; report(owner, 0, status); *output = owner; return 0; }
    owner->primary_error = error;
    const int cleanup_error = unload(owner);
    owner->cleanup_error = cleanup_error;
    report(owner, cleanup_error, status);
    if (cleanup_error) *output = owner; // Retain every blob until the last handle closes.
    else { delete owner; if (status) status->owned_blob_bytes = 0; }
    return error;
}
extern "C" int gmm2_sparse_loader_v2_functions(const Gmm2SparseLibraryV2 *owner, Gmm2SparseFunctions *functions) {
    if (!owner || !functions || owner->stage != 13 || owner->primary_error) return GMM2_LOADER_V2_INVALID_ARGUMENT;
    *functions = {owner->blobs[0].function, owner->blobs[1].function, owner->blobs[2].function};
    return 0;
}
extern "C" int gmm2_sparse_loader_v2_status(const Gmm2SparseLibraryV2 *owner, Gmm2SparseLoaderV2Status *status) {
    if (!owner || !status) return GMM2_LOADER_V2_INVALID_ARGUMENT;
    report(owner, 0, status); return 0;
}
extern "C" int gmm2_sparse_loader_v2_close(Gmm2SparseLibraryV2 **handle, Gmm2SparseLoaderV2Status *status) {
    if (!handle) return GMM2_LOADER_V2_INVALID_ARGUMENT;
    if (!*handle) { if (status) *status = {}; return 0; }
    auto *owner = *handle;
    if (!owner->primary_error) owner->stage = 14;
    const int error = unload(owner);
    owner->cleanup_error = error;
    report(owner, error, status);
    if (!error) { delete owner; *handle = nullptr; if (status) { status->stage = 15; status->owned_blob_bytes = 0; } }
    return error;
}
