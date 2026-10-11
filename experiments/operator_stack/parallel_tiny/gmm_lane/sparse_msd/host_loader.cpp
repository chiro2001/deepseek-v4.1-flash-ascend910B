#include "host_adapter.h"
#include <acl/acl_rt.h>
#include <initializer_list>
#include <new>

struct Gmm2SparseLibrary {
    aclrtBinHandle planner_bin{}, native_bin{}, candidate_bin{};
    Gmm2SparseFunctions functions{};
};

namespace {
int load(const char *path, const char *name, uint32_t magic, bool mixed,
         aclrtBinHandle *binary, void **function) {
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    option.value.magic = magic;
    aclrtBinaryLoadOptions options{&option, 1};
    aclError error = aclrtBinaryLoadFromFile(path, &options, binary);
    if (error) return error;
    error = aclrtBinaryGetFunction(*binary, name, function);
    if (error) return error;
    void *aic{}, *aiv{};
    error = aclrtGetFunctionAddr(*function, &aic, &aiv);
    if (error) return error;
    if (!aiv || (mixed ? !aic : aic != nullptr)) return GMM2_SPARSE_WRONG_ENGINE_PAIR;
    return 0;
}
}

extern "C" int gmm2_sparse_msd_open(const char *planner, const char *native,
                                   const char *candidate, Gmm2SparseLibrary **output) {
    if (!planner || !native || !candidate || !output) return GMM2_SPARSE_INVALID_ARGUMENT;
    *output = nullptr;
    auto *library = new (std::nothrow) Gmm2SparseLibrary{};
    if (!library) return GMM2_SPARSE_INVALID_ARGUMENT;
    int status = load(planner, "gmm2_sparse_row_plan", ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE,
                      false, &library->planner_bin, &library->functions.planner);
    if (!status) status = load(native, "Gmm2NativeMsdDown_control_v6", ACL_RT_BINARY_MAGIC_ELF_AICORE,
                               true, &library->native_bin, &library->functions.native);
    if (!status) status = load(candidate, "Gmm2SparseMsdDown_candidate_v6", ACL_RT_BINARY_MAGIC_ELF_AICORE,
                               true, &library->candidate_bin, &library->functions.candidate);
    if (status) { gmm2_sparse_msd_close(library); return status; }
    *output = library;
    return 0;
}

extern "C" int gmm2_sparse_msd_functions(const Gmm2SparseLibrary *library, Gmm2SparseFunctions *output) {
    if (!library || !output) return GMM2_SPARSE_INVALID_ARGUMENT;
    *output = library->functions;
    return 0;
}

extern "C" int gmm2_sparse_msd_close(Gmm2SparseLibrary *library) {
    if (!library) return 0;
    int first_error = 0;
    for (auto binary : {library->candidate_bin, library->native_bin, library->planner_bin}) {
        if (binary) {
            const aclError status = aclrtBinaryUnLoad(binary);
            if (!first_error && status) first_error = status;
        }
    }
    delete library;
    return first_error;
}
