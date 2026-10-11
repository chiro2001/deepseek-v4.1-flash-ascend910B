#ifndef GMM2_SPARSE_MSD_LOADER_V2_H
#define GMM2_SPARSE_MSD_LOADER_V2_H
#include "../host_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Gmm2SparseLibraryV2 Gmm2SparseLibraryV2;
typedef struct {
    int primary_error, cleanup_error;
    unsigned stage, live_binaries;
    uint64_t owned_blob_bytes;
} Gmm2SparseLoaderV2Status;
enum {
    GMM2_LOADER_V2_READ_ERROR = -910601,
    GMM2_LOADER_V2_ELF_PROFILE_ERROR = -910602,
    GMM2_LOADER_V2_INVALID_ARGUMENT = -910603,
};

// No context/device initialization or launch. Caller must have a legal context.
// On rollback unload failure, *owner remains non-null and owns all blob memory.
int gmm2_sparse_loader_v2_open(const char *planner_object, const char *native_object,
                              const char *candidate_object, Gmm2SparseLibraryV2 **owner,
                              Gmm2SparseLoaderV2Status *status);
int gmm2_sparse_loader_v2_functions(const Gmm2SparseLibraryV2 *owner, Gmm2SparseFunctions *functions);
int gmm2_sparse_loader_v2_status(const Gmm2SparseLibraryV2 *owner, Gmm2SparseLoaderV2Status *status);
// Clears *owner only after all SDK handles have successfully unloaded.
// Stream completion is the caller's responsibility before closing a live module.
int gmm2_sparse_loader_v2_close(Gmm2SparseLibraryV2 **owner, Gmm2SparseLoaderV2Status *status);
const char *gmm2_sparse_loader_v2_stage_name(unsigned stage);

#ifdef __cplusplus
}
#endif
#endif
