#ifndef GMM2_SPARSE_MSD_HOST_ADAPTER_H
#define GMM2_SPARSE_MSD_HOST_ADAPTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed device context and stream only. This library never selects a device.
// Pointer-list descriptors must already reside in GM before either launch.
typedef struct { uint64_t address, bytes; } Gmm2SparseSpan;
typedef struct {
    Gmm2SparseSpan x, weight, assist, scale, counts, token_scale, y;
    Gmm2SparseSpan x_list, weight_list, assist_list, scale_list, y_list;
    Gmm2SparseSpan workspace;
    uint64_t ffts_address; // Caller obtains the current context's C2C control address.
    uint32_t row_capacity, experts, k, n, group_list_type;
} Gmm2SparseBuffers;

typedef struct { void *planner, *native, *candidate; } Gmm2SparseFunctions;

enum {
    GMM2_SPARSE_INVALID_ARGUMENT = -910401,
    GMM2_SPARSE_INVALID_PROFILE = -910402,
    GMM2_SPARSE_INVALID_EXTENT = -910403,
    GMM2_SPARSE_OVERLAPPING_BUFFERS = -910404,
    GMM2_SPARSE_WRONG_ENGINE_PAIR = -910405,
};

// Pure CPU functions, safe to run without ACL initialization.
int gmm2_sparse_msd_validate(const Gmm2SparseBuffers *buffers);
int gmm2_sparse_msd_pack_lists(const Gmm2SparseBuffers *buffers, uint64_t words[20]);
uint64_t gmm2_sparse_msd_workspace_bytes(void);
uint64_t gmm2_sparse_msd_frame_offset(void);

// Runtime functions are unexecuted in this CPU-only lane. The caller supplies
// a previously initialized, authorized NPU context. Open verifies an AIC+AIV
// pair for each mixed binary before returning handles; errors are propagated.
typedef struct Gmm2SparseLibrary Gmm2SparseLibrary;
int gmm2_sparse_msd_open(const char *planner_object, const char *native_object,
                        const char *candidate_object, Gmm2SparseLibrary **library);
int gmm2_sparse_msd_functions(const Gmm2SparseLibrary *library, Gmm2SparseFunctions *functions);
int gmm2_sparse_msd_close(Gmm2SparseLibrary *library);

// sparse=0: native control only. sparse=1: device planner then candidate on the
// same stream. Valid upstream counts48 must sum to <=6; no count D2H occurs.
int gmm2_sparse_msd_launch(const Gmm2SparseFunctions *functions,
                          const Gmm2SparseBuffers *buffers, void *stream, uint32_t sparse);

#ifdef __cplusplus
}
#endif
#endif
