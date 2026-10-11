#ifndef GMM2_LAUNCH_ARGS_V3_H
#define GMM2_LAUNCH_ARGS_V3_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { GMM2_LAUNCH_V3_INVALID_ARGUMENT = -910701 };
typedef struct { uint64_t counts, frame; } Gmm2PlannerArgsV3;
typedef struct {
    uint64_t ffts, x_list, weight_list, assist_list, scale_list;
    uint64_t counts, token_scale, y_list, workspace_system_base;
} Gmm2MsdArgsV3;

// hostArgs contains only the original 16B/72B GM-address values. No placeholders
// are needed because descriptors/tiling are not stored inline in these payloads.
// This helper never selects a device, creates a context/stream or registers ELF.
int gmm2_launch_v3_host_args(void *function, void *stream, void *hostArgs,
                            size_t bytes, unsigned blocks, unsigned schedule_mode);

// sdk_args_handle must be initialized for this exact function by
// aclrtKernelArgsInit. Borrowed: retain it with its function/module owner.
// Checked sequence matches the installed SDK launcher: Append, Finalize, Launch.
int gmm2_launch_v3_with_config(void *function, void *stream, void *sdk_args_handle,
                              void *hostArgs, size_t bytes, unsigned blocks,
                              unsigned schedule_mode, unsigned *failed_stage);
#ifdef __cplusplus
}
#endif
#endif
