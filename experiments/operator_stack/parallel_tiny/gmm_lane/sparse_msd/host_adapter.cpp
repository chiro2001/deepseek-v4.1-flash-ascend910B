// Private raw-ELF adapter. No ACL init, set_device, allocation or synchronization.
#include "host_adapter.h"
#include <acl/acl_rt.h>
#include <array>
#include <cstddef>
#include <limits>

namespace {
constexpr uint64_t SYSTEM_BYTES = 16ULL * 1024 * 1024;
constexpr uint64_t NATIVE_WORKSPACE_BYTES = 112ULL * 1024 * 1024;
constexpr uint64_t FRAME_BYTES = 128;
constexpr uint64_t WORKSPACE_BYTES = NATIVE_WORKSPACE_BYTES + FRAME_BYTES;

struct GmmArgs {
    uint64_t ffts, x_list, weight_list, assist_list, scale_list;
    uint64_t counts, token_scale, y_list, workspace;
};
struct PlannerArgs { uint64_t counts, frame; };
static_assert(sizeof(GmmArgs) == 72 && offsetof(GmmArgs, workspace) == 64);
static_assert(sizeof(PlannerArgs) == 16 && sizeof(void *) == 8);

bool valid_span(const Gmm2SparseSpan &span, uint64_t required, uint64_t alignment) {
    return span.address && span.address % alignment == 0 && span.bytes >= required &&
           span.bytes <= std::numeric_limits<uint64_t>::max() - span.address;
}
bool overlap(const Gmm2SparseSpan &a, const Gmm2SparseSpan &b) {
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}

extern "C" uint64_t gmm2_sparse_msd_workspace_bytes() { return WORKSPACE_BYTES; }
extern "C" uint64_t gmm2_sparse_msd_frame_offset() { return SYSTEM_BYTES; }

extern "C" int gmm2_sparse_msd_validate(const Gmm2SparseBuffers *b) {
    if (!b || !b->ffts_address) return GMM2_SPARSE_INVALID_ARGUMENT;
    if (b->row_capacity != 6 || b->experts != 48 || b->k != 2304 || b->n != 5120 ||
        b->group_list_type != 1) return GMM2_SPARSE_INVALID_PROFILE;
    const std::array<Gmm2SparseSpan, 13> spans = {b->x, b->weight, b->assist, b->scale,
        b->counts, b->token_scale, b->y, b->x_list, b->weight_list, b->assist_list,
        b->scale_list, b->y_list, b->workspace};
    constexpr std::array<uint64_t, 13> sizes = {6 * 2304, 48ULL * 2304 * 5120 / 2,
        48 * 5120 * 4, 48 * 5120 * 8, 48 * 8, 6 * 4, 6 * 5120 * 2,
        32, 32, 32, 32, 32, WORKSPACE_BYTES};
    constexpr std::array<uint64_t, 13> alignments = {16, 16, 4, 8, 8, 4, 16,
        8, 8, 8, 8, 8, 512};
    for (size_t i = 0; i < spans.size(); ++i) {
        if (!valid_span(spans[i], sizes[i], alignments[i])) return GMM2_SPARSE_INVALID_EXTENT;
    }
    for (size_t i = 0; i < spans.size(); ++i) {
        for (size_t j = i + 1; j < spans.size(); ++j) {
            if (overlap(spans[i], spans[j])) return GMM2_SPARSE_OVERLAPPING_BUFFERS;
        }
    }
    return 0;
}

extern "C" int gmm2_sparse_msd_pack_lists(const Gmm2SparseBuffers *b, uint64_t words[20]) {
    if (!words) return GMM2_SPARSE_INVALID_ARGUMENT;
    const int status = gmm2_sparse_msd_validate(b);
    if (status) return status;
    const uint64_t addresses[] = {b->x.address, b->weight.address, b->assist.address,
                                  b->scale.address, b->y.address};
    for (size_t i = 0; i < 5; ++i) {
        words[4 * i] = 24;             // Byte offset of pointer array.
        words[4 * i + 1] = 1ULL << 32; // count=1, dim=0 (pointer-only list).
        words[4 * i + 2] = 0xffffffff; // Native ListTensorDesc pointer-only marker.
        words[4 * i + 3] = addresses[i];
    }
    return 0;
}

extern "C" int gmm2_sparse_msd_launch(const Gmm2SparseFunctions *f,
                                     const Gmm2SparseBuffers *b, void *stream, uint32_t sparse) {
    if (!f || !stream || sparse > 1 || !f->native || (sparse && (!f->planner || !f->candidate)))
        return GMM2_SPARSE_INVALID_ARGUMENT;
    const int status = gmm2_sparse_msd_validate(b);
    if (status) return status;
    if (sparse) {
        const PlannerArgs planner = {b->counts.address, b->workspace.address + SYSTEM_BYTES};
        const aclError error = aclrtLaunchKernel(f->planner, 1, &planner, sizeof(planner), stream);
        if (error) return error;
    }
    const GmmArgs args = {b->ffts_address, b->x_list.address, b->weight_list.address,
        b->assist_list.address, b->scale_list.address, b->counts.address, b->token_scale.address,
        b->y_list.address, b->workspace.address};
    aclrtLaunchKernelAttr schedule{};
    schedule.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    schedule.value.schemMode = 1; // Frozen native/candidate JSON schedule_mode.
    aclrtLaunchKernelCfg cfg{&schedule, 1};
    return aclrtLaunchKernelV2(sparse ? f->candidate : f->native, 24,
                             &args, sizeof(args), &cfg, stream);
}
