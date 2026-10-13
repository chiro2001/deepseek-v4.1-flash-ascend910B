#ifndef M1_PROBE_WORKSPACE_LAYOUT_H
#define M1_PROBE_WORKSPACE_LAYOUT_H
#include <cstdint>
constexpr uint64_t M1_WORKSPACE_BYTES = 117440640;
constexpr uint64_t M1_WORKSPACE_CANARY_BYTES = 128;
constexpr uint64_t M1_WORKSPACE_ALIGNMENT = 512;
// Up to511B alignment slack, two128B canaries; round backing capacity up1B.
constexpr uint64_t M1_WORKSPACE_ALLOCATION_BYTES = M1_WORKSPACE_BYTES + 768;
inline bool m1_workspace_layout(uint64_t allocation, uint64_t allocation_bytes, uint64_t *base) {
    if (!allocation || !base || allocation_bytes < M1_WORKSPACE_ALLOCATION_BYTES ||
        allocation_bytes > UINT64_MAX - allocation) return false;
    const uint64_t aligned = (allocation + M1_WORKSPACE_CANARY_BYTES + M1_WORKSPACE_ALIGNMENT - 1) &
                             ~(M1_WORKSPACE_ALIGNMENT - 1);
    if (aligned < allocation + M1_WORKSPACE_CANARY_BYTES || aligned % M1_WORKSPACE_ALIGNMENT ||
        aligned > allocation + allocation_bytes - M1_WORKSPACE_BYTES - M1_WORKSPACE_CANARY_BYTES) return false;
    *base = aligned;
    return true;
}
#endif
