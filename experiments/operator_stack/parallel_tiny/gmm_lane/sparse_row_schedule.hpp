// CPU/device contract for a future native-MSD scheduling operator.
// No Ascend headers or launch code: this file does not implement matmul.
#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace v41_sparse_gmm {
constexpr std::uint32_t kExperts = 48;
constexpr std::uint32_t kCapacity = 6;
struct GroupDescriptor {
    std::int32_t expert;
    std::uint32_t row_begin;
    std::uint32_t row_count;
    std::uint32_t reserved;
};
struct alignas(64) SparseSchedule {
    std::uint32_t total_rows = 0;
    std::uint32_t active_groups = 0;
    std::uint32_t status = 0;
    std::uint32_t reserved = 0;
    std::array<GroupDescriptor, kCapacity> groups{};
};
static_assert(sizeof(GroupDescriptor) == 16);
static_assert(sizeof(SparseSchedule) == 128);

// CPU reference for one AIV producer. A device implementation must publish the
// whole descriptor before the existing native pre/mid synchronization point.
inline SparseSchedule BuildSchedule(const std::array<std::int64_t, kExperts>& counts) {
    SparseSchedule out;
    for (auto& entry : out.groups) entry.expert = -1;
    for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
        const auto count = counts[expert];
        if (count < 0 || count > kCapacity || out.total_rows + count > kCapacity)
            throw std::invalid_argument("invalid local row capacity");
        if (count == 0) continue;
        out.groups[out.active_groups++] = {static_cast<std::int32_t>(expert),
                                           out.total_rows, static_cast<std::uint32_t>(count), 0};
        out.total_rows += static_cast<std::uint32_t>(count);
    }
    for (std::uint32_t i = out.active_groups; i < kCapacity; ++i)
        out.groups[i].row_begin = out.total_rows;
    return out;
}

// Decode-only planner: input is the original [1,6] global expert IDs. The
// device version uses a fixed sorting network, with non-local IDs at the end.
// This avoids deriving a new prefix representation from all 48 counts.
inline SparseSchedule BuildFromTop6(std::array<std::int32_t, kCapacity> ids, std::uint32_t ep_rank) {
    if (ep_rank >= 8) throw std::invalid_argument("invalid EP rank");
    auto sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    for (std::uint32_t i = 0; i < kCapacity; ++i)
        if (sorted[i] < 0 || sorted[i] >= 384 || (i && sorted[i] == sorted[i - 1]))
            throw std::invalid_argument("top6 must contain unique valid global expert IDs");
    SparseSchedule out;
    for (auto& entry : out.groups) entry.expert = -1;
    const auto first = static_cast<std::int32_t>(ep_rank * kExperts);
    for (auto id : sorted) {
        if (id < first || id >= first + static_cast<std::int32_t>(kExperts)) continue;
        out.groups[out.active_groups++] = {id - first, out.total_rows++, 1, 0};
    }
    for (std::uint32_t i = out.active_groups; i < kCapacity; ++i)
        out.groups[i].row_begin = out.total_rows;
    return out;
}

// Native packed INT4 storage is [E, N/64, K/16, 16, 8] INT32.
// INT32 is a bit container holding eight signed INT4 nibbles, not eight INT8s.
inline std::uint64_t NzWordOffset(std::uint32_t expert, std::uint32_t k, std::uint32_t n,
                                  std::uint32_t K, std::uint32_t N) {
    if (K % 16 || N % 64 || expert >= kExperts || k >= K || n >= N)
        throw std::invalid_argument("invalid packed NZ coordinate");
    return (((static_cast<std::uint64_t>(expert) * (N / 64) + n / 64) * (K / 16)
             + k / 16) * 16 + k % 16) * 8 + (n % 64) / 8;
}
inline std::uint64_t ExpertWeightByteBase(std::uint32_t expert, std::uint32_t K, std::uint32_t N) {
    return static_cast<std::uint64_t>(expert) * K * N / 2;
}
inline std::uint64_t ScaleByteBase(std::uint32_t expert, std::uint32_t N) {
    return static_cast<std::uint64_t>(expert) * N * sizeof(std::uint64_t);
}
inline std::uint64_t AssistByteBase(std::uint32_t expert, std::uint32_t N) {
    return static_cast<std::uint64_t>(expert) * N * sizeof(float);
}
inline std::uint64_t MsdInputByteBase(std::uint32_t row_begin, std::uint32_t K) {
    // high and low INT4 rows are interleaved: [high(r), low(r)].
    return static_cast<std::uint64_t>(row_begin) * K;
}
inline std::uint64_t MsdHalfOutputByteBase(std::uint32_t row_begin, std::uint32_t N) {
    return static_cast<std::uint64_t>(row_begin) * 2 * N * sizeof(std::uint16_t);
}
}  // namespace v41_sparse_gmm
