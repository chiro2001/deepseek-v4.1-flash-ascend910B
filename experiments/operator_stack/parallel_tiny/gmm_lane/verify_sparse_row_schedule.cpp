#include "sparse_row_schedule.hpp"
#include <cassert>
#include <iostream>
#include <random>

int main() {
    using namespace v41_sparse_gmm;
    std::mt19937 rng(20261011);
    std::uint64_t cases = 0;
    for (std::uint32_t total = 0; total <= kCapacity; ++total) {
        for (std::uint32_t trial = 0; trial < 1000; ++trial) {
            std::array<std::int64_t, kExperts> counts{};
            std::array<std::uint32_t, kCapacity> row_experts{};
            for (std::uint32_t i = 0; i < total; ++i) ++counts[rng() % kExperts];
            // Independent scalar routing-order oracle: expand each count.
            std::uint32_t row = 0;
            for (std::uint32_t e = 0; e < kExperts; ++e)
                for (std::int64_t j = 0; j < counts[e]; ++j) row_experts[row++] = e;
            auto schedule = BuildSchedule(counts);
            assert(schedule.total_rows == total && schedule.active_groups <= total);
            row = 0;
            for (std::uint32_t i = 0; i < schedule.active_groups; ++i) {
                const auto d = schedule.groups[i];
                assert(d.row_begin == row && d.row_count > 0 && d.expert >= 0 && d.expert < 48);
                if (i) assert(schedule.groups[i - 1].expert < d.expert);
                for (std::uint32_t j = 0; j < d.row_count; ++j) assert(row_experts[row++] == std::uint32_t(d.expert));
                assert(ExpertWeightByteBase(d.expert, 5120, 4608) == std::uint64_t(d.expert) * 11796480);
                assert(ExpertWeightByteBase(d.expert, 2304, 5120) == std::uint64_t(d.expert) * 5898240);
                assert(MsdInputByteBase(d.row_begin, 5120) == std::uint64_t(2 * d.row_begin) * 5120 / 2);
                assert(MsdHalfOutputByteBase(d.row_begin, 4608) == std::uint64_t(2 * d.row_begin) * 4608 * 2);
            }
            assert(row == total);
            for (std::uint32_t i = schedule.active_groups; i < kCapacity; ++i)
                assert(schedule.groups[i].expert == -1 && schedule.groups[i].row_count == 0);
            ++cases;
        }
    }
    for (const auto dims : {std::array<std::uint32_t, 2>{5120, 4608}, {2304, 5120}}) {
        const auto K = dims[0], N = dims[1];
        for (const auto e : {0u, 47u}) for (const auto k : {0u, 15u, 16u, K - 1})
            for (const auto n : {0u, 7u, 8u, 63u, 64u, N - 1}) {
                auto word = NzWordOffset(e, k, n, K, N);
                assert(word < std::uint64_t(kExperts) * K * N / 8);
                // Invert the blocked storage coordinates independently.
                auto local_word = word - std::uint64_t(e) * K * N / 8;
                auto n0_word = local_word % 8; local_word /= 8;
                auto k0 = local_word % 16; local_word /= 16;
                auto k1 = local_word % (K / 16); local_word /= K / 16;
                auto n1 = local_word;
                assert(k1 * 16 + k0 == k);
                assert(n1 * 64 + n0_word * 8 + n % 8 == n);
            }
    }
    std::uint64_t top6_cases = 0;
    for (std::uint32_t rank = 0; rank < 8; ++rank) for (std::uint32_t trial = 0; trial < 1000; ++trial) {
        std::array<std::int32_t, 6> ids{};
        for (std::uint32_t i = 0; i < 6; ++i) {
            bool duplicate;
            do {
                ids[i] = rng() % 384;
                duplicate = false;
                for (std::uint32_t j = 0; j < i; ++j) duplicate |= ids[i] == ids[j];
            } while (duplicate);
        }
        std::array<std::int64_t, 48> counts{};
        for (auto id : ids) if (id >= std::int32_t(rank * 48) && id < std::int32_t((rank + 1) * 48))
            ++counts[id - rank * 48];
        const auto expected = BuildSchedule(counts), actual = BuildFromTop6(ids, rank);
        assert(expected.total_rows == actual.total_rows && expected.active_groups == actual.active_groups);
        for (std::uint32_t i = 0; i < 6; ++i) {
            assert(expected.groups[i].expert == actual.groups[i].expert);
            assert(expected.groups[i].row_begin == actual.groups[i].row_begin);
            assert(expected.groups[i].row_count == actual.groups[i].row_count);
        }
        ++top6_cases;
    }
    for (auto invalid : {-1, 7}) {
        std::array<std::int64_t, kExperts> counts{};
        counts[47] = invalid;
        bool rejected = false;
        try { (void)BuildSchedule(counts); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    std::cout << "CPU sparse descriptor contract passed: " << cases
              << " row-distribution cases; " << top6_cases
              << " top6/EP cases; formal NZ address boundaries; no NPU launch\n";
}
