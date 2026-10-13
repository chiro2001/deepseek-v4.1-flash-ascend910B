// CPU-only positive/negative contract sweep for the K5 capacity adapter.
// Pure integer/address arithmetic; no ACL/runtime/NPU symbols are referenced
// and no device call is made. Trace: k5_capacity_cpu 2026-10-12.
#include "capacity_adapter.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

struct PlannerArgs { uint64_t counts, frame; };
struct GmmArgs { uint64_t ffts, x, w, bias, scale, counts, token, y, workspace; };
static_assert(sizeof(PlannerArgs) == 16 && sizeof(GmmArgs) == 72 && offsetof(GmmArgs, workspace) == 64,
              "Frozen HostArgs ABI (planner 16B, GMM 72B)");

namespace {
unsigned checks = 0, positives = 0, negatives = 0;
bool failed = false;

void record(bool ok, const std::string &what) {
    ++checks; ++positives;
    if (!ok) { failed = true; std::cerr << "FAIL " << what << std::endl; }
}
void expect_ok(int code, const std::string &what) {
    ++checks; ++positives;
    if (code != 0) { failed = true; std::cerr << "FAIL " << what << " code=" << code << std::endl; }
}
void expect_code(int code, int expected, const std::string &what) {
    ++checks; ++negatives;
    if (code != expected) {
        failed = true;
        std::cerr << "FAIL " << what << " expected=" << expected << " got=" << code << std::endl;
    }
}

struct Fixture {
    Gmm2SparseBuffers b{};
    std::array<Gmm2SparseSpan *, 13> spans{};
    std::array<uint64_t, 13> sizes{};
};

Fixture make_fixture(const CapacityGeometry &g) {
    Fixture f{};
    f.b.row_capacity = g.capacity; f.b.experts = g.experts; f.b.k = g.k; f.b.n = g.n;
    f.b.group_list_type = 1; f.b.ffts_address = 0x1000;
    Gmm2SparseSpan *spans[] = {&f.b.x, &f.b.weight, &f.b.assist, &f.b.scale, &f.b.counts, &f.b.token_scale, &f.b.y,
        &f.b.x_list, &f.b.weight_list, &f.b.assist_list, &f.b.scale_list, &f.b.y_list, &f.b.workspace};
    const uint64_t m = g.capacity, e = g.experts, k = g.k, n = g.n;
    const uint64_t sizes[] = {m * k, e * k * n / 2, e * n * 4, e * n * 8, e * 8, m * 4, m * n * 2,
        32, 32, 32, 32, 32, capacity_workspace_bytes(g)};
    uint64_t address = 0x100000;
    for (unsigned i = 0; i < 13; ++i) {
        address = (address + 511) & ~511ull;
        *spans[i] = {address, sizes[i]};
        f.spans[i] = spans[i]; f.sizes[i] = sizes[i];
        address += sizes[i] + 512;
    }
    return f;
}
}

int main() {
    const std::array<CapacityGeometry, 2> geometry{{
        {36, 48, 2304, 5120, 112ull * 1024 * 1024},
        {15, 16, 2304, 5120, 112ull * 1024 * 1024}}};
    for (const CapacityGeometry &g : geometry) {
        const std::string tag = g.capacity == 36 ? "target36_E48" : "draft15_E16";
        Fixture f = make_fixture(g);

        // Positive path: full valid profile, packed single-tensor lists.
        expect_ok(capacity_validate(&f.b, &g), tag + " positive");
        uint64_t words[20]{};
        expect_ok(capacity_pack_lists(&f.b, &g, words), tag + " pack_lists positive");
        const uint64_t roots[] = {f.b.x.address, f.b.weight.address, f.b.assist.address, f.b.scale.address,
                                  f.b.y.address};
        for (unsigned i = 0; i < 5; ++i) {
            const uint64_t *w = words + 4 * i;
            record(w[0] == 24 && w[1] == (1ull << 32) && w[2] == 0xffffffffu && w[3] == roots[i],
                   tag + " single-tensor list " + std::to_string(i));
        }
        record(capacity_frame_bytes(g) == (g.capacity == 36 ? 640 : 256), tag + " frame_bytes");
        record(capacity_workspace_bytes(g) == g.native_workspace_bytes + capacity_frame_bytes(g),
               tag + " workspace sum");

        // Every span one byte short must be rejected with INVALID_EXTENT.
        for (unsigned i = 0; i < 13; ++i) {
            const auto old = *f.spans[i];
            f.spans[i]->bytes = f.sizes[i] - 1;
            expect_code(capacity_validate(&f.b, &g), GMM2_SPARSE_INVALID_EXTENT,
                        tag + " short span " + std::to_string(i));
            *f.spans[i] = old;
        }
        // Misaligned and overflowing addresses.
        {
            const auto old = f.b.x;
            f.b.x.address += 1;
            expect_code(capacity_validate(&f.b, &g), GMM2_SPARSE_INVALID_EXTENT, tag + " misaligned x");
            f.b.x = old;
        }
        {
            const auto old = f.b.workspace;
            f.b.workspace.address += 1;
            expect_code(capacity_validate(&f.b, &g), GMM2_SPARSE_INVALID_EXTENT, tag + " misaligned workspace");
            f.b.workspace = old;
        }
        {
            const auto old = f.b.x;
            f.b.x.address = UINT64_MAX - 15; // 16B aligned, address+bytes overflows
            expect_code(capacity_validate(&f.b, &g), GMM2_SPARSE_INVALID_EXTENT, tag + " address overflow");
            f.b.x = old;
        }
        // Overlap and profile mismatches.
        {
            auto bad = f.b;
            bad.x.address = bad.weight.address;
            expect_code(capacity_validate(&bad, &g), GMM2_SPARSE_OVERLAPPING_BUFFERS, tag + " overlap x/weight");
        }
        {
            auto bad = f.b;
            bad.row_capacity = g.capacity + 1;
            expect_code(capacity_validate(&bad, &g), GMM2_SPARSE_INVALID_PROFILE, tag + " wrong row_capacity");
        }
        {
            auto bad = f.b;
            bad.experts = g.experts + 1;
            expect_code(capacity_validate(&bad, &g), GMM2_SPARSE_INVALID_PROFILE, tag + " wrong experts");
        }
        {
            auto bad = f.b;
            bad.group_list_type = 0;
            expect_code(capacity_validate(&bad, &g), GMM2_SPARSE_INVALID_PROFILE, tag + " wrong group_list_type");
        }
        {
            CapacityGeometry bad = g;
            bad.k = g.k + 8; // k%16 != 0
            expect_code(capacity_validate(&f.b, &bad), GMM2_SPARSE_INVALID_PROFILE, tag + " k not 16-aligned");
        }
        {
            CapacityGeometry bad = g;
            bad.n = g.n + 8; // n%64 != 0
            expect_code(capacity_validate(&f.b, &bad), GMM2_SPARSE_INVALID_PROFILE, tag + " n not 64-aligned");
        }
        {
            CapacityGeometry bad = g;
            bad.native_workspace_bytes = CAPACITY_SYSTEM_BYTES - 1;
            expect_code(capacity_validate(&f.b, &bad), GMM2_SPARSE_INVALID_PROFILE, tag + " workspace below 16MiB");
        }
        {
            CapacityGeometry bad = g;
            bad.native_workspace_bytes = UINT64_MAX - 100; // > UINT64_MAX - frame_bytes
            expect_code(capacity_validate(&f.b, &bad), GMM2_SPARSE_INVALID_PROFILE, tag + " workspace overflow");
        }
        // Role set: only (36,48) and (15,16) exist in this lane.
        for (const std::array<uint32_t, 2> wrong : {std::array<uint32_t, 2>{36, 16}, {15, 48}}) {
            CapacityGeometry bad = g;
            bad.capacity = wrong[0]; bad.experts = wrong[1];
            auto bad_buffers = f.b;
            bad_buffers.row_capacity = wrong[0]; bad_buffers.experts = wrong[1];
            expect_code(capacity_validate(&bad_buffers, &bad), GMM2_SPARSE_INVALID_PROFILE, tag + " unknown role");
        }
        // Null and missing arguments.
        expect_code(capacity_validate(nullptr, &g), GMM2_SPARSE_INVALID_ARGUMENT, tag + " null buffers");
        expect_code(capacity_validate(&f.b, nullptr), GMM2_SPARSE_INVALID_ARGUMENT, tag + " null geometry");
        {
            auto bad = f.b;
            bad.ffts_address = 0;
            expect_code(capacity_validate(&bad, &g), GMM2_SPARSE_INVALID_ARGUMENT, tag + " missing ffts");
        }
        expect_code(capacity_pack_lists(&f.b, &g, nullptr), GMM2_SPARSE_INVALID_ARGUMENT, tag + " null words");
        {
            auto bad = f.b;
            bad.experts = g.experts + 1; // failure must propagate from validate
            expect_code(capacity_pack_lists(&bad, &g, words), GMM2_SPARSE_INVALID_PROFILE,
                        tag + " pack_lists propagates");
        }
        // Boundary positive: exactly 16MiB of native workspace.
        {
            CapacityGeometry boundary = g;
            boundary.native_workspace_bytes = CAPACITY_SYSTEM_BYTES;
            Fixture fb = make_fixture(boundary);
            expect_ok(capacity_validate(&fb.b, &boundary), tag + " 16MiB boundary");
        }
    }
    std::cout << "{\"passed\":" << (failed ? "false" : "true")
              << ",\"checks\":" << checks
              << ",\"positive_asserts\":" << positives
              << ",\"negative_rejections\":" << negatives
              << ",\"profiles\":[";
    for (unsigned i = 0; i < geometry.size(); ++i) {
        const CapacityGeometry &g = geometry[i];
        std::cout << (i ? "," : "") << "{\"role\":\"" << (g.capacity == 36 ? "target36_E48" : "draft15_E16")
                  << "\",\"capacity\":" << g.capacity << ",\"experts\":" << g.experts
                  << ",\"frame_bytes\":" << capacity_frame_bytes(g)
                  << ",\"workspace_bytes\":" << capacity_workspace_bytes(g) << "}";
    }
    std::cout << "],\"frozen_HostArgs_ABI\":{\"planner_bytes\":16,\"gmm_bytes\":72,\"workspace_word_offset\":64}"
              << ",\"CPU_only\":true,\"real_SDK_calls\":0,\"NPU_calls\":0}" << std::endl;
    return failed ? 1 : 0;
}
