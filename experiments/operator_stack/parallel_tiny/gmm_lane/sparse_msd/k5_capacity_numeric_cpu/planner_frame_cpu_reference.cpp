// CPU-only executable twin of source_factory.planner_source(profile) semantics.
// Same loop as the AscendC kernel template: zero the whole frame, set header
// word[3]=experts, preset slot expert words to 0xffffffff, fill slots in
// expert order, and on invalid counts set word[2]=1 and return before the
// totals word[0]/word[1] are written. Compiled by the host g++ with no device
// headers; this is a semantic CPU reference, not a device build. The frame is
// written little-endian so the Python oracle can byte-compare it.
// Trace: k5_capacity_cpu 2026-10-12.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: planner_frame_cpu_reference <experts> <capacity> <frame_words>"
                     " <counts_i64le.bin> <out_frame.bin>" << std::endl;
        return 2;
    }
    const uint64_t experts = std::strtoull(argv[1], nullptr, 10);
    const uint64_t capacity = std::strtoull(argv[2], nullptr, 10);
    const uint64_t frame_words = std::strtoull(argv[3], nullptr, 10);
    const uint64_t slots_max = capacity < experts ? capacity : experts;
    if (!experts || !capacity || !frame_words || frame_words < 4 + 4 * slots_max) {
        std::cerr << "invalid geometry" << std::endl;
        return 2;
    }
    std::ifstream counts(argv[4], std::ios::binary | std::ios::ate);
    if (!counts) { std::cerr << "counts file unreadable" << std::endl; return 2; }
    const std::streamsize count_bytes = counts.tellg();
    if (count_bytes < 0 || static_cast<uint64_t>(count_bytes) != experts * 8) {
        std::cerr << "counts file must be exactly experts*8 bytes" << std::endl;
        return 2;
    }
    counts.seekg(0);
    std::vector<int64_t> values(experts);
    counts.read(reinterpret_cast<char *>(values.data()), count_bytes);
    if (!counts) { std::cerr << "counts read failed" << std::endl; return 2; }

    std::vector<uint32_t> frame(frame_words, 0);
    frame[3] = static_cast<uint32_t>(experts);
    for (uint64_t i = 0; i < slots_max; ++i) frame[4 + 4 * i] = 0xffffffffu;
    uint32_t rows = 0, slots = 0;
    for (uint64_t expert = 0; expert < experts; ++expert) {
        const int64_t count = values[expert];
        if (count < 0 || static_cast<uint64_t>(count) > capacity ||
            static_cast<uint64_t>(rows) + static_cast<uint64_t>(count) > capacity) {
            frame[2] = 1;
            goto write;
        }
        if (count == 0) continue;
        frame[4 + 4 * slots] = static_cast<uint32_t>(expert);
        frame[4 + 4 * slots + 1] = rows;
        frame[4 + 4 * slots + 2] = static_cast<uint32_t>(count);
        frame[4 + 4 * slots + 3] = 0;
        rows += static_cast<uint32_t>(count);
        slots += 1;
    }
    frame[0] = rows;
    frame[1] = slots;
write:
    std::ofstream out(argv[5], std::ios::binary);
    if (!out) { std::cerr << "frame file unwritable" << std::endl; return 2; }
    for (const uint32_t word : frame) {
        const unsigned char bytes[4] = {static_cast<unsigned char>(word),
                                        static_cast<unsigned char>(word >> 8),
                                        static_cast<unsigned char>(word >> 16),
                                        static_cast<unsigned char>(word >> 24)};
        out.write(reinterpret_cast<const char *>(bytes), 4);
    }
    out.close();
    std::cout << "{\"experts\":" << experts << ",\"capacity\":" << capacity
              << ",\"frame_words\":" << frame_words << ",\"rows\":" << frame[0]
              << ",\"slots\":" << frame[1] << ",\"error_flag\":" << frame[2] << "}" << std::endl;
    return 0;
}
