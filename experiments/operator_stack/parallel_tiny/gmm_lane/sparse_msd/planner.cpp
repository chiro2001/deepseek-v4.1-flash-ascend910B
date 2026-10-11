// Independent device planner. Compiled only; this lane has no NPU permission.
#include "kernel_operator.h"
using namespace AscendC;

// 128-byte frame: totalRows, activeSlots, status, totalExpertNum,
// then six {expert, rowBegin, rowCount, reserved} records plus four padding words.
extern "C" __global__ __aicore__ void gmm2_sparse_row_plan(GM_ADDR counts, GM_ADDR frame) {
    if (GetBlockIdx() != 0) return;
    GlobalTensor<int64_t> input;
    GlobalTensor<uint32_t> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(counts), 48);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(frame), 32);
    for (uint32_t i = 0; i < 32; ++i) output.SetValue(i, 0);
    output.SetValue(3, 48);
    for (uint32_t i = 0; i < 6; ++i) output.SetValue(4 + 4 * i, 0xffffffffU);
    uint32_t rows = 0, slots = 0;
    for (uint32_t expert = 0; expert < 48; ++expert) {
        const int64_t count = input.GetValue(expert);
        if (count < 0 || count > 6 || rows + count > 6) {
            output.SetValue(2, 1);
            return;
        }
        if (count == 0) continue;
        const uint32_t pos = 4 + 4 * slots;
        output.SetValue(pos, expert);
        output.SetValue(pos + 1, rows);
        output.SetValue(pos + 2, static_cast<uint32_t>(count));
        rows += static_cast<uint32_t>(count);
        ++slots;
    }
    output.SetValue(0, rows);
    output.SetValue(1, slots);
}
