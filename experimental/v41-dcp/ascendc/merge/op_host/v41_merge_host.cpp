// Host-side ctypes shims for the V4.1 DCP merge kernels.
//
// Deliberately avoids torch extension / pybind so the library can be loaded with
// ctypes and driven from python that owns the torch_npu stream (required for
// ACL-graph capture) -- same pattern as the neighbour `wo_a` scaffold.

#include <cstdint>
#include "acl/acl.h"

extern "C" void v41_merge_pre(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                              uint8_t* out, uint8_t* lse, uint8_t* olse,
                              uint8_t* pack, uint8_t* tiling);

extern "C" void v41_merge_post(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                               uint8_t* pack, uint8_t* ori_out, uint8_t* out,
                               uint8_t* tiling);

extern "C" int v41_merge_pre_launch(uint32_t grid, void* stream, void* out, void* lse,
                                    void* olse, void* pack, void* tiling)
{
    v41_merge_pre(grid, nullptr, static_cast<aclrtStream>(stream),
                  static_cast<uint8_t*>(out), static_cast<uint8_t*>(lse),
                  static_cast<uint8_t*>(olse), static_cast<uint8_t*>(pack),
                  static_cast<uint8_t*>(tiling));
    return 0;
}

extern "C" int v41_merge_post_launch(uint32_t grid, void* stream, void* pack,
                                     void* ori_out, void* out, void* tiling)
{
    v41_merge_post(grid, nullptr, static_cast<aclrtStream>(stream),
                   static_cast<uint8_t*>(pack), static_cast<uint8_t*>(ori_out),
                   static_cast<uint8_t*>(out), static_cast<uint8_t*>(tiling));
    return 0;
}
