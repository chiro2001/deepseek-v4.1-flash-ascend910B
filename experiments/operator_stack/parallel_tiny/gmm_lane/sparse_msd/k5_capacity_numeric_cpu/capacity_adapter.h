#ifndef K5_CAPACITY_NUMERIC_ADAPTER_H
#define K5_CAPACITY_NUMERIC_ADAPTER_H
#include "borrowed/host_adapter.h"
#include <cstdint>
struct CapacityGeometry {
 uint32_t capacity,experts,k,n;
 uint64_t native_workspace_bytes;
};
constexpr uint64_t CAPACITY_SYSTEM_BYTES=16ull*1024*1024;
inline uint64_t capacity_frame_bytes(const CapacityGeometry &g) {
 const uint64_t slots=g.capacity<g.experts?g.capacity:g.experts;
 return ((4*(4+4*slots)+127)/128)*128;
}
inline uint64_t capacity_workspace_bytes(const CapacityGeometry &g) {
 return g.native_workspace_bytes+capacity_frame_bytes(g);
}
extern "C" int capacity_validate(const Gmm2SparseBuffers *,const CapacityGeometry *);
extern "C" int capacity_pack_lists(const Gmm2SparseBuffers *,const CapacityGeometry *,uint64_t words[20]);
#endif
