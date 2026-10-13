// CPU-only enlargement of the frozen extent/profile gate; native list bytes unchanged.
#include "capacity_adapter.h"
#include <array>
#include <limits>
#include <cstddef>
namespace {
bool span(const Gmm2SparseSpan &s,uint64_t bytes,uint64_t alignment) {
 return s.address&&s.address%alignment==0&&s.bytes>=bytes&&s.bytes<=UINT64_MAX-s.address;
}
bool overlap(const Gmm2SparseSpan&a,const Gmm2SparseSpan&b) {return a.address<b.address+b.bytes&&b.address<a.address+a.bytes;}
}
extern "C" int capacity_validate(const Gmm2SparseBuffers*b,const CapacityGeometry*g) {
 if(!b||!g||!b->ffts_address)return GMM2_SPARSE_INVALID_ARGUMENT;
 const bool role=(g->capacity==36&&g->experts==48)||(g->capacity==15&&g->experts==16);
 if(!role||!g->k||g->k%16||!g->n||g->n%64||g->native_workspace_bytes<CAPACITY_SYSTEM_BYTES||
    g->native_workspace_bytes>UINT64_MAX-capacity_frame_bytes(*g)||
    b->row_capacity!=g->capacity||b->experts!=g->experts||b->k!=g->k||b->n!=g->n||b->group_list_type!=1)
  return GMM2_SPARSE_INVALID_PROFILE;
 const uint64_t m=g->capacity,e=g->experts,k=g->k,n=g->n;
 const std::array<Gmm2SparseSpan,13> spans={b->x,b->weight,b->assist,b->scale,b->counts,b->token_scale,b->y,
  b->x_list,b->weight_list,b->assist_list,b->scale_list,b->y_list,b->workspace};
 const std::array<uint64_t,13> bytes={m*k,e*k*n/2,e*n*4,e*n*8,e*8,m*4,m*n*2,
  32,32,32,32,32,capacity_workspace_bytes(*g)};
 constexpr std::array<uint64_t,13> alignment={16,16,4,8,8,4,16,8,8,8,8,8,512};
 for(size_t i=0;i<spans.size();++i)if(!span(spans[i],bytes[i],alignment[i]))return GMM2_SPARSE_INVALID_EXTENT;
 for(size_t i=0;i<spans.size();++i)for(size_t j=i+1;j<spans.size();++j)
  if(overlap(spans[i],spans[j]))return GMM2_SPARSE_OVERLAPPING_BUFFERS;
 return 0;
}
extern "C" int capacity_pack_lists(const Gmm2SparseBuffers*b,const CapacityGeometry*g,uint64_t words[20]) {
 if(!words)return GMM2_SPARSE_INVALID_ARGUMENT;
 const int code=capacity_validate(b,g);if(code)return code;
 const uint64_t address[]={b->x.address,b->weight.address,b->assist.address,b->scale.address,b->y.address};
 for(size_t i=0;i<5;++i) {words[4*i]=24;words[4*i+1]=1ull<<32;words[4*i+2]=0xffffffff;words[4*i+3]=address[i];}
 return 0;
}
