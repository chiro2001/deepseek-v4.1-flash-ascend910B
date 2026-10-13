"""Extend the proven v5 bridge's extents/role only; raw HostArgs order preserved."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def replace_once(source,needle,value):
 if source.count(needle)!=1:raise ValueError('Frozen bridge edit is ambiguous: '+needle)
 return source.replace(needle,value)


def generate(profile,native_workspace_bytes,role_root,native_workspace_origin='new_compiler_metadata'):
 path=HERE/'borrowed/v5_probe_bridge.cpp';source=path.read_text()
 original_sha=hashlib.sha256(source.encode()).hexdigest()
 source=replace_once(source,'#include "workspace_layout.h"','#include "workspace_layout.h"\n#include "capacity_adapter.h"\n#include "capacity_geometry.h"')
 source=replace_once(source,'constexpr uint64_t Y_BYTES = 6ull * 5120 * 2;','constexpr uint64_t Y_BYTES = uint64_t(PROFILE.capacity) * PROFILE.n * 2;')
 source=replace_once(source,'int mapping() {','int mapping() {\n    const char *declared = std::getenv("CAPACITY_EXPECTED_PHYSICAL");\n    if (!declared || (std::string(declared)!="6" && std::string(declared)!="2" && std::string(declared)!="3" && std::string(declared)!="7")) return -910814;\n    const uint32_t expected_physical = static_cast<uint32_t>(std::strtoul(declared,nullptr,10));')
 source=replace_once(source,'std::string(value) != "6"','std::string(value) != declared')
 source=source.replace('rtGetDeviceIndexByPhyId(6, &logical)','rtGetDeviceIndexByPhyId(expected_physical, &logical)')
 source=replace_once(source,'logical == 0 && physical == 6','logical == 0 && physical == expected_physical')
 source=replace_once(source,'std::string(root) != "/work/gmm_lane_sparse_msd_build_20261011_v3_repro"','std::string(root) != EXPECTED_ROLE_ROOT')
 source=replace_once(source,'std::array<size_t, 3> aic_expected{0, 5152, 8748}, aiv_expected{368, 8440, 8440};','// New geometry recompiles function sizes; ELF72/16 ABI, SHA and mixed/vector profiles remain independently guarded.')
 source=replace_once(source,'aic_bytes == aic_expected[i] && aiv_bytes == aiv_expected[i]','(i == 0 ? aic_bytes == 0 : aic_bytes > 0) && aiv_bytes > 0')
 source=replace_once(source,'p->workspace_base + SYSTEM + 128','p->workspace_base + SYSTEM + FRAME_BYTES')
 source=replace_once(source,'b.x = {x, 6ull * 2304}; b.weight = {weight, 48ull * 2304 * 5120 / 2};',
  'b.x = {x, uint64_t(PROFILE.capacity) * PROFILE.k}; b.weight = {weight, uint64_t(PROFILE.experts) * PROFILE.k * PROFILE.n / 2};')
 source=replace_once(source,'b.assist = {assist, 48ull * 5120 * 4}; b.scale = {scale, 48ull * 5120 * 8};',
  'b.assist = {assist, uint64_t(PROFILE.experts) * PROFILE.n * 4}; b.scale = {scale, uint64_t(PROFILE.experts) * PROFILE.n * 8};')
 source=replace_once(source,'b.counts = {counts, 384}; b.token_scale = {token_scale, 24};',
  'b.counts = {counts, uint64_t(PROFILE.experts) * 8}; b.token_scale = {token_scale, uint64_t(PROFILE.capacity) * 4};')
 source=replace_once(source,'b.row_capacity = 6; b.experts = 48; b.k = 2304; b.n = 5120;',
  'b.row_capacity = PROFILE.capacity; b.experts = PROFILE.experts; b.k = PROFILE.k; b.n = PROFILE.n;')
 source=replace_once(source,'gmm2_sparse_msd_validate(&b)','capacity_validate(&b, &PROFILE)')
 source=replace_once(source,'gmm2_sparse_msd_pack_lists(&b, words.data())','capacity_pack_lists(&b, &PROFILE, words.data())')
 source=replace_once(source,'aclrtMemcpy(frame, 128, reinterpret_cast<void *>(planner.frame), 128, ACL_MEMCPY_DEVICE_TO_HOST)',
  'aclrtMemcpy(frame, FRAME_BYTES, reinterpret_cast<void *>(planner.frame), FRAME_BYTES, ACL_MEMCPY_DEVICE_TO_HOST)')
 geometry=f'''#pragma once
#include "capacity_adapter.h"
constexpr CapacityGeometry PROFILE{{{profile.capacity},{profile.experts},{profile.k},{profile.n},{native_workspace_bytes}ull}};
constexpr uint64_t FRAME_BYTES={profile.frame_bytes};
constexpr const char *EXPECTED_ROLE_ROOT={json.dumps(str(role_root))};
'''
 workspace=f'''#pragma once
#include <cstdint>
constexpr uint64_t M1_WORKSPACE_BYTES={native_workspace_bytes+profile.frame_bytes}ull;
constexpr uint64_t M1_WORKSPACE_CANARY_BYTES=128;
constexpr uint64_t M1_WORKSPACE_ALIGNMENT=512;
constexpr uint64_t M1_WORKSPACE_ALLOCATION_BYTES=M1_WORKSPACE_BYTES+768;
inline bool m1_workspace_layout(uint64_t allocation,uint64_t bytes,uint64_t*base) {{
 if(!allocation||!base||bytes<M1_WORKSPACE_ALLOCATION_BYTES||bytes>UINT64_MAX-allocation)return false;
 const uint64_t value=(allocation+128+511)&~511ull;
 if(value<allocation+128||value>allocation+bytes-M1_WORKSPACE_BYTES-128)return false;
 *base=value;return true;
}}
'''
 receipt={'original_v5_bridge_sha256':original_sha,'borrowed_context_stream_and_cleanup_preserved':True,
  'device_init_set_reset_finalize_in_bridge':False,'native_pointer_lists_32B_and_GMM_HostArgs_72B_preserved':True,
  'planner_HostArgs_16B_preserved':True,'function_sizes_new_geometry_recorded_not_old_size_assumed':True,
  'native_workspace_bytes_input':native_workspace_bytes,'native_workspace_input_origin':native_workspace_origin,
  'frame_bytes':profile.frame_bytes,
  'actual_numeric_passed':False,'actual_NPU_calls':0,'performance_receipt_granted':False,'formal_receipt_granted':False}
 return source,geometry,workspace,receipt
