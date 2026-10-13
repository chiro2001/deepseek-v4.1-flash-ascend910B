"""New private copies: enlarge original scheduling geometry, preserve math/ABI."""
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
FROZEN_FACTORY_SHA='70aa7481d56c0b4ce03e3a3a756ddc87e941a7a357c4ce2c686b241d9f928db2'


def frozen_factory():
 path=HERE/'borrowed/prepare_source.py'
 if hashlib.sha256(path.read_bytes()).hexdigest()!=FROZEN_FACTORY_SHA:raise ValueError('Frozen M1 native-source factory changed')
 spec=importlib.util.spec_from_file_location('_frozen_capacity_factory_reference',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module


def replacement_map(profile):
 """Original M1 sparse literals -> capacity profile geometry.

 Keys are literal substrings of the frozen prepare_source.py output and must
 each occur exactly once in the candidate header before substitution.
 """
 return {
  'tiling->groupNum == 48':f'tiling->groupNum == {profile.experts}',
  'tiling->m == 6':f'tiling->m == {profile.capacity}',
  'tiling->k == 2304':f'tiling->k == {profile.k}',
  'tiling->n == 5120':f'tiling->n == {profile.n}',
  'sparsePlannerGm.GetValue(3) == 48':f'sparsePlannerGm.GetValue(3) == {profile.experts}',
  'sparsePlannerGm.GetValue(0) <= 6':f'sparsePlannerGm.GetValue(0) <= {profile.capacity}',
  'sparsePlannerGm.GetValue(1) <= 6':f'sparsePlannerGm.GetValue(1) <= {profile.slots}',
  'reinterpret_cast<__gm__ uint32_t*>(planner), 32)':f'reinterpret_cast<__gm__ uint32_t*>(planner), {profile.frame_words})'}


def change_header(original,flavor,profile):
 factory=frozen_factory();changed,math=factory.change_header(original,flavor)
 replacements=replacement_map(profile)
 for needle,value in replacements.items():
  if changed.count(needle)!=1:raise ValueError('Original sparse scheduling insertion is ambiguous: '+needle)
  changed=changed.replace(needle,value)
 for name,expected in math.items():
  begin,end=factory.function_span(changed,name)
  if hashlib.sha256(changed[begin:end].encode()).hexdigest()!=expected:raise ValueError('Native mathematical body changed: '+name)
 return changed,{'native_function_body_sha256_unchanged':math,'original_AIV_dense_loop_preserved':True,
  'only_AIC_schedule_geometry_and_frame_extent_changed':True,'profile':profile.to_dict(),
  'actual_numeric_passed':False,'performance_receipt_granted':False,'formal_receipt_granted':False}


def planner_source(profile):
 return f'''// Capacity extension of the original 4-word-header planner; ABI remains two pointers.
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void gmm2_sparse_row_plan(GM_ADDR counts, GM_ADDR frame) {{
 if (GetBlockIdx()!=0) return;
 GlobalTensor<int64_t> input;GlobalTensor<uint32_t> output;
 input.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(counts),{profile.experts});
 output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(frame),{profile.frame_words});
 for(uint32_t i=0;i<{profile.frame_words};++i) output.SetValue(i,0);
 output.SetValue(3,{profile.experts});
 for(uint32_t i=0;i<{profile.slots};++i) output.SetValue(4+4*i,0xffffffffU);
 uint32_t rows=0,slots=0;
 for(uint32_t expert=0;expert<{profile.experts};++expert) {{
  const int64_t count=input.GetValue(expert);
  if(count<0 || count>{profile.capacity} || rows+count>{profile.capacity}) {{output.SetValue(2,1);return;}}
  if(count==0) continue;
  const uint32_t pos=4+4*slots;
  output.SetValue(pos,expert);output.SetValue(pos+1,rows);output.SetValue(pos+2,static_cast<uint32_t>(count));
  rows+=static_cast<uint32_t>(count);++slots;
 }}
 output.SetValue(0,rows);output.SetValue(1,slots);
}}
'''
