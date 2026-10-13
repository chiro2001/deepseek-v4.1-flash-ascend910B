"""Actual role/loaded-weight geometry and unchanged native pointer-list ABI."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
SPARSE=HERE.parent
GEOMETRY_SHA='bbd121e63720ef3d3d20f64c007847b5117f7891b0a74ead6660e864259abee7'
CONFIG_SOURCE_SHA='659d24f4e9bac4543e8408fb64d61205be34afbcef14f5d65a91dfec725fe185'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def false_claims():
 return {'actual_capacity_numeric_passed':False,'actual_consumer_activation_observed':False,
  'actual_target36_consumer_proof':False,'actual_draft15_consumer_proof':False,
  'actual_device_consumer_proof':False,'performance_receipt_granted':False,'formal_receipt_granted':False}


def align(value,alignment):return (value+alignment-1)//alignment*alignment


@dataclass(frozen=True)
class Profile:
 family:str
 capacity:int
 experts:int
 k:int
 n:int
 loaded_W2_shape:tuple
 loaded_W2_stage:str
 FFN_partition:int
 source_reference:str

 @property
 def slots(self):return min(self.capacity,self.experts)
 @property
 def frame_bytes(self):return align(4*(4+4*self.slots),128)
 @property
 def frame_words(self):return self.frame_bytes//4
 def to_dict(self):
  return {**asdict(self),'slots':self.slots,'frame_bytes':self.frame_bytes,'frame_words':self.frame_words,
   'K_derived_from_loaded_W2_and_FFN_partition':True,'native_math_changes':False,**false_claims()}


def actual_profiles():
 path=HERE/'evidence/actual_GMM2_geometry_all_ranks.json'
 config=HERE/'evidence/actual_config_and_profile_source.json'
 if digest(path)!=GEOMETRY_SHA or digest(config)!=CONFIG_SOURCE_SHA:raise ValueError('Frozen actual geometry/config source changed')
 geometry=json.loads(path.read_text());cfg=json.loads(config.read_text())['config_geometry_fields']
 ranks=geometry['rank_profiles']
 if len(ranks)!=8 or geometry['profile_geometry_is_capacity_not_counts_value'] is not True:
  raise ValueError('Original all-rank capacity geometry is required')
 expected={
  'target':(cfg['text_config.n_routed_experts']//8,(cfg['text_config.dspark_block_size']+1)*cfg['text_config.num_experts_per_tok']),
  'draft':(cfg['text_config.dspark_n_routed_experts']//8,cfg['text_config.dspark_block_size']*cfg['text_config.dspark_num_experts_per_tok'])}
 found={}
 for rank in ranks:
  if len(rank['row_groups'])!=2:raise ValueError('Both target and native draft geometry are required')
  local={}
  for group in rank['row_groups']:
   example=group['example'];shapes=example['Input Shapes'].strip('"').split(';')
   x=tuple(int(x) for x in shapes[0].split(','));w=tuple(int(x) for x in shapes[1].split(','))
   if len(x)!=2 or len(w)!=3 or x[1]!=w[1] or example['Output Data Types']!='DT_BF16':
    raise ValueError('Actual native GMM2 logical tensor geometry differs')
   matches=[name for name,(e,m) in expected.items() if (w[0],x[0])==(e,m)]
   if len(matches)!=1:raise ValueError('Actual GMM2 role is ambiguous')
   name=matches[0];local[name]={'capacity':x[0],'experts':w[0],'observed_logical_k':w[1],'n':w[2]}
  if set(local)!={'target','draft'} or (found and local!=found):raise ValueError('All-rank geometry mismatch')
  found=local
 return found


def bind(family,loaded_shape,stage,ffn_partition,source_reference,activation_shape=None):
 actual=actual_profiles()
 if family not in actual or stage not in ('original_CPU_transposed_INT8','original_NPU_NZ_INT32_view'):
  raise ValueError('Exact native role and original loaded W2 stage are required')
 shape=tuple(loaded_shape)
 if len(shape)!=3 or any(type(x)is not int or x<=0 for x in shape):raise ValueError('Actual loaded W2 shape must be positive integer E/K/packedN')
 e,k,packed_n=shape;n=packed_n*(2 if stage=='original_CPU_transposed_INT8' else 8)
 observed=actual[family]
 if e!=observed['experts'] or n!=observed['n'] or k!=observed['observed_logical_k'] or ffn_partition!=k:
  raise ValueError('Loaded W2/FFN role does not match the actual native profile')
 if k%16 or n%64 or not source_reference:raise ValueError('Original packed NZ alignment/provenance is missing')
 if activation_shape is not None and tuple(activation_shape)!=(observed['capacity'],k):
  raise ValueError('Raw prepack activation K/capacity differs from the loaded role')
 return Profile(family,observed['capacity'],e,k,n,shape,stage,ffn_partition,source_reference)


def compile_input(template,profile):
 value=json.loads(json.dumps(template));op=value['op_list'][0]
 inputs=op['inputs'];x,w,bias,scale=inputs[:4]
 if len(inputs)!=9:raise ValueError('Native input arity changed')
 if inputs[4:7]!=[None,None,None]:raise ValueError('Original null offset/antiquant payloads must be preserved')
 expected_dtypes={'x':('int8','ND'),'weight':('int4','FRACTAL_NZ'),'bias':('float32','ND'),
  'scale':('uint64','ND'),'group_list':('int64','ND'),'per_token_scale':('float32','ND')}
 for item in x+w+bias+scale+[inputs[7],inputs[8]]:
  if (item['dtype'],item['format'])!=expected_dtypes[item['name']]:
   raise ValueError('Native descriptor dtype/format changed: '+item['name'])
 for item in op['outputs'][0]:
  if (item['dtype'],item['format'])!=('bfloat16','ND'):raise ValueError('Native output descriptor changed')
 for item in x:item.update(shape=[profile.capacity,profile.k],ori_shape=[profile.capacity,profile.k])
 for item in w:item.update(shape=[profile.experts,profile.n//64,profile.k//16,16,64],ori_shape=[profile.experts,profile.k,profile.n])
 for item in bias:item.update(shape=[profile.experts,profile.n],ori_shape=[profile.experts,profile.n])
 for item in scale:item.update(shape=[profile.experts,1,profile.n],ori_shape=[profile.experts,1,profile.n])
 inputs[7].update(shape=[profile.experts],ori_shape=[profile.experts])
 inputs[8].update(shape=[profile.capacity],ori_shape=[profile.capacity])
 for item in op['outputs'][0]:item.update(shape=[profile.capacity,profile.n],ori_shape=[profile.capacity,profile.n])
 expected={'split_item':2,'dtype':27,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':1,'act_type':0,'tuning_config':[0]}
 if {item['name']:item['value'] for item in op['attrs']}!=expected:raise ValueError('Native attributes changed')
 return value
