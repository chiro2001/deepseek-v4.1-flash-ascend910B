"""Original modelslim transforms for the actual main/draft local W2 role."""
import hashlib
import json
from pathlib import Path
import struct

from .profile import actual_profiles,bind,digest,false_claims

MODEL=Path('/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq')
SOURCE=Path('/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w4a8/w4a8.py')
SOURCE_SHA='f3a8cdd205719453606fa106d47c1e556ef40d2ee4046869a029d5f12a3b6de9'


def nz_expert_bytes(weight_nd,expert):
 import numpy as np
 k,half_n=map(int,weight_nd.shape[1:]);n=half_n*2
 if k%16 or n%64:raise ValueError('Original packed NZ alignment is unsupported')
 return np.ascontiguousarray(weight_nd[expert].numpy().reshape(k//16,16,n//64,32).transpose(2,0,1,3)).tobytes()


def header_record(model,mapping,name,headers):
 path=model/mapping[name]
 if path not in headers:
  with path.open('rb') as stream:
   prefix=stream.read(8);size=struct.unpack('<Q',prefix)[0]
   if not 0<size<=100_000_000:raise ValueError('Invalid actual W2 source header')
   encoded=stream.read(size)
  headers[path]=(8+size,json.loads(encoded),hashlib.sha256(prefix+encoded).hexdigest())
 start,header,header_sha=headers[path];spec=header[name]
 return path,start,header_sha,spec


def load_weights(family,model=MODEL,rank=0,layer=0,read_payloads=True):
 """No torch_npu import. K comes from the actual selected original W2 tensor."""
 if family not in ('target','draft') or rank!=0 or layer!=0:raise ValueError('This mechanism probe binds only original role layer0/rank0')
 model=Path(model)
 if digest(SOURCE)!=SOURCE_SHA:raise ValueError('Original modelslim loader source changed')
 if digest(model/'config.json')!='40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424':
  raise ValueError('Original full checkpoint config owner changed')
 actual=actual_profiles()[family];experts=actual['experts']
 index=model/'quant_model_weights.safetensors.index.json';description=model/'quant_model_description.json'
 mapping=json.loads(index.read_text())['weight_map'];quant=json.loads(description.read_text())
 if quant.get('group_size')!=0:raise ValueError('Original per-channel W4A8 contract is required')
 prefix='layers.0.ffn.experts' if family=='target' else 'mtp.0.ffn.experts'
 headers={};records=[];k=n=None
 for expert in range(experts):
  for suffix in ('weight','weight_scale','scale_bias','weight_offset'):
   name=f'{prefix}.{expert}.w2.{suffix}'
   if name not in mapping:raise ValueError('Actual role W2 source tensor is missing: '+name)
   path,start,header_sha,spec=header_record(model,mapping,name,headers)
   if quant.get(name)!='W4A8_DYNAMIC':raise ValueError('Actual W2 role quantization differs')
   if suffix=='weight':
    if spec['dtype']!='I8' or len(spec['shape'])!=2:raise ValueError('Original packed INT8 W2 tensor is required')
    observed_n,observed_k=spec['shape'][0]*2,spec['shape'][1]
    if k is not None and (k,n)!=(observed_k,observed_n):raise ValueError('Original W2 expert shapes differ')
    k,n=observed_k,observed_n
   expected={'weight':('I8',[n//2,k]),'weight_scale':('F32',[n,1]),'scale_bias':('F32',[n,16]),'weight_offset':('F32',[n,1])}[suffix]
   if (spec['dtype'],spec['shape'])!=expected:raise ValueError('Original modelslim W2 auxiliary shape/dtype differs')
   begin,end=spec['data_offsets']
   records.append({'name':name,'local_expert':expert,'global_expert':expert,'suffix':suffix,'file':str(path),
    'file_bytes':path.stat().st_size,'file_mtime_ns':path.stat().st_mtime_ns,'header_sha256':header_sha,
    'absolute_byte_range':[start+begin,start+end],'dtype':spec['dtype'],'shape':spec['shape'],'bytes':end-begin})
 if (experts,k,n)!=(actual['experts'],actual['observed_logical_k'],actual['n']):
  raise ValueError('Actual original role W2 headers differ from all-rank profile geometry')
 manifest={'schema':'k5_capacity_original_W2_manifest_v1','family':family,'checkpoint':str(model),'rank':rank,'layer':layer,
  'local_to_global_experts':list(range(experts)),'raw_tensor_records':records,
  'header_geometry_proposal':{'capacity':actual['capacity'],'experts':experts,'k_from_actual_W2_header':k,'n':n,
   'actual_W2_CPU_or_NPU_load_executed':False,'K_derived_from_loaded_W2_and_FFN_partition':False},
  'loader_source':str(SOURCE),'loader_source_sha256':SOURCE_SHA,'config_sha256':digest(model/'config.json'),
  'index_sha256':digest(index),'description_sha256':digest(description),'tensor_payloads_read':read_payloads,
  'original_NPU_NZ_cast_executed':False,'synthetic_weights':False,**false_claims()}
 if not read_payloads:return None,None,None,manifest
 import numpy as np
 import torch
 raw_weight=np.empty((experts,n//2,k),dtype=np.int8)
 raw_scale=np.empty((experts,n,1),dtype=np.float32)
 raw_assist=np.empty((experts,n,16),dtype=np.float32)
 offsets_zero=True
 for record in records:
  with Path(record['file']).open('rb') as stream:
   stream.seek(record['absolute_byte_range'][0]);encoded=stream.read(record['bytes'])
  dtype=np.int8 if record['dtype']=='I8' else np.float32
  expected_bytes=int(np.prod(record['shape']))*np.dtype(dtype).itemsize
  if len(encoded)!=record['bytes'] or len(encoded)!=expected_bytes:raise ValueError('Actual source W2 payload length differs')
  value=np.frombuffer(encoded,dtype=dtype).reshape(record['shape'])
  if not np.isfinite(value).all():raise ValueError('Actual original W2/auxiliary contains nonfinite values')
  record.update(source_tensor_sha256=hashlib.sha256(encoded).hexdigest(),finite=True,min=float(value.min()),max=float(value.max()))
  expert=record['local_expert'];suffix=record['suffix']
  if suffix=='weight':raw_weight[expert]=value
  elif suffix=='weight_scale':raw_scale[expert]=value
  elif suffix=='scale_bias':raw_assist[expert]=value
  else:offsets_zero=offsets_zero and bool((value==0).all())
 if not offsets_zero:raise ValueError('Original zero-offset native MSD role is not supported')
 # Identical Torch CPU operations to the frozen original loader/proven M1 path.
 weight_nd=torch.from_numpy(raw_weight).transpose(1,2).contiguous()
 scale_np=torch.from_numpy(raw_scale).transpose(1,2).contiguous().numpy().copy()
 scale_np.dtype=np.uint32
 scale=torch.from_numpy(scale_np.astype(np.int64))
 assist=torch.from_numpy(raw_assist).transpose(1,2).contiguous().sum(axis=1)
 actual_loaded=bind(family,tuple(map(int,weight_nd.shape)),'original_CPU_transposed_INT8',int(weight_nd.shape[1]),str(model)+'/'+prefix)
 nz_hash=hashlib.sha256();expert_hashes=[]
 for expert in range(experts):
  encoded=nz_expert_bytes(weight_nd,expert);nz_hash.update(encoded);expert_hashes.append(hashlib.sha256(encoded).hexdigest())
 def rec(tensor,dtype):
  data=tensor.numpy().tobytes()
  return {'shape':list(tensor.shape),'dtype':dtype,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 manifest.update(loaded_cpu_profile=actual_loaded.to_dict(),all_original_weight_offsets_zero=True,
  original_CPU_loader_transforms_verified=True,
  processed={'weight_ND':rec(weight_nd,'INT8'),'scale':rec(scale,'INT64'),'assist':rec(assist,'FP32'),
   'expected_physical_NZ':{'INT32_storage':[experts,n//64,k//16,16,8],'sha256':nz_hash.hexdigest(),'expert_sha256':expert_hashes}})
 return weight_nd,scale,assist,manifest
