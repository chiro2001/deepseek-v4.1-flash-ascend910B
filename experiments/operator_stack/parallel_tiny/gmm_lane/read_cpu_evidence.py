"""Read-only source/checkpoint samples. Standard library only; never imports NPU code."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


REMOTE_PROGRAM = r'''
from pathlib import Path
import json,hashlib,struct
base=Path('/vllm-workspace/vllm-ascend')
vendor=base/'vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/grouped_matmul_swiglu_quant_v2'
core=Path('/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul')
spec={
 'w4a8':base/'vllm_ascend/quantization/methods/w4a8/w4a8.py',
 'gmm1_pre':vendor/'grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h',
 'gmm1_mid':vendor/'grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h',
 'gmm1_post':vendor/'grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h',
 'gmm1_pipeline':vendor/'grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h',
 'gmm1_cpp':vendor/'grouped_matmul_swiglu_quant_v2.cpp',
 'gmm1_torch_adapter':base/'csrc/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h',
 'gmm1_base_tiling':base/'csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_base_tiling.cpp',
 'gmm1_fusion_tiling':base/'csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_fusion_tiling.cpp',
 'gmm1_api_utils':base/'csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/grouped_matmul_swiglu_quant_v2_utils.h',
 'gmm2_pre':core/'grouped_matmul_antiquant_a8w4_msd_pre.h',
 'gmm2_mid_post':core/'grouped_matmul_antiquant_a8w4_msd.h',
 'gmm2_mid_post_new':core/'grouped_matmul_antiquant_a8w4_msd_new.h',
 'gmm2_generic_utils':core/'grouped_matmul_utils.h',
 'gmm2_entry':core/'grouped_matmul.cpp',
}
sources={}
for key,p in spec.items():
 data=p.read_bytes()
 assert len(data)<1000000
 sources[key]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'content':data.decode()}
model=Path('/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq')
cfg=(model/'config.json').read_bytes()
config=json.loads(cfg)['text_config']
index=json.loads((model/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
quant=json.loads((model/'quant_model_description.json').read_text())
header_cache={}
samples=[]
for layer in [0,20]:
 for expert in [0,47,383]:
  for projection in ['w1','w3','w2']:
   prefix=f'layers.{layer}.ffn.experts.{expert}.{projection}'
   names=[prefix+'.'+suffix for suffix in ['weight','weight_scale','weight_offset','scale_bias']]
   meta={}
   for name in names:
    path=model/index[name]
    if path not in header_cache:
     with path.open('rb') as f:
      hlen=struct.unpack('<Q',f.read(8))[0]
      header_cache[path]=(8+hlen,json.loads(f.read(hlen)))
    start,head=header_cache[path]
    record=head[name]
    begin,end=record['data_offsets']
    with path.open('rb') as f:
     f.seek(start+begin)
     first=f.read(min(end-begin,128))
    info={'dtype':record['dtype'],'shape':record['shape'],'data_bytes':end-begin,'file':str(path),'first_128_bytes_hex':first.hex(),'quantization':quant.get(name)}
    if record['dtype']=='F32':
     info['first_f32_values']=list(struct.unpack('<'+'f'*(len(first)//4),first))
    meta[name]=info
   weight=meta[names[0]]
   assert weight['dtype']=='I8'
   packed_n,k=weight['shape']
   n=packed_n*2
   path=model/index[names[0]]
   start,head=header_cache[path]
   begin=head[names[0]]['data_offsets'][0]
   coords=[]
   with path.open('rb') as f:
    for kk in [0,15,16,k-1]:
     for nn in [0,1,7,8,31,32,63,64,n-1]:
      f.seek(start+begin+(nn//2)*k+kk)
      byte=f.read(1)[0]
      nibble=(byte>>(4*(nn%2)))&15
      coords.append({'k':kk,'n':nn,'raw_byte_offset':(nn//2)*k+kk,'raw_byte':byte,'signed_int4':nibble-16 if nibble>=8 else nibble})
   samples.append({'layer':layer,'global_expert':expert,'projection':projection,'logical_k':k,'logical_n':n,'tensors':meta,'coordinate_samples':coords})
result={'container':'dsv41-real-stack-tp8-20261010-a321','read_only':True,'npu_initialized':False,'checkpoint':str(model),'config_sha256':hashlib.sha256(cfg).hexdigest(),'config':{k:config.get(k) for k in ['hidden_size','moe_intermediate_size','num_hidden_layers','n_routed_experts','num_experts_per_tok']},'quant_group_size':quant.get('group_size'),'sources':sources,'checkpoint_samples':samples}
raw=json.dumps(result)
assert len(raw.encode())<1000000
print(raw)
'''


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert args.output.is_relative_to("experiments/operator_stack/parallel_tiny/gmm_lane")
    assert not args.output.exists()
    command = "docker exec dsv41-real-stack-tp8-20261010-a321 python3 -c " + shlex.quote(REMOTE_PROGRAM)
    data = subprocess.check_output(["ssh", "-o", "ControlPath=none", "-o", "BatchMode=yes", "a3-21", command])
    assert len(data) < 1000000
    result = json.loads(data)
    references = args.output.parent / "references" / "installed"
    references.mkdir(parents=True, exist_ok=False)
    for key, source in result["sources"].items():
        content = source.pop("content")
        assert hashlib.sha256(content.encode()).hexdigest() == source["sha256"]
        reference = references / (key + Path(source["path"]).suffix + ".txt")
        reference.write_text(content)
        source["local_reference"] = str(reference)
    result["ssh_bundle_bytes"] = len(data)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sources": len(result["sources"]), "parameter_sets": len(result["checkpoint_samples"]),
                      "ssh_bundle_bytes": len(data), "npu_initialized": False}))


if __name__ == "__main__":
    main()
