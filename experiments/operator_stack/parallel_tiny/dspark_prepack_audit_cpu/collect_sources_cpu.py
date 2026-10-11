"""Remote installed-source capture, stdlib only; never import SDK/model/device."""
import ast
import hashlib
import json
from pathlib import Path
import sysconfig

ASCEND = Path('/vllm-workspace/vllm-ascend')
SITE = Path(sysconfig.get_paths()['purelib'])
CANN = Path('/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/grouped_matmul')
GM1 = ASCEND / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/grouped_matmul_swiglu_quant_v2'

PYTHON = {
    'w4a8': (ASCEND/'vllm_ascend/quantization/methods/w4a8/w4a8.py',
        ['_get_mlp_weights', '_get_mlp_scale_bias', '_maybe_convert_group_list',
         'apply_gmm1_act_quant', 'apply_gmm2', 'process_weights_after_loading_modelslim']),
    'quant_base': (ASCEND/'vllm_ascend/quantization/methods/base.py', ['_quant_hidden_states']),
    'scheme_adapter': (ASCEND/'vllm_ascend/quantization/method_adapters.py', ['AscendFusedMoEMethod.__init__','AscendFusedMoEMethod.apply']),
    'native_mlp_sequence': (ASCEND/'vllm_ascend/ops/fused_moe/moe_mlp.py', ['apply_moe_mlp']),
    'fused_input_owner': (ASCEND/'vllm_ascend/ops/fused_moe/dataclass/fused_experts.py', ['build_fused_experts_input']),
    'role_model': (ASCEND/'vllm_ascend/models/deepseek_v4/model.py', ['DeepseekV4MoE.__init__', 'DeepseekV4MoE.forward']),
    'draft_override': (ASCEND/'vllm_ascend/patch/platform/patch_speculative_config.py', ['_normalize_deepseek_v4_dspark_draft']),
    'mlp_input': (ASCEND/'vllm_ascend/ops/fused_moe/dataclass/moe_mlp.py', ['MoEMlpComputeInput', 'build_mlp_compute_input']),
    'sdk_stream': (SITE/'torch_npu/npu/utils.py', ['current_stream']),
}
TEXT = {
    'gmm1_schema': (ASCEND/'csrc/torch_binding.cpp', [(2215,2225)]),
    'gmm1_torch': (ASCEND/'csrc/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h', [(20,82)]),
    'native_stream_and_GM': (ASCEND/'csrc/aclnn_torch_adapter/op_api_common.h', [(375,438),(680,748)]),
    'gmm1_pre': (GM1/'grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h', [(75,95),(148,214)]),
    'gmm1_pipeline': (GM1/'grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h', [(65,137)]),
    'gmm1_nz_descriptor': (ASCEND/'csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp', [(105,167)]),
    'gmm1_int4_view': (ASCEND/'csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/gmm_dsq_base.h', [(520,562)]),
    'gmm2_schema': (SITE/'torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml', [(442,454)]),
    'gmm2_pre': (CANN/'grouped_matmul_antiquant_a8w4_msd_pre.h', [(72,85),(128,144),(180,235)]),
    'gmm2_kernel_selector': (CANN/'grouped_matmul.cpp', [(243,282)]),
}


def defs(tree, prefix=''):
    result = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            name=prefix+node.name
            result[name]=node
            result.update(defs(node,name+'.'))
        else:
            result.update(defs(node,prefix))
    return result


def main():
    records={}
    for label,(path,names) in PYTHON.items():
        data=path.read_bytes();source=data.decode();all_defs=defs(ast.parse(source));fragments={}
        for name in names:
            selected=[(q,n) for q,n in all_defs.items() if q==name or ('.' not in name and q.endswith('.'+name))]
            if len(selected)!=1:
                fragments[name]={'missing_or_ambiguous':True,'candidates':[q for q,_ in selected]}
                continue
            q,node=selected[0];segment=ast.get_source_segment(source,node)
            fragments[q]={'first_line':node.lineno,'last_line':node.end_lineno,
                'source':segment,'sha256':hashlib.sha256(segment.encode()).hexdigest()}
        records[label]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'functions':fragments}
    for label,(path,ranges) in TEXT.items():
        data=path.read_bytes();lines=data.decode().splitlines();snippets=[]
        for first,last in ranges:
            snippet='\n'.join(lines[first-1:last]);snippets.append({'first_line':first,'last_line':last,
                'source':snippet,'sha256':hashlib.sha256(snippet.encode()).hexdigest()})
        records[label]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'snippets':snippets}
    print(json.dumps({'CPU_source_only':True,'SDK_imports':0,'NPU_calls':0,'HTTP_inference_requests':0,
        'sources':records},indent=2))


if __name__=='__main__':main()
