"""Private source-derived W4A8 tap at the actual operator call boundary.

The original source AST is kept, except one original consumer call is wrapped
and private return tuples carry all snapshot tensors. No installed file, global
op or original method is modified. This is preparation, not graph/device proof.
"""
import ast
import copy
import hashlib
import re

from .contracts import SOURCES, check_call, no_actual_claims, pack_target_source
from .snapshot_transport import snapshot_before_native_tuple,validate_buffer_owners_eager

METHOD={'gmm1':'apply_gmm1_act_quant','gmm2':'apply_gmm2'}
TARGET={'gmm1':'torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2',
        'gmm2':'torch_npu.npu_grouped_matmul'}


def schema_declaration(stage):
    text='\n'.join(part['source'] for part in SOURCES[stage+'_schema']['snippets'])
    if stage=='gmm1':
        quoted=re.findall(r'"((?:[^"\\]|\\.)*)"',text)
        declaration=''.join(part for part in quoted if part.startswith('grouped_matmul_swiglu_quant_v2(')
            or part.startswith(' ') or part.startswith('('))
    else:
        declaration=next(line.split('func: ',1)[1].strip() for line in text.splitlines()
            if 'func: npu_grouped_matmul(' in line)
    if not declaration or '->' not in declaration:
        raise ValueError('The actual native schema declaration was not discovered')
    return declaration


def verify_runtime_schema(operator, stage):
    schema=getattr(operator,'_schema',None)
    if schema is None:
        raise ValueError('Actual selected OpOverload schema is required, not a profiler label')
    actual=str(schema)
    # Namespace spelling must be supplied by the actual initialized operator.
    # The declaration and all argument/default/return types remain exact.
    unqualified=actual.split('::',1)[-1]
    normalize=lambda value:re.sub(r'\s+','',value)
    if normalize(unqualified)!=normalize(schema_declaration(stage)):
        raise ValueError('Actual selected GMM schema differs from the installed source declaration')
    return {'actual_runtime_schema':actual,'schema_source_path':SOURCES[stage+'_schema']['path'],
        'schema_source_sha256':SOURCES[stage+'_schema']['sha256'],**no_actual_claims()}


def source_tree(stage):
    name='AscendW4A8DynamicFusedMoEMethod.'+METHOD[stage]
    entry=SOURCES['w4a8']['functions'][name]
    original=ast.parse(entry['source']).body[0]
    return original,entry


def compile_private_native_method(stage, namespace, tap):
    original,entry=source_tree(stage)
    transformed=copy.deepcopy(original)
    count=0
    class WrapOneConsumer(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            node=self.generic_visit(node)
            if ast.unparse(node.func)==TARGET[stage]:
                count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_prepack_tap',ctx=ast.Load()),
                    args=[node.func,*node.args],keywords=node.keywords),node)
            return node
    transformed=WrapOneConsumer().visit(transformed)
    if count!=1:
        raise ValueError('The actual source must contain exactly one selected formal consumer')
    # The real source guard permits the SILU/per-channel formal path only.
    # Reject SITU before any quant/consumer instead of silently auditing another op.
    if stage=='gmm1':
        condition=ast.parse("if mlp_compute_input.activation != MoEActivation.SILU:\n    raise ValueError('Only the original SILU fused prepack path is bound')").body[0]
        transformed.body.insert(0,condition)
    class ReturnCarrier(ast.NodeTransformer):
        def visit_Return(self,node):
            return ast.copy_location(ast.Return(value=ast.Call(func=ast.Name(id='_carry_snapshot',ctx=ast.Load()),
                args=[node.value],keywords=[])),node)
    transformed=ReturnCarrier().visit(transformed)
    # Annotations are transport-irrelevant and can name installed SDK types.
    transformed.returns=None;transformed.decorator_list=[]
    for arg in [*transformed.args.posonlyargs,*transformed.args.args,*transformed.args.kwonlyargs]:arg.annotation=None
    tree=ast.fix_missing_locations(ast.Module(body=[transformed],type_ignores=[]))
    collector={'outputs':None}
    def before_native(native,*args,**kwargs):
        result,outputs=tap(native,args,kwargs)
        if collector['outputs'] is not None:
            raise ValueError('Duplicate native consumer in one private invocation')
        collector['outputs']=outputs
        return result
    def carry(result):
        if collector['outputs'] is None:
            raise ValueError('A private return bypassed the actual selected consumer')
        if stage=='gmm1':
            if not isinstance(result,tuple) or len(result)!=2:
                raise ValueError('Actual GMM1 activation/scale return ABI changed')
            return (*result,*collector['outputs'])
        return (result,*collector['outputs'])
    scope=dict(namespace,_prepack_tap=before_native,_carry_snapshot=carry)
    exec(compile(tree,SOURCES['w4a8']['path'],'exec'),scope)
    fn=scope[transformed.name]
    transformed_text=ast.unparse(tree)
    receipt={'private_only':True,'original_function_source_sha256':entry['sha256'],
        'original_file_sha256':SOURCES['w4a8']['sha256'],
        'selected_native_target':TARGET[stage],'wrapped_native_calls':count,
        'source_derived_private_AST_sha256':hashlib.sha256(transformed_text.encode()).hexdigest(),
        'native_math_and_original_call_arguments_rewritten':False,
        'snapshot_tensor_outputs_retained':6,**no_actual_claims()}
    return fn,collector,receipt


class PreparedStageAdapter:
    """Eager integration seam for an audit call. Not installed anywhere.

    An outer graph must explicitly keep the snapshot tuple using the successful
    fresh SDK transport gates. Host metadata/ledger code stays outside capture;
    the pure snapshot/real consumer source sequence needs actual device proof.
    """
    def __init__(self,torch,binding,stage,namespace,format_getter,lifecycle,buffers,
            positions,valid,epoch,borrow_stream,*,actual_operator=None,sdk_pair_evidence=None):
        self.torch,self.binding,self.stage=torch,binding,stage
        self.format_getter,self.lifecycle=format_getter,lifecycle
        self.buffers,self.positions,self.valid,self.epoch=buffers,positions,valid,epoch
        self.borrow_stream=borrow_stream
        self.runtime_schema_receipt=None
        self.sdk_pair_provenance=None
        if actual_operator is not None:
            self.runtime_schema_receipt=verify_runtime_schema(actual_operator,stage)
        if not binding.cpu_mock:
            if self.runtime_schema_receipt is None:
                raise ValueError('Production adapter needs the actual selected OpOverload schema')
            if sdk_pair_evidence is None or not getattr(sdk_pair_evidence,'protocol_passed',False):
                raise ValueError('Fresh TP8 static/nonstatic SDK transport prerequisite is missing')
            if hasattr(sdk_pair_evidence,'as_dict'):
                self.sdk_pair_provenance=sdk_pair_evidence.as_dict()
            elif hasattr(sdk_pair_evidence,'to_dict'):
                self.sdk_pair_provenance=sdk_pair_evidence.to_dict()
            else:
                raise ValueError('SDK prerequisite must retain immutable original job/rank/source provenance')
        self.lease=None;self.mlp_input=None;self.last_call=None
        def tap(native,args,kwargs):
            if args:
                raise ValueError('The actual source used keyword GMM arguments; ABI changed')
            x,x_meta=check_call(binding,stage,self.mlp_input,kwargs,format_getter)
            if self.lease is None or self.lease.x is not x or self.lease.owner_key!=(binding.role,binding.global_layer) or self.lease.stage!=stage:
                raise ValueError('Raw lease belongs to another role/stage/original input')
            stream=self.borrow_stream()
            if stream.get('kind')!='borrowed_aclrtStream' or type(stream.get('handle')) is not int or stream['handle']<=0:
                raise ValueError('Need the actual current native stream handle, not profiler stream47')
            if not stream.get('source_sha256'):
                raise ValueError('Borrowed native stream source provenance is missing')
            if not binding.cpu_mock and (stream.get('source_sha256')!=SOURCES['sdk_stream']['sha256'] or
                    stream.get('cpp_launch_source_sha256')!=SOURCES['native_stream_and_GM']['sha256']):
                raise ValueError('Actual Python/C++ borrowed stream sources differ from the frozen pair')
            validate_buffer_owners_eager(x,buffers)
            self.lifecycle.snapshot(self.lease,x,buffers[0])
            observed=snapshot_before_native_tuple(torch,x,positions,valid,epoch,buffers)
            self.lifecycle.enter_consumer(self.lease)
            self.last_call={'role':binding.role,'global_layer':binding.global_layer,'stage':stage,
                'original_x_metadata':x_meta,'original_x_object_retained':True,
                'borrowed_stream':dict(stream),'pack_target_source':pack_target_source(stage),
                'activation_version_is_content_proof':False,'CPU_mock':binding.cpu_mock,
                'Python_copy_before_consumer_only':True,**no_actual_claims()}
            # Return ORIGINAL call result, carrier and argument owners. No fake
            # arithmetic dependency, x clone substitution, or scale sanitization.
            result=native(**kwargs)
            if stage=='gmm1' and (tuple(result[0].shape)!=(binding.capacity,binding.ffn) or
                    str(result[0].dtype)!='torch.int8'):
                raise ValueError('Actual GMM1 FFN output differs from loaded W2 K')
            return result,observed[1:]
        self.fn,self.collector,self.source_receipt=compile_private_native_method(stage,namespace,tap)

    def call_eager(self,mlp_input,lease,*native_extra):
        # This entry is outside capture only. Pure graph wiring is emitted as
        # a separate source plan, and cannot use this Python side-effect state.
        self.mlp_input,self.lease=mlp_input,lease
        self.collector['outputs']=None
        result=self.fn(self.binding.scheme,mlp_input,*native_extra)
        return result
