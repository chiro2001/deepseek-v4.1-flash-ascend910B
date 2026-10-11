"""Source-derived pure Tensor audit entry for a future private outer graph.

No host ledger, pointer read, file hash, D2H, sanitize, SDK import or live hook.
The original selected consumer expression/arguments are left untouched. All
six raw snapshot tensor outputs must be retained by the owning outer program.
Actual compiler/device copy-before-pack order remains an unpassed device gate.
"""
import ast
import copy
import hashlib

from .contracts import no_actual_claims
from .native_adapter_factory import source_tree,TARGET
from .snapshot_transport import snapshot_before_native_tuple


def make_pure_graph_entry(stage,namespace):
    original,entry=source_tree(stage)
    changed=copy.deepcopy(original)
    inserted=0
    snapshot=ast.parse('_raw_audit = _snapshot_tensor_tuple(torch, hidden_states, audit_positions, audit_valid, audit_epoch, audit_buffers)').body[0]
    class InsertAtOriginalBoundary(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal inserted
            matches=[n for n in ast.walk(node.value) if isinstance(n,ast.Call) and ast.unparse(n.func)==TARGET[stage]]
            if matches:
                if stage!='gmm1' or len(matches)!=1:
                    raise ValueError('Unexpected native assignment consumer structure')
                inserted+=1
                return [copy.deepcopy(snapshot),node]
            return node

        def visit_Return(self,node):
            nonlocal inserted
            matches=[n for n in ast.walk(node.value) if isinstance(n,ast.Call) and ast.unparse(n.func)==TARGET[stage]]
            if matches:
                if stage!='gmm2' or len(matches)!=1:
                    raise ValueError('Unexpected native return consumer structure')
                inserted+=1
                # Keep the entire ORIGINAL return expression including [0].
                assignment=ast.Assign(targets=[ast.Name(id='_native_result',ctx=ast.Store())],value=node.value)
                result=ast.parse('return (_native_result, *_raw_audit[1:])').body[0]
                return [copy.deepcopy(snapshot),assignment,result]
            if stage=='gmm1':
                return ast.Return(value=ast.Tuple(elts=[*node.value.elts,
                    ast.Starred(value=ast.Subscript(value=ast.Name(id='_raw_audit',ctx=ast.Load()),
                        slice=ast.Slice(lower=ast.Constant(value=1)),ctx=ast.Load()),ctx=ast.Load())],ctx=ast.Load()))
            return node
    changed=InsertAtOriginalBoundary().visit(changed)
    if inserted!=1:
        raise ValueError('Expected one original prepack consumer boundary')
    if stage=='gmm1':
        changed.body.insert(0,ast.parse("if mlp_compute_input.activation != MoEActivation.SILU:\n    raise ValueError('Only source-guarded formal SILU GMM1 is prepared')").body[0])
    changed.args.kwonlyargs.extend([ast.arg(arg=name) for name in
        ('audit_positions','audit_valid','audit_epoch','audit_buffers')])
    changed.args.kw_defaults.extend([None]*4)
    changed.returns=None;changed.decorator_list=[]
    for arg in [*changed.args.posonlyargs,*changed.args.args,*changed.args.kwonlyargs]:arg.annotation=None
    tree=ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))
    # Native call AST equality is independent of the added Tensor transport.
    original_calls=[ast.dump(n,include_attributes=False) for n in ast.walk(original)
        if isinstance(n,ast.Call) and ast.unparse(n.func)==TARGET[stage]]
    changed_calls=[ast.dump(n,include_attributes=False) for n in ast.walk(tree)
        if isinstance(n,ast.Call) and ast.unparse(n.func)==TARGET[stage]]
    if original_calls!=changed_calls:
        raise ValueError('The native consumer call/argument AST changed')
    scope=dict(namespace,_snapshot_tensor_tuple=snapshot_before_native_tuple)
    exec(compile(tree,entry.get('path','<source-derived-prepack>'),'exec'),scope)
    text=ast.unparse(tree)
    receipt={'private_source_only':True,'stage':stage,
        'original_function_sha256':entry['sha256'],
        'private_AST_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'original_native_call_AST_byte_equal':True,'six_explicit_snapshot_tensor_outputs':True,
        'actual_graph_captured':False,'actual_device_order_validated':False,
        'graph_compiler_must_not_drop_or_reorder_snapshot':True,**no_actual_claims()}
    return scope[changed.name],text,receipt
