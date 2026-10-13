"""Source-derived private methods: original native AST plus explicit audit taps.

No SDK import or install on import. Native operator calls/kwargs, conditions,
returns and arithmetic must survive stripping the audit expressions exactly.
"""
import ast
import __future__
import copy
import hashlib
import inspect
import linecache
from pathlib import Path
import sys
import types


VALUE_CALLS = {
    'DeepseekV4DSparkModel.combine_hidden_states':{'self.main_proj':'combine.proj','self.main_norm':'combine.norm'},
    'DeepseekV4DSparkModel._project_shared_kv':{'attn.wkv':'shared.wkv','attn.kv_norm':'shared.norm',
                                            '_apply_dsv4_rope':'shared.rope'},
}
WRITE_CALLS = {'torch_npu.npu_scatter_nd_update_','torch.ops._C_ascend.npu_scatter_nd_update_sk',
               'torch.ops._C_ascend.kv_compress_epilog','scatter_cache_sk'}


def function_node(source, qualified):
    cls,method = qualified.split('.')
    found = [node for owner in ast.parse(source).body if isinstance(owner,ast.ClassDef) and owner.name == cls
             for node in owner.body if isinstance(node,ast.FunctionDef) and node.name == method]
    if len(found) != 1:
        raise ValueError('Native source method missing/ambiguous: '+qualified)
    return found[0]


def argument_dict(node):
    names = [arg.arg for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs] if arg.arg != 'self']
    if node.args.vararg:names.append(node.args.vararg.arg)
    if node.args.kwarg:names.append(node.args.kwarg.arg)
    return ast.Dict(keys=[ast.Constant(name) for name in names],values=[ast.Name(name,ast.Load()) for name in names])


class InsertAudit(ast.NodeTransformer):
    def __init__(self, qualified, arguments):
        self.qualified = qualified;self.arguments = arguments

    def meta(self):
        return copy.deepcopy(self.arguments)

    def visit_Return(self,node):
        node = self.generic_visit(node)
        if node.value is None:return node
        node.value = ast.Call(ast.Name('_v10_return',ast.Load()),
            [ast.Name('self',ast.Load()),ast.Constant(self.qualified),node.value or ast.Constant(None),self.meta()],[])
        return node

    def visit_Call(self,node):
        node = self.generic_visit(node)
        label = VALUE_CALLS.get(self.qualified,{}).get(ast.unparse(node.func))
        if label:
            return ast.Call(ast.Name('_v10_value',ast.Load()),
                [ast.Name('self',ast.Load()),ast.Constant(label),node,self.meta()],[])
        return node

    def visit_Expr(self,node):
        node = self.generic_visit(node)
        if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in WRITE_CALLS:
            call = node.value
            args = ast.Tuple([copy.deepcopy(value) for value in call.args],ast.Load())
            kwargs = ast.Dict([ast.Constant(item.arg) for item in call.keywords],
                              [copy.deepcopy(item.value) for item in call.keywords])
            audit = ast.Expr(ast.Call(ast.Name('_v10_before_write',ast.Load()),
                [ast.Name('self',ast.Load()),ast.Constant(self.qualified),ast.Constant(ast.unparse(call.func)),
                 args,kwargs,self.meta()],[]))
            return [audit,node]
        return node


class StripAudit(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id.startswith('_v10_'):
            return None
        return self.generic_visit(node)

    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id in ('_v10_return','_v10_value'):
            return self.visit(node.args[2])
        return self.generic_visit(node)


def transform_method(source,entry):
    data = source.encode()
    if hashlib.sha256(data).hexdigest() != entry['file_sha256']:
        raise ValueError('Native source file SHA changed')
    original = function_node(source,entry['qualified_function'])
    segment = ast.get_source_segment(source,original)
    if hashlib.sha256(segment.encode()).hexdigest() != entry['function_sha256']:
        raise ValueError('Native source function SHA changed')
    node = InsertAudit(entry['qualified_function'],argument_dict(original)).visit(copy.deepcopy(original))
    entry_call = ast.Expr(ast.Call(ast.Name('_v10_enter',ast.Load()),
        [ast.Name('self',ast.Load()),ast.Constant(entry['qualified_function']),argument_dict(original)],[]))
    index = 1 if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str) else 0
    node.body.insert(index,entry_call)
    stripped = StripAudit().visit(copy.deepcopy(node))
    if ast.dump(stripped,include_attributes=False) != ast.dump(original,include_attributes=False):
        raise ValueError('Audit insertion changed original native AST')
    ast.fix_missing_locations(node)
    return node,{'qualified_function':entry['qualified_function'],'file_sha256':entry['file_sha256'],
                 'function_sha256':entry['function_sha256'],'native_AST_preserved':True,
                 'device_audit_passed':False,'full_consumer_audit_passed':False,'safe_formal_receipt':False}


def export_dynamo_guard_aliases(actual,callbacks):
    """Publish the audit entry points where torch.dynamo will look for them.

    A source-derived method keeps the native ``__module__``. When Dynamo traces
    it, every LOAD_GLOBAL audit name is turned into an attribute guard on that
    module object (``getattr(sys.modules[fn.__module__], '_v10_enter')``).
    Without the same binding on the module the guard raises
    ``AttributeError: module ... has no attribute '_v10_enter'`` and the whole
    profile/capture compile fails. Export one stable object per name; a second
    install with different callbacks would silently misroute taps and is
    rejected.
    """
    aliases = {'_v10_enter':callbacks.enter,'_v10_return':callbacks.return_value,
               '_v10_value':callbacks.value,'_v10_before_write':callbacks.before_write}
    module_name = getattr(actual,'__module__',None)
    module = sys.modules.get(module_name) if module_name else None
    if module is None:
        # Fixtures without an importable module fall back to the globals-dict
        # route; nothing is exported and the receipt says so explicitly.
        return aliases,{'dynamo_module':module_name,'dynamo_module_guard_aliases':None,
                        'dynamo_aliases_exported':False}

    def same_callable(first,second):
        if first is second:
            return True
        # Bound methods are re-created on each attribute access; compare the
        # underlying function and instance instead of object identity.
        return (getattr(first,'__self__',None) is not None
                and getattr(first,'__self__',None) is getattr(second,'__self__',None)
                and getattr(first,'__func__',None) is getattr(second,'__func__',None))

    alias_modules = getattr(callbacks,'dynamo_alias_modules',None)
    rebinds = getattr(callbacks,'dynamo_alias_rebinds',None)
    for name,value in aliases.items():
        existing = module.__dict__.get(name)
        if existing is not None and not same_callable(existing,value):
            # A rebind is only acceptable over a previous export of our own tap
            # class (fresh session in the same process, e.g. CPU contracts). A
            # foreign attribute with the same name is a real conflict.
            if type(getattr(existing,'__self__',None)).__name__ != 'ConsumerCallbacks':
                raise ValueError('Native module already defines a conflicting '+name)
            if rebinds is not None:
                rebinds.append({'module':module_name,'name':name})
        if existing is None or not same_callable(existing,value):
            module.__dict__[name] = value
    if alias_modules is not None:
        alias_modules[module_name] = sorted(aliases)
    return aliases,{'dynamo_module':module_name,'dynamo_module_guard_aliases':sorted(aliases),
                    'dynamo_aliases_exported':True}


def make_private_method(original,entry,callbacks):
    actual = inspect.unwrap(getattr(original,'__func__',original))
    if actual.__qualname__ != entry['qualified_function']:
        raise ValueError('Actual native method owner differs')
    source = Path(inspect.getfile(actual)).read_text()
    node,receipt = transform_method(source,entry)
    # Decorators already wrap the supplied native bound method. The standard
    # runner enters inference_mode; the source body itself is preserved here.
    node.decorator_list = []
    private_cls = ast.ClassDef(name='_V10PrivateSourceClass',bases=[],keywords=[],body=[node],decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[private_cls],type_ignores=[]))
    filename = '/CPU-or-private-v10/'+entry['qualified_function']+'.py'
    text = ast.unparse(tree);linecache.cache[filename]=(len(text),None,text.splitlines(keepends=True),filename)
    globals_dict = dict(actual.__globals__)
    aliases,guard_receipt = export_dynamo_guard_aliases(actual,callbacks)
    globals_dict.update(aliases)
    receipt.update(guard_receipt)
    exec(compile(tree,filename,'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),globals_dict)
    candidate = globals_dict['_V10PrivateSourceClass'].__dict__[node.name]
    old_cells = dict(zip(actual.__code__.co_freevars,actual.__closure__ or ()))
    if any(name not in old_cells for name in candidate.__code__.co_freevars):
        raise ValueError('Unsupported native closure in source-derived private method')
    closure = tuple(old_cells[name] for name in candidate.__code__.co_freevars) or None
    function = types.FunctionType(candidate.__code__,globals_dict,actual.__name__,actual.__defaults__,closure)
    function.__kwdefaults__ = actual.__kwdefaults__;function.__annotations__ = actual.__annotations__
    function.__qualname__ = actual.__qualname__;function.__module__ = actual.__module__
    function._v10_original_function = actual;function._v10_source_receipt = receipt
    return function,receipt


def install_private_method(owner,method_name,entry,callbacks):
    original = getattr(owner,method_name)
    function,receipt = make_private_method(original,entry,callbacks)
    setattr(owner,method_name,types.MethodType(function,owner))
    return {'owner':owner,'method_name':method_name,'original':original,'source':receipt,
            'scope':'this retained loaded object in this private audit process only'}


def restore_private_methods(installed):
    for item in reversed(installed):setattr(item['owner'],item['method_name'],item['original'])
