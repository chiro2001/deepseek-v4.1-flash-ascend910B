"""Retain stock loaded native models before the original ACLGraph wrapper."""
import ast
import copy
import hashlib
import inspect
import linecache
from pathlib import Path
import types


RUNNER_SHA = '67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e'


def runner_load_with_callbacks(runner,callbacks):
    original = runner.load_model
    function = inspect.unwrap(getattr(original,'__func__',original))
    path = Path(inspect.getfile(function));source = path.read_text()
    if function.__qualname__ != 'NPUModelRunner.load_model' or hashlib.sha256(path.read_bytes()).hexdigest() != RUNNER_SHA:
        raise ValueError('Need original normal NPUModelRunner load source')
    owner = next(node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name == 'NPUModelRunner')
    node = copy.deepcopy(next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name == 'load_model'))
    original_node = copy.deepcopy(node);insertions = {'target':0,'before_graph':0}

    class Insert(ast.NodeTransformer):
        def visit_AnnAssign(self,statement):
            statement = self.generic_visit(statement)
            if ast.unparse(statement.target) == 'self.model' and isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func) == 'get_model':
                insertions['target'] += 1
                callback = ast.Expr(ast.Call(ast.Name('_v10_retain_target',ast.Load()),
                    [ast.Name('self',ast.Load()),ast.Attribute(ast.Name('self',ast.Load()),'model',ast.Load())],[]))
                return [statement,callback]
            return statement

        def visit_Assign(self,statement):
            statement = self.generic_visit(statement)
            if any(ast.unparse(target) == 'self.model' for target in statement.targets) and isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func) == 'get_model':
                insertions['target'] += 1
                callback = ast.Expr(ast.Call(ast.Name('_v10_retain_target',ast.Load()),
                    [ast.Name('self',ast.Load()),ast.Attribute(ast.Name('self',ast.Load()),'model',ast.Load())],[]))
                return [statement,callback]
            if any(isinstance(target,ast.Name) and target.id == 'cudagraph_mode' for target in statement.targets):
                insertions['before_graph'] += 1
                callback = ast.Expr(ast.Call(ast.Name('_v10_before_graph',ast.Load()),[ast.Name('self',ast.Load())],[]))
                return [callback,statement]
            return statement

    node = Insert().visit(node)
    if insertions != {'target':1,'before_graph':1}:raise ValueError('Normal load/native-before-wrapper boundary changed')
    class Strip(ast.NodeTransformer):
        def visit_Expr(self,statement):
            if isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name) and statement.value.func.id.startswith('_v10_'):return None
            return self.generic_visit(statement)
    if ast.dump(Strip().visit(copy.deepcopy(node)),include_attributes=False) != ast.dump(original_node,include_attributes=False):
        raise ValueError('Load callbacks changed the original native load AST')
    node.decorator_list = []
    tree = ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace = dict(function.__globals__)
    namespace.update(_v10_retain_target=callbacks.retain_target,_v10_before_graph=callbacks.before_graph)
    filename = '/private-v10/NPUModelRunner.load_model.py';text = ast.unparse(tree)
    linecache.cache[filename]=(len(text),None,text.splitlines(keepends=True),filename)
    exec(compile(tree,filename,'exec'),namespace)
    replacement = namespace[node.name]
    replacement.__defaults__ = function.__defaults__;replacement.__kwdefaults__ = function.__kwdefaults__
    runner.load_model = types.MethodType(replacement,runner)
    return {'original':original,'owner':runner,'native_load_AST_preserved':True,
            'before_graph_boundary_verified':True,'source_sha256':RUNNER_SHA}


def adapter_dispatch(wrapped,session,role,args,kwargs):
        result = wrapped(*args,**kwargs)
        # Compiled/ACL-replayed callables may hand back lists where the eager
        # callable returned tuples; only the payload structure is contractual.
        # v15: when a wrapped callable was compiled/captured before the
        # observer-buffer envelope existed, the call returns the raw native
        # value. The owned observer/prepack buffers are module buffers that the
        # callbacks already filled during the call, so the adapter can hand the
        # same owned tensors to the export side while recording explicitly that
        # this envelope was synthesised (never silently, never as a graph claim).
        if isinstance(result,(tuple,list)) and len(result) == 2 and isinstance(result[1],(tuple,list)):
            expected = session.role_tensor_outputs(role)
            if len(result[1]) != len(expected):raise ValueError('Actual graph dropped observer/prepack Tensor outputs')
            if any(not session.torch.is_tensor(tensor) or tensor.shape != original.shape
                   or tensor.dtype != original.dtype or tensor.device != original.device
                   for tensor,original in zip(result[1],expected)):
                raise ValueError('Actual graph changed observer/prepack Tensor output metadata')
            session.returned_outputs[role] = tuple(result[1])
            record = getattr(session,'record_envelope',None)
            if record is not None:record(role,'wrapped_callable_envelope')
            return result[0]
        if not session.torch.is_tensor(result):
            mismatch = getattr(session,'adapter_mismatch',None)
            if mismatch is not None:mismatch(role,result)
            raise ValueError('Actual outer graph did not return explicit observer buffers')
        session.returned_outputs[role] = tuple(session.role_tensor_outputs(role))
        record = getattr(session,'record_envelope',None)
        if record is not None:record(role,'adapter_synthesis_after_raw_native_result')
        return result


class NativeResultAdapter:
    """Plain proxy used by CPU contracts; transparently return native result only."""
    _v10_adapter = True
    def __init__(self,wrapped,session,role):
        self.wrapped = wrapped;self.session = session;self.role = role

    def __getattr__(self,name):return getattr(self.wrapped,name)
    def __dir__(self):return sorted(set(dir(type(self)))|set(dir(self.wrapped)))

    def unwrap_native(self):
        unwrap = getattr(self.wrapped,'unwrap',None)
        return unwrap() if unwrap is not None else self.wrapped

    def __call__(self,*args,**kwargs):
        return adapter_dispatch(self.wrapped,self.session,self.role,args,kwargs)


_ADAPTER_SUBCLASSES = {}


def force_draft_eager(runner,session):
    """Keep the K5 draft on the native eager path for the audit window.

    The draft ACL graph replays recorded ops, so Python-level consumer taps
    inside the draft decoder/attention never run per round (v9 evidence:
    draft decoder.output / shared / cache.draft.read stayed at epoch 0 while
    graph-external taps filled). The v10 design and the tiny v7 baseline both
    keep the draft eager while the target keeps its FULL graph.
    """
    proposer = getattr(runner,'drafter',None)
    receipt = {'scope':'draft forced to the native eager path for the audit window',
               'proposer_present':proposer is not None,'device_audit_passed':False}
    if proposer is None:return receipt
    receipt['use_cuda_graph_before'] = bool(getattr(proposer,'use_cuda_graph',False))
    proposer.use_cuda_graph = False
    wrapper = getattr(proposer,'_runnable',None)
    receipt['runnable_type_before'] = type(wrapper).__name__ if wrapper is not None else None
    unwrap = getattr(wrapper,'unwrap',None)
    if unwrap is not None:
        inner = unwrap();proposer._runnable = inner
        receipt['runnable_unwrapped'] = True
        receipt['runnable_type_after'] = type(inner).__name__
    else:
        receipt['runnable_unwrapped'] = False
        receipt['runnable_type_after'] = receipt['runnable_type_before']
    receipt['use_cuda_graph_after'] = bool(getattr(proposer,'use_cuda_graph',False))
    return receipt


def make_result_adapter(wrapped,session,role):
    """Adapter whose type is a subclass of the wrapped object's own class.

    Native consumers assert ``isinstance(model, _HIDDEN_STATE_DRAFTER_TYPES)``
    on ``proposer.model`` (llm_base_proposer._propose), so a plain proxy is
    rejected by their type guard. The subclass keeps the same ``__dict__``
    (shared submodules/parameters/compile state) and overrides only ``__call__``.
    """
    if getattr(wrapped,'_v10_adapter',False):return wrapped
    base = type(wrapped)
    cls = _ADAPTER_SUBCLASSES.get(base)
    if cls is None:
        def subclass_call(self,*args,**kwargs):
            return adapter_dispatch(self._v10_wrapped,self._v10_session,self._v10_role,args,kwargs)
        def subclass_init(self,wrap,sess,rol):
            self._v10_wrapped=wrap;self._v10_session=sess;self._v10_role=rol
        cls = type('NativeResultAdapter_'+base.__name__,(base,),
                   {'__call__':subclass_call,'__init__':subclass_init,'_v10_adapter':True,
                    '__doc__':'Adapter subclass of '+base.__module__+'.'+base.__qualname__})
        _ADAPTER_SUBCLASSES[base]=cls
    instance = object.__new__(cls)
    instance.__dict__.update(vars(wrapped))
    instance._v10_wrapped=wrapped;instance._v10_session=session;instance._v10_role=role
    return instance


def extend_native_root_forward(root,session,role):
    # The loaded native object's class/interface and all other methods remain
    # the original. Only this private forward's outer return gains the buffers.
    original = root.forward
    tag = 'v10-envelope:'+role
    existing = vars(root).get('forward')
    if getattr(existing,'_v10_tag',None) == tag:
        return {'owner':root,'original':existing,'role':role,'reapplied':False,
                'extension_tag':tag,'explicit_tensor_output_envelope':True}
    def forward(owner,*args,**kwargs):
        positions = kwargs.get('positions',args[1] if len(args) > 1 else None)
        if positions is None:raise ValueError('Actual native root did not supply positions')
        session.callbacks.enter_root(role,positions)
        value = original(*args,**kwargs)
        return value,session.role_tensor_outputs(role)
    forward._v10_tag = tag
    forward._v10_root_id = id(root)
    root.forward = types.MethodType(forward,root)
    return {'owner':root,'original':original,'role':role,'reapplied':True,
            'extension_tag':tag,'explicit_tensor_output_envelope':True}


def extend_compiled_dispatch(root,session,role):
    """Attach the observer envelope to the compiled dispatch points.

    ``@support_torch_compile`` captures ``self.forward`` inside
    ``TorchCompileWithNoGuardsWrapper.__init__`` (model construction time) and
    the class ``__call__`` then dispatches straight to ``_compiled_callable``
    (or ``aot_compiled_fn``), never touching an instance-level ``forward``.
    The envelope therefore has to wrap the compiled callable, whose graph is
    the original forward (all in-graph observer writes still replay), while the
    Python-level return gains the owned observer/prepack buffers.
    """
    events = {}
    def wrap_compiled(holder,name,pass_self):
        current = vars(holder).get(name)
        if current is None:return None
        marker = '_v10_envelope_tag'
        if getattr(current,marker,None) == 'v10-envelope:'+role:
            return 'already'
        inner = current
        def shim(*args,**kwargs):
            if pass_self:model,args = args[0],args[1:]
            else:model = holder
            try:session.callbacks.enter_root(role,kwargs.get('positions',args[1] if len(args)>1 else None))
            except RuntimeError:pass
            value = inner(model,*args,**kwargs) if pass_self else inner(*args,**kwargs)
            events[name] = events.get(name,0)+1
            note = getattr(session,'note_dispatch',None)
            if note is not None:
                note(role,'compiled_callable' if name == '_compiled_callable' else 'compiled_aot')
            return value,session.role_tensor_outputs(role)
        setattr(shim,marker,'v10-envelope:'+role)
        setattr(shim,'_v10_inner',inner)
        vars(holder)[name] = shim
        return 'wrapped'
    receipt = {'role':role}
    receipt['compiled_callable'] = wrap_compiled(root,'_compiled_callable',False)
    receipt['aot_compiled_fn'] = wrap_compiled(root,'aot_compiled_fn',True)
    receipt['device_audit_passed'] = False
    return receipt
