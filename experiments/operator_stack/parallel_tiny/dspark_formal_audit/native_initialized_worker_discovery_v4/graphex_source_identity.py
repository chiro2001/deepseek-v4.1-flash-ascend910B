"""Read the real SDK lazy facade/class/backing-module sources, never call a factory."""
import inspect
from pathlib import Path
import sys
from types import ModuleType

if __package__:
    from .contracts import digest
else:
    from contracts import digest


LAZY_REGISTRAR = 'torch_npu.dynamo'
LAZY_CLASS = '_LazyNpuGraphEx'
BACKING_MODULE = 'torch_npu.dynamo.npugraph_ex'
COMPILER_MODULE = 'npugraph_ex.npu_fx_compiler'


def _module_source(module, name):
    if not isinstance(module,ModuleType) or module.__name__!=name or sys.modules.get(name) is not module:
        raise ValueError('Need the actual imported SDK backing module: '+name)
    path=vars(module).get('__file__')
    spec=vars(module).get('__spec__')
    if (not isinstance(path,str) or not path or spec is None or spec.name!=name
            or not isinstance(spec.origin,str) or Path(spec.origin).resolve()!=Path(path).resolve()):
        raise ValueError('Actual SDK module file/spec source is missing or mismatched: '+name)
    return digest(path)


def lazy_proxy_source(facade):
    """Inspect the exact registered SDK class, never getfile(the lazy instance)."""
    cls=type(facade)
    if (isinstance(facade,ModuleType) or cls.__module__!=LAZY_REGISTRAR or cls.__qualname__!=LAZY_CLASS
            or sys.modules.get('npugraph_ex') is not facade):
        raise ValueError('Need the unchanged actual SDK _LazyNpuGraphEx facade')
    registrar=sys.modules.get(LAZY_REGISTRAR)
    if not isinstance(registrar,ModuleType) or vars(registrar).get(LAZY_CLASS) is not cls:
        raise ValueError('Lazy facade class differs from the actual SDK registrar class')
    source=_module_source(registrar,LAZY_REGISTRAR)
    if str(Path(inspect.getfile(cls)).resolve())!=source['path']:
        raise ValueError('Lazy facade class source differs from its actual SDK registrar')
    return source


def guard_graphex_imports(facade, compiler_module, frozen_sources):
    """Run after the original import_module(facade), import_module(compiler).

    The normal SDK child import loads the lazy object's backing module. Read
    that existing object and its sys.modules identity; do not force a different
    import route, replace the proxy, wrap a factory, or normalize its kwargs.
    """
    lazy_source=lazy_proxy_source(facade)
    try:backing=object.__getattribute__(facade,'_npugraph_ex')
    except AttributeError as exc:raise ValueError('Actual SDK lazy facade has no backing-module field') from exc
    sources={'npugraph_ex_lazy_proxy':lazy_source,
             'npugraph_ex':_module_source(backing,BACKING_MODULE),
             'npu_fx_compiler':_module_source(compiler_module,COMPILER_MODULE)}
    for name,source in sources.items():
        if source!=frozen_sources.get(name):
            raise ValueError('Actual SDK lazy/backing source differs from the frozen manifest: '+name)
    # Attribute reads use the original, now-loaded SDK lazy getter. No call.
    factory=facade.get_npu_backend
    if (not callable(factory) or factory is not vars(backing).get('get_npu_backend')
            or factory is not vars(compiler_module).get('get_npu_backend')
            or facade.CompilerConfig is not vars(backing).get('CompilerConfig')):
        raise ValueError('Actual SDK lazy factory/config binding differs from the imported backing module')
    return {'scope':'SDK source/getter binding only; no compiler factory or device call',
            'facade_type':LAZY_REGISTRAR+'.'+LAZY_CLASS,'facade_is_module':False,
            'backing_module':BACKING_MODULE,'compiler_module':COMPILER_MODULE,
            'sources':sources,'factory_binding_verified':True,'factory_called_by_source_guard':False,
            'sdk_factory_replaced':False,'kwargs_filtered':False,'device_api_calls':0,
            'device_audit_passed':False,'safe_formal_receipt':False,'formal_model_validated':False}
