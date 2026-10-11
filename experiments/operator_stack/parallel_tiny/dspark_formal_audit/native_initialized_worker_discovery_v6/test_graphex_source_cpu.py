"""Actual SDK lazy class AST/getter contracts, no torch/NPU or real factory import."""
import ast
import inspect
from importlib.machinery import ModuleSpec
import os
from pathlib import Path
import sys
import time
from types import ModuleType
import unittest
from unittest.mock import patch

from contracts import digest
from graphex_source_identity import (LAZY_REGISTRAR,LAZY_CLASS,BACKING_MODULE,COMPILER_MODULE,
                                     lazy_proxy_source,guard_graphex_imports)


class GraphExSourceCPU(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent/'evidence/sdk_source_reference'
        self.paths={'npugraph_ex_lazy_proxy':root/'torch_npu_dynamo_init.py.txt',
                    'npugraph_ex':root/'npugraph_ex_init.py.txt',
                    'npu_fx_compiler':root/'npu_fx_compiler.py.txt'}
        self.sources={name:digest(path) for name,path in self.paths.items()}
        self.assertEqual(self.sources['npugraph_ex_lazy_proxy']['sha256'],
                         'd5e39c5c07ae0e94a4ec0f1c7d4a86c008f15fb635458fd996edc692b046a0ce')
        self.assertEqual(self.sources['npugraph_ex']['sha256'],
                         '6ce899dddebf09d3be797865bd93cc4ee2ba39c9b33f8d08bb7887c2d5e85c92')
        self.assertEqual(self.sources['npu_fx_compiler']['sha256'],
                         '926818c66e7fae1cc5191ff994935b746c46fcd9e6fba23e1938a0f84fe1382c')
        def module(name,path):
            value=ModuleType(name);value.__file__=str(path.resolve())
            value.__spec__=ModuleSpec(name,loader=None,origin=value.__file__)
            return value
        self.registrar=module(LAZY_REGISTRAR,self.paths['npugraph_ex_lazy_proxy'])
        self.registrar.__dict__.update(os=os,time=time)
        tree=ast.parse(self.paths['npugraph_ex_lazy_proxy'].read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.ClassDef)
               and node.name in ('_LazyBackend',LAZY_CLASS)]
        self.assertEqual([node.name for node in nodes],['_LazyBackend',LAZY_CLASS])
        # Execute the collected original class bodies, not a lookalike proxy.
        exec(compile(ast.Module(body=nodes,type_ignores=[]),self.registrar.__file__,'exec'),self.registrar.__dict__)
        self.facade=vars(self.registrar)[LAZY_CLASS]('npugraph_ex')
        self.backing=module(BACKING_MODULE,self.paths['npugraph_ex'])
        self.compiler=module(COMPILER_MODULE,self.paths['npu_fx_compiler'])
        self.calls=[];self.sentinel=object()
        def factory(*args,**kwargs):self.calls.append((args,kwargs));return self.sentinel
        self.factory=factory
        self.backing.get_npu_backend=factory;self.compiler.get_npu_backend=factory
        self.backing.CompilerConfig=type('CPUOnlyConfig',(),{})
        self.facade._npugraph_ex=self.backing
        self.modules=patch.dict(sys.modules,{LAZY_REGISTRAR:self.registrar,BACKING_MODULE:self.backing,
                                             COMPILER_MODULE:self.compiler,'npugraph_ex':self.facade})
        self.modules.start();self.addCleanup(self.modules.stop)

    def check(self):return guard_graphex_imports(self.facade,self.compiler,self.sources)

    def test_old_getfile_instance_failure_reproduced_with_original_sdk_lazy_class(self):
        self.assertEqual(type(self.facade).__name__,'_LazyNpuGraphEx')
        with self.assertRaisesRegex(TypeError,'got _LazyNpuGraphEx'):inspect.getfile(self.facade)
        self.assertEqual(lazy_proxy_source(self.facade),self.sources['npugraph_ex_lazy_proxy'])

    def test_class_backing_and_compiler_sources_are_separate_without_factory_call(self):
        receipt=self.check()
        self.assertEqual(receipt['sources'],self.sources)
        self.assertEqual(receipt['backing_module'],BACKING_MODULE)
        self.assertFalse(receipt['facade_is_module']);self.assertEqual(self.calls,[])
        self.assertFalse(receipt['device_audit_passed']);self.assertFalse(receipt['safe_formal_receipt'])
        self.assertFalse(receipt['factory_called_by_source_guard']);self.assertFalse(receipt['kwargs_filtered'])

    def test_getfile_receives_class_only_and_never_lazy_instance(self):
        original=inspect.getfile;seen=[]
        def getter(value):
            self.assertIsNot(value,self.facade);self.assertIs(value,type(self.facade))
            seen.append(value);return original(value)
        with patch('graphex_source_identity.inspect.getfile',side_effect=getter):self.check()
        self.assertEqual(seen,[type(self.facade)])

    def test_each_wrong_lazy_backing_or_compiler_sha_rejected(self):
        for name in self.sources:
            old=self.sources[name];self.sources[name]={**old,'sha256':'0'*64}
            with self.assertRaisesRegex(ValueError,name):self.check()
            self.sources[name]=old
        self.assertEqual(self.calls,[])

    def test_module_file_and_spec_origin_mismatch_rejected(self):
        self.compiler.__spec__.origin=str(self.paths['npugraph_ex_lazy_proxy'])
        with self.assertRaisesRegex(ValueError,'file/spec source'):self.check()
        self.assertEqual(self.calls,[])

    def test_missing_nonmodule_or_other_imported_backing_rejected(self):
        for backing in (None,object(),ModuleType(BACKING_MODULE)):
            self.facade._npugraph_ex=backing
            with self.assertRaisesRegex(ValueError,'actual imported SDK backing module'):self.check()
        self.assertEqual(self.calls,[])

    def test_unregistered_lazy_class_rejected_even_with_matching_name(self):
        vars(self.registrar)[LAZY_CLASS]=type(LAZY_CLASS,(),{'__module__':LAZY_REGISTRAR})
        with self.assertRaisesRegex(ValueError,'registrar class'):self.check()
        self.assertEqual(self.calls,[])

    def test_changed_factory_binding_rejected_even_when_files_match(self):
        self.compiler.get_npu_backend=lambda **kwargs:None
        with self.assertRaisesRegex(ValueError,'factory/config binding'):self.check()
        self.assertEqual(self.calls,[])

    def test_original_lazy_factory_keeps_all_arguments_and_objects_after_guard(self):
        original_backing=self.facade._npugraph_ex;original_factory=self.factory
        self.check();config=object();option=object()
        result=self.facade.get_npu_backend('CPU-only-argument',compiler_config=config,unfiltered_option=option)
        self.assertIs(result,self.sentinel);self.assertIs(self.facade._npugraph_ex,original_backing)
        self.assertIs(self.facade.get_npu_backend,original_factory)
        self.assertEqual(self.calls,[(('CPU-only-argument',),{'compiler_config':config,'unfiltered_option':option})])
        gate=ast.parse((Path(__file__).parent/'initialized_worker_gate.py').read_text())
        raw=next(node for node in ast.walk(gate) if isinstance(node,ast.Assign)
                 and any(isinstance(target,ast.Name) and target.id=='raw' for target in node.targets))
        self.assertIsInstance(raw.value,ast.Call)
        self.assertEqual(ast.unparse(raw.value.func),'nge.get_npu_backend')
        self.assertEqual([(keyword.arg,ast.unparse(keyword.value)) for keyword in raw.value.keywords],
                         [('compiler_config','config')])


if __name__=='__main__':unittest.main()
