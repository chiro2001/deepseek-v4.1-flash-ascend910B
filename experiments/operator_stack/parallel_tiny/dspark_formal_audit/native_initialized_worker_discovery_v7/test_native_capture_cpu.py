"""Original native factory/SDK StreamContext AST lifecycle, no torch/NPU import."""
import ast
from contextlib import contextmanager,nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
from typing import Any,Optional
import unittest
from unittest.mock import patch

from native_capture_scope import NativeCaptureScope,native_factory_source,validate_native_entry,RUNNER_MODULE


class NativeCaptureCPU(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent/'evidence/sdk_source_reference'
        self.events=[];self.state=SimpleNamespace(current=None,context=100,device=0,rank=0,restore=True)
        class Stream:
            def __init__(stream,device=None):
                stream.device=device;stream.npu_stream=200+len(self.events)
                self.events.append(('native_stream_create',device))
            def wait_stream(stream,previous):self.events.append(('wait_original',previous.npu_stream))
        self.original=SimpleNamespace(device=0,npu_stream=10);self.state.current=self.original
        def set_stream(stream):
            self.events.append(('native_set_stream',stream.npu_stream))
            if self.state.restore or stream is not self.original:self.state.current=stream
        self.fake_npu=SimpleNamespace(Stream=Stream,current_stream=lambda:self.state.current,
            default_stream=lambda device=None:self.original,set_stream=set_stream)
        self.torch=SimpleNamespace(npu=self.fake_npu,device=object,jit=SimpleNamespace(is_scripting=lambda:False))
        utility=ModuleType('CPU_only_npu_utils');utility.__dict__.update(torch=self.torch,
            Optional=Optional,Any=Any,_get_device_index=lambda device,optional:0,device=lambda device:nullcontext())
        source=(root/'npu_utils.py.txt').read_text();tree=ast.parse(source)
        nodes=[node for node in tree.body if (isinstance(node,ast.ClassDef) and node.name=='StreamContext')
               or (isinstance(node,ast.FunctionDef) and node.name=='stream')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/'npu_utils.py.txt'),'exec'),utility.__dict__)
        self.fake_npu.stream=utility.stream
        self.module=ModuleType(RUNNER_MODULE);self.module.__file__=str((root/'npu_model_runner.py.txt').resolve())
        self.module.__dict__.update(torch=self.torch,dataclass=dataclass,contextmanager=contextmanager,nullcontext=nullcontext)
        self.module.NPUModelRunner=type('NPUModelRunner',(),{})
        source=(root/'npu_model_runner.py.txt').read_text();tree=ast.parse(source)
        nodes=[node for node in tree.body if (isinstance(node,ast.ClassDef) and node.name=='GraphCaptureContext')
               or (isinstance(node,ast.FunctionDef) and node.name=='graph_capture')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),self.module.__file__,'exec'),self.module.__dict__)
        self.modules=patch.dict(sys.modules,{RUNNER_MODULE:self.module});self.modules.start();self.addCleanup(self.modules.stop)
        self.worker=SimpleNamespace(device=0,model_runner=self.module.NPUModelRunner())
        self.snapshots=0
        def snapshot(*args):
            self.snapshots+=1
            return {'current_initialized':True,'context_ptr':self.state.context,'runtime_logical_device':self.state.device,
                    'torch_current_device':self.state.device,'worker_device_index':0,'physical_device':8,
                    'tp_rank':self.state.rank,'tp_world_size':8,'worker_rank':0,'global_rank':0,
                    'stream_ptr':self.state.current.npu_stream,'capturing':False,'runner_idle':True},self.state.current
        self.runtime=SimpleNamespace(snapshot=snapshot)
        self.baseline=snapshot()[0];self.source=native_factory_source(self.worker)[0]
        self.receipt={};self.resources={}

    def scope(self):return NativeCaptureScope(self.worker,self.runtime,self.torch,None,None,
        self.baseline,self.source,self.receipt,self.resources)

    def test_actual_native_ast_waits_selects_nondefault_and_restores_original(self):
        with self.scope() as scope:
            self.assertNotEqual(scope.stream.npu_stream,self.original.npu_stream)
            self.assertIs(scope.context.stream,self.state.current)
            self.assertEqual(self.events[1],('wait_original',10))
        self.assertIs(self.state.current,self.original)
        self.assertTrue(self.receipt['native_context_creates_stream'])
        self.assertFalse(self.receipt['init_return_stream_borrowed_unchanged'])
        self.assertTrue(self.receipt['normal_exit_restoration_verified'])
        self.assertIs(self.resources['native_capture_context'].stream,scope.stream)

    def test_bad_factory_sha_rejected_before_native_stream_creation(self):
        self.source={**self.source,'function_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'frozen source'):self.scope()
        self.assertEqual(self.events,[])

    def test_wrong_runner_owner_rejected(self):
        self.worker.model_runner=object()
        with self.assertRaisesRegex(ValueError,'normal native'):self.scope()
        self.assertEqual(self.events,[])

    def test_default_or_unselected_stream_is_rejected(self):
        actual={**self.baseline,'stream_ptr':10}
        with self.assertRaisesRegex(ValueError,'non-default'):validate_native_entry(self.baseline,actual,10,10)
        actual['stream_ptr']=30
        with self.assertRaisesRegex(ValueError,'non-default'):validate_native_entry(self.baseline,actual,20,10)

    def test_device_context_and_rank_changes_are_rejected(self):
        for name in ('context_ptr','runtime_logical_device','physical_device','tp_rank'):
            actual={**self.baseline,'stream_ptr':20,name:self.baseline[name]+1}
            with self.assertRaisesRegex(ValueError,'device/context/rank'):validate_native_entry(self.baseline,actual,20,10)

    def test_wrong_returned_context_type_uses_native_exit(self):
        original=self.module.graph_capture
        @contextmanager
        def wrong(device):
            with original(device) as ctx:yield SimpleNamespace(stream=ctx.stream)
        with patch('native_capture_scope.native_factory_source',return_value=(self.source,wrong,self.module.GraphCaptureContext)):
            with self.assertRaisesRegex(ValueError,'owner type'):
                with self.scope():pass
        self.assertIs(self.state.current,self.original)

    def test_normal_restore_failure_is_rejected_without_repair(self):
        with self.assertRaisesRegex(ValueError,'Borrowed worker'):
            with self.scope():self.state.restore=False
        self.assertFalse(self.receipt['normal_exit_restoration_verified'])
        self.assertEqual([event for event in self.events if event==('native_set_stream',10)],[('native_set_stream',10)])

    def test_failed_body_uses_only_native_exit_and_no_device_requery(self):
        entered_count=None
        with self.assertRaisesRegex(RuntimeError,'CPU-only body failure'):
            with self.scope():
                entered_count=self.snapshots
                raise RuntimeError('CPU-only body failure')
        self.assertEqual(self.snapshots,entered_count)
        self.assertIs(self.state.current,self.original)
        self.assertFalse(self.receipt['normal_exit_restoration_verified'])
        self.assertTrue(self.receipt['failure_exit_uses_native_lifecycle_only'])

    def test_enter_scope_change_rejects_and_unwinds_native_context(self):
        self.state.context=999
        with self.assertRaisesRegex(ValueError,'device/context/rank'):
            with self.scope():pass
        self.assertIs(self.state.current,self.original)

    def test_scope_holds_same_native_stream_through_repeated_queries(self):
        with self.scope() as scope:
            for _ in range(32):
                actual,_=self.runtime.snapshot()
                self.assertEqual(actual['stream_ptr'],scope.stream.npu_stream)
                validate_native_entry(self.baseline,actual,scope.stream.npu_stream,10)
        self.assertEqual(len([event for event in self.events if event[0]=='native_stream_create']),1)


if __name__=='__main__':unittest.main()
