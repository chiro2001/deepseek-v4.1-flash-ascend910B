"""CPU standard-init driver contracts; never create an SDK worker process."""
import ast
from contextlib import redirect_stdout
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from parallel_tiny.dspark_formal_audit.native_initialized_worker_discovery_v8.init_only_driver import (
    init_standard_wrapper,_collect,main)


class DriverCPU(unittest.TestCase):
    def test_standard_wrapper_split_init_never_calls_model_loader(self):
        calls=[]
        class Wrapper:
            def __init__(self,**kwargs):calls.append(('wrapper',kwargs))
            def init_worker(self,kwargs):calls.append(('init_worker',kwargs))
            def init_device(self):calls.append(('init_device',))
            def load_model(self):self.fail('no model load')
        config=object();lock=object()
        wrapper=init_standard_wrapper(Wrapper,config,3,'tcp://CPU-only:1',lock)
        self.assertIsInstance(wrapper,Wrapper)
        self.assertEqual([call[0] for call in calls],['wrapper','init_worker','init_device'])
        kwargs=calls[1][1]
        self.assertEqual(len(kwargs),8);self.assertIs(kwargs[3]['vllm_config'],config)
        self.assertEqual(kwargs[3]['rank'],3);self.assertFalse(kwargs[3]['is_driver_worker'])

    def test_failed_actual_rpc_message_persisted_before_throw(self):
        parent,child=multiprocessing.Pipe()
        message={'kind':'result','rank':0,'result':{'passed':False,'scope':'CPU-only-mock','epoch':0}}
        child.send(message)
        try:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'raw_results'
                with self.assertRaisesRegex(RuntimeError,'SDK gate failed'):
                    _collect([parent],[SimpleNamespace(exitcode=None)],'result',1,raw_root=root)
                self.assertEqual(json.loads((root/'rank0.json').read_text()),message)
        finally:parent.close();child.close()

    def test_direct_script_dry_run_imports_no_torch_or_sdk(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('torch_npu',sys.modules)
        model=Path(__file__).parent/'evidence/checkpoint_config'
        for job,variant in (('CPU-only-static-job','static_default'),('CPU-only-nonstatic-job','nonstatic_default')):
            output=io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(['--model',str(model),'--physical-chips','8,9,10,11,12,13,14,15',
                    '--output',f'/{job}/gate','--variant',variant,'--dry-run']),0)
            receipt=json.loads(output.getvalue())
            self.assertEqual(receipt['audit_job_id'],job)
            self.assertEqual(receipt['driver_output_root'],f'/{job}/gate')
            self.assertEqual(receipt['controller_path'],f'/{job}/gate/metadata/controller.json')
            self.assertEqual(receipt['npu_jobs'],0);self.assertFalse(Path(f'/{job}').exists())
        self.assertNotIn('torch',sys.modules);self.assertNotIn('torch_npu',sys.modules)

    def test_readonly_sdk_source_proves_wrapper_split_and_stock_proc_forces_load(self):
        root=Path(__file__).parent/'evidence/sdk_source_reference'
        wrapper=ast.parse((root/'worker_base.py.txt').read_text())
        proc=ast.parse((root/'multiproc_executor.py.txt').read_text())
        def method(tree,cls_name,fn_name):
            cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==cls_name)
            return next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==fn_name)
        def calls(fn):
            return [node.func.attr for node in ast.walk(fn) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        self.assertNotIn('load_model',calls(method(wrapper,'WorkerWrapperBase','init_worker')))
        self.assertNotIn('load_model',calls(method(wrapper,'WorkerWrapperBase','init_device')))
        self.assertIn('load_model',calls(method(proc,'WorkerProc','__init__')))


if __name__=='__main__':unittest.main()
