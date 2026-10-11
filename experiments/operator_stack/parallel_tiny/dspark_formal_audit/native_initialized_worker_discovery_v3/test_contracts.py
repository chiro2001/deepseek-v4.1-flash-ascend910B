"""CPU false-positive contracts only; metadata never grants device evidence."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from contracts import (WorkerGateLifecycle,read_frozen_json,validate_controller,
                       validate_initialized_facts,assert_borrowed_scope_stable,
                       validate_export,validate_copy_aliases)
from initialized_worker_gate import run_in_initialized_worker
from runtime_scope import already_imported_worker_modules,AlreadyLoadedRuntime,guard_worker_sources,CANDIDATE_FILES
from worker_hook import make_worker_class
from controller_cpu import write_controller_bundle


class CPUContracts(unittest.TestCase):
    def setUp(self):
        self.identity={'pid':os.getpid(),'boot_id':'CPU-mock-boot','normal_init_return_ns':1}
        self.controller={'schema':'native_initialized_worker_discovery_v3',
            'operation':'run_one_variant_in_new_formal_audit_job','variant':'static_default',
            'generations':32,'tp_world_size':8,'physical_chips':list(range(8,16)),
            'checkpoint_config_sha256':'40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424',
            'checkpoint_config_path':'CPU-mock/config.json',
            'source_manifest_path':'/CPU-only-new-job/gate/metadata/source_manifest.json',
            'source_manifest_sha256':'a'*64,'output_root':'/CPU-only-new-job/gate/sdk_results',
            'audit_job_id':'CPU-only-new-job','audit_job_root':'/CPU-only-new-job',
            'driver_output_root':'/CPU-only-new-job/gate',
            'controller_path':'/CPU-only-new-job/gate/metadata/controller.json',
            'nonce':'CPU-only-nonce','worker_processes':[
                {'tp_rank':rank,'pid':os.getpid()+rank,'boot_id':'CPU-mock-boot','normal_init_return_ns':1,
                 'loaded_abi_proofs_path':f'/CPU-only-new-job/gate/metadata/loaded_abi_proofs/rank{rank}.json',
                 'loaded_abi_proofs_sha256':'b'*64}
                for rank in range(8)]}
        self.facts={'tp_rank':0,'tp_world_size':8,'worker_rank':0,'global_rank':0,
                    'current_initialized':True,'context_ptr':4096,'stream_ptr':8192,
                    'runtime_logical_device':0,'torch_current_device':0,'worker_device_index':0,
                    'physical_device':8,'runner_idle':True,'capturing':False}
        self.payload={'values':[[1,2]],'positions':[[8]],'valid':[True],
                      'epoch':2,'copy_count':1,'rows':1}
        self.buffers={};self.calls=[];self.marker_sources={}
        for owner in ('producer','reader'):
            for field,shape,stride,dtype in (
                ('values',[6,64],[64,1],'torch.bfloat16'),('positions',[6,1],[1,1],'torch.int64'),
                ('valid',[6],[1],'torch.bool'),('epoch',[1],[1],'torch.int64'),
                ('copy_count',[1],[1],'torch.int64'),('row_count',[1],[1],'torch.int64')):
                pointer=(len(self.buffers)+1)*4096
                stamp={'ptr':pointer,'shape':shape,'stride':stride,'dtype':dtype,'device':'npu:0','format':2}
                self.buffers[owner+'.'+field]=stamp
                if field in ('copy_count','row_count'):continue
                source={**stamp,'ptr':pointer+1048576}
                self.calls.append({'phase':'cold-capture','op':'aten.copy_.default','destination':dict(stamp),
                    'logical_destination':dict(stamp),'source':source,'stream':8192})
                if field!='values':self.marker_sources[pointer]=source

    def facts_check(self):
        return validate_initialized_facts(self.controller,self.facts,self.identity)

    def alias_check(self,returned=None):
        return validate_copy_aliases(self.buffers,self.calls,self.buffers if returned is None else returned,
                                     8192,marker_sources=self.marker_sources)

    def test_matching_cpu_metadata_still_has_false_device_receipt(self):
        for result in (validate_controller(self.controller),self.facts_check(),self.alias_check()):
            self.assertFalse(result['device_audit_passed'])
            self.assertFalse(result['safe_formal_receipt'])

    def test_boolean_override_does_not_authorize_controller(self):
        with self.assertRaisesRegex(ValueError,'controller schema'):
            validate_controller({'allow':True,'current_initialized':True,'device_audit_passed':True})

    def test_p7_or_changed_formal_scope_rejected(self):
        self.controller['physical_chips'][0]=7
        with self.assertRaisesRegex(ValueError,'formal TP8'):
            self.facts_check()

    def test_reused_controller_process_for_two_ranks_rejected(self):
        self.controller['worker_processes'][1]['pid']=self.controller['worker_processes'][0]['pid']
        with self.assertRaisesRegex(ValueError,'same process'):
            self.facts_check()

    def test_initialized_false_or_null_context_cannot_be_repaired(self):
        self.facts['current_initialized']=False
        with self.assertRaisesRegex(ValueError,'already be initialized'):
            self.facts_check()
        self.facts['current_initialized']=True;self.facts['context_ptr']=0
        with self.assertRaisesRegex(ValueError,'context/stream'):
            self.facts_check()

    def test_actual_mapping_wins_over_metadata_boolean(self):
        self.facts['physical_device']=7;self.facts['mapping_passed']=True
        with self.assertRaisesRegex(ValueError,'physical mapping'):
            self.facts_check()

    def test_metadata_boolean_cannot_stand_in_for_context_pointer(self):
        self.facts['context_ptr']=True
        with self.assertRaisesRegex(ValueError,'context/stream'):
            self.facts_check()

    def test_old_worker_pid_and_wrong_logical_device_rejected(self):
        self.identity['pid']+=1000
        with self.assertRaisesRegex(ValueError,'new initialized process'):
            self.facts_check()
        self.identity['pid']-=1000;self.facts['worker_device_index']=1
        with self.assertRaisesRegex(ValueError,'logical device'):
            self.facts_check()

    def test_overlap_with_capture_or_request_rejected(self):
        for key,value in (('capturing',True),('runner_idle',False)):
            facts={**self.facts,key:value}
            with self.assertRaisesRegex(ValueError,'idle worker'):
                validate_initialized_facts(self.controller,facts,self.identity)

    def test_scope_change_in_context_stream_or_rank_rejected(self):
        for key in ('context_ptr','stream_ptr','physical_device','tp_rank'):
            after={**self.facts,key:self.facts[key]+1}
            with self.assertRaisesRegex(ValueError,'changed'):
                assert_borrowed_scope_stable(self.facts,after)

    def test_controller_changed_after_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'controller.json';path.write_text(json.dumps(self.controller))
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            read_frozen_json(path,sha)
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'frozen SHA'):
                read_frozen_json(path,sha)

    def test_failed_worker_lifecycle_is_not_retryable(self):
        worker=object();life=WorkerGateLifecycle(id(worker))
        life.begin(worker,'first');life.finish(False,{'CPU-only-mock':True})
        with self.assertRaisesRegex(ValueError,'no retry'):
            life.begin(worker,'second')
        with self.assertRaisesRegex(ValueError,'different actual worker'):
            WorkerGateLifecycle(id(worker)).begin(object(),'wrong')

    def test_stale_epoch_duplicate_write_and_wrong_payload_rejected(self):
        for key,value,message in (('epoch',1,'Stale'),('copy_count',2,'duplicate'),('values',[[9,9]],'payload')):
            payload={**self.payload,key:value}
            with self.assertRaisesRegex(ValueError,message):
                validate_export(payload,generation=2,expected_values=[[1,2]],expected_positions=[[8]])

    def test_callback_only_or_missing_marker_copy_rejected(self):
        self.calls=[{'python_callback_count':64}]
        with self.assertRaisesRegex(ValueError,'captured copy'):
            self.alias_check()
        self.calls=[]
        with self.assertRaisesRegex(ValueError,'Missing actual'):
            self.alias_check()

    def test_marker_copyback_source_rejected_even_with_original_destination(self):
        for call in self.calls:
            if call['destination']['ptr'] in self.marker_sources:
                call['source']={**call['source'],'ptr':99999999};break
        with self.assertRaisesRegex(ValueError,'copyback'):
            self.alias_check()

    def test_nonowned_destination_and_returned_alias_rejected(self):
        self.calls[0]['destination']['ptr']=999999
        with self.assertRaisesRegex(ValueError,'nonowned'):
            self.alias_check()
        self.calls[0]['destination']=dict(self.buffers['producer.values'])
        returned=copy.deepcopy(self.buffers);returned['reader.values']['ptr']=999999
        with self.assertRaisesRegex(ValueError,'Returned Tensor tuple'):
            self.alias_check(returned)

    def test_counter_copyback_is_rejected(self):
        destination=self.buffers['producer.copy_count']
        self.calls.append({'phase':'cold-capture','op':'aten.copy_.default','stream':8192,
                           'destination':destination,'logical_destination':destination,
                           'source':{**destination,'ptr':999999}})
        with self.assertRaisesRegex(ValueError,'nonowned'):
            self.alias_check()

    def test_missing_normal_imports_do_not_import_or_initialize_npu(self):
        fake=SimpleNamespace(npu=SimpleNamespace(is_initialized=lambda:False,init=lambda:self.fail('init forbidden')))
        with patch.dict(sys.modules,{'torch':SimpleNamespace(),'torch_npu':fake,'vllm.distributed':SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError,'not initialized'):
                already_imported_worker_modules()

    def test_wrong_library_sha_rejected_before_noload_call(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'not_a_library';path.write_text('CPU-only metadata')
            entry={'path':str(path),'sha256':'0'*64}
            with patch('runtime_scope.ctypes.CDLL') as loader:
                with self.assertRaisesRegex(ValueError,'library SHA'):
                    AlreadyLoadedRuntime({'acl':entry,'runtime':entry})
                loader.assert_not_called()

    def test_candidate_source_hash_mismatch_rejected_before_worker_or_sdk_use(self):
        root=Path(__file__).parent
        manifest={'candidate_sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                       for name in CANDIDATE_FILES}}
        manifest['candidate_sources']['tensor_tuple_transport.py']='0'*64
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source_manifest.json';path.write_text(json.dumps(manifest))
            controller={**self.controller,'source_manifest_path':str(path),
                        'source_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError,'Candidate source SHA'):
                guard_worker_sources(None,controller,None,None,None)

    def test_controller_builder_rejects_metadata_evidence_claim_or_source_mismatch(self):
        descriptions=[{**item,'source_manifest':{'CPU-only-mock':True},
                       'device_audit_passed':False,'safe_formal_receipt':False}
                      for item in self.controller['worker_processes']]
        with tempfile.TemporaryDirectory() as folder:
            kwargs={'metadata_root':Path(folder)/'new','output_root':Path(folder)/'results',
                    'checkpoint_config_path':Path(folder)/'unused.json','audit_job_id':'CPU-only-job',
                    'nonce':'CPU-only-nonce','variant':'static_default','physical_chips':list(range(8,16))}
            descriptions[0]['device_audit_passed']=True
            with self.assertRaisesRegex(ValueError,'cannot claim'):
                write_controller_bundle(descriptions,**kwargs)
            descriptions[0]['device_audit_passed']=False
            descriptions[1]['source_manifest']={'different':True}
            with self.assertRaisesRegex(ValueError,'SHA differ'):
                write_controller_bundle(descriptions,**kwargs)
            self.assertFalse(kwargs['metadata_root'].exists())

    def test_minimal_hook_runs_normal_parent_init_and_adds_only_lifecycle(self):
        class Base:
            def init_device(self):self.parent_calls=getattr(self,'parent_calls',0)+1;return 'parent-result'
        worker=make_worker_class(Base)()
        self.assertEqual(worker.init_device(),'parent-result')
        self.assertEqual(worker.parent_calls,1)
        self.assertEqual(worker._native_initialized_gate_lifecycle.state,'prepared_after_normal_init')
        self.assertFalse(hasattr(worker,'device_audit_passed'))
        with self.assertRaisesRegex(ValueError,'already installed'):
            worker.init_device()
        self.assertEqual(worker.parent_calls,1)

    def test_full_entry_without_real_initialized_sdk_is_terminal_false(self):
        with tempfile.TemporaryDirectory() as folder:
            worker=SimpleNamespace()
            worker._native_initialized_gate_lifecycle=WorkerGateLifecycle(id(worker))
            driver_root=Path(folder)/'gate';metadata_root=driver_root/'metadata';metadata_root.mkdir(parents=True)
            path=metadata_root/'controller.json'
            controller={**self.controller,'audit_job_id':Path(folder).name,'audit_job_root':folder,
                        'driver_output_root':str(driver_root),'controller_path':str(path),
                        'source_manifest_path':str(metadata_root/'source_manifest.json'),
                        'output_root':str(driver_root/'sdk_results'),
                        'worker_processes':[{**item,'loaded_abi_proofs_path':str(metadata_root/'loaded_abi_proofs'/f"rank{item['tp_rank']}.json")}
                                            for item in self.controller['worker_processes']]}
            path.write_text(json.dumps(controller))
            self.assertNotIn('torch_npu',sys.modules)
            result=run_in_initialized_worker(worker,controller_path=path,
                controller_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                output=driver_root/'sdk_results/static_default/rank0')
            self.assertFalse(result['passed']);self.assertFalse(result['probe_allocation_reached'])
            self.assertFalse(result['actual_copy_alias_gate_passed']);self.assertFalse(result['device_audit_passed'])
            self.assertEqual(result['worker_lifecycle'],'terminal_failed')
            self.assertFalse((driver_root/'sdk_results').exists())

    def test_source_has_no_self_init_device_selection_or_stream_creation(self):
        forbidden={'set_device','_npu_setDevice','init','empty_cache','reset_device','Stream',
                   'aclInit','aclrtSetDevice','aclrtCreateContext','aclrtCreateStream','aclrtResetDevice','rtSetDevice'}
        root=Path(__file__).parent
        for name in ('initialized_worker_gate.py','runtime_scope.py','mapped_object_proof.py','tensor_tuple_transport.py'):
            for node in ast.walk(ast.parse((root/name).read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,forbidden,(name,node.lineno,node.func.attr))


if __name__=='__main__':unittest.main()
