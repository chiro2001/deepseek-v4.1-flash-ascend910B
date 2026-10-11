"""CPU-only negative contracts for native object/source bindings, never hardware."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

from consumer_bindings import (bind_native_decoders, bind_called_dispatch, bind_cache_tensor,
                               bind_worker_consumers, bind_scheduler_consumer,
                               seal_native_parameters, assert_native_parameter_seals)
from adapters import require_sdk_receipts, install_from_loaded_refs


class BindingContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'synthetic_sdk.py'
        methods = {
            'DeepseekV4DSparkModel': ('combine_hidden_states', '_project_shared_kv', '_store_standard_swa_kv'),
            'DeepseekV41DecoderLayer': ('forward',),
            'AscendFusedTopKRouter': ('_compute_routing',),
            'TokenDispatcherWithAllGather': ('token_dispatch',),
            'TokenDispatcherWithMC2': ('token_dispatch',),
            'DeepseekV41Model': ('forward',),
            'DeepseekV41DSparkModel': ('forward',),
            'DsaAttnKvPlan': ('dsa_kv_compress_scatter',),
            'DeepseekV41EagerAttentionImpl': ('_write_compressed_source', '_attention'),
            'AscendDSAImpl': ('_forward_attention',),
            'AscendDSparkProposer': ('set_inputs_first_pass',),
            'NPUModelRunner': ('_sample', '_bookkeeping_sync'),
            'Scheduler': ('update_from_output',),
        }
        source = ''
        for cls, names in methods.items():
            base = '(DeepseekV4DSparkModel)' if cls == 'DeepseekV41DSparkModel' else ''
            source += f'class {cls}{base}:\n'
            for name in names:
                source += f'    def {name}(self, *args, **kwargs):\n        return None\n'
        self.path.write_text(source)
        self.sdk = ModuleType('_native_binding_cpu_mock')
        self.sdk.__file__ = str(self.path)
        sys.modules[self.sdk.__name__] = self.sdk
        exec(compile(source, str(self.path), 'exec'), self.sdk.__dict__)
        sha = hashlib.sha256(source.encode()).hexdigest()
        hooks = []
        for cls in ast.parse(source).body:
            for function in cls.body:
                hooks.append({'qualified_function': cls.name+'.'+function.name, 'file_sha256': sha,
                              'function_sha256': hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest()})
        self.plan = {'hooks': hooks}
        self.main = self.sdk.DeepseekV41Model()
        self.main.aux_hidden_state_layers = [38, 39, 40]
        self.main.layers = [self.layer('main', index) for index in range(40)]
        self.draft = self.sdk.DeepseekV41DSparkModel()
        self.draft.layers = {str(index+40): self.layer('draft', index) for index in range(3)}
        for key, value in {'target_layer_ids': [37, 38, 39], 'mtp_start_layer_idx': 40,
                           'num_dspark_layers': 3, 'block_size': 5, 'hidden_size': 5120, 'hc_mult': 4}.items():
            setattr(self.draft, key, value)

    def tearDown(self):
        del sys.modules[self.sdk.__name__]
        self.temp.cleanup()

    def layer(self, role, local_index):
        layer = self.sdk.DeepseekV41DecoderLayer()
        layer.mlp = SimpleNamespace(layer_idx=local_index)
        layer.self_attn = SimpleNamespace(layer_idx=local_index, wkv=object(), kv_norm=object(), rotary_emb=object())
        if role == 'main':
            layer.self_attn.v41_impl = self.sdk.DeepseekV41EagerAttentionImpl()
        else:
            layer.self_attn.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=self.sdk.AscendDSAImpl()))
        return layer

    def bind(self):
        return bind_native_decoders(self.main, self.draft, self.plan)

    def test_role_global_and_local_indices_remain_distinct(self):
        result = self.bind()
        draft_rows = [row for row in result.manifest['module_bindings'] if row['role'] == 'draft']
        self.assertEqual([row['global_layer'] for row in draft_rows], [40, 41, 42])
        self.assertEqual([row['native_attention_layer_idx'] for row in draft_rows], [0, 1, 2])
        self.assertIsNot(result.modules[('main', 0)], result.modules[('draft', 40)])
        self.assertFalse(result.manifest['device_audit_passed'])
        self.assertFalse(result.manifest['registration_is_device_coverage'])

    def test_duplicate_native_object_rejected(self):
        self.draft.layers['40'] = self.main.layers[0]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.bind()

    def test_replaced_loaded_object_rejected(self):
        result = self.bind()
        self.main.layers[5] = self.layer('main', 5)
        with self.assertRaisesRegex(ValueError, 'replaced'):
            result.assert_same_loaded_objects()

    def test_shared_context_producer_rejected(self):
        self.draft.layers['41'].self_attn.wkv = self.draft.layers['40'].self_attn.wkv
        with self.assertRaisesRegex(ValueError, 'distinct objects'):
            self.bind()

    def test_wrong_source_bytes_rejected(self):
        self.path.write_text(self.path.read_text()+'\n# Changed bytes\n')
        with self.assertRaisesRegex(ValueError, 'source file'):
            self.bind()

    def test_actual_dispatch_binding_requires_retained_moe_owner(self):
        bindings = self.bind()
        router = self.sdk.AscendFusedTopKRouter()
        dispatcher = self.sdk.TokenDispatcherWithAllGather()
        kwargs = {'owner_key': ('draft', 40), 'native_moe_owner': self.draft.layers['40'].mlp,
                  'bindings': bindings, 'plan': self.plan}
        result = bind_called_dispatch(router, dispatcher, **kwargs)
        self.assertEqual(result['actual_dispatcher_object_id'], id(dispatcher))
        self.assertFalse(result['device_audit_passed'])
        kwargs['native_moe_owner'] = self.main.layers[0].mlp
        with self.assertRaisesRegex(ValueError, 'retained'):
            bind_called_dispatch(router, dispatcher, **kwargs)

    def test_shared_cache_gm_requires_logical_group_and_storage_layout(self):
        tensor = object()
        a = SimpleNamespace(prefix='main.cache', block_size=128, kv_cache=[tensor])
        b = SimpleNamespace(prefix='draft.cache', block_size=128, kv_cache=[tensor])
        metadata = {'ptr': 4096, 'storage_ptr': 4096, 'storage_offset': 0, 'shape': [8, 128, 1, 512],
                    'stride': [65536, 512, 512, 1], 'dtype': 'torch.bfloat16', 'device': 'npu:0', 'format': 2, 'version': 3}
        first = bind_cache_tensor(a, tensor, metadata, owner_key=('main', 0), logical_group=0, registered_prefix=a.prefix)
        second = bind_cache_tensor(b, tensor, metadata, owner_key=('draft', 40), logical_group=1, registered_prefix=b.prefix)
        self.assertEqual(first['tensor']['ptr'], second['tensor']['ptr'])
        self.assertNotEqual(first['logical_group'], second['logical_group'])
        self.assertFalse(first['cache_write_read_proven'])
        with self.assertRaisesRegex(ValueError, 'absent'):
            bind_cache_tensor(a, object(), metadata, owner_key=('main', 0), logical_group=0, registered_prefix=a.prefix)

    def test_worker_and_frontend_scheduler_bindings_are_separate(self):
        bindings=self.bind()
        runner=self.sdk.NPUModelRunner()
        runner._formal_dspark_native_backbone=self.main
        runner._formal_dspark_native_draft_backbone=self.draft
        runner.drafter=self.sdk.AscendDSparkProposer()
        runner.vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method='dspark',num_speculative_tokens=5))
        worker=bind_worker_consumers(runner,self.sdk.DsaAttnKvPlan(),bindings,self.plan)
        scheduler=bind_scheduler_consumer(self.sdk.Scheduler(),self.plan)
        self.assertFalse(worker['bookkeeping_is_device_coverage'])
        self.assertFalse(scheduler['scheduler_is_device_coverage'])
        runner._formal_dspark_native_draft_backbone=self.main
        with self.assertRaisesRegex(ValueError,'load-time references'):
            bind_worker_consumers(runner,self.sdk.DsaAttnKvPlan(),bindings,self.plan)

    def test_weight_version_and_storage_layout_changes_rejected(self):
        class MockParameter:
            _version=1
            shape=(4,4)
            dtype='torch.bfloat16'
            device='npu:0'  # Synthetic metadata; never allocate or grant device proof.
            def data_ptr(self):return 4096
            def untyped_storage(self):return self
            def storage_offset(self):return 0
            def stride(self):return (4,1)
        bindings=self.bind()
        for module in bindings.modules.values():
            module.weight=MockParameter()
            module.named_parameters=lambda module=module: [('weight',module.weight)]
        sealed=seal_native_parameters(bindings,lambda tensor: 2)
        self.assertFalse(assert_native_parameter_seals(bindings,sealed,lambda tensor: 2)['device_audit_passed'])
        self.main.layers[37].weight._version=2
        with self.assertRaisesRegex(ValueError,'pointer/version/storage/layout'):
            assert_native_parameter_seals(bindings,sealed,lambda tensor: 2)


class SDKReceiptContracts(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent
        self.exporter_sha=hashlib.sha256((root/'exporter.py').read_bytes()).hexdigest()
        gate_sha=hashlib.sha256((root/'gate_sdk.py').read_bytes()).hexdigest()
        fields=('payload_equal','positions_equal','device_write_count_one_each','cache_write_read_equal',
                'returned_value_equal','original_buffer_pointers_stable','full_rank_device_fence_observed')
        self.static={'stage':'graphex','physical_chip':7,'static_kernel':True,'passed':True,
                     'actual_copy_alias_gate_passed':True,'outer_keep_inference_input_mutations':False,
                     'actual_capture_copy_destinations_observed':True,'capture_all_observer_destinations_original':True,
                     'capture_nonowned_copy_destinations':[],'capture_observer_copy_calls':[],
                     'generations':[{'epoch':epoch,**{field:True for field in fields}} for epoch in range(1,33)],
                     'sdk_sources':{'npugraph_ex':{'sha256':'a'*64},'compiler':{'sha256':'b'*64}},
                     'environment':{'source_exporter':{'sha256':self.exporter_sha},'source_gate':{'sha256':gate_sha},
                                    'initialized_before':False,'device_count_before':1,
                                    'torch':'CPU-only mock','torch_npu':'CPU-only mock',
                                    'process_identity':{'container_pid':1,'host_boot_id':'CPU-only mock','gate_start_monotonic_ns':1}}}
        self.static['buffers_before']={}
        for owner in ('producer','reader'):
            for field,shape,stride,dtype in (
                ('values',[6,64],[64,1],'torch.bfloat16'),('positions',[6,1],[1,1],'torch.int64'),
                ('valid',[6],[1],'torch.bool'),('epoch',[1],[1],'torch.int64')):
                pointer=(len(self.static['buffers_before'])+1)*4096
                stamp={'ptr':pointer,'shape':shape,'stride':stride,'dtype':dtype,'device':'npu:0','format':2}
                self.static['buffers_before'][owner+'.'+field]=stamp
                self.static['capture_observer_copy_calls'].append({'phase':'cold-capture','op':'aten.copy_.default',
                    'destination':dict(stamp),'logical_destination':dict(stamp),
                    'source':{**stamp,'ptr':pointer+1048576},'stream':1})
        self.nonstatic=copy.deepcopy(self.static)
        self.nonstatic['static_kernel']=False
        self.nonstatic['environment']['process_identity']['container_pid']=2
        self.nonstatic['environment']['process_identity']['gate_start_monotonic_ns']=2

    def check(self):
        return require_sdk_receipts(self.static,self.nonstatic,self.exporter_sha)

    def test_mock_receipt_structure_never_grants_full_device_receipt(self):
        result=self.check()
        self.assertFalse(result['full43_device_audit_passed'])
        self.assertFalse(result['safe_formal_receipt'])

    def test_repeated_process_rejected(self):
        self.nonstatic['environment']['process_identity']=self.static['environment']['process_identity']
        with self.assertRaisesRegex(ValueError,'distinct fresh'):
            self.check()

    def test_duplicate_or_stale_epoch_rejected(self):
        self.static['generations'][-1]['epoch']=31
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.check()

    def test_missing_real_fence_rejected(self):
        self.static['generations'][3]['full_rank_device_fence_observed']=False
        with self.assertRaisesRegex(ValueError,'fence proof'):
            self.check()

    def test_nonowned_alias_or_changed_gate_rejected(self):
        self.static['capture_nonowned_copy_destinations']=[{'temporary_destination':1}]
        with self.assertRaisesRegex(ValueError,'alias evidence'):
            self.check()
        self.static['capture_nonowned_copy_destinations']=[]
        self.static['environment']['source_gate']['sha256']='Changed unvalidated gate'
        with self.assertRaisesRegex(ValueError,'frozen gate source'):
            self.check()

    def test_callback_only_copy_or_missing_epoch_copy_rejected(self):
        self.static['capture_observer_copy_calls']=[{'mock_python_callback':True}]
        with self.assertRaisesRegex(ValueError,'copy source/destination'):
            self.check()
        self.static['capture_observer_copy_calls']=self.nonstatic['capture_observer_copy_calls'][:-1]
        with self.assertRaisesRegex(ValueError,'epoch copy destinations'):
            self.check()

    def test_current_failure_refused_before_binding_or_allocation(self):
        # Deliberately invalid native refs/torch: refusal must precede any use.
        with self.assertRaisesRegex(ValueError,'has not passed'):
            install_from_loaded_refs(None,None,None,None,source_plan={},
                                     static_receipt={'stage':'graphex','static_kernel':True,'passed':False},
                                     nonstatic_receipt={})


if __name__ == '__main__':
    unittest.main()
