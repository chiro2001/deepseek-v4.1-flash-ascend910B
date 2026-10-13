"""CPU contracts for the first private diagnostic; no SDK or NPU imports."""
import os
os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',ASCEND_RT_VISIBLE_DEVICES='',
    ASCEND_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
import __future__
import ast
import copy
from contextlib import nullcontext
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace,MethodType,FunctionType
import unittest
from unittest.mock import patch

class BlockSDK(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname == name or fullname.startswith(name+'.') for name in
               ('vllm','vllm_ascend','torch_npu','torchair','acl')):
            self.attempts.append(fullname)
            raise ImportError('CPU diagnostic rejects SDK/NPU import: '+fullname)
GUARD = BlockSDK();sys.meta_path.insert(0,GUARD)
import torch
from .source_factory import transform_method,make_private_method,install_private_method,restore_private_methods
from .observer import PointSpec,make_observer_class,make_graph_envelope_class
from .runtime_callbacks import ConsumerCallbacks
from .lifecycle import runner_load_with_callbacks,extend_native_root_forward,NativeResultAdapter
from .native_refs import bind_native_decoders,retain_owned_backbones
from .executor_source import derive_executor_init
from .formal_audit_driver import summarize_rank_exports
from .prepack_integration import PrepackIntegration
from .session import AuditSession

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE/'consumer_plan.json').read_text())
ENTRIES = {e['qualified_function']:e for e in PLAN['hooks']}
SOURCE = HERE/'evidence/native_sources'
RECEIPTS = []
TEMP_ROOT=Path('/home/chiro/tmp/dsv41-full43-v16-20261012')
if str(HERE).startswith('/home/chiro/'):
    TEMP_ROOT.mkdir(parents=True,exist_ok=True);os.environ['TMPDIR']=str(TEMP_ROOT)


def source_class(file,cls,methods,bases=(),namespace=None):
    path = SOURCE/(file+'.txt');tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == cls)
    node = ast.ClassDef(name=cls,bases=[ast.Name(name,ast.Load()) for name in bases],keywords=[],
        body=[copy.deepcopy(n) for n in original.body if isinstance(n,ast.FunctionDef) and n.name in methods],decorator_list=[])
    scope = dict(namespace or {});scope.update(torch=torch,nn=torch.nn,__name__=__name__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec',
        flags=__future__.annotations.compiler_flag,dont_inherit=True),scope)
    return scope[cls]


def native_bindings_fixture():
    Decoder = source_class('v41_model.py','DeepseekV41DecoderLayer',['forward'],('Module',),{'Module':torch.nn.Module})
    Main = source_class('v41_model.py','DeepseekV41Model',['forward'],('Module',),{'Module':torch.nn.Module})
    V4 = source_class('v4_dspark.py','DeepseekV4DSparkModel',
        ['combine_hidden_states','_project_shared_kv','_store_standard_swa_kv'],('Module',),{'Module':torch.nn.Module})
    Draft = source_class('v41_dspark.py','DeepseekV41DSparkModel',['forward'],('V4',),{'V4':V4})
    MainReader = source_class('dsa_v41.py','DeepseekV41EagerAttentionImpl',['_attention','_write_compressed_source'])
    DraftReader = source_class('dsa_v1.py','AscendDSAImpl',['_forward_attention'])
    main=Main();draft=Draft();main.aux_hidden_state_layers={38,39,40}
    main.layers=torch.nn.ModuleList();draft.layers=torch.nn.ModuleDict()
    for role,layers in [('main',range(40)),('draft',range(40,43))]:
        for layer in layers:
            index=layer if role=='main' else layer-40;decoder=Decoder();decoder.layer_idx=index
            decoder.mlp=SimpleNamespace(layer_idx=index)
            reader=MainReader() if role=='main' else DraftReader()
            decoder.self_attn=SimpleNamespace(layer_idx=index,wkv=object(),kv_norm=object(),rotary_emb=object())
            if role=='main':decoder.self_attn.v41_impl=reader;main.layers.append(decoder)
            else:
                decoder.self_attn.dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=reader))
                draft.layers[str(layer)]=decoder
    draft.target_layer_ids=[37,38,39];draft.mtp_start_layer_idx=40;draft.num_dspark_layers=3
    draft.block_size=5;draft.hidden_size=5120;draft.hc_mult=4
    return main,draft


def actual_conditional_root_fixture(main):
    class Tower(torch.nn.Module):
        def __init__(self,config):super().__init__();self.weight=torch.nn.Parameter(torch.ones(2,2))
    class Language(torch.nn.Module):
        def __init__(self,**kwargs):super().__init__();self.model=main
        def make_empty_intermediate_tensors(self,*args):return None
        def forward(self,*args,**kwargs):return args[0]
    Conditional=source_class('deepseek_v41_vl_model.py','AscendDeepseekV41ForConditionalGeneration',
        ['__init__','forward'],('Module',),{'Module':torch.nn.Module,'DeepseekV4ViT':Tower,
            'DeepseekV4Aligner':Tower,'maybe_prefix':lambda prefix,label:prefix+'.'+label})
    Conditional.language_model_cls=Language
    Conditional.__module__='vllm_ascend.models.deepseek_v41.vl_model'
    Conditional._mark_tower_model=lambda *args:nullcontext()
    Conditional._mark_language_model=lambda *args:nullcontext()
    config=SimpleNamespace(vision_n_layers=32,hidden_size=5120)
    vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config,
        multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda name:4),dtype=torch.bfloat16))
    return Conditional(vllm_config=vllm_config,prefix='actual_CPU_source_fixture')


class Contracts(unittest.TestCase):
    def callbacks(self):
        return ConsumerCallbacks(torch,device='cpu',epoch=torch.zeros(1,dtype=torch.int64),native_bindings=None)

    def test_all17_original_AST_preserved(self):
        for entry in PLAN['hooks']:
            _,receipt=transform_method((SOURCE/(entry['file']+'.txt')).read_text(),entry)
            self.assertTrue(receipt['native_AST_preserved']);RECEIPTS.append(receipt)

    def test_changed_source_is_rejected(self):
        entry=PLAN['hooks'][0];source=(SOURCE/(entry['file']+'.txt')).read_text()
        with self.assertRaises(ValueError):transform_method(source+'\n',entry)

    def test_actual43_source_bound_modules_and_distinct_producers(self):
        main,draft=native_bindings_fixture();bound=bind_native_decoders(main,draft,PLAN)
        self.assertEqual(len(bound.modules),43);self.assertTrue(bound.assert_same_loaded_objects()['loaded_object_bindings_stable'])
        self.assertFalse(bound.manifest['device_audit_passed'])
        draft.layers['41'].self_attn.wkv=draft.layers['40'].self_attn.wkv
        with self.assertRaises(ValueError):bind_native_decoders(main,draft,PLAN)

    def test_host_resident_cpu_parameter_seals_never_query_npu_format(self):
        from .native_refs import seal_native_parameters,assert_native_parameter_seals,tensor_layout_metadata
        from .native_refs import seal_native_parameters,assert_native_parameter_seals,tensor_layout_metadata
        module=torch.nn.Module();module.weight=torch.nn.Parameter(torch.zeros(4,4))
        class Bindings:
            modules={('main',0):module}
            def assert_same_loaded_objects(self):return {'loaded_object_bindings_stable':True}
        def forbidden(tensor):raise AssertionError('NPU format query must not run for a host-resident CPU parameter')
        sealed=seal_native_parameters(Bindings(),forbidden)
        row=sealed['seals'][('main',0)]['weight']
        self.assertTrue(row['host_resident']);self.assertIsNone(row['format']);self.assertTrue(row['device'].startswith('cpu'))
        self.assertEqual(sealed['host_resident_parameter_count'],1)
        self.assertTrue(assert_native_parameter_seals(Bindings(),sealed,forbidden)['native_weight_metadata_stable'])
        # An NPU tensor still goes through the real SDK format getter exactly once.
        calls=[]
        npu=SimpleNamespace(_version=0,device='npu:3',dtype='torch.bfloat16',shape=(2,2),
            data_ptr=lambda:64,untyped_storage=lambda:SimpleNamespace(data_ptr=lambda:128),
            storage_offset=lambda:0,stride=lambda:(2,1))
        meta=tensor_layout_metadata(npu,lambda tensor:(calls.append(tensor),7)[1])
        self.assertEqual(meta['format'],7);self.assertFalse(meta['host_resident']);self.assertEqual(len(calls),1)
        # Foreign devices are rejected instead of silently sealed.
        foreign=SimpleNamespace(**{**vars(npu),'device':'xla:0'})
        with self.assertRaises(ValueError):tensor_layout_metadata(foreign,lambda tensor:7)
        # A version-less tensor still fails the version seal even on CPU.
        class NoVersion:
            device='cpu';dtype='torch.bfloat16';shape=(2,)
            @property
            def _version(self):raise RuntimeError('inference tensor has no version counter')
            def data_ptr(self):return 64
            def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:64)
            def storage_offset(self):return 0
            def stride(self):return (1,)
        class NoVersionModule:
            def named_parameters(self):return {'w':NoVersion()}
        class NoVersionBindings:
            modules={('main',0):NoVersionModule()}
            def assert_same_loaded_objects(self):return {'loaded_object_bindings_stable':True}
        with self.assertRaises(ValueError):seal_native_parameters(NoVersionBindings(),forbidden)

    def test_dynamo_module_attribute_guard_aliases_exported_for_installed_taps(self):
        import sys as _sys, types as _types
        from .source_factory import export_dynamo_guard_aliases
        callbacks=self.callbacks()
        # (a) real native fixture function whose module is importable
        entry=PLAN['hooks'][2]
        cls_name,method_name=entry['qualified_function'].split('.')
        class Fixture:pass
        module=_types.ModuleType('_v13_alias_fixture_module')
        _sys.modules[module.__name__]=module
        def actual(self,*args, **kwargs):return None
        actual.__qualname__=entry['qualified_function'];actual.__module__=module.__name__
        aliases,receipt=export_dynamo_guard_aliases(actual,callbacks)
        self.assertTrue(receipt['dynamo_aliases_exported'])
        self.assertEqual(receipt['dynamo_module'],module.__name__)
        for name,value in aliases.items():
            self.assertIs(module.__dict__[name],value)
        # Idempotent for the same callbacks; a different callbacks object is a
        # recorded rebind; a foreign attribute with the same name is rejected.
        export_dynamo_guard_aliases(actual,callbacks)
        other=self.callbacks()
        export_dynamo_guard_aliases(actual,other)
        self.assertEqual(len(other.dynamo_alias_rebinds),4)
        self.assertEqual(other.dynamo_alias_rebinds[0]['module'],module.__name__)
        module.__dict__['_v10_enter']=42
        with self.assertRaises(ValueError):export_dynamo_guard_aliases(actual,callbacks)
        # (b) the actual dynamo failure mode this export prevents
        probe_module=_types.ModuleType('_v13_dynamo_probe_module')
        namespace={'__name__':probe_module.__name__}
        exec(compile("class P:\n    def f(self,x):\n        _v10_enter(self)\n        return x+1\n",
                     '/CPU-v13/dynamo_probe.py','exec'),namespace)
        probe_module.P=namespace['P'];_sys.modules[probe_module.__name__]=probe_module
        namespace['_v10_enter']=lambda self:None
        namespace['P'].f.__module__=probe_module.__name__
        with self.assertRaises(Exception) as context:
            torch.compile(namespace['P']().f,backend='eager')(torch.ones(1))
        self.assertIn('_v10_enter',str(context.exception))
        probe_module._v10_enter=namespace['_v10_enter']
        output=torch.compile(namespace['P']().f,backend='eager')(torch.ones(1))
        self.assertEqual(output.item(),2.0)

    def test_actual_conditional_constructor_ownership_preserves_vision_and_language_objects(self):
        main,draft=native_bindings_fixture();target=actual_conditional_root_fixture(main)
        draft_root=torch.nn.Module();draft_root.add_module('model',draft)
        with self.assertRaises(AttributeError):getattr(target,'model')  # Actual v10 ownership assumption.
        vision=target.vision;aligner=target.aligner;language=target.language_model
        parameters={name:(parameter,parameter.data_ptr(),parameter._version) for name,parameter in target.named_parameters()}
        actual_main,actual_draft,receipt=retain_owned_backbones(target,draft_root,PLAN)
        self.assertIs(actual_main,main);self.assertIs(actual_draft,draft)
        self.assertIs(target.vision,vision);self.assertIs(target.aligner,aligner);self.assertIs(target.language_model,language)
        self.assertEqual(receipt['root_ownership_bindings'][0]['actual_owned_module_path'],'language_model.model')
        for name,(original,ptr,version) in parameters.items():
            actual=dict(target.named_parameters())[name]
            self.assertIs(actual,original);self.assertEqual(actual.data_ptr(),ptr);self.assertEqual(actual._version,version)
        carrier=torch.ones(6,8,dtype=torch.bfloat16)
        self.assertIs(target(carrier,torch.arange(6)),carrier)
        self.assertFalse(receipt['device_audit_passed']);self.assertEqual(len(bind_native_decoders(actual_main,actual_draft,PLAN).modules),43)

    def test_owned_backbones_absent_duplicate_and_wrong_role_rejected(self):
        main,draft=native_bindings_fixture();target=actual_conditional_root_fixture(main)
        draft_root=torch.nn.Module();draft_root.add_module('model',draft)
        with self.assertRaises(ValueError):retain_owned_backbones(torch.nn.Module(),draft_root,PLAN)
        second,_=native_bindings_fixture();target.add_module('second_native_model',second)
        with self.assertRaises(ValueError):retain_owned_backbones(target,draft_root,PLAN)
        del target.second_native_model
        with self.assertRaises(ValueError):retain_owned_backbones(target,target,PLAN)

    def test_multimodal_root_wrong_class_or_shadow_owner_rejected(self):
        main,draft=native_bindings_fixture();target=actual_conditional_root_fixture(main)
        draft_root=torch.nn.Module();draft_root.add_module('model',draft)
        original=target.language_model
        object.__setattr__(target,'language_model',torch.nn.Identity())
        with self.assertRaises(ValueError):retain_owned_backbones(target,draft_root,PLAN)
        del target.__dict__['language_model']
        self.assertIs(target.language_model,original)
        Wrong=type('WrongConditionalSubclass',(type(target),),{'__module__':type(target).__module__})
        target.__class__=Wrong
        with self.assertRaises(ValueError):retain_owned_backbones(target,draft_root,PLAN)

    def test_multimodal_root_source_byte_change_rejected(self):
        main,draft=native_bindings_fixture();target=actual_conditional_root_fixture(main)
        draft_root=torch.nn.Module();draft_root.add_module('model',draft);cls=type(target);original=cls.forward
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'changed_vl_model.py';bad.write_text((SOURCE/'deepseek_v41_vl_model.py.txt').read_text()+'\n')
            fn=FunctionType(original.__code__.replace(co_filename=str(bad)),original.__globals__,original.__name__,original.__defaults__,original.__closure__)
            fn.__qualname__=original.__qualname__;cls.forward=fn
            try:
                with self.assertRaises(ValueError):retain_owned_backbones(target,draft_root,PLAN)
            finally:cls.forward=original

    def test_missing_replaced_and_wrong_role_native_modules_rejected(self):
        main,draft=native_bindings_fixture();bound=bind_native_decoders(main,draft,PLAN)
        main.layers[1]=main.layers[0]
        with self.assertRaises(ValueError):bound.assert_same_loaded_objects()
        with self.assertRaises(ValueError):bind_native_decoders(main,draft,PLAN)
        main,draft=native_bindings_fixture();draft.layers['40'].mlp.layer_idx=40
        with self.assertRaises(ValueError):bind_native_decoders(main,draft,PLAN)

    def test_private_original_return_identity_and_shared_attn_owner(self):
        Class=source_class('v4_dspark.py','DeepseekV4DSparkModel',['combine_hidden_states'])
        root=Class();carrier=torch.ones(5,5120,dtype=torch.bfloat16)
        root.main_proj=lambda x:carrier;root.main_norm=lambda x:x
        callbacks=self.callbacks();callbacks.register_owner(root,'draft',40)
        callbacks.begin_after_prior_fence(9,actual_target_rows=5);callbacks.enter_root('main',torch.arange(5,dtype=torch.int64))
        for label in ('combine.proj','combine.norm','combine.output'):callbacks.add_point('draft',40,label,5120,torch.bfloat16)
        installed=install_private_method(root,'combine_hidden_states',ENTRIES['DeepseekV4DSparkModel.combine_hidden_states'],callbacks)
        self.assertIs(root.combine_hidden_states(torch.empty(5,15360)),carrier)
        self.assertEqual(callbacks.points['draft.40.combine.output'].copy_count.item(),1)
        self.assertTrue(torch.equal(callbacks.points['draft.40.combine.proj'].positions,torch.arange(5).reshape(-1,1)))
        other=SimpleNamespace();callbacks.register_owner(other,'draft',41)
        point=callbacks.add_point('draft',41,'shared.norm',512,torch.bfloat16)
        callbacks.current_owner=callbacks.owners[id(root)]
        value=torch.ones(5,512,dtype=torch.bfloat16)
        self.assertIs(callbacks.value(root,'shared.norm',value,{'attn':other,'positions':torch.arange(5)}),value)
        self.assertEqual(point.copy_count.item(),1)
        restore_private_methods([installed]);self.assertIs(root.combine_hidden_states.__func__,Class.combine_hidden_states)

    def test_original_zero_argument_super_closure_retained(self):
        weights=torch.ones(2,6);ids=torch.zeros(2,6,dtype=torch.int32)
        class Parent:
            def _compute_routing(self,*args,**kwargs):return weights,ids
        Router=source_class('actual_router.py','AscendFusedTopKRouter',['_compute_routing'],('Parent',),{'Parent':Parent})
        router=Router();router.custom_routing_function=lambda *args:None
        router.bias_vl=None;router.tid2eid=None
        router.is_fused_supported=lambda *args:False
        callbacks=self.callbacks();private,_=make_private_method(router._compute_routing,ENTRIES['AscendFusedTopKRouter._compute_routing'],callbacks)
        router._compute_routing=MethodType(private,router)
        output=router._compute_routing(torch.zeros(2,5120),torch.zeros(2,384),None)
        self.assertIs(output[0],weights);self.assertIs(output[1],ids)

    def test_six_owned_outputs_keep_original_carrier(self):
        callbacks=self.callbacks();obj=SimpleNamespace();callbacks.register_owner(obj,'main',0)
        point=callbacks.add_point('main',0,'decoder.output',8,torch.bfloat16)
        callbacks.begin_after_prior_fence(13,actual_target_rows=6);callbacks.enter_root('main',torch.arange(6)+77)
        carrier=torch.arange(48,dtype=torch.float32).to(torch.bfloat16).reshape(6,8)
        self.assertIs(callbacks.tap(obj,'decoder.output',carrier,carrier,{}),carrier)
        outputs=point.tensor_tuple();self.assertEqual(len(outputs),6)
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in (*outputs,carrier)}),7)
        self.assertEqual(outputs[3].item(),13);self.assertEqual(outputs[4].item(),1)
        self.assertTrue(torch.equal(outputs[0],carrier));self.assertTrue(torch.equal(outputs[1],torch.arange(6).reshape(-1,1)+77))

    def test_unsupported_shape_dtype_position_uses_original_uncovered_fallback(self):
        point=make_observer_class(torch)(PointSpec('p','main',0,6,8,torch.bfloat16),device='cpu',epoch=torch.tensor([3]))
        for payload,positions,valid in [(torch.ones(7,8,dtype=torch.bfloat16),torch.arange(7).reshape(-1,1),torch.ones(7,dtype=torch.bool)),
            (torch.ones(6,8),torch.arange(6).reshape(-1,1),torch.ones(6,dtype=torch.bool)),
            (torch.ones(6,8,dtype=torch.bfloat16),torch.arange(5).reshape(-1,1),torch.ones(5,dtype=torch.bool))]:
            self.assertIs(point(payload,payload,positions,valid),payload)
        self.assertEqual(point.copy_count.item(),0)

    def test_real_root_positions_and_outer_adapter_complete_outputs(self):
        callbacks=self.callbacks();root=torch.nn.Module();point=callbacks.add_point('main',0,'decoder.output',8,torch.bfloat16)
        callbacks.register_owner(root,'main',0);callbacks.begin_after_prior_fence(17,actual_target_rows=6)
        value=torch.ones(6,8,dtype=torch.bfloat16)
        root.forward=lambda input_ids,positions:callbacks.tap(root,'decoder.output',value,value,{'positions':positions})
        session=SimpleNamespace(torch=torch,callbacks=callbacks,returned_outputs={},
            role_tensor_outputs=lambda role:(*point.tensor_tuple(),*callbacks.role_metadata_outputs(role)))
        extend_native_root_forward(root,session,'main');adapter=NativeResultAdapter(root,session,'main')
        self.assertIs(adapter(torch.zeros(6,dtype=torch.int64),torch.arange(6)+101),value)
        self.assertEqual(len(session.returned_outputs['main']),10)
        self.assertTrue(all(actual is expected for actual,expected in zip(session.returned_outputs['main'],session.role_tensor_outputs('main'))))
        self.assertTrue(torch.equal(callbacks.target_positions,torch.arange(6).reshape(-1,1)+101))
        broken=NativeResultAdapter(lambda *args:(value,()),session,'main')
        with self.assertRaises(ValueError):broken()

    def test_native_BF16_byte_export_and_raw_retention(self):
        callbacks=self.callbacks();owner=SimpleNamespace();callbacks.register_owner(owner,'main',0)
        point=callbacks.add_point('main',0,'decoder.output',8,torch.bfloat16)
        callbacks.begin_after_prior_fence(19,actual_target_rows=6);callbacks.enter_root('main',torch.arange(6))
        values=torch.arange(48,dtype=torch.float32).to(torch.bfloat16).reshape(6,8)
        callbacks.tap(owner,'decoder.output',values,values,{})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw.pt';record=callbacks.export_after_fence(19,path)
            raw=torch.load(path,weights_only=True);self.assertTrue(torch.equal(raw['main.0.decoder.output']['values'],values))
            self.assertEqual(record['raw_device_export']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record['records']['main.0.decoder.output']['values_sha256'],hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest())
            self.assertTrue(record['complete_point_exports']);self.assertFalse(record['device_audit_passed'])

    def test_CPU_compiled_outer_keeps_all_six_original_alias_outputs(self):
        epoch=torch.zeros(1,dtype=torch.int64)
        point=make_observer_class(torch)(PointSpec('compiled','main',0,6,8,torch.bfloat16),device='cpu',epoch=epoch)
        class Native(torch.nn.Module):
            def __init__(self):super().__init__();self.point=point
            def forward(self,carrier,positions,valid):return self.point(carrier,carrier,positions,valid)
        native=torch.compile(Native(),backend='aot_eager',fullgraph=True)
        outer=torch.compile(make_graph_envelope_class(torch)(native,[point]),backend='aot_eager',fullgraph=True)
        for generation in range(1,33):
            point.reset_after_fence();epoch.fill_(generation)
            carrier=torch.full((6,8),generation,dtype=torch.bfloat16)
            positions=(torch.arange(6)+generation*6).reshape(-1,1);valid=torch.ones(6,dtype=torch.bool)
            result,outputs=outer(carrier,positions,valid)
            self.assertIs(result,carrier);self.assertEqual(len(outputs),6)
            self.assertTrue(all(actual.data_ptr()==original.data_ptr() and actual.storage_offset()==original.storage_offset()
                for actual,original in zip(outputs,point.tensor_tuple())))
            self.assertTrue(torch.equal(outputs[0],carrier));self.assertTrue(torch.equal(outputs[1],positions))
            self.assertEqual(outputs[3].item(),generation);self.assertEqual(outputs[4].item(),1)

    def test_CPU_compiled_actual_root_entry_and_adapter(self):
        callbacks=self.callbacks();point=callbacks.add_point('main',0,'decoder.output',8,torch.bfloat16)
        class Native(torch.nn.Module):
            def forward(self,carrier,positions):
                callbacks.enter(self,'DeepseekV41DecoderLayer.forward',{'positions':positions})
                return callbacks.tap(self,'decoder.output',carrier,carrier,{'positions':positions})
        root=Native();root.add_module('_v10_observer_buffers',torch.nn.ModuleList([point]))
        callbacks.register_owner(root,'main',0)
        session=SimpleNamespace(torch=torch,callbacks=callbacks,returned_outputs={},
            role_tensor_outputs=lambda role:(*point.tensor_tuple(),*callbacks.role_metadata_outputs(role)))
        extend_native_root_forward(root,session,'main')
        adapter=NativeResultAdapter(torch.compile(root,backend='aot_eager',fullgraph=True),session,'main')
        for generation in range(1,33):
            callbacks.begin_after_prior_fence(generation,actual_target_rows=6)
            carrier=torch.full((6,8),generation,dtype=torch.bfloat16);positions=torch.arange(6)+generation*6
            self.assertIs(adapter(carrier,positions),carrier)
            self.assertEqual(point.copy_count.item(),1);self.assertEqual(point.epoch.item(),generation)
            self.assertEqual(len(session.returned_outputs['main']),10)
            self.assertTrue(torch.equal(callbacks.target_positions,positions.reshape(-1,1)))
            self.assertTrue(all(actual.data_ptr()==original.data_ptr() for actual,original in
                zip(session.returned_outputs['main'],session.role_tensor_outputs('main'))))
        # Compiled containers may arrive as lists; the adapter accepts both
        # shapes, and an unexpected structure must be rejected (not coerced).
        def listify(*args,**kwargs):
            value,outputs=root(*args,**kwargs)
            return [value,list(outputs)]
        list_adapter=NativeResultAdapter(listify,session,'main')
        callbacks.begin_after_prior_fence(33,actual_target_rows=6)
        carrier=torch.full((6,8),33,dtype=torch.bfloat16);positions=torch.arange(6)+33*6
        self.assertIs(list_adapter(carrier,positions),carrier)
        self.assertEqual(len(session.returned_outputs['main']),10)
        # A wrapped callable that predates the envelope returns the raw native
        # tensor: the adapter hands back the same owned buffers and records the
        # synthesis explicitly.
        sink=[]
        session.record_envelope=lambda role,source:sink.append((role,source))
        raw_adapter=NativeResultAdapter(lambda *args,**kwargs:torch.full((6,8),7,dtype=torch.bfloat16),session,'main')
        raw_result=raw_adapter(carrier,positions)
        self.assertEqual(raw_result.shape,(6,8));self.assertEqual(raw_result[0,0].item(),7.0)
        self.assertEqual(len(session.returned_outputs['main']),10)
        self.assertIn(('main','adapter_synthesis_after_raw_native_result'),sink)
        broken_adapter=NativeResultAdapter(lambda *args,**kwargs:'not-a-tensor-nor-envelope',session,'main')
        with self.assertRaises(ValueError):broken_adapter(carrier,positions)

    def test_compiled_dispatch_envelope_reaches_the_callable_the_class_invokes(self):
        from .lifecycle import extend_compiled_dispatch
        callbacks=self.callbacks();point=callbacks.add_point('draft',40,'decoder.output',8,torch.bfloat16)
        events=[]
        class DecoratedLike(torch.nn.Module):
            def __init__(self):
                super().__init__();self.forward_ran=0
            def forward(self,carrier,positions):
                self.forward_ran+=1;return torch.full((5,8),3.0,dtype=torch.bfloat16)
            def __call__(self,*args,**kwargs):
                if getattr(self,'aot_compiled_fn',None) is not None:
                    return self.aot_compiled_fn(self,*args,**kwargs)
                return (self._compiled_callable(*args,**kwargs)
                        if self._compiled_callable is not None else self.forward(*args,**kwargs))
        root=DecoratedLike()
        root._compiled_callable=lambda carrier,positions:torch.full((5,8),4.0,dtype=torch.bfloat16)
        session=SimpleNamespace(torch=torch,callbacks=callbacks,returned_outputs={},
            role_tensor_outputs=lambda role:(*point.tensor_tuple(),*callbacks.role_metadata_outputs(role)),
            note_dispatch=lambda role,source:events.append((role,source)))
        callbacks.begin_after_prior_fence(5,actual_target_rows=6)
        carrier=torch.ones(5,8,dtype=torch.bfloat16);positions=torch.arange(5)
        receipt=extend_compiled_dispatch(root,session,'draft')
        self.assertEqual(receipt['compiled_callable'],'wrapped');self.assertIsNone(receipt['aot_compiled_fn'])
        value,outputs=root(carrier,positions)
        self.assertEqual(value[0,0].item(),4.0);self.assertEqual(root.forward_ran,0)  # compiled path really used
        self.assertEqual(len(outputs),len(session.role_tensor_outputs('draft')))
        self.assertEqual(events,[('draft','compiled_callable')])
        again=extend_compiled_dispatch(root,session,'draft')
        self.assertEqual(again['compiled_callable'],'already')
        value,_=root(carrier,positions);self.assertEqual(value[0,0].item(),4.0)
        # AOT path: the compiled fn receives the model as its first argument.
        root2=DecoratedLike();root2._compiled_callable=None
        root2.aot_compiled_fn=lambda model,carrier,positions:(model.forward_ran,model.forward(carrier,positions))[1]
        extend_compiled_dispatch(root2,session,'draft')
        value2,outputs2=root2(carrier,positions)
        self.assertEqual(value2[0,0].item(),3.0);self.assertEqual(root2.forward_ran,1)
        self.assertEqual(len(outputs2),len(session.role_tensor_outputs('draft')))
        # Envelope receipts contain bound methods: they must go through the
        # JSON-safe projection before landing on disk.
        from .lifecycle import extend_native_root_forward
        root3=torch.nn.Module()
        envelope=extend_native_root_forward(root3,session,'draft')
        self.assertFalse(isinstance(envelope['original'],(str,int,float,bool,type(None),dict,list)))
        safe={key:(value if isinstance(value,(str,int,float,bool,type(None),dict,list)) else repr(value))
              for key,value in envelope.items()}
        json.dumps({'receipts':safe})

    def test_original_load_AnnAssign_boundary_and_original_MP_AST(self):
        Runner=source_class('runner.py','NPUModelRunner',['load_model'])
        runner=Runner();callbacks=SimpleNamespace(retain_target=lambda *args:None,before_graph=lambda *args:None)
        receipt=runner_load_with_callbacks(runner,callbacks)
        self.assertTrue(receipt['native_load_AST_preserved']);self.assertTrue(receipt['before_graph_boundary_verified'])
        _,receipt=derive_executor_init((SOURCE/'multiproc_executor.py.txt').read_text())
        self.assertEqual(receipt['counts'],{'birth':1,'failure':1});self.assertTrue(receipt['native_bootstrap_AST_preserved'])

    def test_real_IDS_source_entries_outer_six_buffers_and_original_fallback(self):
        from parallel_tiny.dspark_prepack_audit_cpu.test_component_cpu import fixture,ast_function
        from parallel_tiny.dspark_prepack_audit_cpu.contracts import SOURCES
        for role in ('main','draft'):
            f=fixture(role,ffn=1280);callbacks=self.callbacks()
            callbacks.begin_after_prior_fence(23,actual_target_rows=6)
            callbacks.enter_root('main',torch.arange(6)+200);callbacks.enter_root('draft',torch.arange(5)+201)
            session=SimpleNamespace(torch=torch,npu=SimpleNamespace(get_npu_format=lambda t:2),callbacks=callbacks)
            integration=PrepackIntegration(session)
            for method in ('apply_gmm1_act_quant','apply_gmm2'):
                fn=ast_function(SOURCES['w4a8']['functions']['AscendW4A8DynamicFusedMoEMethod.'+method],f.namespace)
                setattr(f.scheme,method,MethodType(fn,f.scheme))
            with patch('parallel_tiny.dspark_prepack_audit_cpu.contracts.bind_owner',return_value=f.binding):
                integration.bind_quant_layer(role,f.binding.global_layer,f.decoder,f.layer,f.scheme)
            inp,x,scale=f.input_for('gmm1');q,scale=f.scheme.apply_gmm1_act_quant(inp)
            self.assertEqual(tuple(q.shape),(f.binding.capacity,1280))
            g1=integration.buffers[(id(f.layer),'gmm1')]
            self.assertTrue(torch.equal(g1[0],x));self.assertEqual(g1[4].item(),1)
            before=q.clone();out=f.scheme.apply_gmm2(inp,q,scale);g2=integration.buffers[(id(f.layer),'gmm2')]
            self.assertTrue(torch.equal(g2[0],before));self.assertEqual(g2[4].item(),1)
            self.assertEqual(tuple(out.shape),(f.binding.capacity,5120))
            self.assertIs(f.native_records[-1]['x'][0],q)
            self.assertEqual(len(tuple(t for p in callbacks.points.values() for t in p.tensor_tuple())),12)
            # One native unsupported geometry call still returns the original
            # native carriers; it does not earn a snapshot copy.
            inp,x,_=f.input_for('gmm1');inp.hidden_states=x[:1];inp.dynamic_scale=inp.dynamic_scale[:1]
            count=g1[4].item();result=f.scheme.apply_gmm1_act_quant(inp)
            self.assertEqual(result[0].shape[0],1);self.assertEqual(g1[4].item(),count)

    def test_driver_dry_run_constructs_no_model_or_SDK(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/'model';model.mkdir();(model/'config.json').write_bytes((HERE/'evidence/checkpoint_config_CPU_fixture.json').read_bytes())
            command=[sys.executable,str(HERE/'formal_audit_driver.py'),'--model',str(model),
                '--physical-chips','8,9,10,11,12,13,14,15','--output',str(Path(tmp)/'output')]
            result=subprocess.run(command,capture_output=True,text=True,check=True)
            data=json.loads(result.stdout);self.assertEqual(data['npu_jobs'],0);self.assertFalse(data['strict_full43_admission'])
            self.assertFalse(data['config']['audit_enabled']);self.assertFalse((Path(tmp)/'output').exists())

    def test_generation_argument_and_registered43_cannot_grant_actual32(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=summarize_rank_exports(tmp,32)
            self.assertEqual(set(data),{str(i) for i in range(8)})
            self.assertTrue(all(not row['diagnostic_export_contract_complete'] for row in data.values()))


def main():
    runner=unittest.TextTestRunner(stream=sys.stderr,verbosity=2)
    result=runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    forbidden=[name for name in sys.modules if any(name==s or name.startswith(s+'.') for s in ('vllm','vllm_ascend','torch_npu','torchair','acl'))]
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if any(name in line.lower() for name in ('libascendcl','libtorch_npu','libhccl','libopapi'))})
    receipt={'scope':'v16 CPU diagnostic contracts only','torch':torch.__version__,'tests':result.testsRun,
        'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'original_consumer_functions_AST_verified':len(RECEIPTS),'mock_native_consumers':True,
        'SDK_imports':forbidden,'blocked_SDK_import_attempts':GUARD.attempts,'NPU_calls':0,
        'NPU_dynamic_libraries':libraries,
        'device_audit_passed':False,'full_consumer_audit_passed':False,'safe_formal_receipt':False,
        'formal_model_validated':False,'actual_prepack_consumer_order_validated':False}
    print(json.dumps(receipt,indent=2))
    return 0 if result.wasSuccessful() and not forbidden and not GUARD.attempts and not result.skipped and not libraries else 1

if __name__=='__main__':raise SystemExit(main())
