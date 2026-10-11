"""CPU torch contract tests with original source AST and explicit fake consumers."""
import os
os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',ASCEND_RT_VISIBLE_DEVICES='',
    ASCEND_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',
    PYTHONDONTWRITEBYTECODE='1')

import ast
import ctypes
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


class BlockNPU(importlib.abc.MetaPathFinder):
    blocked=[]
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==n or fullname.startswith(n+'.') for n in ('torch_npu','vllm','vllm_ascend','acl','torchair')):
            self.blocked.append(fullname)
            raise ImportError('CPU-only prepack test rejects NPU/SDK import: '+fullname)
        return None
GUARD=BlockNPU();sys.meta_path.insert(0,GUARD)
import torch

from .contracts import (GEOMETRY,SOURCES,bind_owner,check_call,metadata,no_actual_claims,pack_target_source)
from .native_adapter_factory import PreparedStageAdapter,schema_declaration,verify_runtime_schema
from .pure_graph_factory import make_pure_graph_entry
from .raw_lifecycle import RawLifecycle
from .snapshot_transport import allocate,snapshot_before_native_tuple

HERE=Path(__file__).resolve().parent
NEGATIVES=[];POSITIVES=[];PURE_RECEIPTS=[]


def ast_function(entry,namespace):
    tree=ast.parse(entry['source']);node=tree.body[0]
    node.returns=None;node.decorator_list=[]
    for a in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:a.annotation=None
    ast.fix_missing_locations(tree);scope=dict(namespace)
    exec(compile(tree,'<actual-source-AST-CPU-fixture>','exec'),scope)
    return scope[node.name]


class WeightMetadata:
    next_ptr=40000
    def __init__(self,shape,dtype,format=2):
        self.shape=tuple(shape);self.dtype=dtype;self.device='cpu';self._version=0;self.format=format
        WeightMetadata.next_ptr+=4096;self.ptr=WeightMetadata.next_ptr
        running=1;strides=[]
        for value in reversed(shape):strides.insert(0,running);running*=value
        self.strides=tuple(strides);self.bytes=running*torch.empty((),dtype=dtype).element_size()
    def data_ptr(self):return self.ptr
    def untyped_storage(self):return self
    def nbytes(self):return self.bytes
    def storage_offset(self):return 0
    def stride(self):return self.strides


class TorchFacade:
    def __init__(self,operators):self.ops=SimpleNamespace(_C_ascend=operators)
    def __getattr__(self,name):return getattr(torch,name)


def format_getter(tensor):return getattr(tensor,'format',2)


def fixture(role='main',global_layer=None,ffn=2304):
    if global_layer is None:global_layer=0 if role=='main' else 40
    e,top,local,_,capacity=GEOMETRY[role]
    events=[];native_records=[];quant_records=[]
    class FakeNative:
        def gmm1(self,**kwargs):
            events.append('CPU_fake_GMM1_workspace_prepack')
            native_records.append(kwargs)
            x=kwargs['x']
            # GMM1's source packs into workspace, so this fake does not mutate x.
            ffn_=kwargs['weight'][0].shape[2]*4
            return torch.full((x.shape[0],ffn_),7,dtype=torch.int8),torch.ones(x.shape[0],dtype=torch.float32)
        def gmm2(self,**kwargs):
            events.append('CPU_fake_GMM2_original_x_GM_prepack')
            native_records.append(kwargs)
            x=kwargs['x'][0]
            # An actual CPU raw-pointer write bypasses ATen version counting.
            ctypes.memset(x.data_ptr(),0x31,x.numel())
            return [torch.zeros((x.shape[0],kwargs['weight'][0].shape[2]*8),dtype=torch.bfloat16)]
        def quant(self,x,**kwargs):
            events.append('actual_quant_source_called_CPU_fake_quant')
            out=torch.arange(x.numel(),dtype=torch.int32).remainder(251).to(torch.int8).reshape(x.shape)
            scale=torch.ones(x.shape[0],dtype=torch.float32)
            quant_records.append((out,scale))
            return out,scale
    fake=FakeNative()
    operators=SimpleNamespace(grouped_matmul_swiglu_quant_v2=fake.gmm1)
    facade=TorchFacade(operators)
    enums=SimpleNamespace(SILU='SILU',SITU='SITU')
    quant_types=SimpleNamespace(**{name:name for name in ('W4A8','W4A16','W4A16MXFP','W8A8MXFP','W4A4MXFP','W4A8MXFP')})
    npu=SimpleNamespace(npu_grouped_matmul=fake.gmm2,npu_dynamic_quant=fake.quant)
    namespace={'torch':facade,'torch_npu':npu,'MoEActivation':enums,
        'QuantType':quant_types,'dispose_tensor':lambda value:events.append('original_dispose_position_CPU_fake')}
    methods={}
    for name,entry in SOURCES['w4a8']['functions'].items():
        if name.endswith(('._get_mlp_weights','._get_mlp_scale_bias','._maybe_convert_group_list')):
            methods[name.rsplit('.',1)[1]]=ast_function(entry,namespace)
    quant_entry=next(iter(SOURCES['quant_base']['functions'].values()))
    methods['_quant_hidden_states']=ast_function(quant_entry,namespace)
    Scheme=type('CPU_AST_native_scheme',(),methods)
    scheme=Scheme();scheme.quant_type=quant_types.W4A8;scheme.act_quant_type=torch.int8;scheme.use_expert_weight_list=False
    router=SimpleNamespace(top_k=top)
    layer=SimpleNamespace(quant_method=scheme,router=router,intermediate_size_per_partition=ffn,
        w13_weight=WeightMetadata((local,5120,ffn//4),torch.int32,29),
        w2_weight=WeightMetadata((local,ffn,640),torch.int32,29),
        w13_weight_scale=WeightMetadata((local,2*ffn),torch.int64),
        w2_weight_scale=WeightMetadata((local,1,5120),torch.int64),
        w13_scale_bias=WeightMetadata((local,2*ffn),torch.float32),
        w2_scale_bias=WeightMetadata((local,5120),torch.float32))
    experts=SimpleNamespace(modules=lambda:[layer])
    local_layer=global_layer if role=='main' else global_layer-40
    moe=SimpleNamespace(layer_idx=local_layer,experts=experts,n_routed_experts=e,
        n_local_physical_experts=local,n_physical_experts=e,ep_size=8,ep_rank=0,
        gate=SimpleNamespace(weight=WeightMetadata((e,5120),torch.float32)))
    decoder=SimpleNamespace(mlp=moe,layer_idx=local_layer)
    binding=bind_owner(role=role,global_layer=global_layer,retained_decoder=decoder,
        quant_layer=layer,router=router,scheme=scheme,format_getter=format_getter,cpu_mock=True)
    source_current_stream=SOURCES['sdk_stream']['sha256']
    stream=lambda:{'kind':'borrowed_aclrtStream','handle':123456789,'source_sha256':source_current_stream,
        'cpp_launch_source_sha256':SOURCES['native_stream_and_GM']['sha256'],'CPU_mock':True}
    def input_for(stage,bf16=False):
        width=5120 if stage=='gmm1' else ffn
        x=torch.arange(capacity*width,dtype=torch.int32).remainder(251).to(torch.int8).reshape(capacity,width)
        hidden=x.to(torch.bfloat16) if bf16 else x
        scale=None if bf16 else torch.ones(capacity,dtype=torch.float32)
        inp=SimpleNamespace(hidden_states=hidden,dynamic_scale=scale,layer=layer,
            group_list=torch.zeros(local,dtype=torch.int64),group_list_type=1,
            activation=enums.SILU,swiglu_limit=17.0,dynamic_eplb=False,lora_context=None)
        return inp,x,scale
    return SimpleNamespace(binding=binding,layer=layer,moe=moe,decoder=decoder,router=router,
        scheme=scheme,namespace=namespace,facade=facade,stream=stream,events=events,
        native_records=native_records,quant_records=quant_records,input_for=input_for)


class PrepackContract(unittest.TestCase):
    def reject(self,name,callback,error=(ValueError,RuntimeError)):
        with self.assertRaises(error):callback()
        NEGATIVES.append(name)

    def adapter(self,f,stage,x,inp):
        b=f.binding;width=b.hidden if stage=='gmm1' else b.ffn
        buffers=allocate(torch,b.capacity,width,device='cpu')
        positions=torch.arange(b.capacity,dtype=torch.int64).reshape(-1,1)
        valid=torch.arange(b.capacity)%2==0
        epoch=torch.tensor([91],dtype=torch.int64)
        tracker=RawLifecycle();lease=tracker.producer_returned((b.role,b.global_layer),stage,x,91)
        obj=PreparedStageAdapter(f.facade,b,stage,f.namespace,format_getter,tracker,buffers,
            positions,valid,epoch,f.stream)
        return obj,tracker,lease,buffers

    def test_all_four_role_stage_source_AST_positive_and_same_carrier(self):
        for role in ('main','draft'):
            for stage in ('gmm1','gmm2'):
                f=fixture(role);inp,x,scale=f.input_for(stage)
                expected=x.clone();version=x._version;ptr=x.data_ptr()
                adapter,tracker,lease,buffers=self.adapter(f,stage,x,inp)
                result=adapter.call_eager(inp,lease,*(() if stage=='gmm1' else (x,scale)))
                self.assertTrue(torch.equal(buffers[0],expected))
                self.assertEqual(x.data_ptr(),ptr);self.assertEqual(x._version,version)
                self.assertTrue(all(result[-6+i] is buffers[i] for i in range(6)))
                self.assertIs(f.native_records[-1]['x'] if stage=='gmm1' else f.native_records[-1]['x'][0],x)
                self.assertEqual(lease.state,'consumed')
                self.assertEqual(adapter.last_call['original_x_metadata']['ptr'],ptr)
                self.assertTrue(all(value is False for value in no_actual_claims().values()))
                changed=not torch.equal(x,expected)
                self.assertEqual(changed,stage=='gmm2')
                if stage=='gmm1':self.assertEqual(f.events[-1],'original_dispose_position_CPU_fake')
                self.assertEqual(buffers[3].item(),91)
                POSITIVES.append({'CPU_mock':True,'role':role,'stage':stage,'M':f.binding.capacity,
                    'K':expected.shape[1],'original_GM_same':True,'fixed_raw_before_pack_equal':True,
                    'raw_pointer_write_mutated_x':changed,'ATen_version_unchanged':True,
                    'pack_target_source':pack_target_source(stage),**no_actual_claims()})

    def test_pure_graph_entries_preserve_original_call_AST_and_tensor_outputs(self):
        for role in ('main','draft'):
            for stage in ('gmm1','gmm2'):
                f=fixture(role);inp,x,scale=f.input_for(stage)
                expected=x.clone();width=x.shape[1]
                buffers=allocate(torch,f.binding.capacity,width,device='cpu')
                positions=torch.arange(f.binding.capacity,dtype=torch.int64).reshape(-1,1)
                valid=torch.zeros(f.binding.capacity,dtype=torch.bool) # full RAW still copied, no sanitization
                epoch=torch.tensor([117],dtype=torch.int64)
                fn,text,receipt=make_pure_graph_entry(stage,f.namespace)
                result=fn(f.scheme,inp,*(() if stage=='gmm1' else (x,scale)),
                    audit_positions=positions,audit_valid=valid,audit_epoch=epoch,audit_buffers=buffers)
                self.assertTrue(torch.equal(buffers[0],expected))
                self.assertEqual(len(result),8 if stage=='gmm1' else 7)
                self.assertTrue(all(result[-6+i] is buffers[i] for i in range(6)))
                self.assertTrue(receipt['original_native_call_AST_byte_equal'])
                self.assertNotIn('data_ptr(',text);self.assertNotIn('_version',text)
                path=HERE/'evidence'/f'private_{stage}_tensor_entry.py.txt'
                if not path.exists():path.write_text(text+'\n')
                PURE_RECEIPTS.append({'role':role,**receipt,'CPU_mock_execution_passed':True})

    def test_snapshot_boundary_is_after_actual_quantization_not_BF16_entry(self):
        f=fixture();inp,_,_=f.input_for('gmm1',bf16=True)
        fn,_,_=make_pure_graph_entry('gmm1',f.namespace)
        buffers=allocate(torch,36,5120,device='cpu')
        result=fn(f.scheme,inp,audit_positions=torch.arange(36,dtype=torch.int64).reshape(-1,1),
            audit_valid=torch.ones(36,dtype=torch.bool),audit_epoch=torch.tensor([1]),audit_buffers=buffers)
        self.assertTrue(torch.equal(buffers[0],f.quant_records[0][0]))
        self.assertIs(f.native_records[0]['x'],f.quant_records[0][0])
        self.assertEqual(f.events[0],'actual_quant_source_called_CPU_fake_quant')

    def test_role_specific_loaded_FFN_is_not_hardcoded_K2304(self):
        f=fixture('draft',ffn=1792)
        self.assertEqual(f.binding.ffn,1792)
        inp,x,scale=f.input_for('gmm2')
        adapter,_,lease,buffers=self.adapter(f,'gmm2',x,inp)
        adapter.call_eager(inp,lease,x,scale)
        self.assertEqual(buffers[0].shape,(15,1792))
        wrong=torch.zeros((15,2304),dtype=torch.int8)
        inp.hidden_states=wrong
        tracker=RawLifecycle();bad_lease=tracker.producer_returned(('draft',40),'gmm2',wrong,2)
        self.reject('old_K2304_cannot_override_loaded_draft_FFN',lambda:adapter.call_eager(inp,bad_lease,wrong,scale))

    def test_actual_modelslim_scheme_adapter_chain_is_bound(self):
        f=fixture('draft')
        f.layer.quant_method=SimpleNamespace(quant_method=f.scheme)
        bound=bind_owner(role='draft',global_layer=40,retained_decoder=f.decoder,
            quant_layer=f.layer,router=f.router,scheme=f.scheme,format_getter=format_getter,cpu_mock=True)
        self.assertIs(bound.scheme,f.scheme)
        f.layer.quant_method.quant_method=object()
        self.reject('replaced_nested_actual_scheme_owner',lambda:bind_owner(role='draft',global_layer=40,
            retained_decoder=f.decoder,quant_layer=f.layer,router=f.router,scheme=f.scheme,
            format_getter=format_getter,cpu_mock=True))

    def test_pack_after_clone_is_not_original_raw_and_version_is_false_negative(self):
        x=torch.full((6,2304),-127,dtype=torch.int8)
        before=x.clone();version=x._version;ptr=x.data_ptr()
        ctypes.memset(ptr,0x31,x.numel())
        after=x.clone()
        self.assertEqual(x._version,version)
        self.assertFalse(torch.equal(before,after))
        self.assertEqual(ptr,x.data_ptr())
        tracker=RawLifecycle();lease=tracker.producer_returned(('main',0),'gmm2',x,1)
        tracker.snapshot(lease,x,before);tracker.enter_consumer(lease)
        self.reject('after_pack_clone_cannot_register_before_pack_snapshot',lambda:tracker.snapshot(lease,x,after))
        self.reject('unchanged_version_cannot_rearm_consumed_raw_GM',lambda:tracker.producer_returned(('main',0),'gmm2',x,2,prior_fence=lambda:None))

    def test_AA_restore_calls_fence_and_restores_original_same_GM(self):
        f=fixture();inp,x,scale=f.input_for('gmm2');expected=x.clone();ptr=x.data_ptr()
        adapter,tracker,lease,buffers=self.adapter(f,'gmm2',x,inp)
        adapter.call_eager(inp,lease,x,scale)
        self.reject('reusing_packed_x_without_restore',lambda:adapter.call_eager(inp,lease,x,scale))
        self.reject('restore_without_prior_fence',lambda:tracker.restore_same_original_GM(lease,prior_fence=None))
        ledger=[]
        tracker.restore_same_original_GM(lease,prior_fence=lambda:ledger.append('prior_completed_work_fence_CPU_fake'))
        self.assertEqual(ledger,['prior_completed_work_fence_CPU_fake'])
        self.assertEqual(x.data_ptr(),ptr);self.assertTrue(torch.equal(x,expected))
        adapter.call_eager(inp,lease,x,scale)
        self.assertTrue(torch.equal(buffers[0],expected))

    def test_fresh_producer_generation_requires_executed_refeed_callback(self):
        x=torch.zeros((15,17),dtype=torch.int8);tracker=RawLifecycle()
        lease=tracker.producer_returned(('draft',40),'gmm2',x,1)
        raw=x.clone();tracker.snapshot(lease,x,raw);tracker.enter_consumer(lease)
        self.reject('new_epoch_label_without_producer_is_not_raw_refeed',lambda:tracker.producer_returned(('draft',40),'gmm2',x,2,prior_fence=lambda:None))
        self.reject('refeed_another_carrier_not_same_original_GM',lambda:tracker.producer_returned(('draft',40),'gmm2',x,2,prior_fence=lambda:None,producer_callback=lambda:x.clone()))
        actions=[]
        def producer():actions.append('real_callback_CPU_fake_refill');x.fill_(3);return x
        next_lease=tracker.producer_returned(('draft',40),'gmm2',x,2,prior_fence=lambda:actions.append('fence'),producer_callback=producer)
        self.assertEqual(actions,['fence','real_callback_CPU_fake_refill'])
        self.assertEqual(next_lease.state,'fresh')

    def test_role_global_local_router_EP_and_owner_mismatch_rejected(self):
        changes={'wrong_topk':lambda f:setattr(f.router,'top_k',3),
            'wrong_local_E':lambda f:setattr(f.moe,'n_local_physical_experts',16),
            'wrong_global_E':lambda f:setattr(f.moe,'n_routed_experts',128),
            'wrong_EP':lambda f:setattr(f.moe,'ep_size',4),
            'wrong_layer':lambda f:setattr(f.decoder,'layer_idx',1),
            'wrong_ffn_partition':lambda f:setattr(f.layer,'intermediate_size_per_partition',1792)}
        for label,change in changes.items():
            f=fixture();change(f)
            self.reject(label,lambda:bind_owner(role='main',global_layer=0,retained_decoder=f.decoder,
                quant_layer=f.layer,router=f.router,scheme=f.scheme,format_getter=format_getter,cpu_mock=True))
        f=fixture()
        self.reject('main_cannot_be_global40_draft_owner',lambda:bind_owner(role='main',global_layer=40,
            retained_decoder=f.decoder,quant_layer=f.layer,router=f.router,scheme=f.scheme,format_getter=format_getter,cpu_mock=True))
        foreign=SimpleNamespace(**vars(f.layer))
        self.reject('foreign_quant_layer_with_same_shapes',lambda:bind_owner(role='main',global_layer=0,
            retained_decoder=f.decoder,quant_layer=foreign,router=f.router,scheme=f.scheme,format_getter=format_getter,cpu_mock=True))

    def test_actual_call_shape_dtype_format_counts_weights_owner_rejections(self):
        cases=['wrong_x_shape','wrong_x_dtype','wrong_x_format','wrong_counts_shape','wrong_counts_dtype',
            'foreign_layer','replaced_weight','changed_weight_version','changed_weight_format','EPLB','LoRA']
        for label in cases:
            f=fixture();inp,x,scale=f.input_for('gmm2')
            if label=='wrong_x_shape':x=torch.zeros((15,2304),dtype=torch.int8)
            elif label=='wrong_x_dtype':x=x.float()
            elif label=='wrong_x_format':x.format=29
            elif label=='wrong_counts_shape':inp.group_list=torch.zeros(16,dtype=torch.int64)
            elif label=='wrong_counts_dtype':inp.group_list=inp.group_list.int()
            elif label=='foreign_layer':inp.layer=SimpleNamespace(**vars(f.layer))
            elif label=='replaced_weight':f.layer.w2_weight=WeightMetadata((48,2304,640),torch.int32,29)
            elif label=='changed_weight_version':f.layer.w2_weight._version+=1
            elif label=='changed_weight_format':f.layer.w2_weight.format=2
            elif label=='EPLB':inp.dynamic_eplb=True
            elif label=='LoRA':inp.lora_context=object()
            adapter,_,lease,_=self.adapter(f,'gmm2',x,inp)
            self.reject(label,lambda:adapter.call_eager(inp,lease,x,scale))
            self.assertFalse(f.native_records)

    def test_buffer_alias_wrong_marker_fields_and_lost_outputs_rejected(self):
        f=fixture();inp,x,scale=f.input_for('gmm2')
        adapter,_,lease,buffers=self.adapter(f,'gmm2',x,inp)
        bad=(x,*buffers[1:])
        obj=PreparedStageAdapter(f.facade,f.binding,'gmm2',f.namespace,format_getter,adapter.lifecycle,bad,
            adapter.positions,adapter.valid,adapter.epoch,f.stream)
        self.reject('snapshot_buffer_aliases_original_x',lambda:obj.call_eager(inp,lease,x,scale))
        self.reject('missing_snapshot_output_field',lambda:snapshot_before_native_tuple(torch,x,
            adapter.positions,adapter.valid,adapter.epoch,buffers[:-1]))
        bad_epoch=(buffers[0],buffers[1],buffers[2],buffers[3].int(),buffers[4],buffers[5])
        self.reject('wrong_snapshot_marker_dtype',lambda:snapshot_before_native_tuple(torch,x,
            adapter.positions,adapter.valid,adapter.epoch,bad_epoch))
        aliased=(buffers[0],buffers[1],buffers[2],buffers[3],buffers[3],buffers[5])
        obj=PreparedStageAdapter(f.facade,f.binding,'gmm2',f.namespace,format_getter,adapter.lifecycle,aliased,
            adapter.positions,adapter.valid,adapter.epoch,f.stream)
        self.reject('two_mutable_snapshot_fields_alias',lambda:obj.call_eager(inp,lease,x,scale))

    def test_stream_profiler_label_zero_handle_and_missing_source_rejected(self):
        for label,stream in [('profiler47',{'kind':'profiler_stream_id','handle':47,'source_sha256':'x'}),
                ('zero_stream',{'kind':'borrowed_aclrtStream','handle':0,'source_sha256':'x'}),
                ('no_stream_source',{'kind':'borrowed_aclrtStream','handle':123})]:
            f=fixture();inp,x,scale=f.input_for('gmm2');adapter,_,lease,_=self.adapter(f,'gmm2',x,inp)
            adapter.borrow_stream=lambda s=stream:s
            self.reject(label,lambda:adapter.call_eager(inp,lease,x,scale))
            self.assertFalse(f.native_records)

    def test_wrong_raw_lease_role_stage_and_owner_rejected(self):
        for label,owner,stage in [('lease_role',('draft',40),'gmm2'),('lease_layer',('main',1),'gmm2'),('lease_stage',('main',0),'gmm1')]:
            f=fixture();inp,x,scale=f.input_for('gmm2');adapter,tracker,_,_=self.adapter(f,'gmm2',x,inp)
            lease=tracker.producer_returned(owner,stage,x,1)
            self.reject(label,lambda:adapter.call_eager(inp,lease,x,scale))

    def test_original_nan_scale_and_swiglu_limit_are_not_sanitized(self):
        f=fixture();inp,x,scale=f.input_for('gmm1');scale[0]=float('nan');inp.swiglu_limit=float('inf')
        adapter,_,lease,_=self.adapter(f,'gmm1',x,inp);adapter.call_eager(inp,lease)
        self.assertIs(f.native_records[-1]['x_scale'],scale)
        self.assertTrue(torch.isnan(scale[0]))
        self.assertEqual(f.native_records[-1]['swiglu_limit'],float('inf'))

    def test_original_prefix_to_dense_counts_conversion_preserved(self):
        f=fixture();inp,x,scale=f.input_for('gmm2')
        inp.group_list=torch.arange(48,dtype=torch.int64);inp.group_list_type=0
        adapter,_,lease,_=self.adapter(f,'gmm2',x,inp);adapter.call_eager(inp,lease,x,scale)
        self.assertTrue(torch.equal(f.native_records[-1]['group_list'],torch.tensor([0]+[1]*47)))
        self.assertEqual(f.native_records[-1]['group_list_type'],1)

    def test_unsupported_SITU_path_and_wrong_schema_do_not_grant_admission(self):
        f=fixture();inp,x,_=f.input_for('gmm1');inp.activation='SITU'
        adapter,_,lease,_=self.adapter(f,'gmm1',x,inp)
        self.reject('SITU_is_another_consumer_path',lambda:adapter.call_eager(inp,lease))
        self.assertFalse(f.native_records)
        for stage in ('gmm1','gmm2'):
            good=SimpleNamespace(_schema='test_runtime_namespace::'+schema_declaration(stage))
            self.assertFalse(verify_runtime_schema(good,stage)['actual_device_consumer_proof'])
            bad=SimpleNamespace(_schema=str(good._schema).replace(' group_list',' changed_group_list',1))
            self.reject('wrong_actual_schema_'+stage,lambda:verify_runtime_schema(bad,stage))
        self.reject('missing_actual_selected_schema',lambda:verify_runtime_schema(object(),'gmm2'))


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrepackContract))
    forbidden=[name for name in sys.modules if any(name==n or name.startswith(n+'.') for n in ('torch_npu','vllm','vllm_ascend','acl','torchair'))]
    assert not forbidden and not GUARD.blocked
    maps=Path('/proc/self/maps').read_text()
    libraries=sorted({line.split()[-1] for line in maps.splitlines() if any(n in line.lower() for n in ('libascendcl','libtorch_npu','libhccl','libopapi'))})
    assert not libraries
    report={'CPU_only':True,'CPU_torch_version':torch.__version__,'CPU_mock_native_consumers':True,
        'tests':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),
        'negative_cases':NEGATIVES,'negative_case_count':len(NEGATIVES),'positive_role_stage_cases':POSITIVES,
        'pure_Tensor_entry_CPU_cases':PURE_RECEIPTS,'actual_NPU_calls':0,'actual_HTTP_inference_requests':0,
        'services_started':0,'NPU_SDK_modules_imported':forbidden,'NPU_runtime_libraries_mapped':libraries,
        'backend_autoload':os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'],
        'device_visibility':{n:os.environ[n] for n in ('ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')},
        'actual_claims':no_actual_claims(),
        'limitations':['All GMM/quant/workspace/stream callbacks are explicit CPU fakes; no native math or model executed.',
            'Pure Tensor source order is not actual compiled/device copy-before-prepack proof.',
            'GMM1 actual workspace addresses and GMM2 actual selected tiling/kernel still require a device audit job.',
            'Host raw lifecycle/provenance needs actual producer/device epoch evidence per replay; _version is never that evidence.']}
    (HERE/'evidence/CPU_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():sys.exit(1)


if __name__=='__main__':main()
