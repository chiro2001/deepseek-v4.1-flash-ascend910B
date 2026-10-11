import base64,json,linecache,sys,traceback,types
assert 'torch_npu' not in sys.modules
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch.utils._python_dispatch import TorchDispatchMode
source=base64.b64decode('IiIiRXhwbGljaXQgVGVuc29yIGlucHV0L291dHB1dCBib3VuZGVkIHRyYW5zcG9ydCwgbm8gdG9yY2ggaW1wb3J0IG9uIGltcG9ydC4iIiIKZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgZGF0YWNsYXNzCgoKRklFTERTID0gKCd2YWx1ZXMnLCdwb3NpdGlvbnMnLCd2YWxpZCcsJ2Vwb2NoJywnY29weV9jb3VudCcsJ3Jvd19jb3VudCcpCgoKQGRhdGFjbGFzcyhmcm96ZW49VHJ1ZSkKY2xhc3MgU25hcHNob3RTcGVjOgogICAgbWF4X3Jvd3M6IGludAogICAgd2lkdGg6IGludAogICAgcG9zaXRpb25fd2lkdGg6IGludCA9IDEKCiAgICBkZWYgY2hlY2soc2VsZik6CiAgICAgICAgaWYgbWluKHNlbGYubWF4X3Jvd3Msc2VsZi53aWR0aCxzZWxmLnBvc2l0aW9uX3dpZHRoKSA8PSAwOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdOZWVkIGZpeGVkIHBvc2l0aXZlIGJvdW5kZWQgc25hcHNob3QgZGltZW5zaW9ucycpCgoKZGVmIGFsbG9jYXRlX2J1ZmZlcnModG9yY2gsIHNwZWMsICosIGRldmljZSwgZHR5cGUpOgogICAgc3BlYy5jaGVjaygpCiAgICByZXR1cm4gKHRvcmNoLnplcm9zKChzcGVjLm1heF9yb3dzLHNwZWMud2lkdGgpLGRldmljZT1kZXZpY2UsZHR5cGU9ZHR5cGUpLAogICAgICAgICAgICB0b3JjaC5mdWxsKChzcGVjLm1heF9yb3dzLHNwZWMucG9zaXRpb25fd2lkdGgpLC0xLGRldmljZT1kZXZpY2UsZHR5cGU9dG9yY2guaW50NjQpLAogICAgICAgICAgICB0b3JjaC56ZXJvcygoc3BlYy5tYXhfcm93cywpLGRldmljZT1kZXZpY2UsZHR5cGU9dG9yY2guYm9vbCksCiAgICAgICAgICAgIHRvcmNoLnplcm9zKCgxLCksZGV2aWNlPWRldmljZSxkdHlwZT10b3JjaC5pbnQ2NCksCiAgICAgICAgICAgIHRvcmNoLnplcm9zKCgxLCksZGV2aWNlPWRldmljZSxkdHlwZT10b3JjaC5pbnQ2NCksCiAgICAgICAgICAgIHRvcmNoLnplcm9zKCgxLCksZGV2aWNlPWRldmljZSxkdHlwZT10b3JjaC5pbnQ2NCkpCgoKZGVmIHNuYXBzaG90X3R1cGxlKHRvcmNoLCBjYXJyaWVyLCB2YWx1ZXMsIHBvc2l0aW9ucywgdmFsaWQsIGVwb2NoLCBidWZmZXJzKToKICAgIGlmIGxlbihidWZmZXJzKSAhPSBsZW4oRklFTERTKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdFeHBsaWNpdCBzbmFwc2hvdCBUZW5zb3IgdHVwbGUgQUJJIGNoYW5nZWQnKQogICAgb3V0X3ZhbHVlcyxvdXRfcG9zaXRpb25zLG91dF92YWxpZCxvdXRfZXBvY2gsb3V0X2NvdW50LG91dF9yb3dzID0gYnVmZmVycwogICAgcm93cyA9IHZhbHVlcy5zaGFwZVswXQogICAgaWYgKHZhbHVlcy5uZGltICE9IDIgb3Igcm93cyA+IG91dF92YWx1ZXMuc2hhcGVbMF0gb3IgdmFsdWVzLnNoYXBlWzFdICE9IG91dF92YWx1ZXMuc2hhcGVbMV0KICAgICAgICAgICAgb3IgcG9zaXRpb25zLnNoYXBlICE9IChyb3dzLG91dF9wb3NpdGlvbnMuc2hhcGVbMV0pIG9yIHZhbGlkLnNoYXBlICE9IChyb3dzLCkpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ0JvdW5kZWQgc25hcHNob3QgZXh0ZW50L3Bvc2l0aW9ucy92YWxpZCBtaXNtYXRjaCcpCiAgICBpZiBlcG9jaC5zaGFwZSAhPSAoMSwpIG9yIGVwb2NoLmR0eXBlICE9IHRvcmNoLmludDY0OgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ05lZWQgZXhwbGljaXQgZGV2aWNlIGludDY0WzFdIGVwb2NoIFRlbnNvcicpCiAgICBpZiAodmFsdWVzLmR0eXBlICE9IG91dF92YWx1ZXMuZHR5cGUgb3IgcG9zaXRpb25zLmR0eXBlICE9IHRvcmNoLmludDY0IG9yIHZhbGlkLmR0eXBlICE9IHRvcmNoLmJvb2wKICAgICAgICAgICAgb3IgYW55KHRlbnNvci5kZXZpY2UgIT0gb3V0X3ZhbHVlcy5kZXZpY2UgZm9yIHRlbnNvciBpbiAodmFsdWVzLHBvc2l0aW9ucyx2YWxpZCxlcG9jaCwqYnVmZmVycykpKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdBY3R1YWwgc25hcHNob3QgZHR5cGUvZGV2aWNlIGRpZmZlcnMgZnJvbSB0aGUgcHJlYWxsb2NhdGVkIHR1cGxlJykKICAgIG91dF92YWx1ZXMuemVyb18oKTtvdXRfdmFsdWVzWzpyb3dzXS5jb3B5Xyh2YWx1ZXMpCiAgICBvdXRfcG9zaXRpb25zLmZpbGxfKC0xKTtvdXRfcG9zaXRpb25zWzpyb3dzXS5jb3B5Xyhwb3NpdGlvbnMpCiAgICBvdXRfdmFsaWQuemVyb18oKTtvdXRfdmFsaWRbOnJvd3NdLmNvcHlfKHZhbGlkKQogICAgb3V0X3Jvd3MuZmlsbF8ocm93cyk7b3V0X2NvdW50LmFkZF8oMSk7b3V0X2Vwb2NoLmNvcHlfKGVwb2NoKQogICAgIyBFdmVyeSBtdXRhdGVkIFRlbnNvciByZW1haW5zIGFuIGV4cGxpY2l0IGdyYXBoIG91dHB1dC4gT3JpZ2luYWwgY2FycmllcgogICAgIyBhbmQgYnVmZmVyIGFkZHJlc3NlcyBtdXN0IGFsc28gYmUgY2hlY2tlZCBpbiB0aGUgcmVhbCBjb21waWxlZCBTREsgZ2F0ZS4KICAgIHJldHVybiAoY2FycmllciwqYnVmZmVycykKCgpkZWYgcmVzZXRfYWZ0ZXJfcHJpb3JfZmVuY2UoYnVmZmVycyk6CiAgICAjIFRoZSBjYWxsZXIgcGVyZm9ybXMgYSByZWFsIGRldmljZSBmZW5jZSBiZWZvcmUgcmVzZXQ7IG5vIHByZXRlbmQgZmxhZy4KICAgIHZhbHVlcyxwb3NpdGlvbnMsdmFsaWQsZXBvY2gsY291bnQscm93cyA9IGJ1ZmZlcnMKICAgIHZhbHVlcy56ZXJvXygpO3Bvc2l0aW9ucy5maWxsXygtMSk7dmFsaWQuemVyb18oKTtlcG9jaC56ZXJvXygpO2NvdW50Lnplcm9fKCk7cm93cy56ZXJvXygpCg==').decode();filename='/CPU-only-v5/tensor_tuple_transport.py'
module=types.ModuleType('tensor_tuple_transport');module.__file__=filename;sys.modules['tensor_tuple_transport']=module
linecache.cache[filename]=(len(source),None,source.splitlines(keepends=True),filename)
exec(compile(source,filename,'exec'),module.__dict__)
from tensor_tuple_transport import SnapshotSpec,allocate_buffers,snapshot_tuple
program_source=base64.b64decode('ZGVmIHByb2dyYW0oKmFjdHVhbCk6CiAgICAgICAgICAgIHgsd2VpZ2h0LHBvc2l0aW9ucyx2YWxpZCxlcG9jaCxpbmRpY2VzLGNhY2hlPWFjdHVhbFs6N10KICAgICAgICAgICAgcHJvZHVjZXJfaW5wdXRzPWFjdHVhbFs3OjEzXTtyZWFkZXJfaW5wdXRzPWFjdHVhbFsxMzoxOV0KICAgICAgICAgICAgdmFsdWU9dG9yY2gubm4uZnVuY3Rpb25hbC5saW5lYXIoeCx3ZWlnaHQpCiAgICAgICAgICAgIHByb2R1Y2VyX291dHB1dHM9c25hcHNob3RfdHVwbGUodG9yY2gsdmFsdWUsdmFsdWUscG9zaXRpb25zLHZhbGlkLGVwb2NoLHByb2R1Y2VyX2lucHV0cykKICAgICAgICAgICAgdmFsdWU9cHJvZHVjZXJfb3V0cHV0c1swXQogICAgICAgICAgICBjYWNoZS5pbmRleF9jb3B5XygwLGluZGljZXMsdmFsdWUpCiAgICAgICAgICAgIGNvbnN1bWVkPWNhY2hlLmluZGV4X3NlbGVjdCgwLGluZGljZXMpCiAgICAgICAgICAgIHJlYWRlcl9vdXRwdXRzPXNuYXBzaG90X3R1cGxlKHRvcmNoLHZhbHVlLGNvbnN1bWVkLHBvc2l0aW9ucyx2YWxpZCxlcG9jaCxyZWFkZXJfaW5wdXRzKQogICAgICAgICAgICByZXR1cm4gKHJlYWRlcl9vdXRwdXRzWzBdKjIsY2FjaGUsKnByb2R1Y2VyX291dHB1dHNbMTpdLCpyZWFkZXJfb3V0cHV0c1sxOl0p').decode();filename='/CPU-only-v5/program.py'
linecache.cache[filename]=(len(program_source),None,program_source.splitlines(keepends=True),filename)
exec(compile(program_source,filename,'exec'),globals())
records=[]
def shape(value,depth=0):
    if isinstance(value,torch.Tensor):return {'type':type(value).__name__,'shape':list(value.shape),'stride':list(value.stride())}
    if isinstance(value,(tuple,list)):return {'type':type(value).__name__,'length':len(value),'items':[shape(item,depth+1) for item in value[:3]] if depth<2 else []}
    return {'type':type(value).__name__,'is_none':value is None}
class Mode(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):return func(*args,**(kwargs or {}))
class CPUCompiledFxGraph:
    def __init__(self,gm,examples):self.gm=gm;self.examples=examples
    def __call__(self,*args,**kwargs):
        records.append({'boundary':'generated_kernel_entry','runtime_args':shape(args),'kwargs':list(kwargs)})
        torch._C._dynamo.guards.assert_size_stride(args[0],tuple(self.examples[0].shape),tuple(self.examples[0].stride()))
        return self.gm(*args,**kwargs)
def sdk_like_backend(gm,examples):
    def compiler(inner_gm,inner_examples):
        records.append({'boundary':'inner_fw_compile','example_args':shape(inner_examples)})
        return CPUCompiledFxGraph(inner_gm,inner_examples)
    result=aot_module_simplified(gm,examples,fw_compiler=compiler,inference_compiler=compiler,keep_inference_input_mutations=True)
    records.append({'boundary':'sdk_backend_return','boxed':getattr(result,'_boxed_call',None),'signature':str(__import__('inspect').signature(result))})
    return result
def fw(gm,args):
    records.append({'boundary':'outer_fw_compile','example_args':shape(args)})
    sdk=sdk_like_backend(gm,args)
    def boxed(actual,**kwargs):
        records.append({'boundary':'outer_fw_boxed_adapter','runtime_args':shape(actual),'kwargs':list(kwargs)})
        if not isinstance(actual,list) or len(actual)!=len(args) or not all(isinstance(value,torch.Tensor) for value in actual):
            raise TypeError('Expected one boxed list of19 Tensor inputs')
        return sdk(*actual,**kwargs)
    boxed._boxed_call=True
    return boxed
outer=aot_autograd(fw_compiler=fw,keep_inference_input_mutations=False)
def backend(gm,args):
    inner=outer(gm,args)
    records.append({'boundary':'outer_backend_return','boxed':getattr(inner,'_boxed_call',None),'signature':str(__import__('inspect').signature(inner))})
    def invoke(*actual):
        records.append({'boundary':'copymode_invoke','runtime_args':shape(actual),'inner_boxed_now':getattr(inner,'_boxed_call',None)})
        with Mode():return inner(*actual)
    return invoke
spec=SnapshotSpec(6,64)
x=torch.zeros((6,64),dtype=torch.bfloat16);weight=torch.eye(64,dtype=torch.bfloat16)
positions=torch.arange(6,dtype=torch.int64).reshape(6,1);valid=torch.ones((6,),dtype=torch.bool)
epoch=torch.zeros((1,),dtype=torch.int64);indices=torch.arange(6,dtype=torch.int64)
cache=torch.zeros((16,64),dtype=torch.bfloat16)
producer=allocate_buffers(torch,spec,device='cpu',dtype=torch.bfloat16)
reader=allocate_buffers(torch,spec,device='cpu',dtype=torch.bfloat16)
inputs=(x,weight,positions,valid,epoch,indices,cache,*producer,*reader)
receipt={'scope':'CPU double-AOT/current invoke boxing reproduction only; no SDK/NPU','torch':torch.__version__,
         'all_original_inputs_are_tensor':all(isinstance(value,torch.Tensor) for value in inputs),'records':records,
         'npu_jobs':0,'device_audit_passed':False,'safe_formal_receipt':False,'passed':False}
prior=torch._inductor.config.enable_auto_functionalized_v2;torch._inductor.config.enable_auto_functionalized_v2=False
try:
    compiled=torch.compile(program,backend=backend,fullgraph=True,dynamic=False)
    with torch.no_grad():output=compiled(*inputs)
    receipt['passed']=True;receipt['output_count']=len(output)
except BaseException as exc:receipt.update(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
finally:torch._inductor.config.enable_auto_functionalized_v2=prior
assert 'torch_npu' not in sys.modules
print(json.dumps(receipt,indent=2))
