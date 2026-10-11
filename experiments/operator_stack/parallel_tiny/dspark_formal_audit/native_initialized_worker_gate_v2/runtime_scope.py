"""Read existing initialized SDK context/mapping; never initialize or select it."""
import ctypes
import inspect
import os
from pathlib import Path
import sys

if __package__:
    from .contracts import digest,read_frozen_json
else:
    from contracts import digest,read_frozen_json


CANDIDATE_FILES = ('contracts.py','runtime_scope.py','tensor_tuple_transport.py',
                   'initialized_worker_gate.py','worker_hook.py','hook_tuple_adapters.py',
                   'init_only_driver.py','sdk_worker.py','controller_cpu.py')


def already_imported_worker_modules():
    names = ('torch','torch_npu','vllm.distributed')
    if any(name not in sys.modules for name in names):
        raise ValueError('Normal NPUWorker imports must precede this initialized-worker gate')
    torch,npu,distributed = (sys.modules[name] for name in names)
    if not npu.npu.is_initialized():
        raise ValueError('Normal worker NPU context is not initialized; no self-initialization allowed')
    return torch,npu,distributed


def guard_worker_sources(worker, controller, torch, npu, distributed):
    manifest = read_frozen_json(controller['source_manifest_path'],controller['source_manifest_sha256'])
    candidates = manifest.get('candidate_sources',{})
    if set(candidates) != set(CANDIDATE_FILES):
        raise ValueError('Need the exact initialized-worker candidate source set')
    root = Path(__file__).parent
    for name in CANDIDATE_FILES:
        if digest(root/name)['sha256'] != candidates[name]:
            raise ValueError('Candidate source SHA mismatch: '+name)
    actual = {
        'torch': torch.__file__, 'torch_npu': npu.__file__,
        'npu_worker_init': inspect.getfile(inspect.unwrap(worker._init_device)),
        'npu_runner': inspect.getfile(type(worker.model_runner)),
        'tp_group_getter': inspect.getfile(inspect.unwrap(distributed.get_tp_group)),
        'tp_rank_getter': inspect.getfile(inspect.unwrap(distributed.get_tensor_model_parallel_rank)),
    }
    sources = manifest.get('sdk_sources',{})
    required = set(actual)|{'npugraph_ex','npu_fx_compiler'}
    if set(sources) != required:
        raise ValueError('Actual native worker/rank/SDK source manifest is incomplete')
    receipts = {}
    for name,entry in sources.items():
        path = Path(actual.get(name,entry['path']))
        receipt = digest(path)
        if receipt['path'] != str(Path(entry['path']).resolve()) or receipt['sha256'] != entry['sha256']:
            raise ValueError('Actual native worker/rank/SDK source mismatch: '+name)
        receipts[name] = receipt
    if digest(controller['checkpoint_config_path'])['sha256'] != controller['checkpoint_config_sha256']:
        raise ValueError('Formal checkpoint config SHA differs from the controller')
    return manifest,receipts


def guard_formal_worker_config(worker):
    config = worker.vllm_config
    hf = config.model_config.hf_text_config
    spec = config.speculative_config
    if (hf.hidden_size != 5120 or hf.num_hidden_layers != 40
            or hf.n_routed_experts != 384 or hf.num_experts_per_tok != 6
            or list(hf.engram_layer_ids) != [1,14] or hf.num_nextn_predict_layers != 3
            or list(hf.dspark_target_layer_ids) != [37,38,39] or hf.dspark_block_size != 5
            or spec is None or spec.method != 'dspark' or spec.num_speculative_tokens != 5
            or config.load_config.load_format == 'dummy'
            or config.scheduler_config.async_scheduling
            or config.parallel_config.tensor_parallel_size != 8
            or config.parallel_config.pipeline_parallel_size != 1
            or config.parallel_config.data_parallel_size != 1):
        raise ValueError('Actual initialized worker differs from unchanged formal TP8/full40/DSparkK5')
    if (os.environ.get('HCCL_DETERMINISTIC') != 'strict'
            or os.environ.get('STACK_PHYSICAL_CHIPS') != '8,9,10,11,12,13,14,15'
            or os.environ.get('STACK_TP8_ARM') != 'tp8base'
            or os.environ.get('STACK_FP32_DECODE_REDUCTION') != '0'):
        raise ValueError('Actual worker process differs from the formal deterministic environment')


def describe_source_metadata(worker, torch, npu, distributed, library_paths):
    """Read files/source identity only; do not dlopen or query a device mapping."""
    if set(library_paths)!={'acl','runtime'}:
        raise ValueError('Controller supplies exact ACL/runtime file paths; no library-path fallback')
    root=Path(__file__).parent
    nge_root=Path(npu.__file__).parent/'dynamo/npugraph_ex'
    paths={'torch':torch.__file__,'torch_npu':npu.__file__,
           'npu_worker_init':inspect.getfile(inspect.unwrap(worker._init_device)),
           'npu_runner':inspect.getfile(type(worker.model_runner)),
           'tp_group_getter':inspect.getfile(inspect.unwrap(distributed.get_tp_group)),
           'tp_rank_getter':inspect.getfile(inspect.unwrap(distributed.get_tensor_model_parallel_rank)),
           'npugraph_ex':nge_root/'__init__.py','npu_fx_compiler':nge_root/'npu_fx_compiler.py'}
    return {'candidate_sources':{name:digest(root/name)['sha256'] for name in CANDIDATE_FILES},
            'sdk_sources':{name:digest(path) for name,path in paths.items()},
            'libraries':{name:digest(path) for name,path in library_paths.items()}}


class AlreadyLoadedRuntime:
    """RTLD_NOLOAD handles only; a missing already-loaded library is terminal."""
    def __init__(self, libraries):
        if set(libraries) != {'acl','runtime'} or not hasattr(os,'RTLD_NOLOAD'):
            raise ValueError('Need exact already-loaded ACL/runtime ABI paths and NOLOAD support')
        self.receipts = {}
        self.handles = {}
        for name,entry in libraries.items():
            receipt = digest(entry['path'])
            if receipt['sha256'] != entry['sha256']:
                raise ValueError('Existing worker SDK ABI library SHA mismatch: '+name)
            stat=Path(receipt['path']).stat()
            mapped=[]
            for line in Path('/proc/self/maps').read_text().splitlines():
                columns=line.split(None,5)
                if len(columns)!=6 or columns[5]!=receipt['path']:continue
                major,minor=(int(value,16) for value in columns[3].split(':'))
                if (int(columns[4])==stat.st_ino and major==os.major(stat.st_dev) and minor==os.minor(stat.st_dev)):
                    mapped.append(columns[0])
            if not mapped:
                raise ValueError('Frozen SDK ABI file is not the actual already-mapped inode: '+name)
            receipt['mapped_inode']=stat.st_ino;receipt['mapped_segments']=mapped
            # Never fall back to a normal CDLL load or another toolkit path.
            self.handles[name] = ctypes.CDLL(receipt['path'],mode=os.RTLD_NOW|os.RTLD_NOLOAD)
            self.receipts[name] = receipt
        acl, runtime = self.handles['acl'], self.handles['runtime']
        self.get_context = acl.aclrtGetCurrentContext
        self.get_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.get_context.restype = ctypes.c_int
        self.get_device = acl.aclrtGetDevice
        self.get_device.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.get_device.restype = ctypes.c_int
        self.get_physical = runtime.rtGetDevicePhyIdByIndex
        self.get_physical.argtypes = [ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
        self.get_physical.restype = ctypes.c_int

    def snapshot(self, worker, torch, npu, distributed):
        if not npu.npu.is_initialized():
            raise ValueError('Existing normal worker initialization was lost; no repair')
        context = ctypes.c_void_p()
        logical = ctypes.c_int32()
        physical = ctypes.c_uint32()
        code = self.get_context(ctypes.byref(context))
        if code:raise RuntimeError('aclrtGetCurrentContext failed on the existing context: '+str(code))
        if not context.value:
            raise ValueError('Current runtime context is null; the gate cannot create one')
        code = self.get_device(ctypes.byref(logical))
        if code:raise RuntimeError('aclrtGetDevice failed on the existing context: '+str(code))
        code = self.get_physical(logical.value,ctypes.byref(physical))
        if code:raise RuntimeError('rtGetDevicePhyIdByIndex failed: '+str(code))
        tp = distributed.get_tp_group()
        rank = distributed.get_tensor_model_parallel_rank()
        if rank != tp.rank_in_group:
            raise ValueError('Actual loaded TP group rank getter disagrees with the group')
        stream = torch.npu.current_stream()
        capture_query = getattr(torch.npu,'is_current_stream_capturing',None)
        if capture_query is None:
            raise ValueError('Actual SDK cannot report current stream capture state')
        counter = getattr(worker.model_runner,'_formal_dspark_counter',None)
        idle = (worker.model_runner.execute_model_state is None
                and (counter is None or counter.tag is None))
        facts = {'current_initialized':True,'context_ptr':int(context.value),
                 'runtime_logical_device':int(logical.value),'physical_device':int(physical.value),
                 'torch_current_device':int(torch.npu.current_device()),
                 'worker_device_index':worker.device.index,'stream_ptr':int(stream.npu_stream),
                 'tp_rank':int(rank),'tp_world_size':int(tp.world_size),'worker_rank':int(worker.rank),
                 'global_rank':int(tp.ranks[rank]),'runner_idle':idle,'capturing':bool(capture_query())}
        return facts,stream
