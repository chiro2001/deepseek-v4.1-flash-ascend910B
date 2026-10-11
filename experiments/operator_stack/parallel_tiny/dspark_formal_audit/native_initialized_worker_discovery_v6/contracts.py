"""CPU controller/lifecycle contracts. No torch or SDK import on import."""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import time


VARIANTS = {
    'static_default': (True, False),
    'nonstatic_default': (False, False),
    'static_keep': (True, True),
    'nonstatic_keep': (False, True),
}


def digest(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def read_frozen_json(path, sha256):
    path = Path(path)
    data = path.read_bytes()
    if len(sha256) != 64 or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError('Controller/source manifest differs from the supplied frozen SHA')
    return json.loads(data)


def validate_controller(controller):
    if controller.get('schema') != 'native_initialized_worker_discovery_v6':
        raise ValueError('Need an explicit initialized-worker controller schema')
    if controller.get('operation') != 'run_one_variant_in_new_formal_audit_job':
        raise ValueError('Controller must authorize a new private audit job')
    if controller.get('variant') not in VARIANTS:
        raise ValueError('Need one explicit GraphEx static/nonstatic mutation variant')
    if type(controller.get('generations')) is not int or controller['generations'] < 32:
        raise ValueError('Need at least32 changing hardware generations')
    if (controller.get('tp_world_size') != 8 or controller.get('physical_chips') != list(range(8,16))
            or controller.get('checkpoint_config_sha256') != '40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424'):
        raise ValueError('Controller differs from the unchanged formal TP8/checkpoint scope')
    if not all(isinstance(controller.get(key),str) and controller[key] for key in
               ('audit_job_id','nonce','audit_job_root','driver_output_root','controller_path',
                'output_root','source_manifest_path','source_manifest_sha256','checkpoint_config_path')):
        raise ValueError('Controller job/output/source/checkpoint identity is incomplete')
    job_root=Path(controller['audit_job_root']);driver_root=Path(controller['driver_output_root'])
    metadata_root=Path(controller['controller_path']).parent
    if (not job_root.is_absolute() or not driver_root.is_absolute()
            or job_root.name!=controller['audit_job_id'] or driver_root.parent!=job_root
            or Path(controller['output_root']).parent!=driver_root or metadata_root.parent!=driver_root
            or Path(controller['controller_path']).name!='controller.json'
            or Path(controller['source_manifest_path'])!=metadata_root/'source_manifest.json'):
        raise ValueError('Audit job ID/full output/controller paths are not bound to the same new job')
    processes = controller.get('worker_processes',[])
    if len(processes) != 8 or {item.get('tp_rank') for item in processes} != set(range(8)):
        raise ValueError('Controller needs all eight actual initialized worker identities')
    identities = set()
    for item in processes:
        if (type(item.get('pid')) is not int or item['pid'] <= 0
                or type(item.get('normal_init_return_ns')) is not int or item['normal_init_return_ns'] <= 0
                or not item.get('boot_id')):
            raise ValueError('Actual initialized worker PID/boot/init-return identity is missing')
        if (item.get('loaded_abi_proofs_path')!=str(metadata_root/'loaded_abi_proofs'/f"rank{item['tp_rank']}.json")
                or not isinstance(item.get('loaded_abi_proofs_sha256'),str)
                or len(item['loaded_abi_proofs_sha256'])!=64):
            raise ValueError('Actual initialized rank needs controller-bound mapped ABI proof path/SHA')
        identities.add((item['boot_id'],item['pid']))
    if len(identities) != 8:
        raise ValueError('Controller worker ranks cannot reuse the same process identity')
    return {'controller_structure_passed': True, 'device_audit_passed': False,
            'safe_formal_receipt': False}


def validate_initialized_facts(controller, facts, process_identity):
    validate_controller(controller)
    rank = facts.get('tp_rank')
    if type(rank) is not int or rank not in range(8) or facts.get('tp_world_size') != 8:
        raise ValueError('Actual TP rank/world does not match the formal controller')
    process = next(item for item in controller['worker_processes'] if item['tp_rank'] == rank)
    if any(process[key] != process_identity[key] for key in ('pid','boot_id','normal_init_return_ns')):
        raise ValueError('This worker is not the new initialized process armed by the controller')
    if facts.get('current_initialized') is not True:
        raise ValueError('Normal NPUWorker must already be initialized; the gate cannot initialize it')
    if (any(type(facts.get(key)) is not int or facts[key]<=0 for key in ('context_ptr','stream_ptr'))
            or any(type(facts.get(key)) is not int for key in
                   ('runtime_logical_device','torch_current_device','worker_device_index','physical_device'))
            or facts.get('runtime_logical_device') != facts.get('torch_current_device')
            or facts.get('worker_device_index') != facts.get('torch_current_device')):
        raise ValueError('Current context/stream/worker logical device is missing or mismatched')
    if facts.get('physical_device') != controller['physical_chips'][rank]:
        raise ValueError('Actual runtime physical mapping differs from the assigned TP rank; no reselection')
    if facts.get('worker_rank') != facts.get('global_rank'):
        raise ValueError('Actual worker rank differs from the loaded TP group global rank')
    if facts.get('runner_idle') is not True or facts.get('capturing') is not False:
        raise ValueError('Gate needs an idle worker outside model execution/capture')
    return {'initialized_scope_structure_passed': True,
            'device_audit_passed': False, 'safe_formal_receipt': False}


def assert_borrowed_scope_stable(before, after):
    fields = ('context_ptr','stream_ptr','runtime_logical_device','torch_current_device',
              'worker_device_index','physical_device','tp_rank','tp_world_size','worker_rank','global_rank')
    if after.get('current_initialized') is not True or any(before[key] != after.get(key) for key in fields):
        raise ValueError('Borrowed worker context/stream/device/rank changed; gate cannot repair it')
    return {'borrowed_scope_structure_stable': True, 'device_audit_passed': False}


@dataclass
class WorkerGateLifecycle:
    worker_object_id: int
    pid: int = field(default_factory=os.getpid)
    boot_id: str = field(default_factory=lambda: Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    normal_init_return_ns: int = field(default_factory=time.monotonic_ns)
    state: str = 'prepared_after_normal_init'
    nonce: str | None = None
    resources: object = None
    _lock: object = field(default_factory=threading.Lock,repr=False)

    def identity(self):
        return {'pid':self.pid,'boot_id':self.boot_id,'normal_init_return_ns':self.normal_init_return_ns}

    def begin(self, worker, nonce):
        with self._lock:
            if id(worker) != self.worker_object_id or os.getpid() != self.pid:
                raise ValueError('Lifecycle is bound to a different actual worker process/object')
            if self.state != 'prepared_after_normal_init':
                raise ValueError('One variant per fresh initialized worker; no retry or process reuse')
            self.nonce = nonce
            self.state = 'running'

    def finish(self, passed, resources):
        with self._lock:
            if self.state != 'running':
                raise ValueError('Worker gate is not running')
            # Own only probe resources. The normal SDK owns context and stream.
            # Retain graph/tensors until this fresh audit process terminates.
            self.resources = resources
            self.state = 'terminal_passed' if passed else 'terminal_failed'


def validate_export(payload, *, generation, expected_values, expected_positions, expected_writes=1):
    if payload.get('epoch') != generation:
        raise ValueError('Stale/missing actual device epoch')
    if payload.get('copy_count') != expected_writes:
        raise ValueError('Missing/duplicate actual device GM writes')
    if (payload.get('rows') != len(expected_values) or payload.get('values') != expected_values
            or payload.get('positions') != expected_positions or payload.get('valid') != [True]*len(expected_values)):
        raise ValueError('Actual bounded payload/extent/positions/valid values differ')
    return {'payload_structure_passed':True,'device_audit_passed':False,'safe_formal_receipt':False}


def validate_copy_aliases(buffers, calls, returned_buffers, borrowed_stream, *, marker_sources):
    fields = ('values','positions','valid','epoch','copy_count','row_count')
    if set(buffers) != {f'{owner}.{field}' for owner in ('producer','reader') for field in fields}:
        raise ValueError('Need all original producer/reader Tensor tuple buffer layouts')
    required = {key for key in buffers if key.endswith(('.values','.positions','.valid','.epoch'))}
    if any(type(value.get('ptr')) is not int or value['ptr'] <= 0 or not value.get('device','').startswith('npu:')
           or not value.get('shape') or len(value['shape']) != len(value.get('stride',[])) for value in buffers.values()):
        raise ValueError('Original pointer/layout is not device allocation metadata')
    pointers = {value['ptr']:value for value in buffers.values()}
    payload_pointers = {buffers[key]['ptr'] for key in required}
    marker_pointers = {buffers[key]['ptr'] for key in required if not key.endswith('.values')}
    if set(marker_sources) != marker_pointers:
        raise ValueError('Need original input sources for all positions/valid/epoch copies')
    if len(pointers) != len(buffers):
        raise ValueError('Distinct owned buffers share an unexpected original address')
    seen = set()
    for call in calls:
        destination = call.get('destination',{})
        ptr = destination.get('ptr')
        if (call.get('phase') != 'cold-capture' or call.get('stream') != borrowed_stream
                or ptr not in payload_pointers or destination != pointers[ptr]
                or call.get('logical_destination') != destination):
            raise ValueError('Actual captured copy uses a temporary/nonowned/changed destination or stream')
        source=call.get('source',{})
        if (call.get('op') not in ('aten.copy_.default','aten.copy.default')
                or source.get('device') != destination['device'] or source.get('ptr',0) <= 0
                or source.get('dtype') != destination.get('dtype')):
            raise ValueError('No real native device copy evidence')
        if ptr in marker_sources and source != marker_sources[ptr]:
            raise ValueError('Positions/valid/epoch source is a functionalization temporary or copyback')
        seen.add(ptr)
    if not required or {buffers[key]['ptr'] for key in required}-seen:
        raise ValueError('Missing actual payload/positions/valid/epoch copy destinations')
    if returned_buffers != buffers:
        raise ValueError('Returned Tensor tuple does not preserve the original owned pointers/layout')
    return {'actual_copy_alias_structure_passed':True,'device_audit_passed':False,'safe_formal_receipt':False}
