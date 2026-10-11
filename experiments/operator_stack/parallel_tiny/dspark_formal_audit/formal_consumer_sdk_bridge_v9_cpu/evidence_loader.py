"""Read-only TP8 dual SDK gate protocol. No Torch, SDK, RPC or device imports."""
import ast
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from types import MappingProxyType


SOURCE_CONTRACT_SHA256 = '5f9904d4d826dc33a2617d2a661b9986c695f74777c97bcac5540dc6ccdb88b9'
FALSE_RECEIPTS = ('device_audit_passed','safe_formal_receipt','formal_model_validated',
                  'full_consumer_audit_passed','actual_prepack_consumer_order_validated')
SEMANTICS = ('payload_equal','positions_equal','device_write_count_one_each',
             'native_cache_write_read_equal','returned_value_equal','original_buffers_stable',
             'original_owned_destination_seals_stable','borrowed_scope_stable')
PROOF_FIELDS = ('pid','resolved_path','segments','map_files_link','mapped_file_stat','current_path_stat',
                'mapped_file_sha256','mapped_file_bytes','overlay_mount','executable_memory','symbol_addresses')


class EvidenceError(ValueError):
    def __init__(self, code, message):
        super().__init__(code+': '+message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise EvidenceError(code,message)


def positive_int(value):
    return type(value) is int and value > 0


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key,value in pairs:
        require(key not in result,'json','duplicate JSON key: '+key)
        result[key] = value
    return result


def parse_json(data):
    try:
        return json.loads(data,object_pairs_hook=_pairs,
                          parse_constant=lambda value:require(False,'json','non-finite JSON value'))
    except (UnicodeError,json.JSONDecodeError) as exc:
        raise EvidenceError('json','invalid UTF-8 JSON') from exc


def immutable(value):
    if isinstance(value,dict):
        return MappingProxyType({key:immutable(item) for key,item in value.items()})
    if isinstance(value,list):
        return tuple(immutable(item) for item in value)
    return value


def plain(value):
    if is_dataclass(value):
        return {field.name:plain(getattr(value,field.name)) for field in fields(value)}
    if isinstance(value,(dict,MappingProxyType)):
        return {key:plain(item) for key,item in value.items()}
    if isinstance(value,tuple):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class FileReference:
    path: str
    bytes: int
    sha256: str
    raw_bytes_transferred: bool = True


@dataclass(frozen=True)
class RankGateEvidence:
    rank: int
    physical_chip: int
    identity: object
    result: FileReference
    loaded_abi_proofs: FileReference
    generated_kernels: tuple
    initialized_scope: object
    native_scope: object
    owner_seals: object
    epochs: int
    copy_classes: object


@dataclass(frozen=True)
class JobGateEvidence:
    job_dir: str
    job_id: str
    remote_job_root: str
    variant: str
    archive_receipt: FileReference
    fetch_manifest: FileReference
    driver_compact: FileReference
    original_driver: FileReference
    controller: FileReference
    source_manifest: FileReference
    cleanup: FileReference
    driver_identity: object
    ranks: tuple
    protocol_passed: bool = True
    actual_sdk_gate_passed: bool = False
    device_audit_passed: bool = False
    full_consumer_audit_passed: bool = False
    safe_formal_receipt: bool = False
    formal_model_validated: bool = False
    actual_prepack_consumer_order_validated: bool = False

    def to_dict(self):
        return plain(self)

    as_dict = to_dict


@dataclass(frozen=True)
class Tp8SdkPairEvidence:
    static: JobGateEvidence
    nonstatic: JobGateEvidence
    source_root: str
    source_freeze: FileReference
    source_contract: FileReference
    protocol_passed: bool = True
    npu_jobs: int = 0
    actual_sdk_gate_passed: bool = False
    device_audit_passed: bool = False
    full_consumer_audit_passed: bool = False
    safe_formal_receipt: bool = False
    formal_model_validated: bool = False
    actual_prepack_consumer_order_validated: bool = False

    def to_dict(self):
        return plain(self)

    as_dict = to_dict


def _file(path, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),'file','missing/nonregular/symlink evidence: '+str(path))
    data = path.read_bytes()
    reference = FileReference(str(path.resolve()),len(data),sha(data))
    if expected is not None:
        require(reference.bytes == expected['bytes'] and reference.sha256 == expected['sha256'],
                'hash','evidence bytes/SHA mismatch: '+str(path))
    return data,reference


def source_policy(source_root=None):
    root = Path(source_root) if source_root is not None else Path(__file__).parent.parent/'native_initialized_worker_discovery_v8'
    root = root.resolve()
    contract_data,contract_ref = _file(Path(__file__).with_name('source_contract.json'))
    require(contract_ref.sha256 == SOURCE_CONTRACT_SHA256,'source','bridge source contract changed')
    contract = parse_json(contract_data)
    data,freeze_ref = _file(root/'freeze.json')
    require(freeze_ref.sha256 == contract['source_freeze_sha256'],'source','need exact frozen v8 source package')
    freeze = parse_json(data)
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    require(actual == set(freeze['files'])|{'freeze.json','SHA256SUMS'},'source','v8 frozen source file set changed')
    require(not any(path.is_symlink() for path in root.rglob('*')),'source','v8 source has symlink')
    for name,digest in freeze['files'].items():
        require(sha((root/name).read_bytes()) == digest,'source','v8 source SHA changed: '+name)
    require(sha((root/'SHA256SUMS').read_bytes()) == contract['source_checksum_sha256'],
            'source','v8 checksum file changed')
    module_name = '_frozen_v8_tp8_bridge_contracts_'+freeze['files']['contracts.py']
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name,root/'contracts.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Execute only the hash-checked stdlib contract bytes. Avoid writing
        # __pycache__ into the immutable v8 source package.
        exec(compile((root/'contracts.py').read_bytes(),str(root/'contracts.py'),'exec'),module.__dict__)
    else:
        module = sys.modules[module_name]
    return root,freeze_ref,contract_ref,freeze,contract,module


def canonical_relative(name):
    require(isinstance(name,str),'file','evidence name must be a string')
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and '..' not in path.parts and name not in ('','.'),
            'file','noncanonical/outside evidence path')
    return path


class JobFiles:
    def __init__(self, root, anchor):
        self.root = Path(root).resolve()
        require(self.root.is_dir(),'missing_job','missing complete SDK job: '+str(self.root))
        data,self.archive_ref = _file(self.root/'archive_receipt.json')
        require(self.archive_ref.sha256 == anchor['archive_receipt_sha256'],'collection','collection receipt differs from trusted root anchor')
        self.archive = parse_json(data)
        data,self.manifest_ref = _file(self.root/'fetch_manifest.json')
        require(self.manifest_ref.sha256 == anchor['fetch_manifest_sha256'] == self.archive['fetch_manifest_sha256'],
                'collection','fetch manifest differs from trusted root anchor')
        self.manifest = parse_json(data)
        require(self.archive['job'] == anchor['job_id'] and self.archive['run_exit'] == 0,
                'terminal','archive job/exit is wrong')
        require(self.archive['files_sha256_verified'] is True,'collection','collection SHA verification missing')
        for name,entry in self.manifest.items():
            canonical_relative(name)
            require(positive_int(entry['bytes']) or entry['bytes'] == 0,'collection','invalid collected file length')
            require(isinstance(entry['sha256'],str) and re.fullmatch('[0-9a-f]{64}',entry['sha256']),
                    'collection','invalid collected file SHA')
        self.cache = {};self.refs = {}

    def file(self, name):
        canonical_relative(name)
        require(name in self.manifest,'file','required original evidence absent: '+name)
        if name not in self.cache:
            path = self.root/name
            require(path.resolve().is_relative_to(self.root),'file','evidence escapes job directory')
            require(not any(parent.is_symlink() for parent in [path,*path.parents] if parent.is_relative_to(self.root)),
                    'file','symlink evidence ancestor')
            data,reference = _file(path,self.manifest[name])
            self.cache[name] = data;self.refs[name] = reference
        return self.cache[name]

    def json(self, name):
        return parse_json(self.file(name))


def _bf16(value):
    bits = struct.unpack('>I',struct.pack('>f',float(value)))[0]
    bits = (bits+0x7fff+((bits >> 16)&1))&0xffff0000
    return struct.unpack('>f',struct.pack('>I',bits))[0]


def expected_values(generation, rank):
    values = [_bf16(_bf16(_bf16(index)/512+generation/8)+rank/32) for index in range(384)]
    return [values[row*64:(row+1)*64] for row in range(6)]


def false_formal_receipts(receipt):
    require(all(receipt.get(name) is False for name in FALSE_RECEIPTS),'authority','small SDK gate claimed consumer/formal authority')
    require(receipt.get('performance_claim') is None,'authority','small gate has a performance claim')


def validate_native_scope(rank, receipt, factory):
    baseline = receipt['initialized_worker_scope'];native = receipt['native_capture_scope']
    required = {'current_initialized','context_ptr','runtime_logical_device','physical_device','torch_current_device',
                'worker_device_index','stream_ptr','tp_rank','tp_world_size','worker_rank','global_rank','runner_idle','capturing'}
    require(set(baseline) == required,'scope','baseline context/rank proof missing or changed')
    require(native['init_return_baseline'] == baseline == native['restored_baseline'],
            'scope','init-return baseline not strictly restored')
    require(native['owner'] == factory['qualified_function'] and native['source'] == factory,
            'source','native capture owner/source changed')
    for key in ('native_context_creates_stream','normal_exit_restoration_verified','native_context_exit_completed'):
        require(native.get(key) is True,'scope','native lifecycle proof missing: '+key)
    for key in ('init_return_stream_borrowed_unchanged','gate_stream_constructor_called','gate_stream_patch_applied'):
        require(native.get(key) is False,'scope','native stream ownership falsely described: '+key)
    for key in ('context_object_id','stream_object_id','native_stream_ptr','default_stream_ptr'):
        require(positive_int(native.get(key)),'scope','missing actual native handle: '+key)
    actual = native['native_scope']
    require(set(actual) == required and all(actual[key] == baseline[key] for key in required-{'stream_ptr'}),
            'scope','native capture changed original context/device/rank')
    require(positive_int(baseline['context_ptr']) and positive_int(baseline['stream_ptr']),
            'scope','null/bool context or baseline stream')
    require(actual['stream_ptr'] == native['native_stream_ptr'] != native['default_stream_ptr']
            and actual['stream_ptr'] != baseline['stream_ptr'],'scope','native current/default/init stream confused')
    require(actual['current_initialized'] is True and actual['runner_idle'] is True and actual['capturing'] is False,
            'scope','native scope not initialized/idle/noncapturing')
    require(all(type(actual[key]) is int and actual[key] == rank for key in
                ('runtime_logical_device','torch_current_device','worker_device_index','tp_rank','worker_rank','global_rank'))
            and actual['physical_device'] == rank+8 and actual['tp_world_size'] == 8,
            'scope','actual TP8 rank/device assignment changed')
    return baseline,native


def validate_rank_payload(rank, receipt, controller, source_manifest, profile, upstream, proof, worker):
    false_formal_receipts(receipt)
    require(receipt['passed'] is True and receipt['stage_semantics_passed'] is True
            and receipt['actual_copy_alias_gate_passed'] is True and receipt['alias_error'] is None,
            'rank_gate','rank semantic/copy/alias gate not passed')
    require(receipt['stage'] == 'graphex' and receipt['variant'] == controller['variant']
            and receipt['static_kernel'] is (controller['variant'] == 'static_keep')
            and receipt['outer_keep_inference_input_mutations'] is True,
            'variant','need matching actual GraphEx static/nonstatic keep')
    require(receipt['compiler_options'] == profile['compiler_options_by_variant'][controller['variant']],
            'variant','actual SDK compiler policy differs from the source-verified variant')
    require(receipt['phase'] == 'terminal-complete' and receipt['worker_lifecycle'] == 'terminal_passed',
            'terminal','rank lifecycle not terminal passed')
    identity = receipt['process_identity']
    require({key:worker[key] for key in ('pid','boot_id','normal_init_return_ns')} == identity,
            'identity','rank/controller fresh identity mismatch')
    require(receipt['controller_nonce'] == controller['nonce']
            and receipt['audit_job_id'] == controller['audit_job_id']
            and receipt['audit_job_root'] == controller['audit_job_root']
            and receipt['driver_output_root'] == controller['driver_output_root']
            and receipt['controller_output_root'] == controller['output_root'],
            'binding','rank result has another job/controller binding')
    require(receipt['sources'] == profile['sdk_sources']
            and receipt['torch'] == profile['torch'] and receipt['torch_npu'] == profile['torch_npu']
            and receipt['graphex_source_identity'] == profile['graphex_source_identity'],
            'source','actual rank SDK/source getter identity differs')
    baseline,native = validate_native_scope(rank,receipt,profile['native_capture_factory'])
    upstream.validate_initialized_facts(controller,baseline,identity)
    require(set(proof) == set(profile['libraries']) == set(receipt['runtime_library_sources']),
            'abi','mapped ABI proof/library set missing')
    for name,entry in receipt['runtime_library_sources'].items():
        require({key:entry[key] for key in ('path','bytes','sha256')} == profile['libraries'][name],
                'abi','mapped ABI source differs')
        actual = entry['loaded_abi_proof']
        require(all(actual.get(key) == proof[name].get(key) for key in PROOF_FIELDS),
                'abi','mapped ABI changed after controller discovery')
        require(actual['pid'] == identity['pid'] and actual['mapped_file_sha256'] == entry['sha256']
                and actual['mapped_file_bytes'] == entry['bytes'] and actual['resolved_path'] == entry['path'],
                'abi','mapped ABI owner/path/content proof is wrong')
        require(actual.get('device_api_calls') == 0 and actual.get('new_library_loads') == 0,
                'abi','ABI provenance performed a new load/device API')
        require(actual['executable_memory'] and all(row['byte_equal'] is True and positive_int(row['bytes'])
                and row['memory_sha256'] == row['mapped_file_segment_sha256'] for row in actual['executable_memory']),
                'abi','live executable memory differs from mapped source')
        require(actual['symbol_addresses'] and all(positive_int(value) for value in actual['symbol_addresses'].values()),
                'abi','actual ABI symbol addresses missing')
    buffers = receipt['buffers_before']
    require(receipt['buffers_after_capture'] == buffers,'owner','original tuple layouts changed after capture')
    markers = receipt['marker_sources_before_capture']
    require(all(isinstance(key,str) and re.fullmatch('[1-9][0-9]*',key) for key in markers),
            'owner','noncanonical marker source pointer key')
    markers = {int(key):value for key,value in markers.items()}
    require(receipt['capture_calls'] == [call for call in receipt['copy_calls'] if call['phase'] == 'cold-capture'],
            'copy','capture copies are not the complete original runtime trace')
    try:
        structure = upstream.validate_copy_aliases(buffers,receipt['capture_calls'],receipt['returned_buffers_after_capture'],
            native['native_stream_ptr'],marker_sources=markers,
            owned_seals_before=receipt['owned_copy_destination_seals_before'],
            owned_seals_after=receipt['owned_copy_destination_seals_after_capture'],
            returned_native_cache=receipt['returned_native_cache_after_capture'])
    except (ValueError,KeyError,TypeError) as exc:
        raise EvidenceError('owner','strict v8 owner/marker/copy predicate rejected evidence') from exc
    require(structure == receipt['copy_alias_structure'],'copy','reported copy classification differs from original trace')
    abi = receipt['runtime_call_abi']
    require(len(abi) == 2 and [row['boundary'] for row in abi] ==
            ['dynamo_unboxed_entry','outer_aot_boxed_fw_to_sdk_unboxed'],
            'call_abi','missing real warmup Tensor ABI boundaries')
    require(all(row['arity'] == 19 and row['container'] == 'tuple' and row['tensor_mask'] == [True]*19
                and row['keyword_names'] == [] and len(row['object_ids']) == 19 for row in abi)
            and abi[0]['object_ids'] == abi[1]['object_ids'],'call_abi','Tensor arguments replaced/reordered/not19')
    require(abi[1]['tensor_objects_replaced'] is False and abi[1]['tensor_order_changed'] is False
            and abi[1]['kwargs_filtered'] is False and abi[1]['sdk_call_abi'] == 'unboxed',
            'call_abi','boxed adapter ABI/kwargs policy changed')
    epochs = receipt['generations']
    require(len(epochs) == controller['generations'] == 32
            and [row['epoch'] for row in epochs] == list(range(1,33)),'epoch','need original contiguous32 epochs')
    last_fence = identity['normal_init_return_ns']
    for row in epochs:
        require(all(row.get(key) is True for key in SEMANTICS),'epoch','epoch semantic/seal/scope proof missing')
        fence = row['device_fence_monotonic_ns']
        require(positive_int(fence) and fence > last_fence,'fence','device fence missing/duplicate/before init')
        last_fence = fence
        exports = row['exports']
        require(len(exports) == 2 and [item['owner'] for item in exports] == ['producer','reader'],
                'epoch','both actual producer/reader exports missing')
        generation = row['epoch'];expected = expected_values(generation,rank)
        for item in exports:
            require(all(type(value) in (int,float) for values in item['values'] for value in values),
                    'epoch','payload contains bool/non-numeric values')
            try:
                upstream.validate_export(item,generation=generation,expected_values=expected,
                                         expected_positions=[[generation*8+index] for index in range(6)])
            except (ValueError,KeyError,TypeError) as exc:
                raise EvidenceError('epoch','original export values/positions/epoch/count invalid') from exc
    return baseline,native,structure


def validate_job_payload(driver, controller, cleanup, source_manifest, ranks, proofs, freeze, profile, upstream):
    """Protocol core for CPU rejection tests. File-backed callers use load_tp8_sdk_pair."""
    false_formal_receipts(driver)
    require(driver['passed'] is True and driver['worker_exit_codes_complete'] is True,
            'terminal','whole driver job did not pass with complete exitcodes')
    require(driver['worker_exit_codes'] == {str(rank):0 for rank in range(8)},'terminal','need eight actual reaped zeros')
    require(cleanup['all_terminal'] is True and cleanup['pending_owned_pids'] == []
            and cleanup['uncaptured_owned_pids'] == [] and cleanup['worker_exit_codes'] == driver['worker_exit_codes']
            and cleanup['normal_success_grace_seconds'] == 45,'terminal','owned cleanup incomplete/not current bounded grace')
    require(driver['owned_worker_cleanup'] == cleanup,'terminal','driver and original cleanup disagree')
    require(set(ranks) == set(proofs) == set(range(8)),'rank','missing/extra/duplicate rank results or ABI proofs')
    require(controller['variant'] in ('static_keep','nonstatic_keep') and driver['variant'] == controller['variant'],
            'variant','need static_keep/nonstatic_keep current jobs')
    upstream.validate_controller(controller)
    require(source_manifest['candidate_sources'] == freeze['candidate_source_sha256']
            and source_manifest['sdk_sources'] == profile['sdk_sources']
            and source_manifest['native_capture_factory'] == profile['native_capture_factory']
            and source_manifest['libraries'] == profile['libraries'],'source','current candidate/SDK/native/ABI source differs')
    require(driver['bootstrap_sources'] == profile['bootstrap_sources'],'source','normal bootstrap source changed')
    require(driver['model_load_called'] is False,'scope','small gate loaded model weights')
    require(driver['model_kwargs']['worker_cls'].endswith('native_initialized_worker_discovery_v8.sdk_worker.InitializedSdkGateWorker')
            and driver['model_kwargs']['tensor_parallel_size'] == 8
            and driver['model_kwargs']['gpu_memory_utilization'] == 0.7
            and driver['model_kwargs']['load_format'] != 'dummy','scope','formal init-only TP8 config changed')
    for key in ('audit_job_id','audit_job_root','driver_output_root','controller_path'):
        require(driver[key] == controller[key],'binding','driver/controller path or job mismatch')
    require(driver['controller_nonce'] == controller['nonce'],'binding','driver/controller nonce mismatch')
    workers = controller['worker_processes'];rows = cleanup['workers']
    require(len(workers) == len(rows) == 8 and {row['rank'] for row in rows} == set(range(8)),
            'identity','need eight unique captured cleanup child identities')
    owner = cleanup['owner']
    require(all(positive_int(owner[key]) for key in ('pid','ppid','start_ticks')) and owner['boot_id']
            and re.fullmatch('[0-9a-f]{64}',owner['cmdline_sha256']), 'identity','driver birth/cmdline identity absent')
    require(len({worker['pid'] for worker in workers}) == 8
            and driver['owned_worker_pids'] == [worker['pid'] for worker in workers],
            'identity','worker PID reused/owned registry differs')
    require(owner['pid'] not in driver['owned_worker_pids'] and owner['ppid'] not in driver['owned_worker_pids'],
            'identity','driver/launcher PID reused as a worker')
    ordered = {worker['tp_rank']:worker for worker in workers}
    require(set(ordered) == set(range(8)),'rank','controller rank set incomplete')
    normalized = {}
    for row in rows:
        rank = row['rank'];worker = ordered[rank];birth = row['birth_identity']
        require(row['terminal'] is True and row['exitcode'] == 0 and row['hard_kill_sent'] is False,
                'terminal','successful child not normally reaped zero')
        require(row['pid'] == birth['pid'] == worker['pid'] and birth['ppid'] == owner['pid']
                and birth['boot_id'] == owner['boot_id'] == worker['boot_id']
                and positive_int(birth['start_ticks']) and birth['start_ticks'] >= owner['start_ticks']
                and positive_int(worker['normal_init_return_ns']), 'identity','child/parent/boot/start identity differs')
        normalized[rank] = validate_rank_payload(rank,ranks[rank],controller,source_manifest,profile,upstream,proofs[rank],worker)
    return normalized


def load_job(job_dir, variant, policy):
    root,freeze_ref,contract_ref,freeze,contract,upstream = policy
    anchor = contract['collections'][variant];bundle = JobFiles(job_dir,anchor)
    require(bundle.file('run.exit').strip() == b'0','terminal','original runner exit is not0')
    driver = bundle.json('gate/result_compact.json');controller = bundle.json('gate/metadata/controller.json')
    cleanup = bundle.json('gate/owned_failure_cleanup.json');manifest = bundle.json('gate/metadata/source_manifest.json')
    require(driver['original_result'] == anchor['original_driver_result'],'collection','derived compact lost original large result binding')
    original = driver['original_result']
    require(original['retained_remotely'] is True and original['raw_bytes_transferred'] is False
            and original['remote_path'] == anchor['remote_job_root']+'/gate/result.json',
            'collection','original driver reference/path is not the retained raw result')
    retained = bundle.archive['oversized_files_retained_remotely']['gate/result.json']
    require(all(retained[key] == original[key] for key in ('bytes','sha256','retained_remotely','raw_bytes_transferred')),
            'collection','archive/compact original driver SHA differs')
    require(driver['audit_job_id'] == anchor['job_id'] and driver['audit_job_root'] == anchor['remote_job_root'],
            'binding','trusted job identity/path differs')
    require(controller['source_manifest_sha256'] == bundle.refs['gate/metadata/source_manifest.json'].sha256,
            'hash','controller source manifest binding changed')
    require(driver['controller']['controller_sha256'] == bundle.refs['gate/metadata/controller.json'].sha256,
            'hash','driver controller binding changed')
    require(len(driver['rank_results']) == 8
            and len({row['process_identity']['pid'] for row in driver['rank_results']}) == 8,
            'rank','derived driver summary has missing/extra/duplicate ranks')
    profile = contract['profile'];ranks = {};proofs = {};kernel_refs = {}
    for rank in range(8):
        result_name = f'gate/sdk_results/{variant}/rank{rank}/result.json'
        ranks[rank] = bundle.json(result_name)
        proof_name = f'gate/metadata/loaded_abi_proofs/rank{rank}.json';proofs[rank] = bundle.json(proof_name)
        worker = next(item for item in controller['worker_processes'] if item['tp_rank'] == rank)
        require(worker['loaded_abi_proofs_sha256'] == bundle.refs[proof_name].sha256
                and worker['loaded_abi_proofs_path'] == controller['driver_output_root']+'/metadata/loaded_abi_proofs/'+f'rank{rank}.json',
                'hash','rank ABI proof reference changed')
        receipt = ranks[rank]
        require(receipt['controller'] == {'path':controller['controller_path'],
                'bytes':bundle.refs['gate/metadata/controller.json'].bytes,'sha256':bundle.refs['gate/metadata/controller.json'].sha256},
                'hash','rank controller bytes/path/SHA binding differs')
        require(receipt['source_gate']['sha256'] == freeze['files']['initialized_worker_gate.py']
                and receipt['source_gate']['bytes'] == (root/'initialized_worker_gate.py').stat().st_size
                and receipt['source_gate']['path'].endswith('/native_initialized_worker_discovery_v8/initialized_worker_gate.py'),
                'source','rank actual gate source/path/size changed')
        require(receipt['controller_bound_loaded_abi_proofs'] == {
                'path':worker['loaded_abi_proofs_path'],'sha256':worker['loaded_abi_proofs_sha256']},
                'hash','rank controller-bound ABI proof differs')
        rpc = bundle.json(f'gate/raw_results/rank{rank}.json')
        require(rpc['kind'] == 'result' and rpc['rank'] == rank
                and {key:rpc['result'][key] for key in receipt} == receipt
                and rpc['result']['model_load_called'] is False,'binding','raw RPC and original rank result differ')
        compact = next(item for item in driver['rank_results'] if item['process_identity']['pid'] == worker['pid'])
        require(all(compact[key] == receipt[key] for key in compact),'binding','derived driver rank summary differs from raw rank')
        kernels = []
        require(receipt['generated_kernels'],'kernel','actual generated kernel source absent')
        for kernel in receipt['generated_kernels']:
            path = kernel['path'];prefix = controller['output_root']+'/'+variant+'/rank'+str(rank)+'/'
            require(path.startswith(prefix),'kernel','generated kernel belongs to another rank/job')
            name = result_name.rsplit('/',1)[0]+'/'+path[len(prefix):]
            data = bundle.file(name);ref = bundle.refs[name]
            require(ref.bytes == kernel['bytes'] and ref.sha256 == kernel['sha256'],'kernel','generated kernel source SHA changed')
            tree = ast.parse(data)
            require(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                        and node.func.id == 'assert_size_stride' for node in ast.walk(tree)),
                    'kernel','original generated-kernel size/stride assert missing')
            kernels.append(ref)
        kernel_refs[rank] = tuple(kernels)
    normalized = validate_job_payload(driver,controller,cleanup,manifest,ranks,proofs,freeze,profile,upstream)
    evidence = tuple(RankGateEvidence(rank,rank+8,immutable(ranks[rank]['process_identity']),
        bundle.refs[f'gate/sdk_results/{variant}/rank{rank}/result.json'],
        bundle.refs[f'gate/metadata/loaded_abi_proofs/rank{rank}.json'],kernel_refs[rank],
        immutable(normalized[rank][0]),immutable(normalized[rank][1]),
        immutable(ranks[rank]['owned_copy_destination_seals_before']),32,immutable(normalized[rank][2]['captured_copy_classes']))
        for rank in range(8))
    return JobGateEvidence(str(bundle.root),driver['audit_job_id'],driver['audit_job_root'],variant,
        bundle.archive_ref,bundle.manifest_ref,bundle.refs['gate/result_compact.json'],
        FileReference(original['remote_path'],original['bytes'],original['sha256'],False),
        bundle.refs['gate/metadata/controller.json'],bundle.refs['gate/metadata/source_manifest.json'],
        bundle.refs['gate/owned_failure_cleanup.json'],immutable(cleanup['owner']),evidence)


def validate_pair_identities(static, nonstatic):
    require(static.variant == 'static_keep' and nonstatic.variant == 'nonstatic_keep','variant','pair order/outer keep differs')
    require(static.job_id != nonstatic.job_id and static.remote_job_root != nonstatic.remote_job_root
            and static.job_dir != nonstatic.job_dir,'fresh','static/nonstatic reused a job')
    require(static.driver_identity['boot_id'] == nonstatic.driver_identity['boot_id'],
            'fresh','two physical assignments refer to different host boots')
    a = {(row.identity['boot_id'],row.identity['pid']) for row in static.ranks}
    b = {(row.identity['boot_id'],row.identity['pid']) for row in nonstatic.ranks}
    a.add((static.driver_identity['boot_id'],static.driver_identity['pid']))
    b.add((nonstatic.driver_identity['boot_id'],nonstatic.driver_identity['pid']))
    require(not a&b and static.driver_identity['pid'] != nonstatic.driver_identity['pid'],
            'fresh','static/nonstatic reused a worker/driver PID')
    require(static.source_manifest.sha256 == nonstatic.source_manifest.sha256,
            'source','two actual job source manifests differ')
    require([row.physical_chip for row in static.ranks] == [row.physical_chip for row in nonstatic.ranks] == list(range(8,16)),
            'scope','static/nonstatic physical TP8 assignment differs')


def load_tp8_sdk_pair(static_job_dir, nonstatic_job_dir, source_root=None):
    """Verify trusted root-collected original evidence and return immutable refs.

    This protocol object is a prerequisite for independent consumers. Its own
    actual SDK/device/consumer/formal flags stay false; original job results are
    preserved as references and are never rewritten to the old v1 schema.
    """
    require(static_job_dir is not None and nonstatic_job_dir is not None,'missing_job','both complete fresh SDK jobs are mandatory')
    policy = source_policy(source_root)
    try:
        static = load_job(static_job_dir,'static_keep',policy)
        nonstatic = load_job(nonstatic_job_dir,'nonstatic_keep',policy)
        validate_pair_identities(static,nonstatic)
    except EvidenceError:
        raise
    except (KeyError,TypeError,ValueError,StopIteration) as exc:
        raise EvidenceError('schema','missing/invalid original TP8 evidence field') from exc
    return Tp8SdkPairEvidence(static,nonstatic,str(policy[0]),policy[1],policy[2])
