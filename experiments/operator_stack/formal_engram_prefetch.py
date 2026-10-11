"""Guarded full-MatMul Engram prefetch prototype; CPU/mock validated only.

No NPU, model, GraphEx, precision or performance validation has been performed.
Importing this module does not import torch or create device objects. ``install``
does not patch a runner, model, ARM or existing graph; integration is explicit.

The required runner order is acquire/reset BEFORE device_metadata submit and
native prepare_engram_inputs, then publish AFTER all persistent lookup/mask
copies, submit the aux graph, bind the matching main graph, and retire only
after BOTH real device fences complete. M=1 decode only; all other requests
use the native path after draining any old use of the persistent inputs.
An exception after acquire/partial producer enqueue poisons the slot; this
prototype deliberately has no unsafe cancel/reset escape hatch. Validate
exception cleanup before integration. Keep ENPU soft-segmentation's existing
record-before-wait and _update_full_graph_params_if_needed order unchanged.

Both original nn.Linear(6144,25600,bias=False,BF16) calls remain whole. The
consumer adapter is an UNVALIDATED wait-and-copy(cache,out,event)->None op with
explicit mutation of an independent out Tensor, consumed by the original gate.
This adds two 51,200-byte copies. A second UNVALIDATED mutating op copies an
end-of-model dependency to a marker, increments a device epoch, and records an
ExternalEvent. Python capture callbacks never count per-replay generations.
Completion uses an ordinary Event recorded on the validated main replay
submission stream. Never enqueue a fresh host consumed.wait each generation:
graph replay does not renew the host record/wait association (SDK rejection).
No-op fake implementations return None and never return a cache alias, but
functionalization, preservation of these nodes, graph-pool aliasing, lazy event
initialization and per-generation reset still need dedicated SDK evidence.

All those unknowns default to disabled. A receipt claiming CPU/mock checks
cannot enable an NPU backend. Registration of two weights is NOT coverage.
The main graph must explicitly forward the cache/out/marker/epoch/audit tensors,
use a separate identity from the native graph, and reach actual two-layer
consumer/gate observation hooks before an audit can report success.
"""
from dataclasses import dataclass, field
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any

LAYERS = (1, 14)
K, N, H, HC = 6144, 25600, 5120, 4
CONTRACT_SHA256 = 'bed680a3c032d89c65c1974ce5f2db28d8e419c5e12e0ef85dafdf877e22f62b'
DESIGN_SHA256 = 'c282163fe462fb8e76308803bdd735d79ff1bda771341188b57c1fe87ecdfadd'
_EVENTS = {}  # Stable Python object identities, never lazy zero SDK handles.
_WAIT_VISITS = {}
_CONSUMED_VISITS = {}
_GATE_BEFORE_VISITS = {}
_GATE_AFTER_VISITS = {}
_OPS = None


class GuardViolation(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GuardViolation(message)


def _ledger_module(path=None):
    path = Path(path) if path else Path(__file__).parent/'parallel_tiny/engram_lane/prefetch_lifecycle_contract.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == CONTRACT_SHA256,
            'CPU lifecycle contract source changed; review and refreeze it')
    name = '_formal_engram_prefetch_locked_contract'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class GraphSafetyReceipt:
    """Assertions backed by dedicated evidence, not auto-detected capabilities."""
    scope: str = 'unvalidated'
    evidence_id: str = ''
    external_event_replay: bool = False
    reset_device_completion: bool = False
    wait_copy_preserved: bool = False
    fake_mutation_and_alias: bool = False
    consumed_marker_preserved: bool = False
    separate_main_identity: bool = False
    acquire_before_metadata: bool = False
    known_submission_streams: bool = False
    completion_fence: str = 'unvalidated'

    def permits(self, backend):
        flags = (self.external_event_replay, self.reset_device_completion,
                 self.wait_copy_preserved, self.fake_mutation_and_alias,
                 self.consumed_marker_preserved, self.separate_main_identity,
                 self.acquire_before_metadata, self.known_submission_streams)
        expected = 'cpu-mock' if backend.kind == 'cpu-mock' else 'npu-sdk-validated'
        return (self.scope == expected and bool(self.evidence_id) and all(flags)
                and self.completion_fence == 'submission-stream')


@dataclass(frozen=True)
class TensorStamp:
    object_id: int
    pointer: int
    shape: tuple
    stride: tuple
    dtype: str
    device: str
    version: int
    format: int


@dataclass
class FrozenProjection:
    layer: int
    name: str
    module: Any
    native: Any
    weight: TensorStamp

    def check(self, backend):
        require(self.module.bias is None and backend.stamp(self.module.weight, weight=True) == self.weight,
                f'layer {self.layer}: original weight pointer/version/layout changed')


@dataclass
class Slot:
    identity: Any
    ledger: Any
    lookup: dict
    mask: Any
    cache: dict
    out: dict
    ready: dict
    lookup_written: Any
    producer_done: Any
    consumer_done: Any
    reset_done: Any
    consumed: Any
    marker: Any
    epoch: Any
    aux_stream: Any
    main_stream: Any
    graph: Any = None
    producer_capture: set = field(default_factory=set)
    consumer_capture: set = field(default_factory=set)
    gate_capture: set = field(default_factory=set)
    consumed_capture: bool = False
    audit_buffers: dict = field(default_factory=dict)
    audit_generation: Any = None
    after_main_enqueued: bool = False
    main_capture_fenced: bool = False
    expected_epoch: int = 0
    gate_before_capture: set = field(default_factory=set)
    resources_signature: tuple = ()
    sample_metadata: dict = field(default_factory=dict)
    audit_metadata: dict = field(default_factory=dict)
    input_signature: tuple = ()
    audit_signature: tuple = ()


class PrefetchPrototype:
    def __init__(self, model, *, source_paths, expected_sha256, target_arm,
                 backend, safety=None, contract_path=None):
        self.backend = backend
        self.safety = safety or GraphSafetyReceipt()
        self.contract = _ledger_module(contract_path)
        require(bool(target_arm), 'integration must supply its target ARM')
        self.target_arm = target_arm
        required_sources = {'model', 'runner', 'gate', 'sdk_events'}
        require(set(source_paths) == set(expected_sha256) == required_sources,
                'freeze model/runner/gate/SDK Event source SHA before use')
        self.source = {}
        for name, path in source_paths.items():
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            require(digest == expected_sha256[name], f'{name} source SHA mismatch')
            self.source[name] = digest
        self.source.update(adapter=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           cpu_contract=CONTRACT_SHA256, design=DESIGN_SHA256)
        self.projections = {}
        for name, module in model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.engram\.wkv$', name)
            if not match:
                continue
            layer = int(match[1])
            require(layer in LAYERS and layer not in self.projections, 'unexpected/duplicate Engram projection')
            backend.validate_linear(module)
            stamp = backend.stamp(module.weight, weight=True)
            require(stamp.shape == (N, K) and stamp.dtype == 'bfloat16' and module.bias is None,
                    'original full BF16 N25600/K6144 MatMul required')
            self.projections[layer] = FrozenProjection(layer, name, module, module.forward, stamp)
        require(set(self.projections) == set(LAYERS), 'exact original layers 1 and 14 required')
        self.weight_version = hashlib.sha256(json.dumps(
            {layer: vars(p.weight) for layer, p in self.projections.items()}, sort_keys=True).encode()).hexdigest()
        self.slots = {}
        self.fallbacks = {}
        self.native_done = None

    def _weights_unchanged(self):
        for projection in self.projections.values():
            projection.check(self.backend)

    def register_slot(self, slot_id, lookups, mask, *, graph_key, main_stream, audit=False):
        """Cold setup only. SDK operations remain guarded by the safety receipt."""
        require(self.safety.permits(self.backend), 'unverified graph wait/reset/alias; keep native fallback')
        require(slot_id not in self.slots and set(lookups) == set(LAYERS), 'slot/layers invalid')
        self._weights_unchanged()
        for value in lookups.values():
            stamp = self.backend.stamp(value)
            require(len(stamp.shape) == 2 and stamp.shape[0] >= 1 and stamp.shape[1] == K
                    and stamp.dtype == 'bfloat16', 'persistent full-width BF16 lookup required')
        require(self.backend.stamp(mask).dtype == 'bool' and mask.shape[0] >= 1, 'persistent mask required')
        cache = {layer: self.backend.empty(lookups[layer], (1, N)) for layer in LAYERS}
        out = {layer: self.backend.empty(lookups[layer], (1, N)) for layer in LAYERS}
        ready = {layer: self.backend.event(external=True) for layer in LAYERS}
        consumed = self.backend.event(external=True)
        identities = tuple(id(ready[layer]) for layer in LAYERS)
        for event in (*ready.values(), consumed):
            _EVENTS[id(event)] = event
        identity = self.contract.GraphIdentity(
            slot_id, 1, tuple(self.backend.ptr(lookups[l]) for l in LAYERS), self.backend.ptr(mask),
            tuple(self.backend.ptr(out[l]) for l in LAYERS), self.weight_version, identities, graph_key)
        addresses = (*identity.lookup_ptrs, identity.mask_ptr, *identity.kv_ptrs,
                     *(self.backend.ptr(cache[l]) for l in LAYERS))
        require(len(set(addresses)) == len(addresses), 'lookup/mask/cache/out aliases forbidden')
        for old in self.slots.values():
            require(graph_key != old.identity.graph_key, 'each slot needs a separately captured main graph identity')
            old_addresses = (*old.identity.lookup_ptrs, old.identity.mask_ptr, *old.identity.kv_ptrs,
                             *(self.backend.ptr(old.cache[l]) for l in LAYERS))
            require(not set(addresses).intersection(old_addresses),
                    'multiple slots need independent lookup/mask/cache/out and separate main graphs')
        slot = Slot(identity, self.contract.SlotLedger(identity), dict(lookups), mask, cache, out, ready,
                    self.backend.event(), self.backend.event(), self.backend.event(), self.backend.event(),
                    consumed, self.backend.empty(lookups[1], (1,)),
                    self.backend.empty(lookups[1], (1,), dtype='int64'),
                    self.backend.stream(), main_stream)
        slot.input_signature = self._input_signature(lookups, mask)
        if audit:
            for layer in LAYERS:
                slot.audit_buffers[layer] = {
                    'lookup': self.backend.empty(lookups[layer], (1, K)),
                    'mask': self.backend.empty(mask, (1,), dtype='bool'),
                    'before': self.backend.empty(lookups[layer], (1, HC, H)),
                    'kv': self.backend.empty(lookups[layer], (1, N)),
                    'after': self.backend.empty(lookups[layer], (1, HC, H)),
                    'parameters': None}

        def full_projection_body():
            for layer in LAYERS:
                # Original bound nn.Linear call: no slicing of N/K or new reduction.
                value = self.projections[layer].native(slot.lookup[layer][:1])
                self.backend.copy(slot.cache[layer], value)
                self.backend.record(slot.ready[layer], slot.aux_stream)
                if self.backend.is_real_capture():
                    slot.producer_capture.add(layer)

        slot.graph = self.backend.capture_aux(full_projection_body, slot.aux_stream)
        self.backend.finish_cold_capture(slot.aux_stream)
        require(slot.producer_capture == set(LAYERS), 'registered weights are not actual projection capture coverage')
        self.slots[slot_id] = slot
        slot.resources_signature = self._resources_signature(slot)
        for layer in LAYERS:
            _WAIT_VISITS[id(ready[layer])] = (self, slot_id, layer)
            key = (id(self),slot_id,layer)
            _GATE_BEFORE_VISITS[key] = (self,slot_id,layer)
            _GATE_AFTER_VISITS[key] = (self,slot_id,layer)
        _CONSUMED_VISITS[id(consumed)] = (self,slot_id)
        return identity

    def _resources_signature(self, slot):
        tensors = [slot.cache[l] for l in LAYERS]+[slot.out[l] for l in LAYERS]+[slot.marker,slot.epoch]
        stamps = [self.backend.stamp(t) for t in tensors]
        require(len({s.pointer for s in stamps}) == len(stamps), 'cache/out/marker/epoch must not alias')
        return (tuple((s.object_id,s.pointer,s.shape,s.stride,s.dtype,s.device,s.format) for s in stamps),
                tuple(id(e) for e in (*slot.ready.values(),slot.lookup_written,slot.producer_done,
                                     slot.consumer_done,slot.reset_done,slot.consumed)))

    def _layout(self, tensor):
        stamp = self.backend.stamp(tensor)
        # Views may have new Python identities; mutable values/versions change.
        return (stamp.pointer, stamp.shape, stamp.stride, stamp.dtype, stamp.device, stamp.format)

    def _input_signature(self, lookups, mask):
        return tuple(self._layout(lookups[layer]) for layer in LAYERS) + (self._layout(mask),)

    def _audit_signature(self, slot):
        rows = []
        for layer in LAYERS:
            row = slot.audit_buffers[layer]
            require(row['parameters'] is not None, 'audit parameters must be allocated before capture')
            channel, rotation, eps = row['parameters']
            tensors = [row[name] for name in ('lookup', 'mask', 'before', 'kv', 'after')]
            rows.append((tuple(self._layout(tensor) for tensor in (*tensors, channel, rotation)), eps))
        return tuple(rows)

    def _check_consumer_view(self, slot, layer, lookup, mask=None):
        layout = self._layout(lookup)
        frozen = slot.input_signature[LAYERS.index(layer)]
        require(layout[0] == frozen[0] and layout[1] == (1, K) and layout[2:] == frozen[2:],
                'actual consumer lookup pointer/dtype/stride/device/format changed')
        if mask is not None:
            layout, frozen = self._layout(mask), slot.input_signature[-1]
            require(layout[0] == frozen[0] and layout[1] in ((1,), frozen[1]) and layout[2:] == frozen[2:],
                    'actual consumer mask pointer/dtype/stride/device/format changed')

    def prepare_gate_audit(self, slot_id, parameters):
        """Allocate private parameter snapshots outside capture, never keep mutable aliases."""
        require(not self.backend.is_real_capture(), 'audit allocation must precede main capture')
        slot = self.slots[slot_id]
        require(slot.audit_buffers and set(parameters) == set(LAYERS), 'two-layer audit parameters required')
        require(not slot.main_capture_fenced and not slot.gate_before_capture and not slot.audit_signature,
                'audit destinations are sealed; cannot reallocate after preparation/capture')
        for layer, (channel, rotation, eps) in parameters.items():
            require(channel.shape == (HC, H) and len(rotation.shape) == 2
                    and rotation.shape[0] == rotation.shape[1] and H % rotation.shape[0] == 0,
                    'original gate parameter shapes changed')
            slot.audit_buffers[layer]['parameters'] = (
                self.backend.empty(channel, channel.shape), self.backend.empty(rotation, rotation.shape), eps)
        slot.audit_signature = self._audit_signature(slot)

    def finalize_main_capture(self, slot_id, *, before_metadata_submit):
        """Fence initial capture/warmup waiters before the first ExternalEvent reset."""
        require(before_metadata_submit and not self.backend.is_real_capture(), 'cold main fence must be graph-external')
        slot = self.slots[slot_id]
        self._check_binding(slot, slot.lookup, slot.mask)
        require(slot.consumer_capture == slot.gate_capture == set(LAYERS) and slot.consumed_capture,
                'actual main capture evidence missing')
        self.backend.record(slot.producer_done, slot.aux_stream)
        self.backend.record(slot.consumer_done, slot.main_stream)
        self.backend.synchronize_event(slot.producer_done)
        self.backend.synchronize_event(slot.consumer_done)
        require(self.backend.query(slot.producer_done) and self.backend.query(slot.consumer_done),
                'initial producer/consumer record/wait tasks not complete')
        slot.expected_epoch = self.backend.read_epoch(slot.epoch)
        slot.main_capture_fenced = True

    def _check_binding(self, slot, lookups, mask):
        self._weights_unchanged()
        require(self._resources_signature(slot) == slot.resources_signature,
                'captured cache/out/marker/epoch/event object or layout changed')
        require(self._input_signature(lookups, mask) == slot.input_signature,
                'persistent lookup/mask pointer/dtype/shape/stride/device/format changed')
        if slot.audit_signature:
            require(self._audit_signature(slot) == slot.audit_signature,
                    'sealed audit destination/parameter pointer or layout changed')
        require(tuple(self.backend.ptr(lookups[l]) for l in LAYERS) == slot.identity.lookup_ptrs
                and self.backend.ptr(mask) == slot.identity.mask_ptr, 'persistent input pointers changed')
        require(tuple(self.backend.ptr(slot.out[l]) for l in LAYERS) == slot.identity.kv_ptrs
                and tuple(id(slot.ready[l]) for l in LAYERS) == slot.identity.ready_event_ids,
                'static output/event identity changed')

    def select(self, *, arm, bucket_m, actual_tokens, decode, profile=False, slot_id=0, for_capture=False):
        reason = None
        if arm != self.target_arm:
            reason = 'non-target-arm'
        elif not decode or profile or bucket_m != 1 or actual_tokens != 1:
            reason = 'non-M1-decode-or-profile'
        elif not self.safety.permits(self.backend):
            reason = 'unverified-SDK-graph-contract'
        elif slot_id not in self.slots:
            reason = 'unprepared-slot'
        elif not for_capture and not self.slots[slot_id].main_capture_fenced:
            reason = 'unfenced-main-capture'
        if reason:
            self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        return reason is None

    def drain_before_native_prepare(self, *, before_metadata_submit):
        """Changing ARM/prefill still must protect old persistent buffer readers."""
        require(before_metadata_submit and not self.backend.is_real_capture(), 'drain must precede metadata submit')
        for slot in self.slots.values():
            require(slot.audit_generation is None, 'audit lease also forbids native prepare/sample overwrite')
            if slot.ledger.phase != 'FREE':
                require(slot.after_main_enqueued, 'incomplete submission poisoned slot; cannot silently overwrite/fallback')
                self.retire(slot.ledger.ticket, wait=True)

    def after_native_main_submitted(self, main_stream):
        """Integration must also fence native/prefill readers across an ARM change."""
        require(not self.backend.is_real_capture(), 'native submission fence is graph-external')
        if self.native_done is None:
            self.native_done = self.backend.event()
        self.backend.record(self.native_done, main_stream)

    def acquire_before_prepare(self, slot_id, generation, *, before_metadata_submit, sample_metadata=None):
        require(before_metadata_submit and not self.backend.is_real_capture(), 'acquire must precede metadata submit')
        slot = self.slots[slot_id]
        require(slot.main_capture_fenced, 'initial main capture waiters must complete before reset')
        if slot.ledger.phase != 'FREE':
            self.retire(slot.ledger.ticket, wait=True)
        require(slot.audit_generation is None, 'audit sample leased; do not overwrite before export/release')
        self._weights_unchanged()
        self._check_binding(slot,slot.lookup,slot.mask)
        if slot.audit_buffers:
            require(sample_metadata is not None and set(sample_metadata) == {'rank','request_id','positions'}
                    and len(sample_metadata['positions']) == 1 and bool(sample_metadata['request_id']),
                    'actual rank/request/host positions metadata required for a decode audit')
            slot.sample_metadata = json.loads(json.dumps(sample_metadata))
        if self.native_done is not None:
            self.backend.synchronize_event(self.native_done)
            require(self.backend.query(self.native_done), 'old native/prefill consumers still use persistent inputs')
        ticket = slot.ledger.acquire(generation, actual_tokens=1)
        for event in (*slot.ready.values(), slot.consumed):
            self.backend.reset(event, slot.aux_stream)
        self.backend.record(slot.reset_done, slot.aux_stream)
        # Observed device completion, not reset enqueue success or a sticky flag.
        self.backend.synchronize_event(slot.reset_done)
        require(self.backend.query(slot.reset_done), 'ready reset completion not observed')
        slot.ledger.arm(ticket, ready_flags_clear=True)
        slot.after_main_enqueued = False
        return ticket

    def publish_after_prepare(self, ticket, lookups, mask, *, copy_stream):
        slot = self.slots[ticket.identity.slot]
        slot.ledger.check_ticket(ticket)
        self._check_binding(slot, lookups, mask)
        require(self.backend.same_stream(copy_stream,slot.main_stream), 'record input-ready on the actual native copy producer stream')
        self.backend.record(slot.lookup_written, copy_stream)
        slot.ledger.publish_lookup(ticket, all_copies_event_recorded=True, after_free_fence=True)

    def submit_aux(self, ticket):
        slot = self.slots[ticket.identity.slot]
        slot.ledger.check_ticket(ticket)
        require(slot.ledger.phase == 'LOOKUP_PUBLISHED', 'lookup-ready must precede producer enqueue')
        self._weights_unchanged()
        self.backend.wait(slot.lookup_written, slot.aux_stream)
        self.backend.replay(slot.graph, slot.aux_stream)
        self.backend.record(slot.producer_done, slot.aux_stream)
        slot.ledger.submit_producer(ticket, waits_input_ready=True,
                                    captured_ready_layers=tuple(sorted(slot.producer_capture)))

    def bind_main(self, ticket, *, identity):
        slot = self.slots[ticket.identity.slot]
        self._check_binding(slot,slot.lookup,slot.mask)
        require(slot.consumer_capture == slot.gate_capture == set(LAYERS) and slot.consumed_capture,
                'main capture must reach both actual consumers/gates and an observable completion marker')
        slot.ledger.bind_consumer(ticket, identity, external_wait_layers=tuple(sorted(slot.consumer_capture)))
        return {'engram_prefetch_cache': dict(slot.cache), 'engram_full_kv': dict(slot.out),
                'engram_consumed_marker': slot.marker, 'engram_consumed_epoch': slot.epoch,
                'engram_prefetch_audit': dict(slot.audit_buffers)}

    def consumer_or_native(self, layer, lookup, *, arm, slot_id=0, prepared=False):
        """Metadata routing for opaque/cold validation only; do not trace these guards.

        Dynamo integration must select the native/candidate bank graph-externally
        and call graph_consumer with the explicitly forwarded static Tensor pair.
        """
        projection = self.projections[layer]
        projection.check(self.backend)
        if not prepared or not self.select(arm=arm, bucket_m=lookup.shape[0], actual_tokens=lookup.shape[0],
                                           decode=True, slot_id=slot_id, for_capture=self.backend.is_real_capture()):
            return projection.native(lookup)
        slot = self.slots[slot_id]
        self._check_binding(slot,slot.lookup,slot.mask)
        self._check_consumer_view(slot, layer, lookup)
        require(self.backend.ptr(lookup) == slot.identity.lookup_ptrs[LAYERS.index(layer)], 'consumer lookup alias changed')
        self.backend.wait_copy(slot.cache[layer], slot.out[layer], id(slot.ready[layer]))
        if self.backend.is_real_capture():
            slot.consumer_capture.add(layer)
        return slot.out[layer]

    def graph_consumer(self, slot_id, layer, cache, out):
        """Minimal Tensor adapter after graph-external identity/eligibility guards."""
        self.backend.wait_copy(cache,out,id(self.slots[slot_id].ready[layer]))
        return out  # Outside the custom op; the op itself has no alias return.

    def graph_gate_before(self, slot_id, layer, lookup, mask, hidden, kv, channel, rotation, eps):
        """Opaque bounded observation adapter; no data_ptr/version reads in Dynamo."""
        row=self.slots[slot_id].audit_buffers[layer]
        require(row['parameters'] is not None, 'cold audit parameter allocation missing')
        self.backend.audit_before(lookup,mask,hidden,kv,channel,rotation,eps,row,id(self),slot_id,layer)

    def graph_gate_after(self, slot_id, layer, hidden):
        self.backend.audit_after(hidden,self.slots[slot_id].audit_buffers[layer]['after'],id(self),slot_id,layer)

    def graph_main_consumed(self, slot_id, final_hidden, marker, epoch):
        self.backend.mark_consumed(final_hidden,marker,epoch,id(self.slots[slot_id].consumed))

    def observe_gate_before(self, slot_id, layer, lookup, mask, hidden, kv, *, parameters):
        """Actual consumer hook; bounded copies only. No host reads/logging/budget."""
        slot = self.slots[slot_id]
        require(self.backend.is_real_capture(), 'consumer observation must be a real capture, not registration/fake')
        require(hidden.shape == (1, HC, H) and kv.shape == (1, N), 'actual M1 complete gate inputs required')
        require(layer in slot.consumer_capture, 'gate needs actual consumer wait/copy evidence')
        self._check_consumer_view(slot, layer, lookup, mask)
        require(self.backend.ptr(lookup) == slot.identity.lookup_ptrs[LAYERS.index(layer)]
                and self.backend.ptr(mask) == slot.identity.mask_ptr
                and self.backend.ptr(kv) == slot.identity.kv_ptrs[LAYERS.index(layer)],
                'actual gate tensors do not belong to this slot')
        if slot.audit_buffers:
            row = slot.audit_buffers[layer]
            for name, value in (('lookup', lookup), ('mask', mask), ('before', hidden), ('kv', kv)):
                self.backend.copy(row[name], value)
            frozen = row['parameters']
            require(frozen is not None and frozen[2] == parameters[2], 'preallocated original gate parameters required')
            self.backend.copy(frozen[0], parameters[0])
            self.backend.copy(frozen[1], parameters[1])
        slot.gate_before_capture.add(layer)

    def observe_gate_after(self, slot_id, layer, hidden):
        slot = self.slots[slot_id]
        require(self.backend.is_real_capture(), 'post-gate observation must be real capture')
        require(layer in slot.gate_before_capture and hidden.shape == (1,HC,H), 'pre-gate observation missing')
        if slot.audit_buffers:
            self.backend.copy(slot.audit_buffers[layer]['after'], hidden)
        slot.gate_capture.add(layer)

    def record_main_consumed(self, slot_id, final_hidden):
        """Must be inside the main graph AFTER actual gate/residual writes."""
        slot = self.slots[slot_id]
        if self.backend.is_real_capture():
            require(slot.gate_capture == set(LAYERS), 'completion marker precedes an actual gate')
        self.backend.mark_consumed(final_hidden, slot.marker, slot.epoch, id(slot.consumed))
        if self.backend.is_real_capture():
            slot.consumed_capture = True

    def after_main_submitted(self, ticket, *, main_stream):
        slot = self.slots[ticket.identity.slot]
        slot.ledger.check_ticket(ticket)
        require(slot.ledger.phase == 'CONSUMER_BOUND' and self.backend.same_stream(main_stream,slot.main_stream) and not slot.after_main_enqueued,
                'completion fence must follow matching main graph enqueue')
        require(self.safety.completion_fence == 'submission-stream', 'unverified main graph completion fence')
        # GraphEx/ENPU submission stream ordering needs dedicated evidence.
        # Replaying a captured consumed.record cannot renew a host wait lease.
        self.backend.record(slot.consumer_done, main_stream)
        slot.expected_epoch += 1
        slot.after_main_enqueued = True

    def retire(self, ticket, *, wait=False, retain_audit=False):
        require(not self.backend.is_real_capture(), 'fence query/export must be graph-external')
        slot = self.slots[ticket.identity.slot]
        slot.ledger.check_ticket(ticket)
        require(slot.after_main_enqueued, 'main submission/graph completion marker not fenced')
        if wait:
            self.backend.synchronize_event(slot.producer_done)
            self.backend.synchronize_event(slot.consumer_done)
        producer, consumer = self.backend.query(slot.producer_done), self.backend.query(slot.consumer_done)
        require(producer and consumer, 'both real completion fences required before reset/overwrite')
        require(self.backend.read_epoch(slot.epoch) == slot.expected_epoch,
                'device replay epoch differs: completion may be stale or main graph replay count changed')
        slot.ledger.observe_done(ticket, producer=producer, consumer=consumer)
        if retain_audit:
            require(slot.audit_buffers, 'audit buffers absent')
            slot.audit_generation = ticket.generation
            slot.audit_metadata = dict(slot.sample_metadata, generation=ticket.generation,
                                       bucket_m=slot.identity.bucket_m, slot=slot.identity.slot)
        slot.ledger.retire(ticket)

    def audit(self, slot_id, *, generation, original_gate):
        """Graph-external snapshot audit; full logits/routes remain separate gates."""
        slot = self.slots[slot_id]
        require(slot.audit_generation == generation and slot.ledger.phase == 'FREE', 'fenced sample lease required')
        require(slot.producer_capture == slot.consumer_capture == slot.gate_capture == set(LAYERS),
                'missing actual two-layer projection/gate capture evidence')
        self._check_binding(slot, slot.lookup, slot.mask)
        require(hashlib.sha256(Path(inspect.getfile(original_gate)).read_bytes()).hexdigest() == self.source['gate'],
                'audit must call the frozen original gate source')
        results = []
        for layer in LAYERS:
            row = slot.audit_buffers[layer]
            require(row['parameters'] is not None, 'original gate parameters absent')
            expected_kv = self.projections[layer].native(row['lookup'])
            key, value = self.backend.split_kv(expected_kv)
            expected_gate = original_gate(row['before'], key, value, *row['parameters'][:2],
                                          row['mask'], row['parameters'][2])
            projection = self.backend.compare_bf16(row['kv'], expected_kv)
            gate = self.backend.compare_bf16(row['after'], expected_gate)
            require(projection['max_bf16_ulp'] <= 1 and gate['max_bf16_ulp'] <= 1,
                    'original BF16 1 ULP gate exceeded')
            results.append({'layer': layer, 'projection': projection, 'gate': gate})
        return {'passed': True, 'scope': 'Actual fenced two-layer full projection/gate snapshots; no E2E claim',
                'backend_scope': self.backend.kind, 'npu_validated': self.backend.kind != 'cpu-mock',
                'generation': generation, 'sample':dict(slot.audit_metadata),
                'weight_versions':{layer:vars(p.weight) for layer,p in self.projections.items()},
                'rows': results, 'source': self.source}

    def release_audit(self, slot_id, generation):
        slot = self.slots[slot_id]
        require(slot.audit_generation == generation, 'audit sample lease changed')
        slot.audit_generation = None
        slot.audit_metadata = {}

    def status(self):
        return {'prototype_only': True, 'npu_or_model_validated': False,
                'registered_weights': len(self.projections), 'safety_permits_backend': self.safety.permits(self.backend),
                'extra_main_kv_copy_bytes': 2*N*2, 'source': self.source, 'fallbacks': dict(self.fallbacks),
                'slots': {sid: {'phase': s.ledger.phase, 'graph_key': s.identity.graph_key,
                               'cold_main_capture_fenced': s.main_capture_fenced,
                               'actual_producer_capture_layers': sorted(s.producer_capture),
                               'actual_consumer_capture_layers': sorted(s.consumer_capture),
                               'actual_gate_capture_layers': sorted(s.gate_capture),
                               'consumed_marker_capture': s.consumed_capture}
                          for sid, s in self.slots.items()}}


def install(model, **kwargs):
    """Freeze a freshly loaded model without monkey-patching any existing object."""
    return PrefetchPrototype(model, **kwargs)


class NPUBridge:
    """Lazy SDK adapter. Construct only in an authorized fresh-model validation.

    ``capture_check`` must reject fake/meta execution and report real SDK capture.
    The custom mutation schemas below are proposals, NOT validated GraphEx code.
    """
    kind = 'npu'

    def __init__(self, capture_check):
        import torch
        import torch_npu
        self.torch, self.npu = torch, torch_npu
        self.capture_check = capture_check
        self.ops = self._register_ops()

    def _register_ops(self):
        global _OPS
        if _OPS is not None:
            return _OPS
        torch, npu = self.torch, self.npu

        @torch.library.custom_op('formal_engram_prefetch::wait_copy', mutates_args={'out'})
        def wait_copy(cache: torch.Tensor, out: torch.Tensor, event_id: int) -> None:
            _EVENTS[event_id].wait(npu.npu.current_stream())
            out.copy_(cache)
            owner,slot_id,layer=_WAIT_VISITS[event_id]
            if owner.backend.is_real_capture():
                slot=owner.slots[slot_id]
                require(cache.data_ptr()==slot.cache[layer].data_ptr() and out.data_ptr()==slot.out[layer].data_ptr(),
                        'actual wait/copy capture uses wrong static tensors')
                slot.consumer_capture.add(layer)

        @wait_copy.register_fake
        def wait_copy_fake(cache, out, event_id):
            return None

        @torch.library.custom_op('formal_engram_prefetch::consumed', mutates_args={'marker', 'epoch'})
        def consumed(dependency: torch.Tensor, marker: torch.Tensor, epoch: torch.Tensor, event_id: int) -> None:
            marker.copy_(dependency.reshape(-1)[:1])
            epoch.add_(1)
            _EVENTS[event_id].record(npu.npu.current_stream())
            owner,slot_id=_CONSUMED_VISITS[event_id]
            if owner.backend.is_real_capture():
                require(owner.slots[slot_id].gate_capture==set(LAYERS), 'consumed node precedes actual gates')
                owner.slots[slot_id].consumed_capture=True

        @consumed.register_fake
        def consumed_fake(dependency, marker, epoch, event_id):
            return None

        @torch.library.custom_op('formal_engram_prefetch::audit_before',
                                mutates_args={'lookup_copy','mask_copy','hidden_copy','kv_copy','channel_copy','rotation_copy'})
        def audit_before(lookup:torch.Tensor,mask:torch.Tensor,hidden:torch.Tensor,kv:torch.Tensor,
                         channel:torch.Tensor,rotation:torch.Tensor,eps:float,
                         lookup_copy:torch.Tensor,mask_copy:torch.Tensor,hidden_copy:torch.Tensor,
                         kv_copy:torch.Tensor,channel_copy:torch.Tensor,rotation_copy:torch.Tensor,
                         owner_id:int,slot_id:int,layer:int)->None:
            owner,slot_id,layer=_GATE_BEFORE_VISITS[(owner_id,slot_id,layer)]
            row=owner.slots[slot_id].audit_buffers[layer]
            require(tuple(t.data_ptr() for t in (lookup_copy,mask_copy,hidden_copy,kv_copy,channel_copy,rotation_copy))
                    ==tuple(t.data_ptr() for t in (row['lookup'],row['mask'],row['before'],row['kv'],*row['parameters'][:2])),
                    'opaque audit schema outputs do not match private buffers')
            if owner.backend.is_real_capture():
                owner.observe_gate_before(slot_id,layer,lookup,mask,hidden,kv,parameters=(channel,rotation,eps))
            else:
                for dst,src in ((lookup_copy,lookup),(mask_copy,mask),(hidden_copy,hidden),
                                (kv_copy,kv),(channel_copy,channel),(rotation_copy,rotation)):
                    dst.copy_(src)  # Warmup is not consumer capture evidence.

        @audit_before.register_fake
        def audit_before_fake(*args): return None

        @torch.library.custom_op('formal_engram_prefetch::audit_after',mutates_args={'out'})
        def audit_after(hidden:torch.Tensor,out:torch.Tensor,owner_id:int,slot_id:int,layer:int)->None:
            owner,slot_id,layer=_GATE_AFTER_VISITS[(owner_id,slot_id,layer)]
            require(out.data_ptr()==owner.slots[slot_id].audit_buffers[layer]['after'].data_ptr(),
                    'opaque gate output copy uses wrong private buffer')
            if owner.backend.is_real_capture():
                owner.observe_gate_after(slot_id,layer,hidden)
            else:
                out.copy_(hidden)

        @audit_after.register_fake
        def audit_after_fake(*args): return None

        _OPS = (wait_copy, consumed, audit_before, audit_after)
        return _OPS

    def stamp(self, tensor, weight=False):
        # _version errors for inference tensors fail the guard rather than fabricating a version.
        try:
            version=tensor._version
        except RuntimeError:
            require(not weight, 'real projection weight has no version counter; cannot freeze it')
            version=-1  # Mutable inference buffers are bound by address/shape/layout, not their version.
        return TensorStamp(id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
                           str(tensor.dtype).removeprefix('torch.'), str(tensor.device),
                           version, int(self.npu.get_npu_format(tensor)))

    def validate_linear(self, module):
        require(type(module) is self.torch.nn.Linear, 'only original complete nn.Linear supported')

    def ptr(self, tensor): return tensor.data_ptr()
    def stream(self): return self.npu.npu.Stream()
    def event(self, external=False): return self.npu.npu.ExternalEvent() if external else self.npu.npu.Event()
    def record(self, event, stream): event.record(stream)
    def reset(self, event, stream): event.reset(stream)
    def wait(self, event, stream): event.wait(stream)
    def query(self, event): return bool(event.query())
    def synchronize_event(self, event): event.synchronize()
    def read_epoch(self, tensor): return int(tensor.item())
    def copy(self, output, value): output.copy_(value)
    def is_real_capture(self): return bool(self.capture_check())
    def same_stream(self,a,b): return bool(a==b)
    def wait_copy(self, cache, out, event_id): self.ops[0](cache, out, event_id)
    def mark_consumed(self, dependency, marker, epoch, event_id): self.ops[1](dependency, marker, epoch, event_id)
    def audit_before(self,lookup,mask,hidden,kv,channel,rotation,eps,row,owner_id,slot_id,layer):
        self.ops[2](lookup,mask,hidden,kv,channel,rotation,eps,row['lookup'],row['mask'],row['before'],row['kv'],
                    *row['parameters'][:2],owner_id,slot_id,layer)
    def audit_after(self,hidden,out,owner_id,slot_id,layer): self.ops[3](hidden,out,owner_id,slot_id,layer)

    def empty(self, like, shape, dtype=None):
        dtype = getattr(self.torch, dtype) if dtype else like.dtype
        value = self.torch.empty(shape, dtype=dtype, device=like.device)
        if dtype == self.torch.int64:
            value.zero_()
        return value

    def capture_aux(self, body, stream):
        with self.torch.inference_mode():
            with self.npu.npu.stream(stream):
                body()  # Full native warmup before capture; never N/K fragments.
            stream.synchronize()
            graph = self.npu.npu.NPUGraph()
            with self.npu.npu.graph(graph, stream=stream):
                body()
        return graph

    def finish_cold_capture(self, stream): stream.synchronize()
    def replay(self, graph, stream):
        with self.npu.npu.stream(stream):
            graph.replay()

    def split_kv(self, kv):
        key, value = kv.split([HC*H, H], -1)
        return key.view(1, HC, H), value

    def compare_bf16(self, actual, expected):
        torch = self.torch
        require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.bfloat16,
                'original BF16 consumer contract changed')
        require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), 'nonfinite output')
        a, b = actual.view(torch.int16).to(torch.int32), expected.view(torch.int16).to(torch.int32)
        a, b = torch.where(a < 0, -a-32768, a), torch.where(b < 0, -b-32768, b)
        return {'bitwise_equal': bool(torch.equal(actual.view(torch.int16), expected.view(torch.int16))),
                'max_bf16_ulp': int((a-b).abs().max().item())}


def cpu_selfcheck():
    """Real ordering/ownership counterexamples, no torch import or arithmetic claim."""
    contract = _ledger_module()
    results = []

    class Tensor:
        counter = 1000
        def __init__(self, shape, dtype='bfloat16', value=None):
            Tensor.counter += 1
            self.pointer, self.shape, self.dtype, self.value, self._version = Tensor.counter, tuple(shape), dtype, value, 0
        def __getitem__(self, key): return self

    class Event:
        def __init__(self): self.complete = True; self.flag = False

    class Backend:
        kind = 'cpu-mock'
        capture = False
        auto_complete = True
        reset_completes = True
        def stamp(self, t, weight=False):
            stride = getattr(t, 'stride_override', (t.shape[1], 1) if len(t.shape) == 2 else (1,))
            return TensorStamp(id(t), t.pointer, t.shape, stride, t.dtype,
                               getattr(t, 'device_override', 'cpu-mock'), t._version, getattr(t, 'format_override', 2))
        def validate_linear(self, m): require(m.bias is None, 'mock bias')
        def ptr(self, t): return t.pointer
        def empty(self, like, shape, dtype=None): return Tensor(shape, dtype or like.dtype)
        def stream(self): return object()
        def event(self, external=False): return Event()
        def record(self, event, stream): event.flag = True; event.complete = self.auto_complete
        def reset(self, event, stream): event.flag = False
        def wait(self, event, stream): require(event.flag, 'mock wait before record')
        def query(self, event): return event.complete
        def synchronize_event(self, event): event.complete = self.reset_completes
        def read_epoch(self, tensor): return int(tensor.value or 0)
        def copy(self, out, value): out.value = value.value
        def is_real_capture(self): return self.capture
        def same_stream(self,a,b): return a is b
        def capture_aux(self, body, stream):
            self.capture = True
            body()
            self.capture = False
            return body
        def finish_cold_capture(self, stream): pass
        def replay(self, graph, stream): graph()
        def wait_copy(self, cache, out, event_id): self.wait(_EVENTS[event_id], None); self.copy(out, cache)
        def mark_consumed(self, dep, marker, epoch, event_id):
            self.copy(marker, dep); epoch.value = (epoch.value or 0)+1; self.record(_EVENTS[event_id], None)
        def split_kv(self, kv):
            return Tensor((1,HC,H),value=('key',kv.value)), Tensor((1,H),value=('value',kv.value))
        def compare_bf16(self, a, b):
            require(a.shape == b.shape and a.value == b.value, 'CPU tuple dataflow differs (no numerical claim)')
            return {'bitwise_equal':False,'max_bf16_ulp':0,'scope':'CPU tuple dataflow only; not numerical evidence'}

    class Linear:
        bias = None
        def __init__(self, layer): self.layer = layer; self.weight = Tensor((N, K))
        def forward(self, x): return Tensor((1, N), value=('full-linear', self.layer, x.value))

    class Model:
        def __init__(self): self.linear = {layer: Linear(layer) for layer in LAYERS}
        def named_modules(self): return [(f'model.layers.{l}.engram.wkv', m) for l, m in self.linear.items()]

    path = Path(__file__)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs = dict(source_paths={k: path for k in ('model','runner','gate','sdk_events')},
                  expected_sha256={k: sha for k in ('model','runner','gate','sdk_events')}, target_arm='mock-prefetch')
    backend, model = Backend(), Model()
    disabled = install(model, backend=backend, **kwargs)
    require(not disabled.select(arm='mock-prefetch', bucket_m=1, actual_tokens=1, decode=True), 'default guard bypassed')
    results.append('unverified_graph_contract_native_fallback')
    legacy_receipt = GraphSafetyReceipt('cpu-mock', 'old-host-consumed-wait', *([True]*8))
    require(not legacy_receipt.permits(backend), 'old host wait completion contract must stay disabled')
    results.append('legacy_host_consumed_wait_receipt_disabled')
    receipt = GraphSafetyReceipt('cpu-mock', 'pure-CPU-ordering-only', *([True]*8),
                                completion_fence='submission-stream')
    proto = install(model, backend=backend, safety=receipt, **kwargs)
    lookups = {layer: Tensor((1,K), value='cold') for layer in LAYERS}
    mask, main_stream = Tensor((1,), 'bool', True), object()
    identity = proto.register_slot(0, lookups, mask, graph_key='candidate-slot0', main_stream=main_stream, audit=True)
    parameters = {layer:(Tensor((HC,H),'float32','channel'),Tensor((256,256),'float32','rotation'),1e-20) for layer in LAYERS}
    proto.prepare_gate_audit(0,parameters)
    for name, args in (('prefill',dict(decode=False)),('profile',dict(decode=True,profile=True)),
                       ('M2',dict(decode=True,bucket_m=2,actual_tokens=2)),('other_arm',dict(decode=True,arm='native'))):
        call = dict(arm='mock-prefetch', bucket_m=1, actual_tokens=1, decode=True); call.update(args)
        require(not proto.select(**call), name+' wrongly selected')
        results.append(name+'_fallback')

    def rejected(name, operation):
        try:
            operation()
        except (GuardViolation, contract.ContractViolation):
            results.append(name)
        else:
            raise AssertionError(name+' incorrectly allowed')

    rejected('reset_before_cold_main_fence',lambda: proto.acquire_before_prepare(0,1,before_metadata_submit=True))
    require(not proto.select(arm='mock-prefetch',bucket_m=1,actual_tokens=1,decode=True), 'unfenced graph selected')
    results.append('unfenced_main_capture_fallback')

    def original_gate(hidden,key,value,channel,rotation,mask,eps):
        return Tensor((1,HC,H),value=('gate',hidden.value,key.value,value.value,channel.value,rotation.value,mask.value,eps))

    def captured_main_body():
        hidden = Tensor((1,HC,H), value='actual-residual')
        for layer in LAYERS:
            kv = proto.consumer_or_native(layer,lookups[layer],arm='mock-prefetch',prepared=True)
            proto.observe_gate_before(0,layer,lookups[layer],mask,hidden,kv,parameters=parameters[layer])
            key,value=backend.split_kv(kv)
            hidden=original_gate(hidden,key,value,*parameters[layer][:2],mask,parameters[layer][2])
            proto.observe_gate_after(0,layer,hidden)
        proto.record_main_consumed(0,hidden)
        return hidden

    backend.capture = True
    hidden=captured_main_body()
    backend.capture = False
    proto.finalize_main_capture(0,before_metadata_submit=True)
    rejected('sealed_audit_reallocation', lambda: proto.prepare_gate_audit(0, parameters))
    metadata={'rank':0,'request_id':'CPU-mock-not-model','positions':[1]}
    ticket = proto.acquire_before_prepare(0,1,before_metadata_submit=True,sample_metadata=metadata)
    for layer in LAYERS: lookups[layer].value=('generation',1,layer)
    proto.publish_after_prepare(ticket,lookups,mask,copy_stream=main_stream)
    proto.submit_aux(ticket)
    proto.bind_main(ticket, identity=identity)
    backend.capture = True  # CPU stand-in for replaying already-captured copies, not SDK evidence.
    hidden=captured_main_body()
    backend.capture = False
    proto.after_main_submitted(ticket, main_stream=main_stream)
    rejected('reset_in_capture', lambda: proto.acquire_before_prepare(0,2,before_metadata_submit=False))
    slot = proto.slots[0]
    slot.producer_done.complete = True; slot.consumer_done.complete = False
    rejected('producer_only_fence', lambda: proto.retire(ticket))
    slot.producer_done.complete = False; slot.consumer_done.complete = True
    rejected('consumer_only_fence', lambda: proto.retire(ticket))
    slot.producer_done.complete = slot.consumer_done.complete = True
    proto.retire(ticket, retain_audit=True)
    audit=proto.audit(0,generation=1,original_gate=original_gate)
    require(audit['passed'] and not audit['npu_validated'], 'CPU audit mislabeled as NPU evidence')
    results.append('fenced_two_layer_tuple_snapshot_bookkeeping')
    rejected('overwrite_leased_audit', lambda: proto.acquire_before_prepare(0,2,before_metadata_submit=True))
    rejected('native_prepare_overwrites_audit_lease',lambda:proto.drain_before_native_prepare(before_metadata_submit=True))
    proto.release_audit(0,1)
    second = proto.acquire_before_prepare(0,2,before_metadata_submit=True,sample_metadata=metadata)
    rejected('stale_ticket', lambda: proto.publish_after_prepare(ticket,lookups,mask,copy_stream=main_stream))
    changed = dict(lookups); changed[14] = Tensor((1,K))
    rejected('pointer_mismatch_before_producer',lambda: proto.publish_after_prepare(second,changed,mask,copy_stream=main_stream))
    for field, value in (('dtype', 'float32'), ('shape', (2, K)), ('stride_override', (K+1, 1)),
                         ('device_override', 'different-device'), ('format_override', 29)):
        original = getattr(lookups[14], field, None)
        setattr(lookups[14], field, value)
        rejected('same_pointer_lookup_' + field, lambda: proto.publish_after_prepare(second, lookups, mask, copy_stream=main_stream))
        if original is None:
            delattr(lookups[14], field)
        else:
            setattr(lookups[14], field, original)
    mask.dtype = 'int8'
    rejected('same_pointer_mask_dtype', lambda: proto.publish_after_prepare(second, lookups, mask, copy_stream=main_stream))
    mask.dtype = 'bool'
    old_audit = slot.audit_buffers[14]['after']
    slot.audit_buffers[14]['after'] = Tensor((1, HC, H))
    rejected('sealed_audit_destination_replaced', lambda: proto.publish_after_prepare(second, lookups, mask, copy_stream=main_stream))
    slot.audit_buffers[14]['after'] = old_audit
    model.linear[14].weight._version += 1
    rejected('weight_version_changed',lambda: proto.publish_after_prepare(second,lookups,mask,copy_stream=main_stream))
    model.linear[14].weight._version -= 1
    old_cache=slot.cache[14]; slot.cache[14]=Tensor((1,N))
    rejected('captured_cache_pointer_changed',lambda:proto.publish_after_prepare(second,lookups,mask,copy_stream=main_stream))
    slot.cache[14]=old_cache
    proto.publish_after_prepare(second,lookups,mask,copy_stream=main_stream)
    proto.submit_aux(second); proto.bind_main(second,identity=identity)
    proto.record_main_consumed(0,hidden); proto.after_main_submitted(second,main_stream=main_stream)
    slot.epoch.value+=1
    rejected('stale_or_duplicate_device_epoch',lambda:proto.retire(second))
    slot.epoch.value-=1
    proto.retire(second)
    results.append('two_generations_joint_fence_then_reset_reuse')
    rejected('shared_pointer_dual_slot',lambda: proto.register_slot(1,lookups,mask,graph_key='candidate-slot1',main_stream=main_stream))
    backend.reset_completes = False
    rejected('reset_enqueue_not_device_complete',lambda: proto.acquire_before_prepare(0,3,before_metadata_submit=True,sample_metadata=metadata))
    require('torch' not in sys.modules and 'torch_npu' not in sys.modules, 'CPU selfcheck imported accelerator packages')
    require(not receipt.permits(type('NPUKind',(),{'kind':'npu'})()), 'CPU receipt enabled NPU')
    results.append('CPU_receipt_cannot_enable_NPU')
    return {'scope':'Pure CPU/mock lifecycle only; no SDK graph/alias, numerical or model validation',
            'passed_cases':len(results),'cases':results,'npu_started':False,'torch_imported':False,
            'adapter_sha256':sha,'contract_sha256':CONTRACT_SHA256}


if __name__ == '__main__':
    require(sys.argv[1:] == ['--cpu-selfcheck'], 'only --cpu-selfcheck is supported; no NPU CLI')
    print(json.dumps(cpu_selfcheck(), indent=2))
