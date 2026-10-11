"""Pure CPU reference ledger for full Engram prefetch; no torch or NPU calls.

Observed completion/cleared flags must come from the real integration. This
ledger validates metadata/order; it cannot prove CANN capture or reset safety.
"""
from dataclasses import dataclass, replace
import json


class ContractViolation(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractViolation(message)


@dataclass(frozen=True)
class GraphIdentity:
    slot: int
    bucket_m: int
    lookup_ptrs: tuple[int, int]
    mask_ptr: int
    kv_ptrs: tuple[int, int]
    weight_version: str
    ready_event_ids: tuple[int, int]
    graph_key: str
    output_n: int = 25600
    reduction_k: int = 6144

    def __post_init__(self):
        require(self.bucket_m == 1, 'first prototype supports M=1 decode only')
        require((self.output_n, self.reduction_k) == (25600, 6144), 'original full MatMul shape required')
        require(self.slot >= 0 and bool(self.weight_version) and bool(self.graph_key), 'identity incomplete')
        require(len(self.lookup_ptrs) == len(self.kv_ptrs) == len(self.ready_event_ids) == 2, 'two layers required')
        require(len(set(self.ready_event_ids)) == 2, 'ready events must be per layer')
        require(all(p > 0 for p in (*self.lookup_ptrs, self.mask_ptr, *self.kv_ptrs)), 'persistent addresses required')


@dataclass(frozen=True)
class Ticket:
    generation: int
    actual_tokens: int
    identity: GraphIdentity


class SlotLedger:
    """One stable captured slot, with explicit observed producer/consumer fences."""

    def __init__(self, identity):
        self.identity = identity
        self.phase = 'FREE'
        self.last_generation = -1
        self.ticket = None
        self.producer_done = True
        self.consumer_done = True

    def check_ticket(self, ticket):
        require(ticket == self.ticket, 'stale generation or changed slot/bucket/pointers/version/event identity')

    def acquire(self, generation, actual_tokens=1):
        require(self.phase == 'FREE' and self.producer_done and self.consumer_done,
                'cannot overwrite slot before both old fences are observed complete')
        require(generation > self.last_generation, 'generation must increase')
        require(0 < actual_tokens <= self.identity.bucket_m, 'actual tokens must fit graph bucket')
        self.ticket = Ticket(generation, actual_tokens, self.identity)
        self.last_generation = generation
        self.phase = 'ACQUIRED'
        return self.ticket

    def arm(self, ticket, ready_flags_clear):
        self.check_ticket(ticket)
        require(self.phase == 'ACQUIRED' and self.producer_done and self.consumer_done,
                'reset/arm requires previous record and wait tasks completed')
        require(ready_flags_clear, 'ExternalEvent ready must be cleared before new generation')
        self.phase = 'ARMED'

    def publish_lookup(self, ticket, all_copies_event_recorded, after_free_fence):
        self.check_ticket(ticket)
        require(self.phase == 'ARMED', 'lookup write requires armed free slot')
        require(all_copies_event_recorded and after_free_fence,
                'lookup/mask/padding copies and overwrite fence must precede input-ready event')
        self.phase = 'LOOKUP_PUBLISHED'

    def submit_producer(self, ticket, waits_input_ready, captured_ready_layers=(1, 14)):
        self.check_ticket(ticket)
        require(self.phase == 'LOOKUP_PUBLISHED', 'producer needs current input-ready event')
        require(waits_input_ready and tuple(captured_ready_layers) == (1, 14),
                'aux must wait input-ready and issue both ready-record nodes')
        self.producer_done = False
        self.phase = 'PRODUCER_SUBMITTED'

    def bind_consumer(self, ticket, main_identity, external_wait_layers=(1, 14)):
        self.check_ticket(ticket)
        require(self.phase == 'PRODUCER_SUBMITTED', 'producer must be submitted before main external waits')
        require(main_identity == self.identity and tuple(external_wait_layers) == (1, 14),
                'consumer must bind same tensors/events and wait both layers')
        self.consumer_done = False
        self.phase = 'CONSUMER_BOUND'

    def observe_done(self, ticket, producer=False, consumer=False):
        self.check_ticket(ticket)
        require(self.phase == 'CONSUMER_BOUND', 'completion belongs to bound current generation')
        require(producer or consumer, 'an actual completion observation is required')
        self.producer_done |= producer
        self.consumer_done |= consumer

    def retire(self, ticket):
        self.check_ticket(ticket)
        require(self.phase == 'CONSUMER_BOUND' and self.producer_done and self.consumer_done,
                'retire requires producer_done and gate/residual consumer_done observed complete')
        self.phase = 'FREE'
        self.ticket = None


def selfcheck():
    identity = GraphIdentity(0, 1, (101, 102), 103, (201, 202), 'formal-weight-v1', (301, 302), 'decode-slot0-v1')
    results = []

    def rejected(name, operation):
        try:
            operation()
        except ContractViolation:
            results.append({'case': name, 'expected_rejection': True, 'passed': True})
        else:
            raise AssertionError(name + ' incorrectly allowed')

    ledger = SlotLedger(identity)
    ticket = ledger.acquire(1)
    rejected('ready_sticky_without_reset', lambda: ledger.arm(ticket, False))
    ledger.arm(ticket, True)
    rejected('publish_before_full_copy_fence', lambda: ledger.publish_lookup(ticket, False, True))
    ledger.publish_lookup(ticket, True, True)
    rejected('consumer_before_producer_submission', lambda: ledger.bind_consumer(ticket, identity))
    rejected('missing_layer14_ready_record', lambda: ledger.submit_producer(ticket, True, (1,)))
    ledger.submit_producer(ticket, True)
    rejected('changed_kv_address', lambda: ledger.bind_consumer(ticket, replace(identity, kv_ptrs=(211, 212))))
    rejected('changed_event_or_weight_identity', lambda: ledger.bind_consumer(ticket, replace(identity, weight_version='other')))
    ledger.bind_consumer(ticket, identity)
    rejected('overwrite_before_old_gate_done', lambda: ledger.acquire(2))
    rejected('reset_while_old_waiter_inflight', lambda: ledger.arm(ticket, True))
    ledger.observe_done(ticket, producer=True)
    rejected('retire_with_producer_only', lambda: ledger.retire(ticket))
    ledger.observe_done(ticket, consumer=True)
    ledger.retire(ticket)
    second = ledger.acquire(2)
    rejected('stale_generation_ticket', lambda: ledger.arm(ticket, True))
    ledger.arm(second, True)
    ledger.publish_lookup(second, True, True)
    ledger.submit_producer(second, True)
    ledger.bind_consumer(second, identity)
    ledger.observe_done(second, consumer=True)
    rejected('retire_with_consumer_only', lambda: ledger.retire(second))
    ledger.observe_done(second, producer=True)
    ledger.retire(second)
    results.append({'case': 'two_generations_serial_slot_reuse', 'passed': True})
    rejected('unregistered_m_bucket', lambda: replace(identity, bucket_m=2))
    rejected('n_dimension_sharding_forbidden', lambda: replace(identity, output_n=3200))
    return {'scope': 'Pure CPU metadata/order model; no SDK Event/capture/stream execution validated',
            'torch_imported': False, 'npu_started': False, 'passed_cases': len(results), 'cases': results}


if __name__ == '__main__':
    print(json.dumps(selfcheck(), indent=2))
