"""Return native-signature candidate callables without installing model hooks.

Default binding/entries are absent, so every call delegates to bound native.
This file contains no capture implementation, graph API or device selection.
Future prepared entries must be created under separate SDK/consumer evidence.
"""
from dataclasses import dataclass
from context_wiring_cpu import dispatch_original_site,LeaseError

@dataclass(frozen=True)
class PreparedEntry:
    signature:tuple
    invoke:object
    owner:object

class ContextAdapters:
    def __init__(self,native_combine,native_context_kv,binding_provider=None,prepared_entries=None):
        self.native={'combine':native_combine,'context_kv':native_context_kv}
        self.binding_provider=binding_provider
        self.prepared_entries={} if prepared_entries is None else dict(prepared_entries)
        self.active_leases=[]
        self.last_receipts=[]

    def _at_original_site(self,stage,args):
        native=self.native[stage]
        if self.binding_provider is None:
            self.last_receipts.append({'stage':stage,'mode':'native','reason':'bindings_not_provided'})
            return native(*args)
        current=self.binding_provider(stage,args)
        if current is None:
            self.last_receipts.append({'stage':stage,'mode':'native','reason':'actual_binding_unavailable'})
            return native(*args)
        entry=self.prepared_entries.get(stage)
        signature=entry.signature if entry is not None else None
        def prepared(lease):
            # Keep real positional/keyword-equivalent tensor arguments, native
            # bound model/weights/cache owners, prepared entry/graph/pool owner
            # and the descriptor bundle alive through caller completion.
            lease.keepalive=(args,tuple(self.native.values()),entry,current)
            self.active_leases.append(lease)
            return entry.invoke(args,lease)
        result,receipt=dispatch_original_site(stage,current,signature,lambda:native(*args),prepared)
        self.last_receipts.append({'stage':stage,**receipt})
        return result

    def combine_hidden_states(self,aux_hidden_states):
        return self._at_original_site('combine',(aux_hidden_states,))

    def precompute_and_store_context_kv(self,context_states,context_positions,context_slot_mapping=None):
        return self._at_original_site('context_kv',(context_states,context_positions,context_slot_mapping))

    def assert_can_refresh_or_rebind(self):
        if self.active_leases:raise LeaseError('inputs/cache/entry owners remain pinned to a pending consumer')

    def release_completed_lease(self,lease):
        if lease not in self.active_leases:raise LeaseError('unknown lease')
        if lease.poisoned:raise LeaseError('partial component failure needs explicit parent terminal cleanup proof')
        lease.release()
        lease.keepalive=()
        self.active_leases.remove(lease)
