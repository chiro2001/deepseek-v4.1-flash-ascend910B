"""Host control protocol only. Device epochs/order need a later actual audit.

The ledger never observes contents by _version. A native consume dirties the
lease conservatively, even if metadata/version stay identical or R is zero.
This ledger is not placed inside a replay graph or used to claim device proof.
"""
from dataclasses import dataclass


@dataclass
class RawLease:
    owner_key:tuple
    stage:str
    x:object
    original_ptr:int
    generation:int
    state:str='fresh'
    fixed_snapshot:object=None


class RawLifecycle:
    def __init__(self):
        self.leases={}

    def producer_returned(self, owner_key, stage, x, generation, *, prior_fence=None, producer_callback=None):
        key=(tuple(owner_key),stage,int(x.data_ptr()))
        previous=self.leases.get(key)
        if previous is not None:
            if generation<=previous.generation or prior_fence is None or producer_callback is None:
                raise ValueError('Reusing raw GM needs an invoked new producer after a prior fence')
            prior_fence()  # callbacks, not asserted True flags or _version
            if producer_callback() is not x:
                raise ValueError('Actual refeed returned another raw GM carrier')
        lease=RawLease(tuple(owner_key),stage,x,int(x.data_ptr()),generation)
        self.leases[key]=lease
        return lease

    def snapshot(self, lease, x, fixed_snapshot):
        if lease.state!='fresh' or x is not lease.x or int(x.data_ptr())!=lease.original_ptr:
            raise ValueError('Snapshot must precede the consumer at the bound original GM/owner')
        if fixed_snapshot is x or fixed_snapshot.untyped_storage().data_ptr()==x.untyped_storage().data_ptr():
            raise ValueError('Fixed raw snapshot must have independent storage')
        lease.fixed_snapshot=fixed_snapshot
        lease.state='snapshotted'

    def enter_consumer(self, lease):
        if lease.state!='snapshotted':
            raise ValueError('Native prepack cannot start without its fixed before-pack snapshot')
        lease.state='consumed'  # do not infer preservation from x._version

    def restore_same_original_GM(self, lease, *, prior_fence):
        if lease.state!='consumed' or lease.fixed_snapshot is None or prior_fence is None:
            raise ValueError('A/A reuse needs completed prior native work and the original raw snapshot')
        prior_fence()
        lease.x.copy_(lease.fixed_snapshot)
        if int(lease.x.data_ptr())!=lease.original_ptr:
            raise ValueError('Restore replaced the original GM carrier')
        lease.state='fresh'
