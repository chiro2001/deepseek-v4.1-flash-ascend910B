"""Future K5 planner CPU protocol, not a device implementation or GMM hook."""
from dataclasses import dataclass
from numbers import Integral

MAGIC=0x47534D44
VERSION=1
HEADER_WORDS=8
SLOT_WORDS=4

@dataclass(frozen=True)
class Role:
    name:str
    experts:int
    capacity:int
    k:int=2304
    n:int=5120

    @property
    def frame_bytes(self):
        return ((4*(HEADER_WORDS+SLOT_WORDS*self.capacity)+127)//128)*128

TARGET=Role('target',48,36)
DRAFT=Role('draft',16,15)

class NativeInputContractError(ValueError):pass

def native_counts(role,counts):
    if len(counts)!=role.experts:raise NativeInputContractError('full native expert count vector required')
    if any(isinstance(n,bool) or not isinstance(n,Integral) or not 0<=int(n)<2**63 for n in counts):
        raise NativeInputContractError('native counts require nonnegative signed-int64 values')
    values=tuple(int(n) for n in counts)
    if sum(values)>role.capacity:raise NativeInputContractError('native row capacity exceeded')
    return values

def make_frame(role,counts):
    counts=native_counts(role,counts);R=sum(counts);active=sum(c>0 for c in counts)
    words=[MAGIC,VERSION,role.experts,role.capacity,R,active,0,0]
    begin=0
    for expert,count in enumerate(counts):
        if count:words.extend([expert,begin,count,0])
        begin+=count
    words.extend([0]*(role.frame_bytes//4-len(words)))
    return tuple(words)

def select_groups(role,counts,frame):
    """Validate the entire frame before choosing sparse or full native visits.

    Bad frame on valid native input visits every original native group. Illegal
    native counts are rejected separately; fallback cannot legalize them.
    """
    counts=native_counts(role,counts)
    native=tuple((expert,sum(counts[:expert]),count) for expert,count in enumerate(counts))
    fallback={'mode':'full_native_fallback','groups':native,'group_visit_ids':tuple(range(role.experts))}
    try:
        if len(frame)!=role.frame_bytes//4:return fallback
        if any(not isinstance(w,Integral) or isinstance(w,bool) or not 0<=int(w)<2**32 for w in frame):return fallback
        frame=tuple(int(w) for w in frame)
        R=sum(counts);active=sum(c>0 for c in counts)
        if frame[:8]!=(MAGIC,VERSION,role.experts,role.capacity,R,active,0,0):return fallback
        if active>min(role.capacity,role.experts,R):return fallback
        expected=[];begin=0
        for expert,count in enumerate(counts):
            if count:expected.append((expert,begin,count,0))
            begin+=count
        # This pass completes before any MMCompute/vector handshake is issued.
        for slot,entry in enumerate(expected):
            if frame[8+4*slot:12+4*slot]!=entry:return fallback
        if any(frame[8+4*active:]):return fallback
        sparse=tuple(entry[:3] for entry in expected)
        return {'mode':'sparse','groups':sparse,'group_visit_ids':tuple(g[0] for g in sparse)}
    except (TypeError,ValueError,IndexError):return fallback

def aic_work_trace(groups,*,single_m,single_n,cores,n=5120):
    """Original two-piece MSD scheduling algebra; no arithmetic body executes."""
    result=[];pre_count=0;blocks_n=(n+single_n-1)//single_n
    for expert,begin,count in groups:
        blocks_m=(2*count+single_m-1)//single_m
        current=pre_count+blocks_n*blocks_m
        for core in range(cores):
            block=core if core>=pre_count else core+cores
            while block<current:
                result.append((core,expert,2*begin,2*count,(block-pre_count)//blocks_n,(block-pre_count)%blocks_n))
                block+=cores
        pre_count=current%cores
    return tuple(result)
