"""Bounded same-worker Python/C call timing, separate from paired performance."""
import cProfile
import pstats

PROFILE = None


def start(worker):
    global PROFILE
    assert PROFILE is None
    PROFILE = cProfile.Profile()
    PROFILE.enable()
    return {'started':True,'scope':'Current RPC/execute thread; no NPU profiler'}


def stop(worker):
    global PROFILE
    assert PROFILE is not None
    profile=PROFILE;profile.disable();PROFILE=None
    stats=pstats.Stats(profile)
    rows=[]
    for (filename,line,name),(primitive,calls,self_s,cumulative_s,_) in stats.stats.items():
        rows.append({'file':filename,'line':line,'function':name,'calls':calls,
                     'self_ms':self_s*1000,'cumulative_ms':cumulative_s*1000})
    rows.sort(key=lambda r:r['cumulative_ms'],reverse=True)
    from vllm.distributed import get_tensor_model_parallel_rank
    return {'rank':get_tensor_model_parallel_rank(),'functions':rows[:100],
            'total_self_ms':stats.total_tt*1000,'total_calls':stats.total_calls,
            'scope':'Diagnostic only: cProfile perturbs timing; nested cumulative times cannot be added; '
                    'synchronization includes device/arrival waits and is not pure CPU work'}
