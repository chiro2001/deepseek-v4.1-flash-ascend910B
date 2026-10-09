"""TP1 synchronous tiny replay diagnostics and guarded pre-replay barrier trial."""
import time

import torch

import goal20_patches as goal

ENABLED=False
DIAGNOSTIC=False
RECORDS=[]
COUNTS={'skip':0,'normal':0}


def install(model):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    original=ACLGraphWrapper.__call__
    goal.CONFIGS['nosync']={'hcstatic','hcpost','route','gmm1','gmm1act'}
    def call(self,*args,**kwargs):
        ctx=get_forward_context()
        entry=self.concrete_aclgraph_entries.get(ctx.batch_descriptor)
        eligible=(self.runtime_mode==CUDAGraphMode.FULL and ctx.cudagraph_runtime_mode==self.runtime_mode
                  and entry is not None and entry.aclgraph is not None and not ctx.capturing
                  and not _EXTRA_CTX.is_draft_model and not self.enable_enpu
                  and self.vllm_config.parallel_config.tensor_parallel_size==1
                  and not self.vllm_config.scheduler_config.async_scheduling
                  and getattr(ctx.batch_descriptor,'num_tokens',None)==1)
        if not eligible:return original(self,*args,**kwargs)
        if goal.ARM=='nosync':
            COUNTS['skip']+=1
            entry.aclgraph.replay()
            return entry.output
        COUNTS['normal']+=1
        if DIAGNOSTIC:
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record();start=time.perf_counter()
            result=original(self,*args,**kwargs)
            wall=(time.perf_counter()-start)*1000
            end.record();RECORDS.append((begin,end,wall));return result
        return original(self,*args,**kwargs)
    ACLGraphWrapper.__call__=call


def diagnostics(worker,enabled):
    global DIAGNOSTIC
    torch.npu.synchronize();DIAGNOSTIC=enabled
    if enabled:RECORDS.clear()
    return {'diagnostic':enabled}


def summarize(worker):
    import statistics
    torch.npu.synchronize()
    gpu=[a.elapsed_time(b) for a,b,_ in RECORDS]
    cpu=[wall for _,_,wall in RECORDS]
    return {'records':len(gpu),'replay_device_event_ms':gpu,'replay_call_wall_ms':cpu,
            'device_event_median_ms':statistics.median(gpu) if gpu else None,
            'call_wall_median_ms':statistics.median(cpu) if cpu else None,
            'counts':dict(COUNTS),'note':'Diagnostic event span includes queues/host gaps between records; not formal throughput'}
