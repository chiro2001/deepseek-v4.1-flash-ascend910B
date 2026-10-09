"""Audit-only fix for single-DP TP8 sequence-sharded route export."""
import hashlib
import inspect

INSTALLED = False
RECORDS = []
CALLS = []
SOURCE_SHA256 = None


def install():
    global INSTALLED, SOURCE_SHA256
    if INSTALLED: return
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
    from vllm.forward_context import get_forward_context
    from vllm.distributed import get_tp_group
    original = RoutedExpertsCapturer.capture
    SOURCE_SHA256 = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()

    def capture(self, layer_id, topk_ids):
        ctx = get_forward_context()
        if len(CALLS) < 80:
            CALLS.append({'layer':layer_id,'local_tokens':topk_ids.shape[0],
                          'ctx_tokens':getattr(ctx,'num_tokens',None),'dp_none':ctx.dp_metadata is None,
                          'attn_type':type(ctx.attn_metadata).__name__})
        if ctx.dp_metadata is None and self.tp_size == 8:
            local = topk_ids.shape[0]
            total = int(getattr(ctx,'num_tokens',local))
            if local != total and total > 0:
                assert local == (total+self.tp_size-1)//self.tp_size, (local,total)
                assert total <= self.device_buffer.shape[0], (total,self.device_buffer.shape)
                topk_ids = get_tp_group().all_gather(topk_ids,dim=0)[:total]
                RECORDS.append({'layer':layer_id,'local_tokens':local,'global_tokens':total})
        return original(self, layer_id, topk_ids)

    RoutedExpertsCapturer.capture = capture; INSTALLED = True
    print('AUDIT_ROUTE_CAPTURE_PATCH', SOURCE_SHA256, flush=True)


def status(worker):
    return {'installed': INSTALLED, 'original_capture_sha256': SOURCE_SHA256,
            'actual_id_gathers': len(RECORDS), 'layouts': sorted({(r['local_tokens'],r['global_tokens']) for r in RECORDS}),
            'call_samples':CALLS[:5],
            'note': 'Audit-only route export; model routing and formal timing paths are unchanged'}
