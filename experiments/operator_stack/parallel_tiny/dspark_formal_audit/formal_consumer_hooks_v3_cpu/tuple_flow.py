"""Explicit Tensor-only native consumer envelopes for a NEW audit graph.

No install or SDK import. Every envelope returns native Tensor carriers plus
all owned buffers; the private outer graph must preserve those outputs.
"""
from parallel_tiny.dspark_formal_audit.native_initialized_worker_gate_v2.tensor_tuple_transport import snapshot_tuple


def tap(torch, carrier, payload, positions, valid, epoch, buffers):
    return snapshot_tuple(torch,carrier,payload,positions,valid,epoch,buffers)


def native_layer_output_tuple(torch, native_result, positions, valid, epoch, buffers):
    hidden,mix=native_result
    out=tap(torch,hidden,hidden.reshape(hidden.shape[0],-1),positions.reshape(-1,1),valid,epoch,buffers)
    return (out[0],mix,*out[1:])


def actual_aux_entry_tuple(torch, actual_aux_tensor, positions, valid, epoch, buffers):
    # actual_aux_tensor is the stock hidden_states.mean(dim=1) result entering
    # target37/38/39. Do not recompute/shift the mean to another layer.
    return tap(torch,actual_aux_tensor,actual_aux_tensor,positions.reshape(-1,1),valid,epoch,buffers)


def combine_tuple(torch, draft_model, actual_aux_concat, positions, valid, epoch, input_buffers, proj_buffers, norm_buffers):
    before=tap(torch,actual_aux_concat,actual_aux_concat,positions.reshape(-1,1),valid,epoch,input_buffers)
    projected=draft_model.main_proj(before[0])
    proj=tap(torch,projected,projected,positions.reshape(-1,1),valid,epoch,proj_buffers)
    normalized=draft_model.main_norm(proj[0])
    norm=tap(torch,normalized,normalized,positions.reshape(-1,1),valid,epoch,norm_buffers)
    return (norm[0],*before[1:],*proj[1:],*norm[1:])


def context_wkv_norm_rope_tuple(torch, attn, original_apply_rope, context_hidden, positions, valid, epoch,
                                input_buffers,wkv_buffers,norm_buffers,kv_buffers):
    before=tap(torch,context_hidden,context_hidden,positions.reshape(-1,1),valid,epoch,input_buffers)
    wkv=attn.wkv(before[0])
    projected=tap(torch,wkv,wkv,positions.reshape(-1,1),valid,epoch,wkv_buffers)
    normalized=attn.kv_norm(projected[0])
    norm=tap(torch,normalized,normalized,positions.reshape(-1,1),valid,epoch,norm_buffers)
    k_nope,k_pe=norm[0].split([attn.nope_head_dim,attn.rope_head_dim],dim=-1)
    k_pe=original_apply_rope(attn.rotary_emb,positions,k_pe.unsqueeze(1)).squeeze(1)
    shared_kv=torch.cat([k_nope,k_pe],dim=-1).view(-1,1,attn.head_dim).contiguous()
    final=tap(torch,shared_kv,shared_kv.reshape(shared_kv.shape[0],-1),positions.reshape(-1,1),valid,epoch,kv_buffers)
    return (final[0],*before[1:],*projected[1:],*norm[1:],*final[1:])


def actual_route_pair_tuple(torch, native_route_result, positions, valid, epoch, id_buffers, weight_buffers):
    weights,ids=native_route_result
    selected=tap(torch,ids,ids,positions.reshape(-1,1),valid,epoch,id_buffers)
    weighted=tap(torch,weights,weights,positions.reshape(-1,1),valid,epoch,weight_buffers)
    return (weighted[0],selected[0],*selected[1:],*weighted[1:])


def before_native_pack_tuple(torch, original_x, expanded_positions, expanded_valid, epoch, activation_buffers):
    # Capture the actual GMM input BEFORE raw INT4 pack writes its GM in place.
    # Independent storage + epoch is essential; versions/references are not.
    return tap(torch,original_x,original_x.reshape(original_x.shape[0],-1),
               expanded_positions,expanded_valid,epoch,activation_buffers)


def before_scatter_tuple(torch, live_updates, actual_coordinates, epoch, buffers):
    valid=(actual_coordinates[:,0]>=0)&(actual_coordinates[:,1]>=0)
    return tap(torch,live_updates,live_updates.reshape(actual_coordinates.shape[0],-1),
               actual_coordinates.to(torch.int64),valid,epoch,buffers)


def after_write_before_reader_tuple(torch, q, actual_cache, actual_coordinates, epoch, buffers):
    valid=((actual_coordinates[:,0]>=0)&(actual_coordinates[:,1]>=0)
           &(actual_coordinates[:,0]<actual_cache.shape[0])&(actual_coordinates[:,1]<actual_cache.shape[1]))
    blocks=actual_coordinates[:,0].long().clamp(0,actual_cache.shape[0]-1)*valid.long()
    offsets=actual_coordinates[:,1].long().clamp(0,actual_cache.shape[1]-1)*valid.long()
    values=actual_cache[blocks,offsets].reshape(actual_coordinates.shape[0],-1)
    return tap(torch,q,values,actual_coordinates.to(torch.int64),valid,epoch,buffers)


def per_group_metadata_tuple(torch, native_carrier, payload, actual_positions_and_slots, valid, epoch, buffers):
    return tap(torch,native_carrier,payload,actual_positions_and_slots,valid,epoch,buffers)


def sampled_tokens_tuple(torch, sampled_ids, request_position, epoch, buffers):
    # Native sampled IDs include rejected/padded -1 cells. Their actual values
    # are copied unchanged; CPU correspondence is checked only after a fence.
    rows=sampled_ids.shape[0]
    valid=torch.ones((rows,),dtype=torch.bool,device=sampled_ids.device)
    return tap(torch,sampled_ids,sampled_ids,request_position,valid,epoch,buffers)
