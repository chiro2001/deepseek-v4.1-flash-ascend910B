"""Read stock CPU scheduling metadata; never infer steps from frontend polls."""
def is_real_decode(scheduler, requests):
    if not scheduler.num_scheduled_tokens:
        if scheduler.total_num_scheduled_tokens != 0:
            raise ValueError('Empty schedule with nonzero token total')
        return False
    if len(scheduler.num_scheduled_tokens) != 1:
        raise ValueError('Profile requires exactly one actual scheduled request')
    req_id = next(iter(scheduler.num_scheduled_tokens))
    new = {row.req_id: row for row in scheduler.scheduled_new_reqs}
    state = new.get(req_id) or requests.get(req_id)
    if state is None or not state.prompt_token_ids:
        raise ValueError('Profile has no actual CPU prompt binding')
    cached = scheduler.scheduled_cached_reqs
    if len(cached.req_ids) != len(cached.num_computed_tokens):
        raise ValueError('Unknown cached scheduler ABI')
    done = state.num_computed_tokens if req_id in new else dict(zip(cached.req_ids, cached.num_computed_tokens)).get(req_id)
    if type(done) is not int or done < 0:
        raise ValueError('Profile has no actual computed-token boundary')
    return done >= len(state.prompt_token_ids)
