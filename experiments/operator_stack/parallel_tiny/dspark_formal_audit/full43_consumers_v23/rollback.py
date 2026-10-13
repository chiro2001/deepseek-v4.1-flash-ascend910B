"""Existing CPU rollback correspondence only, never a device consumer proof."""


def validate_rollback(*, actual_inputs,positions,drafts,sampled,computed_before,computed_after,
                      scheduled_inputs,remaining_output_budget,scheduler_usable_output):
    if not sampled or actual_inputs[1:]!=drafts or len(actual_inputs)!=len(drafts)+1:
        raise ValueError('Actual target inputs are not last actual + the real proposal')
    accepted=len(sampled)-1
    if accepted>len(drafts) or sampled[:accepted]!=drafts[:accepted]:
        raise ValueError('Native accepted output is not the real draft prefix')
    if scheduled_inputs!=len(actual_inputs) or positions!=list(range(computed_before,computed_before+scheduled_inputs)):
        raise ValueError('Actual cache input positions/scheduling differ')
    rejected=len(drafts)-accepted
    if computed_after!=computed_before+scheduled_inputs-rejected:
        raise ValueError('Native scheduler rejected-token rollback differs')
    usable=sampled[:remaining_output_budget]
    if scheduler_usable_output!=usable:
        raise ValueError('Actual scheduler stop/remaining-budget trim differs')
    return {'accepted_drafts':accepted,'rejected_drafts':rejected,
            'committed_cache_input_tokens':actual_inputs[:accepted+1],
            'committed_cache_positions':positions[:accepted+1],
            'next_round_bonus_input':sampled[-1],
            'next_round_bonus_position':computed_before+accepted+1,
            'cpu_correspondence_passed':True,'actual_cache_write_read_proven':False,
            'device_audit_passed':False,'full_consumer_audit_passed':False,
            'safe_formal_receipt':False,'formal_model_validated':False}
