"""Event protocol for an explicitly authorized future same-instance M1 gate.

This module neither initializes/selects a device nor starts a standalone run.
Caller supplies original SDK tensor/stream owners, validated bridge handle,
fresh preflight and numeric callbacks. No NPU use during CPU preparation.
"""
import time


METHODS = ('SDK', 'RAW_NATIVE', 'CANDIDATE')


def interleaved_order(round_index):
    # Every block rotates first/last order; no method is always measured first.
    shift = round_index % 3
    return METHODS[shift:] + METHODS[:shift]


def measure_one(torch, stream, method, restore_same_gm_input, submit,
                finish_and_verify, verify_readonly_seals):
    """Three events: restoration-inclusive and operator windows share one end.

    Binding/lists/scratch initialization occurs before this function. Restore
    performs only same-GM x refresh on this stream. submit performs original
    SDK call or HostArgs native or planner->candidate, with no sync/D2H/clear.
    Stop event precedes all snapshots and checks; descriptor/args/module/GM
    owners must outlive stop completion and finish_and_verify.
    """
    begin_fresh = torch.npu.Event(enable_timing=True)
    begin_operator = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    begin_fresh.record(stream)
    restore_same_gm_input()
    begin_operator.record(stream)
    host_begin = time.perf_counter_ns()
    lifetime_owner = submit(method)
    host_submit_end = time.perf_counter_ns()
    end.record(stream)
    end.synchronize()
    host_complete = time.perf_counter_ns()
    # Numeric/frame/canary/seal reads are outside both device-event windows.
    finish_and_verify(method, lifetime_owner)
    verify_readonly_seals()
    result = {'method': method,
              'fresh_input_and_operator_event_ms': begin_fresh.elapsed_time(end),
              'operator_event_ms': begin_operator.elapsed_time(end),
              'host_submit_ns': host_submit_end - host_begin,
              'host_submit_through_stop_sync_ns': host_complete - host_begin,
              'planner_in_operator_window': method == 'CANDIDATE',
              'restored_same_GM_x_before_call': True,
              'D2H_oracle_canaries_seals_outside_timing': True,
              'scope': 'same-instance M1 capacity6 operator/queue event durations'}
    # Events on one stream include its queued idle time while the host submits
    # work. These are not an isolated instruction/core-time measurement.
    return result
