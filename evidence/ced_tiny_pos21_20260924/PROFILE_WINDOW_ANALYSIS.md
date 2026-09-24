# CED 1+1 eager / graph profiler window comparison

Date: 2026-09-24. This is a small-model (`model-tiny`), TP=1, D-role diagnostic
on A3-22 chip 7. Engram is disabled. D uses `MooncakeHybridConnector` as a KV
consumer; P remains on chip 6. It is not a production-model or long-context
benchmark. The concurrent 8+8 run is on A3-21 and shares no device with this
test.

## Request windows

The window bounds use the controller's request start time and HTTP wall time.
The request IDs match those in the saved D `serve.log`. CANN task rows do not
carry vLLM request IDs, so the trace attribution is by serialized wall-time
window, not a request tag embedded in the device trace.

| Mode | Request ID | Window | HTTP wall time | `SparseFlashMla` kernels by CANN stream | Metadata kernels |
|---|---|---:|---:|---|---:|
| Eager | `cmpl-c008829a-d829-4df8-816a-3faea922e551` | M1 | 412 ms | stream 47: 69 | stream 43: 6 |
| Eager | `cmpl-c91f4790-f081-4cdd-879f-c419a722c7a1` | M8 | 1,542 ms | stream 47: 350 | stream 43: 27 |
| Graph warmup | `cmpl-d3b55ddd-92d7-4220-867a-9eddcc171ad3` | M1 | 4,232 ms | stream 2: 40; stream 47: 40 | stream 41: 6 |
| Graph | `cmpl-bab7da9f-8c3a-4ebc-9b76-e1ad14193af3` | M1 | 332 ms | stream 47: 38 | stream 41: 3 |
| Graph | `cmpl-f6662a93-4e86-4cb2-a893-787b551b2e60` | M8 | 419 ms | stream 2: 240; stream 47: 40 | stream 41: 21 |

The graph M8 window contains six 40-kernel `SparseFlashMla` groups on stream 2.
It also contains 2,400 `EVENT_RESET` tasks on CANN connection ID `90704`, and
2,400 `EVENT_RECORD` tasks plus 1,920 `EVENT_WAIT` tasks on connection ID
`90707`. Graph warmup M1 contains one corresponding 400-reset group. The
measured graph M1 window contains no `EVENT_RESET` group. These are observed
per-window trace patterns; CANN stream IDs do not identify a logical model
stage by themselves.

The separately captured full profiler files include activity outside the HTTP
windows as well: eager has 24,344 kernel rows and 690,956 trace events; graph
has 28,817 kernel rows and 336,051 trace events. Do not use those totals as
per-request counts.

## Event and KV observations

For measured M1 / M8 respectively, the host ACL API counts and device task
counts are:

| Mode | Host `Record / Wait / MemcpyAsync` | Device `EVENT_RECORD / EVENT_WAIT / MEMCPY_ASYNC` |
|---|---|---|
| Eager M1 | 716 / 569 / 180 | 716 / 495 / 184 |
| Eager M8 | 3,621 / 2,875 / 951 | 3,621 / 2,550 / 983 |
| Graph M1 | 388 / 307 / 82 | 388 / 271 / 82 |
| Graph M8 | 468 / 348 / 510 | 2,860 / 2,227 / 791 |

`connection_id` correlates many individual ACL API events with CANN device
task events. In graph M8, replay tasks reuse connection IDs, which is why the
device task count greatly exceeds host API calls. The trace does not expose
the recorded event handle: it cannot establish which `EVENT_RECORD` a given
`EVENT_WAIT` consumes. `MEMCPY_ASYNC` appears as a `PCIE_DMA_SQE` device task,
but the trace does not report copy direction or bytes, so these events alone
cannot be called H2D or assigned to the Mooncake KV transfer.

The server log does confirm the transport order for all three graph requests:
the Mooncake KV transfer completes, then the D scheduler logs CED replay for
`prompt=22 loaded=21`, with replay positions `0..20`, then returns HTTP 200.
Transfer times are 210.74 ms for the first warmup request, 0.92 ms for measured
M1, and 0.85 ms for measured M8. The eager measured requests report 1.09 ms and
0.98 ms. Those connector log durations are direct evidence of completed KV
loads; they are not inferred from the CANN `MEMCPY_ASYNC` count.

The source-level schedule makes the intended boundary explicit:

- `experimental/ced/core_scheduler_replay.patch` validates the loaded prefix,
  replays positions `0..20`, and limits the replay chunk before the uncached
  prompt token at position 21.
- `experimental/ced/dsa_v41.py` selects the replay-chunk path for multi-token
  prefill and calls sparse-index selection, attention and O projection. Its
  log marks the reuse of the global source and C2 ring.

The trace shows cross-stream record/wait activity around the attention
kernels, and the server log shows KV load then replay then request completion.
The trace cannot prove the exact producer-to-consumer event edge for the final
prompt token because the event handle and vLLM request ID are absent. Therefore
these profiler results describe the execution and its ordering; they do not
identify a correctness root cause. The independent real-weight A/B that forces
only the one-token prompt tail eager is stronger root-cause evidence and should
be evaluated separately from stream-ID changes here.

## Capture quality and artifacts

Both `/start_profile` and `/stop_profile` returned HTTP 200. Both runs emitted
the profiler warning that stopping in `RECORD` may leave incomplete parsed
data; CANN export nevertheless completed and generated `kernel_details.csv`,
`trace_view.json` and the other parser outputs. The graph run also logged an
external callback thread warning, while the service stayed up and all three
profile-window requests returned HTTP 200. Keep these warnings attached to any
interpretation of the trace.

- Window analysis: `profile_window_summary.json`
- Reproducible summarizer: `analyze_prof_windows.py`
- Raw archive hashes and download instructions: `PROFILE_ARCHIVES.md`
- Graph raw trace: private COS object `share/xfer/ced_tiny_stream_prof_d_graph_20260924.tar.gz`
- Graph control bundle: private COS object `share/xfer/ced_tiny_stream_prof_d_graph_20260924_controls.tar.gz`
- Raw graph trace archive SHA-256: `e59f6081406f34a36fdf4f24c514a1bbd085403165e12d9b37dda437f8439abf`
- Graph control bundle SHA-256: `d8a2b47f2f26c32de5009d51f5e52729c191ab2eda8aa1774d8447b2f1be47a6`
