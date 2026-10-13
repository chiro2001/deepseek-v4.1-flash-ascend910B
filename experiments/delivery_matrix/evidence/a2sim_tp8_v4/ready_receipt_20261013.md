# A2v4 init-only ready receipt (2026-10-13)

The principal service is initialized and ready, but this is **not** a generation,
quality, or performance result. Only the two authorized GET ownership checks
below were sent; no chat/completions or other generation request was issued.

## Live service identity

- Target container: `dsv41-ced-p0-7-20261013-v2`, CID
  `16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`,
  `running`, restart count `0`; host device cgroup grants physical `/dev/davinci0`
  through `/dev/davinci7` only.
- Runner PID `342556`, API PID `342558`, EngineCore PID `343435`; all eight
  `VLLM::Worker_TP{0..7}_EP{0..7}` processes were alive at ready verification.
  `run.exit` is absent because the service is still running.
- Listener: `127.0.0.1:18960` (`ss` confirmed). No separate P prefill service
  or proxy was started.
- Actual runtime plan path:
  `/home/l00886679/tmp/dsv41-a2sim-v4-runtime/a4/service/plan.json`, SHA-256
  `5716d78594ba91c17c14a86b0eebaaf148bef43752c8fa07c1c26a4a60b7b02b`.
  Actual runtime fingerprint is
  `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`;
  this is the service output plan, not the CPU-only preflight plan.
- Actual service selection path:
  `/home/l00886679/tmp/dsv41-a2sim-v4-runtime/a4/service/service_selection.json`,
  SHA-256 `70548f84722bee9d70ad54df538cdcfe48880aaa2b2c6c028564cde4cb9e2616`.
  It reports `engine_initialized=true`, `ASGI_startup_completed=true`, 8 rank
  descriptions, served model `dsv41-delivery-a2sim-tp8-v4`, and SID
  `9d2ba214732b4eed84dc4f416f27c6ae`.

## GET ownership checks

- `GET http://127.0.0.1:18960/v1/models` returned HTTP 200, 525 bytes, SHA-256
  `2b9807e42d0d8572f6def5585dd6d2a588339e7d8878307886f946e6803e5fba`; the only
  model id was `dsv41-delivery-a2sim-tp8-v4`.
- `GET http://127.0.0.1:18960/delivery/identity` returned HTTP 200, 12,349 bytes,
  SHA-256 `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`;
  it reported the same model, SID, runtime fingerprint, engine/ASGI readiness,
  and all eight rank descriptions.

## Eight-rank runtime gate

All ranks `0–7` reported the expected A2-sim form, TP `8`, DCP `1`, physical
chips `0,1,2,3,4,5,6,7`, async scheduling enabled, draft graph enabled, five
speculative tokens, BI=`1`, and Engram device index `0`. Across every rank:

- Actual loaded base alias:
  `/work/delivery_matrix_generic_20261013_v7/delivery_matrix/tp8_physical_0_7_20261013_v1/worker.py`.
- The source derivation reported original worker SHA-256
  `8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d`, derived
  worker SHA-256 `7cabe528dc5df34d8851637deb1cd4ea2f13624ba5dbd427455d6d0fe5dc9ad0`,
  manifest SHA-256 `df5461194e84c713352d85e3e92b03459c947c1541b3c045da1e676cb879c554`,
  and `original_worker_unchanged=true`.
- The plan retains `max_model_len=196608`, `kv_cache_memory_bytes=4294967296`,
  `gpu_memory_utilization=0.7`, `max_num_batched_tokens=8192`, Engram enabled,
  and both multistream overlap options. Startup logs reported KV capacity
  `526,010` tokens and maximum concurrency `2.68x` at 196,608 tokens.

## Caveats and scope

`actual_quality_passed=false` and `actual_performance_observed=false` remain in
the plan. This is an A3 physical 0–7 run simulating A2, not A2-hardware
performance. Startup logged the unavailable `causal_conv1d_update_npu` fallback
to PyTorch and a shared-memory broadcast warning during static compilation;
static compilation then reached 126/126, and the engine, ASGI, and all-rank
identity gates completed. No canary or client scoring was run. The service is
ready for the separately controlled single-canary step only.
