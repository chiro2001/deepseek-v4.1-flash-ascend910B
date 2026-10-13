# A2 simulated TP8 v4 Dihuo 32K result

## Run identity and configuration

- Model/form: `dsv41-delivery-a2sim-tp8-v4` / `a2-tp8-a3sim`
- Service instance: `9d2ba214732b4eed84dc4f416f27c6ae`
- Runtime fingerprint: `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`
- Target input/output: 32,768 / 256 tokens; one measured round, 8 serial requests; one warmup excluded
- Prompt calibration: exact remote `/tokenize` chat-template method; 148 queries; eight measured HTTP usage prompt counts all equal the 32,768-token target
- Frozen client v2 SHA-256: `benchmark.py` `33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816`; `workloads.py` `62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35`
- Run exit: 0; all 8 measured outputs clean; no client process remained after exit

Exact remote invocation:

```sh
TMPDIR=/home/l00886679/tmp/dsv41-a2sim-v4-bench python3 /home/l00886679/tmp/dsv41-ced-runtime-20261013/client_source_ced_20261013_v2/delivery_matrix/benchmark.py --base-url http://127.0.0.1:18960 --model dsv41-delivery-a2sim-tp8-v4 --expected-form a2-tp8-a3sim --tasks dihuo --input-tokens 32768 --output-tokens 256 --rounds 1 --warmups 1 --request-timeout 600 --corpus /home/l00886679/cedpd-repo/data/dihuo.txt --suffix-dir /home/l00886679/cedpd-repo/data/dihuo_local --json-out /home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_32k_a2sim_v4_20261013.json
```

Both the original receipt's `config.input_tokens_target` and `workload_manifest.input_tokens_target` are **32768**. This is also confirmed by all eight final HTTP usage records; the earlier progress message that stated 2048 was a reporting transcription error, not the run configuration or receipt.

## HTTP result and TTFT

Every request returned HTTP 200, `finish_reason=length`, and final SSE usage from `stream_options.include_usage`. Prefix caching was disabled by the bound service identity; Dihuo requests used `ignore_eos=true`.

| Request | HTTP | Finish | Prompt / completion / total tokens | TTFT (s) | Graph calls per rank (ranks 0–7) | Eager calls per rank (ranks 0–7) |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 200 | length | 32768 / 256 / 33024 | 4.749806 | 81 each | 1 each |
| 1 | 200 | length | 32768 / 256 / 33024 | 4.742072 | 80 each | 1 each |
| 2 | 200 | length | 32768 / 256 / 33024 | 4.736698 | 99 each | 1 each |
| 3 | 200 | length | 32768 / 256 / 33024 | 4.738305 | 77 each | 1 each |
| 4 | 200 | length | 32768 / 256 / 33024 | 4.728266 | 102 each | 1 each |
| 5 | 200 | length | 32768 / 256 / 33024 | 4.728766 | 76 each | 1 each |
| 6 | 200 | length | 32768 / 256 / 33024 | 4.717108 | 81 each | 1 each |
| 7 | 200 | length | 32768 / 256 / 33024 | 4.724392 | 107 each | 1 each |

TTFT mean/median/min/max: 4.733177 / 4.732732 / 4.717108 / 4.749806 s.

## Prefill boundary and server-fenced timing

The sourced server prefill boundary for each request is **first scheduled prefill → terminal prefill CPU bookkeeping**, using the per-request `GET /delivery/metrics` cumulative `prefill_seconds` delta (max rank). The separate server-fenced boundary is **terminal prefill CPU bookkeeping → worker completion fence**; it may include metrics RPC delay.

| Request | Server prefill (s) | Server-fenced (ms/step) |
|---:|---:|---:|
| 0 | 4.632292 | 30.691697 |
| 1 | 4.623227 | 30.675237 |
| 2 | 4.620581 | 30.252123 |
| 3 | 4.620983 | 30.622490 |
| 4 | 4.610535 | 30.400601 |
| 5 | 4.613182 | 30.524844 |
| 6 | 4.598580 | 30.536728 |
| 7 | 4.607164 | 30.276737 |

Across the 8 measured requests, sourced server prefill sums to **36.926546 s**. The aggregate server-fenced result is **30.479052 ms/step**. These are separate counters/scopes; neither is inferred from TTFT.

## Aggregate

- HTTP decode rate: **98.051972 tok/s**; `A=2.901849`; output-rate consistency delta: 0
- Client SSE decode: **29.595011 ms/step**, excluding TTFT
- Measured decode rounds: 703 total; **each rank 0–7 recorded 703 draft graph calls and 8 draft eager calls**

The request receipts and split manifest preserve per-request HTTP/usage, TTFT, prefill and server-fenced metrics, and before/after rank counters used for graph/eager deltas.
