# A2 simulated TP8 v4 Dihuo 147456 result

## Run identity and configuration

- Model/form: `dsv41-delivery-a2sim-tp8-v4` / `a2-tp8-a3sim`
- Service instance: `9d2ba214732b4eed84dc4f416f27c6ae`
- Runtime fingerprint: `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`
- Target input/output: 147,456 / 256 tokens; one measured round, 8 serial requests; one warmup excluded
- Prompt calibration: exact remote `/tokenize` chat-template method; 157 queries; eight measured HTTP usage prompt counts all equal the 147,456-token target
- Frozen client v2 SHA-256: `benchmark.py` `33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816`; `workloads.py` `62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35`
- Target plus output cap is 147,712, within service max model length 196,608
- Run exit: 0; all 8 measured outputs clean; no client process remained after exit

Exact remote invocation:

```sh
TMPDIR=/home/l00886679/tmp/dsv41-a2sim-v4-bench python3 /home/l00886679/tmp/dsv41-ced-runtime-20261013/client_source_ced_20261013_v2/delivery_matrix/benchmark.py --base-url http://127.0.0.1:18960 --model dsv41-delivery-a2sim-tp8-v4 --expected-form a2-tp8-a3sim --tasks dihuo --input-tokens 147456 --output-tokens 256 --rounds 1 --warmups 1 --request-timeout 600 --corpus /home/l00886679/cedpd-repo/data/dihuo.txt --suffix-dir /home/l00886679/cedpd-repo/data/dihuo_local --json-out /home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_147456_a2sim_v4_20261013.json
```

## HTTP result and TTFT

Every request returned HTTP 200, `finish_reason=length`, and final SSE usage from `stream_options.include_usage`. Prefix caching was disabled by the bound service identity; Dihuo requests used `ignore_eos=true`.

| Request | HTTP | Finish | Prompt / completion / total tokens | TTFT (s) | Graph calls per rank (ranks 0–7) | Eager calls per rank (ranks 0–7) |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 200 | length | 147456 / 256 / 147712 | 23.064101 | 79 each | 1 each |
| 1 | 200 | length | 147456 / 256 / 147712 | 23.025216 | 68 each | 1 each |
| 2 | 200 | length | 147456 / 256 / 147712 | 22.995748 | 85 each | 1 each |
| 3 | 200 | length | 147456 / 256 / 147712 | 23.019140 | 69 each | 1 each |
| 4 | 200 | length | 147456 / 256 / 147712 | 22.966287 | 112 each | 1 each |
| 5 | 200 | length | 147456 / 256 / 147712 | 22.940219 | 74 each | 1 each |
| 6 | 200 | length | 147456 / 256 / 147712 | 22.958525 | 72 each | 1 each |
| 7 | 200 | length | 147456 / 256 / 147712 | 22.935568 | 79 each | 1 each |

TTFT mean/median/min/max: 22.988100 / 22.981017 / 22.935568 / 23.064101 s.

## Prefill boundary and server-fenced timing

The sourced server prefill boundary for each request is **first scheduled prefill → terminal prefill CPU bookkeeping**, using the per-request `GET /delivery/metrics` cumulative `prefill_seconds` delta (max rank). The separate server-fenced boundary is **terminal prefill CPU bookkeeping → worker completion fence**; it may include metrics RPC delay.

| Request | Server prefill (s) | Server-fenced (ms/step) |
|---:|---:|---:|
| 0 | 22.639160 | 34.293826 |
| 1 | 22.603633 | 34.127612 |
| 2 | 22.580551 | 33.978584 |
| 3 | 22.587369 | 34.258263 |
| 4 | 22.546100 | 33.667905 |
| 5 | 22.519688 | 34.012406 |
| 6 | 22.531736 | 34.112920 |
| 7 | 22.513785 | 33.773333 |

Across the 8 measured requests, sourced server prefill sums to **180.522022 s**. The aggregate server-fenced result is **34.002879 ms/step**. These are separate counters/scopes; neither is inferred from TTFT.

## Aggregate

- HTTP decode rate: **98.743217 tok/s**; `A=3.197492`; output-rate consistency delta: 0
- Client SSE decode: **32.381892 ms/step**, excluding TTFT
- Measured decode rounds: 638 total; **each rank 0–7 recorded 638 draft graph calls and 8 draft eager calls**

The request receipts and split manifest preserve per-request HTTP/usage, TTFT, prefill and server-fenced metrics, and before/after rank counters used for graph/eager deltas.
