# CED P8+D8 v4 benchmark preparation (CPU-only; not released)

## Candidate binding

- New model id: `dsv41-ced-pd-v4`
- Proxy endpoint: `http://127.0.0.1:18992`
- Expected form in frozen client v2: `a3-ced-pd`
- Frozen client revision: `client_source_ced_20261013_v2`
- Local task benchdir: `/home/chiro/tmp/dsv41-ced-pd-v4-bench`
- a3-21 task benchdir: `/home/l00886679/tmp/dsv41-ced-pd-v4-bench`
- Client source on local/a3-21: `client_source_ced_20261013_v2/delivery_matrix/`

No request was sent to port 18992 as part of this preparation. No current proxy model catalogue, pair identity, SID, or runtime fingerprint was queried or assumed. The previous failed CED pair and all A2 identity fields are explicitly out of scope.

## Frozen-source verification and CPU-only checks

Both source copies were taken from the existing frozen v2 source and hash-matched before copying. `manifest.json` SHA-256 is `042c344bc7024d31dbd2384ca9ff0ead5654f69cd592e95becb361123c573653`.

| Frozen client file | Bytes | SHA-256 |
|---|---:|---|
| `delivery_matrix/benchmark.py` | 117,170 | `33c4580e2e458de466a2fd48f20259d97c2d856e29a5fa35c25bbff750d23816` |
| `delivery_matrix/workloads.py` | 35,289 | `62db3507dd74ea69406a940963f93414ae2e9f87aa77424f8204f5c9b7ce4d35` |
| `delivery_matrix/tests/test_workloads.py` | 33,006 | `a14291dd3d04a451d4e1faee87a76c302223c416bca5897c9070ee7a9694bad1` |
| `delivery_matrix/ced_proxy.py` | 11,478 | `813c5a0e1a8b7e7988ee9f02fd3a83d8fa38ede9a88713a9c65e5cf0cf99eb33` |
| `delivery_matrix/test_ced_proxy.py` | 10,379 | `33c91883c1b74ec64af621464170b84e7f0211a980dcc07feae3cb5113570915` |

On the new local task copy, CPU-only contract tests passed: `test_workloads.py` 21/21 and `test_ced_proxy.py` 12/12. `benchmark.py --help` also exited successfully. Tests used `TMPDIR=/home/chiro/tmp/dsv41-ced-pd-v4-bench` and `PYTHONDONTWRITEBYTECODE=1`; proxy tests use in-process `httpx.MockTransport`/ASGI transport. No network or model requests were made.

## Mandatory gate before the one canary

Wait for site to provide the current **new v4 pair** readiness evidence. Root must independently check the exact model id, proxy form/topology, both nested P and D SIDs/runtime fingerprints, pair SID binding, engine/ASGI readiness, and same-pair consistency. Then wait for root to explicitly authorize exactly one canary. Do not query the endpoint, send tokenizer calls, or generate until that authorization arrives. The canary must bind to the identity returned for this new pair at execution time; do not copy SID/fingerprint values from the failed CED pair or A2.

If and only if root has approved the current pair identity and released the canary, run once on a3-21 with the frozen v2 client:

```sh
TMPDIR=/home/l00886679/tmp/dsv41-ced-pd-v4-bench PYTHONDONTWRITEBYTECODE=1 python3 /home/l00886679/tmp/dsv41-ced-pd-v4-bench/client_source_ced_20261013_v2/delivery_matrix/benchmark.py --base-url http://127.0.0.1:18992 --model dsv41-ced-pd-v4 --expected-form a3-ced-pd --tasks code --input-tokens 2048 --canary-only --canary-output-tokens 128 --rounds 1 --warmups 0 --request-timeout 600 --json-out /home/l00886679/tmp/dsv41-ced-pd-v4-bench/evidence/code_canary_ced_pd_v4_20261013.json
```

The command is a prepared plan only; it has not been run. It performs the client's normal model/identity binding, exact prompt-length calibration, and one armed code generation. Any failure must be kept as evidence and stop further generation—no automatic retry. It must not stop, restart, or edit the service.
