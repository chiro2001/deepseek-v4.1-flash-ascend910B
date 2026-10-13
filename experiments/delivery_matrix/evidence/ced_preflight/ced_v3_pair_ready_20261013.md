# CED v3b D readiness, guard negative control, and P/D pair receipt (2026-10-13)

Observation window: 2026-10-13 05:49–06:05 CST on `a3-21` (`192.168.45.21`).
This receipt establishes startup, source/OPP identity, the D request-boundary
negative control, and proxy pair readiness. It does **not** establish successful
KV transfer, inference quality, or performance.

## Selected services and source identity

- P: `dsv41-ced-p0-7-20261013-v2`, container ID
  `16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`, physical
  chips 0–7, API `127.0.0.1:18990`, KV producer group beginning at 19090.
- D: existing container `dsv41-real-stack-tp8-20261010-a321`, container ID
  `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`, physical
  chips 8–15, API `127.0.0.1:18991`, consumer configured for P's producer group
  at 19091. The 19091 listener is P producer TP rank 1, not a D listener.
- P uses the frozen CED service v2 package; D uses the separately authorized
  v3 package `/home/chiro/tmp/dsv41-delivery-matrix/ced_delivery_20261013_v3.tgz`,
  SHA-256 `d4850d9e63285885767e54d1321f8eaec82d44b0b17cf8698b5e7df0a940d1d5`.
  D's actual short-path job root is `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3`
  inside the existing D container. This is not the failed long-path v3 job or its
  runtime fingerprint.
- Both `/delivery/identity` responses identify form `a3-ced-pd`, model
  `dsv41-ced-pd-v2`, source candidate fingerprint
  `8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`,
  `engine_initialized=true`, `ASGI_startup_completed=true`, and rank descriptions
  0–7. Both report `actual_quality_passed=false` and
  `actual_performance_observed=false`.
- P service ID `53af26c59ea9402ba83283a4865f8057`, runtime fingerprint
  `752b0b6fe6d85b8b502d908dac7ee6040fe5bc75c846f02a7a1105b0bcf03c8c`.
  D service ID `b975c203b8f54d4794c39603563e4286`, runtime fingerprint
  `d877f65c0dbae3a0d269dd58fb944d4ecc6320003830ea3821eed95725ba34d3`.
  The overlay manifest SHA-256 is identical on both sides:
  `7fc7e2dd1aa4014fa5ff04e022de73473f4b0a50dee0b000c585408eee484989`.

The actual plan and selection files remain on the selected containers. Their
paths and SHA-256 values were read directly from those containers:

| Side | File | SHA-256 | Bytes |
|---|---|---|---:|
| P | `jobs/prefill_18990/service/plan.json` | `e5a6234cf8227d64bcf287a483fb1b17d4a9659696751eed4af0ab9394b4a748` | 6,937 |
| P | `jobs/prefill_18990/service/service_selection.json` | `214bddc0e8fe596b6fdfca354b743e0c252101a2b6188df9c54e93ccf47c7588` | 12,539 |
| D | `d3/service/plan.json` | `59fd95fab2a7ceb8250996de27b7ddcb21ee3e9b093c09bdf4dce6caa221a6d5` | 7,169 |
| D | `d3/service/service_selection.json` | `014ea629ba20d7f64134f16683d179194f9329ccc3f8000fb40c47b27f2b1fd8` | 12,631 |

Within those plans, both roles set `VLLM_BATCH_INVARIANT=1`, the same overlay
manifest and the same OPP artifact fingerprint. The P overlay path is
`/work/delivery_matrix_ced_20261013_v2/delivery_matrix/ced_runtime_20261013_v1`;
D's is `/work/delivery_matrix_ced_20261013_v3/delivery_matrix/ced_runtime_20261013_v1`.
D's plan includes
`http_middleware=["delivery_matrix.ced_decode_boundary.ced_decode_guard"]`,
so the middleware is part of this D runtime plan/fingerprint.

## Actual BI OPP identity

Both runtime identities report actual root
`/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant` and matching artifact
fingerprint `0167b055e442435b2790f1a77933fe389bdaa45db67a6f2045646606609d4a6b`.
The runtime gate covers `version.info` and the three ABI libraries below; it
explicitly does not claim a full-tree verification. P's separate read-only full
`vendors/` tree, BI subtree, and `config.ini` receipts are recorded in
[`opp_config_ini_addendum_20261013.md`](opp_config_ini_addendum_20261013.md) and
[`opp_vendors_manifest_20261013.json`](opp_vendors_manifest_20261013.json).

| Actual BI OPP file | SHA-256 |
|---|---|
| `version.info` | `cc3e7001ed18e748e400d78bbcd3a8c51c6d292bd7f252a3ed9c4268dc1a5371` |
| `op_api/lib/libcust_opapi.so` | `ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069` |
| `op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so` | `f8308be503301621117cd421e411240a077dc4fcf04972321e403e7efc818010` |
| `op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so` | `68221652dffb7fa13351616ecdf4e3445425697452fed29ae147f0cc335b02c1` |

## D startup and missing-contract negative control

D runner PID in the D container namespace was `1238305`; API process PID was
`1238311`, EngineCore PID `1239181`, with TP0–TP7 worker processes present.
The short-path `d3/run.log` reached `Application startup complete` and
`HTTP server started`; `d3/run.exit` was absent. The D middleware source selected
by the configured module is
`/work/delivery_matrix_ced_20261013_v3/delivery_matrix/ced_decode_boundary.py`,
SHA-256 `beda894149162c3ddeb227dd6943a0ff49168fa762db164bfa6c74065510bda6`.
The startup log records `[V41-DECODE-GUARD] middleware loaded`.

One authorized guard-only negative control was sent directly to D at
`/v1/chat/completions`: a long prompt with **no** `kv_transfer_params`. It
returned HTTP 400 with code `ced_decode_role_requires_kv_transfer_params`. The
D service log recorded `[V41-DECODE-GUARD] rejected ... 没有 kv_transfer_params`
and `累计拦截 1 次`. The loaded middleware calls the original decode guard,
which returns this 400 before `call_next`; this request was rejected before
entering the model engine. Afterward `/health`, `/v1/models`, and
`/delivery/identity` on D each returned HTTP 200; the same API PID and EngineCore
plus TP0–TP7 remained alive, and `d3/run.exit` remained absent. This is only a
negative boundary control, not a KV-transfer canary.

## Proxy pair readiness

Before launch, host port 18992 had no listener. The single-pair proxy was then
started inside the P container, whose network mode is `host`, using the byte-
identical module
`/work/delivery_matrix_ced_20261013_v2/delivery_matrix/ced_proxy.py`, SHA-256
`813c5a0e1a8b7e7988ee9f02fd3a83d8fa38ede9a88713a9c65e5cf0cf99eb33`.
It listens only on `127.0.0.1:18992`; proxy process PID in the P container is
`339897`. Its log is
`/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/proxy_18992/proxy.log`
inside P. The proxy's `GET /health`, `GET /v1/models`, and
`GET /delivery/identity` each returned HTTP 200. Health reported ready for
`dsv41-ced-pd-v2`; both model catalogs returned that same ID. The proxy pair
identity independently rechecked both ready identities, the matching overlay,
and matching actual OPP fingerprint:

- Pair service ID: `3762dc9b083a4b33c045c877273d19a7cb5fc927dee2960b0181948d81424f41`.
- Pair runtime fingerprint: `c70b47fc3e303f58e33e3b1fa0f8960565eb5eefdce3900fb5a394754c9aec8b`.
- Topology: `P8+D8; DCP=1`.

At receipt time both service `run.exit` markers were absent. No generation
request has been sent through the proxy; a real KV-transfer canary and all
quality/performance work remain pending and must stay serialized.
