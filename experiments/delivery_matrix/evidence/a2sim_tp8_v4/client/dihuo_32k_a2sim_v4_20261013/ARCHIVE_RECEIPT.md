# A2 v4 Dihuo 32K archive receipt

The large raw benchmark receipt remains on a3-21; it was not transferred over SSH. It was split there into eight deterministic gzip request receipts and one manifest, then copied one file per `scp`. Every transferred file is below 1 MB. The archived request artifacts preserve the HTTP status, finish reason, final usage triple, TTFT, server prefill boundary, separate server-fenced values, and full before/after metrics snapshots with per-rank counters.

## Source, prompt target, and identity

- Remote raw receipt: `/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_32k_a2sim_v4_20261013.json`
- Raw receipt: 6,133,283 bytes; SHA-256 `86886e31ebd27556fabe3f8db4c7d1c4bc1cbe7fba660d8c12314b39cb6227d6`
- Verified target in both raw `config` and `workload_manifest`: `input_tokens_target=32768`, `dihuo_output_tokens=256`; all eight HTTP usage prompt counts are 32768 and completion counts are 256
- Tokenizer: exact remote chat-template `/tokenize` method; 148 queries; no completion-token fallback
- Corpus SHA-256: `00aefd13430d4be45424db3c3f832b6c012d9c60c199518f45feed5d0a57e52b`
- Split helper: `/home/l00886679/tmp/dsv41-a2sim-v4-bench/split_delivery_evidence_a2_v4.py`; SHA-256 `e676ab3bdbca1ac235b76f03be71b4b801fef2ebc31c81c56c4a9833897574e2`
- Split mode: gzip; 8 request artifacts; maximum compressed request file 147,925 bytes, maximum uncompressed request receipt 772,460 bytes; manifest 22,970 bytes
- Post-run identity GET: `identity_after_dihuo_32k_20261013.json`, 12,349 bytes; SHA-256 `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`
- The post-run GET reported model `dsv41-delivery-a2sim-tp8-v4`, form `a2-tp8-a3sim`, SID `9d2ba214732b4eed84dc4f416f27c6ae`, runtime fingerprint `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`, initialized/ASGI-ready true, PID 342558.
- Frozen v2 client hashes and run configuration are recorded in `RESULT.md` and in the split source manifest.

## Paths and hashes

The local staging directory was `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_32k_a2sim_v4_20261013_requests`; the archived destination is `experiments/delivery_matrix/evidence/a2sim_tp8_v4/client/dihuo_32k_a2sim_v4_20261013/`. Remote split-file SHA-256 was compared with local staging; archived bytes were then rehashed and matched.

| Artifact | Staged / archived bytes | SHA-256 |
|---|---:|---|
| `manifest.json` | 22,970 | `20a870425389f8f78d9141b8f5c3242c3b07975a8318688bf7bd24c3124ad402` |
| `requests/dihuo_round00_request00.json.gz` | 143,962 | `e9df6533178f07aa16ce0d04529fcd56fbba3163d4fe841f27625974becc204e` |
| `requests/dihuo_round00_request01.json.gz` | 144,816 | `fc369f6581c856d6154a54b187998ce48584847f0fbaa63e98426cd112df8612` |
| `requests/dihuo_round00_request02.json.gz` | 145,425 | `dd5924881999e0eeaedc6e33dce8687333b61c1d455ee37008c81578b7defa26` |
| `requests/dihuo_round00_request03.json.gz` | 145,834 | `c81ef60264dd9fccb878f5dc3576c9ee97fe8167342c04424d8b11b9ae343702` |
| `requests/dihuo_round00_request04.json.gz` | 146,531 | `38a58a56c0d24f83c0bc9ea57ade13c458fc9c66b42b9c8f196b94096ba89fad` |
| `requests/dihuo_round00_request05.json.gz` | 146,711 | `2206595459abb802a4597b7d8379dad0d78ad94321118b217aba4888198110dd` |
| `requests/dihuo_round00_request06.json.gz` | 147,081 | `615fb5e045d0506346fcfb391b93eefdd8b6ad91ab214d45743d14808731eb7e` |
| `requests/dihuo_round00_request07.json.gz` | 147,925 | `b6791fbff685f580227a7afdb1a902a4e321cdcbdaa8e935f606d47cfc73a5b9` |
| `identity_after_dihuo_32k_20261013.json` | 12,349 | `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef` |

The raw receipt's recorded remote path, size, and SHA-256 are the provenance for these split artifacts. No v2/v3 results were used or carried forward.
