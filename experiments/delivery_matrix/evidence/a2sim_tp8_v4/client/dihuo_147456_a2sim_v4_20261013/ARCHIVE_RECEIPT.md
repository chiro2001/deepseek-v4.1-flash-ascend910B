# A2 v4 Dihuo 147456 archive receipt

The large raw benchmark receipt remains on a3-21; it was not transferred over SSH. It was split there into eight deterministic gzip request receipts and one manifest, then copied one file per `scp`. Every transferred file is below 1 MB. The archived request artifacts preserve the HTTP status, finish reason, final usage triple, TTFT, server prefill boundary, separate server-fenced values, and full before/after metrics snapshots with per-rank counters.

## Source, prompt target, and identity

- Remote raw receipt: `/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_147456_a2sim_v4_20261013.json`
- Raw receipt: 16,657,226 bytes; SHA-256 `07d19104aeff8ced36aaf922893d386b041c1d675a954cd55e02df375a5277d2`
- Verified target in both raw `config` and `workload_manifest`: `input_tokens_target=147456`, `dihuo_output_tokens=256`; all eight HTTP usage prompt counts are 147456 and completion counts are 256
- Tokenizer: exact remote chat-template `/tokenize` method; 157 queries; no completion-token fallback
- Corpus SHA-256: `00aefd13430d4be45424db3c3f832b6c012d9c60c199518f45feed5d0a57e52b`
- Split helper: `/home/l00886679/tmp/dsv41-a2sim-v4-bench/split_delivery_evidence_a2_v4.py`; SHA-256 `e676ab3bdbca1ac235b76f03be71b4b801fef2ebc31c81c56c4a9833897574e2`
- Split mode: gzip; 8 request artifacts; maximum compressed request file 658,094 bytes, maximum uncompressed request receipt 2,004,006 bytes; manifest 22,966 bytes
- Post-run identity GET: `identity_after_dihuo_147456_20261013.json`, 12,349 bytes; SHA-256 `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`
- The post-run GET reported model `dsv41-delivery-a2sim-tp8-v4`, form `a2-tp8-a3sim`, SID `9d2ba214732b4eed84dc4f416f27c6ae`, runtime fingerprint `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`, initialized/ASGI-ready true, PID 342558.
- Frozen v2 client hashes and run configuration are recorded in `RESULT.md` and in the split source manifest.

## Paths and hashes

The local staging directory was `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_147456_a2sim_v4_20261013_requests`; the archived destination is `experiments/delivery_matrix/evidence/a2sim_tp8_v4/client/dihuo_147456_a2sim_v4_20261013/`. Remote split-file SHA-256 was compared with local staging; archived bytes were then rehashed and matched.

| Artifact | Staged / archived bytes | SHA-256 |
|---|---:|---|
| `manifest.json` | 22,966 | `365880e5cd26f14bea085bd8fa67c596683c12c169e215d190b0f6bef0128995` |
| `requests/dihuo_round00_request00.json.gz` | 614,022 | `ae0779300ac5eb7a5f7935415857647afdc6ed2a422a44d11919b448dbc8b241` |
| `requests/dihuo_round00_request01.json.gz` | 614,413 | `35a77c62a3f64430323495ffd23212575ea6643202a22dddba50a18861421532` |
| `requests/dihuo_round00_request02.json.gz` | 615,038 | `f782df9daff373aebeeed390f544da2eff34ed4a6a749f4ece21f4e42e3dd4be` |
| `requests/dihuo_round00_request03.json.gz` | 615,376 | `4c08570a305536f4b8dfe07a07f76332e6a23dc5601b46b2d51a29baaeb15b06` |
| `requests/dihuo_round00_request04.json.gz` | 617,965 | `eafe8004490d6cad3c17241cec26b395087981cca80b949a5c6f4603982a28de` |
| `requests/dihuo_round00_request05.json.gz` | 636,506 | `ac06fb8de2fcf17c3f396519adf5c92f65b6eb5acd6416422e07a56c5a28f8f4` |
| `requests/dihuo_round00_request06.json.gz` | 656,633 | `51783aa4a9f550e10997897ecd44488901859a2e5eb48d6fe5b10857e2b30ed6` |
| `requests/dihuo_round00_request07.json.gz` | 658,094 | `32e00140bff886b5078e035381b642fe70cb58da3b5c777f9cdcfffb443b2a26` |
| `identity_after_dihuo_147456_20261013.json` | 12,349 | `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef` |

The raw receipt's recorded remote path, size, and SHA-256 are the provenance for these split artifacts. No v2/v3 results were used or carried forward.
