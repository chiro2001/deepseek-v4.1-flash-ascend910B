# A2 v4 Dihuo 2K archive receipt

The large raw benchmark receipt remains on a3-21; it was not transferred over SSH. It was split there into eight deterministic gzip request receipts and one manifest, then copied one file per `scp`. Every transferred file is below 1 MB. The archived request artifacts preserve the HTTP status, finish reason, final usage triple, TTFT, server prefill boundary, separate server-fenced values, and full before/after metrics snapshots with per-rank counters.

## Source and identity

- Remote raw receipt: `/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_2k_a2sim_v4_20261013.json`
- Raw receipt: 2,439,806 bytes; SHA-256 `7ca31c18ea7832f95396b9109a2378b0c23ae411e2ae52dd02e2085c7812a057`
- Split helper: `/home/l00886679/tmp/dsv41-a2sim-v4-bench/split_delivery_evidence_a2_v4.py`; SHA-256 `e676ab3bdbca1ac235b76f03be71b4b801fef2ebc31c81c56c4a9833897574e2`
- Split mode: gzip; 8 request artifacts; maximum compressed request file 20,616 bytes, maximum uncompressed request receipt 343,342 bytes; manifest 22,955 bytes
- Post-run identity GET: `identity_after_dihuo_2k_20261013.json`, 12,349 bytes; SHA-256 `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef`
- Post-run GET still reported model `dsv41-delivery-a2sim-tp8-v4`, form `a2-tp8-a3sim`, SID `9d2ba214732b4eed84dc4f416f27c6ae`, fingerprint `61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9`, initialized/ASGI-ready true, PID 342558.
- The frozen v2 client hashes and run options are recorded in `RESULT.md` and in each split artifact's source manifest.

## Paths and hashes

The local staging directory was `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/dihuo_2k_a2sim_v4_20261013_requests`; the archived destination is `experiments/delivery_matrix/evidence/a2sim_tp8_v4/client/dihuo_2k_a2sim_v4_20261013/`. Remote split-file SHA-256 was compared with local staging; archived bytes were then rehashed and matched.

| Artifact | Staged / archived bytes | SHA-256 |
|---|---:|---|
| `manifest.json` | 22,955 | `afb141d345f61c84e9ab7a0b5f0682f4bb2ea918dee029636e2fbbc65cbb94ec` |
| `requests/dihuo_round00_request00.json.gz` | 16,670 | `c4a2ee604ca5b5b9097f58c0048265740a157d7646b46b2fdb419378b91da161` |
| `requests/dihuo_round00_request01.json.gz` | 17,325 | `575ed90736ac7eb3403e588bd1a188905bede7713002482bf3514f6b3ae5bee3` |
| `requests/dihuo_round00_request02.json.gz` | 17,369 | `4659011ec2d791e9dde59d6b1caebb3aad081ef2d8e2a0f5a3fb7f7b4ce0ffbf` |
| `requests/dihuo_round00_request03.json.gz` | 17,863 | `23b81740d3dda8a69963372d54c5eb94d8e9d0f214931170752c6b3881327732` |
| `requests/dihuo_round00_request04.json.gz` | 18,594 | `c5b97208f2e2636c01afd4c288cf21cf63df819cdf52a41689d554563e575196` |
| `requests/dihuo_round00_request05.json.gz` | 19,365 | `47297fe7fc61a7208cd0072f01cd8462d9cb1b5e5bbe6c4803417f72974ede27` |
| `requests/dihuo_round00_request06.json.gz` | 19,950 | `b0ffda01baad49aab9ead87a8f544d527582ba131b2b4124bccf7eec8a96a5f4` |
| `requests/dihuo_round00_request07.json.gz` | 20,616 | `eb5c86cc644464f6ac96c1f241ebd29029b4c81c23640907160a09310ab929a7` |
| `identity_after_dihuo_2k_20261013.json` | 12,349 | `95bb15a8f5b7c45c4699df70832b98af445ab4fbe0fc00d16e2478dc0a8239ef` |

The raw receipt's recorded remote path, size, and SHA-256 are the provenance for these split artifacts. No v2/v3 results were used or carried forward.
