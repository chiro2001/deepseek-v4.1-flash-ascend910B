# A2 simulated TP8 v4 code8 client evidence archive

The local split receipts, split manifest, and bwrap scoring receipt were copied byte-for-byte from the existing task temp directory. SHA-256 and byte counts match old and new paths. The 1,419,839-byte raw receipt remains on a3-21 and was not copied over SSH.

Remote raw receipt:

- Path: `/home/l00886679/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013.json`
- Size: 1,419,839 bytes
- SHA-256: `4c9196d15121358fe0b35ecd77d5b61c43095c49e09a39e9eca3258ec729d0d3`

| Artifact | Old path | Archived path | Bytes | SHA-256 |
|---|---|---|---:|---|
| Split manifest | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/manifest.json` | `manifest.json` | 22,442 | `6ec9d9267695a4b6e7d5e27a1beaf0823df8498c9762b4c7204d2892474a8704` |
| Local bwrap score | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_local_scoring.json` | `local_scoring.json` | 26,932 | `9f232da2041c374ffe8d9451070f981f09a2b5944b63b374dbe1ad42a6d57c7e` |
| Request 00 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request00.json` | `requests/code_round00_request00.json` | 108,823 | `7c1ad824a4b4873135d9d79bd5db4df34ce08853141e3578cba342b3e5cc3bb1` |
| Request 01 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request01.json` | `requests/code_round00_request01.json` | 120,772 | `aba6899031c61a23a2b2fe601452e481ffbe3e858edbd50a6d9a507da485e978` |
| Request 02 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request02.json` | `requests/code_round00_request02.json` | 137,074 | `222fe58f1b1114c096cfc71a9572a8813bd8f477345522ebafcc06bab045efaf` |
| Request 03 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request03.json` | `requests/code_round00_request03.json` | 148,186 | `845f70e30f59a854eda56acb0d7fe4b59d818caa2e67cba25c46b9a3ae161369` |
| Request 04 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request04.json` | `requests/code_round00_request04.json` | 166,524 | `0059a12163e9e2a5dffb29140cb7e04c0242ddf3fee5d7fe77efe80881490415` |
| Request 05 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request05.json` | `requests/code_round00_request05.json` | 183,112 | `3da5874737e65bfb88142b09e9e156c557dd68c16850b91a105841e45e811203` |
| Request 06 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request06.json` | `requests/code_round00_request06.json` | 203,502 | `2ca9c85fd1932f4c377daa14260fb37e42828df1c4f8e55ef9b208f8415b1929` |
| Request 07 | `/home/chiro/tmp/dsv41-a2sim-v4-bench/evidence/code_formal8_a2sim_v4_20261013_requests/code_round00_request07.json` | `requests/code_round00_request07.json` | 217,439 | `b2c3584d497459554088eb66e81724c3a7426e83dddcad38c1cdb40872fea9a2` |

Code8 completed with 8/8 outputs clean; local isolated scoring passed 8/8 tasks and 128/128 cases. Remote scorer status remains `sandbox_error` for all eight outputs. This archive does not mark A2 v4 quality as passed; it is handed to the site accuracy gate.
