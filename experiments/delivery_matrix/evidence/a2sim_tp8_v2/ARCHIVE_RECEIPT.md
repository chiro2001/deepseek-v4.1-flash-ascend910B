# v2 小证据归档收据

- 归档日期：2026-10-13
- 来源（容器内）：`/work/results/delivery_a2sim_tp8_v2`
- 来源（宿主 bind mount）：`/home/l00886679/projects/dsv41-real-stack-tp8-20261010/results/delivery_a2sim_tp8_v2`
- 本地目标：`operator-stack-tp8/experiments/delivery_matrix/evidence/a2sim_tp8_v2`
- 来源原件未修改或删除；复制文件逐项与远端 SHA-256 相同。
- `run.log`（1,161,093 B）按约定未复制，原件仍保留在远端。

| 相对路径 | SHA-256 |
|---|---|
| `launched.json` | `4a26a9f8ec199c18ff2f28953a3b2eed255d3f6358087342c6de64d7253c3c6b` |
| `runner.sh` | `05068ef74092921995fdfb593250e73d79107e52d6773336320f2cb02b232ad3` |
| `service/plan.json` | `29a228870c03c3fcf99dcc493676191130a11598b7d888d15b05927662584ea8` |
| `service/service_selection.json` | `1e5306a5ea8f04c91304aecd439889462c3e863bdf1bcd981774c44847174783` |
| `service_plan_cpu_only/plan.json` | `cc03334ec2b81001d0b04685c9220cbc3e0ff392b578f1c94c9760e196e0198f` |
| `quality/strict_quality_receipt.json` | `39644ed478e6e53bb580353a54a16be4ab865e11a04c4a60696db75e2baefeeb` |
| `quality/source_guard.json` | `720303230970b745bc2a78c88682fb37735c716c26f8a61c2dd2ead6d35c434e` |
| `quality/client/gsm100.json` | `b6a1eba2f73627f52f77eb3f67cf4dc1e853c8774db4843c7682af704b5e64c1` |
| `quality/client/gsm100.json.jsonl` | `60efd59d3f6e6a4e4452dfb6d22b796e85b6b6723a940c0b8063207870c11c3b` |
| `quality/client/gsm100.log` | `5ac8882423b208ecb4ff616bf942e64534c53d81120976ab1d9dfae1545e8ea6` |
| `quality/client/gsm100.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `quality/client/gsm100.pid` | `f0613ae239188ce21491f9f29357ca34ddec5ebd8ce03e25903f75697fb279df` |
| `quality/client/vision23.raw.json` | `c047f44ab30bde205c5b02289be888d6cea91cb971a74a192fa78991f9aa89cd` |
| `quality/client/vision23.log` | `1455395ea793e9067c2f65cc9c102da4c8f751dd6d619685c385a252b0587409` |
| `quality/client/vision23.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `quality/client/vision23.pid` | `2515fbee3e836cd814386c0adbdd8331939aa0f2f3c917aad20583ee99761da2` |

## 后续更正

保留原 `quality/strict_quality_receipt.json` 不变（SHA-256
`39644ed478e6e53bb580353a54a16be4ab865e11a04c4a60696db75e2baefeeb`）。更正副本
`quality/strict_quality_receipt_corrected.json`（SHA-256
`0af8bd9f9d65a33686269c72da9ed19342aad68ed1f23feb6469ed29b9be27c4`）明确记录
`actual_device_consumer_proof=false`、`host_npu_process_observed=true`；GSM100 与
Vision23 的原质量通过结果保持为 true，正式/性能授权仍为 false。
