# 现场冻结源收据

以下两个文件是 2026-10-13 现场只读预检时从正在使用的服务候选中读取的原始源文件副本。采集时间取现场预检快照记录的 `TARGET_CLOCK_UTC`：`2026-10-13T04:40:40.214302+00:00`。文件大小及 SHA-256 与同一预检输出 `frozen_source_remote_sha.txt` 中远端读数一致。

| 本地冻结副本 | 现场原始路径 | 现场字节数 | 现场 SHA-256 | 本地字节数 | 本地 SHA-256 |
|---|---|---:|---|---:|---|
| `worker.py` | `/work/src_dspark_general_quality_v11/stack/parallel_tiny/dspark_formal_prep/worker.py` | 3,423 | `8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d` | 3,423 | `8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d` |
| `formal_dspark_counters.py` | `/work/src_dspark_general_quality_v11/stack/formal_dspark_counters.py` | 42,510 | `4877900b492640e292d9daa9755d8699767deeae7eec17a8c2edcf85659b8e8c` | 42,510 | `4877900b492640e292d9daa9755d8699767deeae7eec17a8c2edcf85659b8e8c` |

这些副本用于 `build_runtime.py` 的精确来源校验和有界文本派生。派生只调整 DP worker admission 与 active DP 的 TP 终端组宽度/口径；模型加载、SDK、draft graph observer、request/round 计数和 device fence 逻辑仍须按实际运行重新核验。

这不是 DP2TP4、DP4TP2 或 DP4TP2DCP2 的服务启动、精度、容量或性能收据。终端 join 只表示 request 所属 DP engine 的 TP group CPU Gloo join，不是跨 DP 的 global EP barrier，也不证明空闲 DP peers 已完成各自 work。现场快照、文件副本和历史 TP8 收据不能替代任何新布局的实际质量验证。
