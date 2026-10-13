# A2v4 停服后收据（2026-10-13）

本收据只记录已获授权的 API `SIGTERM` 后只读状态；没有停止/重启容器，
没有清缓存、清理进程或发送 generation。P 容器仍为
`dsv41-ced-p0-7-20261013-v2`（CID
`16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`），
`running`, restart count `0`。Docker 没有为该容器配置 Healthcheck（`Health=not-configured`），
因此不把不存在的 `.State.Health` 字段当作检查失败；物理 NPU 的逐芯片 Health 另见
完整 `npu-smi info` 输出，16 芯片均为 `OK`。

## API / 日志 / teardown

- 此前只对经过 `/proc` 与 service identity 双重匹配的 API container PID `342558`
  （host PID `2898801`）发送了一次 `SIGTERM`；未直接 signal EngineCore 或 workers。
- `run.exit` 为 `0\n`。服务 API/EngineCore 已退出；宿主监听表与 P、D 容器内的
  `connect_ex` 检查均确认端口 `18960` 已释放。
- 08:37:35 的原始 post-stop 进程快照保留了 runner PID `342556` 与 16 个 worker
  zombie（其中 8 个为此前已有的条目，8 个是本次 A2v4 worker）。条目状态为 `Z`、
  RSS/VSZ 均为 0。随后的容器内只读采样计数到 P 容器合计 434 个、D 容器合计
  1117 个历史 zombie；这两个总数是容器累计残留，不是本次 A2v4 worker 数量。
  本次 A2 runner/worker 与旧 D4 worker 的代表性 `/proc` 采样显示其没有可计量的
  address space / `smaps_rollup`，故 PSS 不适用（RSS=0）；没有尝试 reap 或重启。
- `run.log`、`run.exit`、原 post-stop snapshot（含原 `.Health` 模板错误）及不含该模板
  错误的 corrected snapshot 都被原样归档。原件未覆盖。

## 芯片、内存、容器与端口

- 停服后的 NPU-SMI 进程表对逻辑 NPU `0–7` 均报告无运行进程，对应物理芯片
  `0–15`；16 芯片 Health 全 `OK`，HBM 约回到 `2.87–3.13 GB` 空闲基线。
- post-stop 主机快照的 `MemAvailable` 约 `2.07e9 kB`；后续转场快照约 `1.93 TiB`。
  没有执行 `drop_caches`。
- 新 CED API/KV 端口 `18990–18992`、`19090–19097` 在宿主无 listener，且从 P、D
  容器逐端 `connect_ex` 均返回 `111`（`ECONNREFUSED`）。
- 当前只读 Docker inspect 仍显示 P 与 D 容器为 `running`, restart count `0`；
  `xp_mk` 与其他外部容器未进入、未修改。

## 归档件与校验

本目录中的 `artifact_hashes.sha256` 校验四个原样归档件：

- `a2_runtime/run.log`（778,032 B）：
  `35e693c05d499fe09cc32e14f3986fd422e9dbff0acdab51f8fc27bac533cace`
- `a2_runtime/run.exit`（2 B）：
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`
- `site_poststop_snapshot.txt`（12,983 B，保留原模板报错）：
  `e6d790d80851158dbc3351868e3e6c9d554ddaa73148a8a5e925a2c007675912`
- `site_poststop_corrected_state.txt`（12,542 B）：
  `9b3947c0905f0dcb3409def6cdfb5c0a850ae8b2f10d8c630db08a62f3820648`

`artifact_hashes.sha256` 校验上列四个归档件，不包含自身或本收据；本收据的 SHA
在归档收尾时单独计算并记录到启动门收据。
