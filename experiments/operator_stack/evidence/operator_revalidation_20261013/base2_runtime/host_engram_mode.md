# base2 Engram JIT / local-owner 只读核查

日期：2026-10-13。本文记录的是启动阶段的只读核查；该核查本身没有发送 HTTP 或 generation 请求，没有读取目标容器环境，没有调用 Torch、NPU 或 worker RPC，也没有改配置、modefile 或服务状态。本文引用的 `base2_run.log.gz` 是后续 client 窗口之前的启动日志快照；服务之后已由独立 client 完成 8+1 次 generation。这里的“本核查未发请求”不表示整个服务运行期间没有 generation。

## 结论

- 启动阶段归档 `base2_run.log.gz` 对精确标记 `[ENGRAM-JIT]`、`[local-owner]`、`[local-metadata]`、`selftest`、`falling back to stock` 的限范围检索均无命中；因此该预-client 快照未记录 JIT selftest fallback 或 local-owner mode-change/validate。
- 无命中不等于 JIT 已成功：工作区交付的 hash JIT 默认由 `V41_ENGRAM_JIT=1` 开启（默认关闭），且成功 selftest 不打印成功标记；若未开启则同样没有 fallback 行。当前日志证据不能区分这两种情况。
- local-owner 的 mode getter 由 `route_many()` 调用并在请求路由时轮询文件。上述归档快照早于后续 8+1 client generation，不能用它判断那些请求期间实际采用的 mode。默认 modefile 缺失且运行时路径可被 Env 覆盖；未读取 Env，也未对 client 窗口作 local-owner 运行时核验，因此有效 mode 与独立性能收益均未证实。
- 启动阶段核查时，通过既有 SSH alias `a3-21` 在远端 host 和目标 D 容器 `dsv41-real-stack-tp8-20261010-a321` 分别检查默认路径 `/tmp/v41_engram_localowner`：两处均不存在。此结果不代表后续 client 窗口时的文件状态。`V41_ENGRAM_LOCAL_OWNER_FILE` 可能覆盖容器默认路径；按限制未读取 Env。因此目标 D 的有效路径及请求期间实际采用的 mode 仍未知。

## 代码行为依据

- 目标 D frozen source `engram_jit_kernel.py` 将 hash JIT 门控为 `V41_ENGRAM_JIT=1`，代码默认 `0`；`engram_hash.py` 仅在 JIT 开启时执行 selftest，异常才打印 `selftest failed, falling back to stock`。这是代码路径说明，不是本次实际启用状态证据；未读取 Env。
- 目标 D frozen source `engram_hbm.py` 的 `_LocalOwnerMode.PATH` 默认 `/tmp/v41_engram_localowner`，可由 `V41_ENGRAM_LOCAL_OWNER_FILE` 覆盖；环境初始 mode 默认 `off`。`get()` 只接受 `off/validate/on/gather/fast/b2g`，且由 `route_many()` 调用；`fast` 才转到 NumPy/JIT plan 分支。modefile 的存在或内容本身不能证明它已在本次请求中被读取采用。

## 检索范围

- 只读归档：client 窗口前的 `base2_run.log.gz`（解压后 179,181 bytes，SHA-256 `85935b1528a0dec0e8e70313d1f1d0b2fb9e5ae215de8e86194cfd1c0517513c`）。上述精确标记在该快照中检索 0 条；不外推到后续 client generation 窗口。
- 启动阶段检查时，a3-21 host 与目标 D 容器默认 modefile 均不存在。目标 D frozen source 的 size/SHA-256：`engram_hbm.py` 61,745 bytes / `b422e9899204b7a55af05832073bdea41e24cdc14900ea739c1dcd82822c77ae`；`engram_hash.py` 32,728 bytes / `acf1aa4a9a75eef5ebe6dbf69bbeb8dd9195797c739ecc46751e3a79776ff4a8`；`engram_jit_kernel.py` 14,709 bytes / `1394bbe7e9752a0bcf3cc177f3ff3baf10c9c1d6847569855f5d5ed5237f38cf`。
- 此核查未补发请求、重启或停止任何进程；后续 client generation 属于独立运行窗口。
