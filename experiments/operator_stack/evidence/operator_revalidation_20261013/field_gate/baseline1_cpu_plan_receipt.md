# baseline1 CPU 计划与一次启动回执

日期：2026-10-13（Asia/Shanghai）  
范围：operator TP8 重新验证的启动闸门及一次性 base1 启动。  
结论：实时启动前闸门通过；按 root 的明确授权启动一次；vLLM EngineCore 在 worker 初始化期间因设备可用显存不足而失败。未重试、未重启容器、未发生成或客户端请求。

## 冻结配方

- generic-v7：`/work/delivery_matrix_generic_20261013_v7`，36/36 文件 `sha256sum -c` 通过。
- source：`/work/src_dspark_general_quality_v11`，220/220 文件 `sha256sum -c` 通过；candidate fingerprint `8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`。
- 源候选 contract SHA-256：`5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`；`FormalDSparkWorker` SHA-256：`8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d`；模型 `config.json` SHA-256：`40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。
- 配方为 `a2-tp8-a3sim`、`--device-index 0`、physical chips `8–15`，未传 `--physical-worker`；端口 `18960`，served model `dsv41-operator-revalidation-tp8-20261013`，`max_model_len=196608`。
- CPU service plan 参数：TP=8、DCP=1、batch=8192、seq=1、KV=4 GiB、Engram enabled 且 `V41_ENGRAM_DEVICE_INDEX=0`、async scheduling、DSpark draft 5 tokens、真实 draft graph 配置、BI 环境开启、双流 overlap 开启。因 Engram index 覆盖为 0，历史 BEST_SERVICE 指标不自动继承。
- frozen launcher 大小 3,986 B，SHA-256：`e67df9cbadd10112d3de3ea8a77e595d3404bf05240ee5d7585da9415c66c1f9`。CPU-only 计划阶段 `service_launch_requested=false`、`NPU_calls=0`、`torch_imported=false`；计划里的 `--execute` 参数本身不是运行凭证。
- CPU runtime fingerprint `3311e9928ccc7950d38337f420d7d9ab7319b2169ba67722dae63d56f611755a` 仅为 CPU 计划值，不是实际服务 runtime fingerprint。

## 实时启动前闸门

2026-10-13 10:20–10:24+08，SSH target `a3-21`：

- `npu-smi info` 完整表中 physical 0–15 全部 `Health=OK`；启动前 process table 全空，physical 8–15 无外部 NPU 进程。
- D 容器 `dsv41-real-stack-tp8-20261010-a321` 的 CID 与授权现场相同：`b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`，状态 running、restart=0。
- 端口 18960 无 listener。冻结 generic-v7、source-v11、model config 哈希均与本地证据一致；BI OPP 目录 `/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant` 存在。`version.info` SHA-256 `cc3e7001ed18e748e400d78bbcd3a8c51c6d292bd7f252a3ed9c4268dc1a5371`；`op_api/lib/libcust_opapi.so` SHA-256 `ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069`。
- D 内新 jobroot `/home/l00886679/tmp/dsv41-operator-revalidation-20261013/jobs/base1` 在创建前不存在；父路径可写。按 `run_service.sh` 的 `DELIVERY_TMPDIR → TMPDIR` 规则实测：TMPDIR 67 bytes；带 UUID 的 Unix socket pathname 104 bytes，含 NUL 为 105/107；`socket.bind()` 成功并清理，临时写入测试成功。
- root 复核配方并授权的边界：仅复制这份 <1 MB launcher 并启动一次；限定 physical 8–15；不触碰 0–7、不停止外部 `fyf_1009`、不改 DCP/CED、不发 generation/client 请求。

## 一次启动结果

- 约 10:24+08 将唯一文件 `base1_launch.sh` 复制到新 jobroot，远端 SHA 与冻结值一致；随后用 `docker exec -d ... bash .../base1_launch.sh` 启动一次，未执行第二次。
- vLLM 启动日志确认 served id、`tensor_parallel_size=8`、world/local world size=8、`decode_context_parallel_size=1`、`max_model_len=196608`。Worker rank 0–3、5–6 出现 BI enable；worker 级 `[BI-META] installed=7 skipped=` 有记录（父进程的早期 Meta skip 与既有 v14 日志一致，不单独判作失败）。rank 4 初始化失败，rank 7 未完成；八 rank 服务未就绪。
- 根因日志（10:27:40+08）：`Free memory on device (10.97/61.27 GiB) on startup is less than desired GPU memory utilization (0.7, 42.89 GiB)`。EngineCore 于 10:28:14+08 failed to start；D 内 `run.exit=1`。失败进程 PID 1625867 位于 rank3 之后、rank5 之前，可能是 rank4 worker，但日志未显式记录其 rank/device。启动时运行时显存检查失败；不能据此认定为启动前存在外部冲突。错误设备/映射根因尚未诊断。
- 服务实例 ID：`a1e248a17c744195a120354545ec9d87`；service PID `1624804`。`engine_initialized=false`、`ASGI_startup_completed=false`、`actual_quality_passed=false`、`actual_performance_observed=false`。端口 18960 始终无 listener，因此未发送 GET identity/models/metrics，也未发送任何生成请求。
- `service_selection.json` 中的 runtime fingerprint 仍是上述 CPU-plan 值且 `simulated_hardware=true`；本次没有取得实际 runtime fingerprint。不要把它报告成真实运行指纹或质量/性能结果。
- 失败后于 10:28:47+08 复核：D 容器仍为同一 CID、running、restart=0；18960 无 listener；NPU process table 全空且 HBM 回到基线附近。服务进程自然退出；未人工 stop、未重试/重启，未动外部 `fyf_1009`。
- Full43 证据仍是独立 diagnostic snapshot（v26，20 个 strict uncovered 项），不构成本次 formal audit 或启动成功证据。
- 原始失败日志、service plan/selection、run.exit 与启动期 PID/NPU 归属边界见 [`base1_failure/README.md`](../base1_failure/README.md)；该目录中的 `run.log` 与 D 容器原件 SHA 一致。

## 远端留存索引

以下文件留在 D 容器 jobroot/service 中；这里只记录路径与 SHA，不将其误记为本地副本：

| D 容器文件 | SHA-256 |
|---|---|
| `jobs/base1/run.exit`（内容 `1`） | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `jobs/base1/run.log`（46,825 bytes） | `192558e37f8be4e3d6e40528a5cafc5bd5d8143c2081e0ddcac8b7b34c1c491c` |
| `jobs/base1/service/plan.json` | `242295357597407e36eb865bb3ecf551f4add662ed7e8a6dbcfc2257944c495f` |
| `jobs/base1/service/service_selection.json` | `e25982c728d09a78abe023e983fc0d77bf7dc9558ae11eb824ab6fe12f17278d` |

本地准备产物在 `~/tmp/dsv41-operator-revalidation/baseline1/`；本地证据副本位于本目录。新任务遵循 `~/tmp/dsv41-<任务名>` 约定；未迁移或删除其他历史临时产物。
