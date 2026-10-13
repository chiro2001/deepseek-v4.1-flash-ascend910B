# base2 TP8 受监控启动回执

日期：2026-10-13（Asia/Shanghai）。按 root 授权，base2 launcher 仅启动一次；服务当前在线，未发 generation 请求，也未触碰/停止外部容器。root 已收到 READY，可按其独立计划继续客户端工作。

## 启动前门禁与隔离

- base2 配方与 CPU plan 通过 root 独立核验；最终 launcher SHA-256 `ef04c733256a907c7b7fd834e0e5aba742246193dbfffcc2277ccb6a4d60cfb1`，3,986 bytes。D 侧只复制这一文件到新 jobroot；base1 未覆盖、未复用。
- D `npu-smi` 预启动快照（10:54:10+08 及 10:57:14+08）：physical 0–15 全部 Health=OK；启动前 NPU process rows 全空。physical 8–15 HBM 约 2.9–3.2 GB / 65,536 MB，满足计划 `gpu_memory_utilization=0.7` 所需 headroom；18960 无 listener。
- D 容器始终为 CID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`，running、restart=0。generic-v7 36/36、source-v11 220/220 SHA 检查通过；candidate contract SHA `5ed2e17c2057caee65a17195a4e13f92187b3c8e93dd03b63de8269b3c745cf2`；FormalDSparkWorker SHA `8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d`；model config SHA `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。BI OPP `version.info` SHA `cc3e7001ed18e748e400d78bbcd3a8c51c6d292bd7f252a3ed9c4268dc1a5371`，`libcust_opapi.so` SHA `ff82d965a8a08bfa2862567022a04a1fd04df56d142f4393f1fb527592111069`。
- 新 D jobroot `/home/l00886679/tmp/dsv41-operator-revalidation-20261013/jobs/base2` 原先不存在，父目录可写；以该 jobroot 作为 `DELIVERY_TMPDIR`/`TMPDIR` 做实际 socket bind 成功：root path 67 bytes，UUID socket pathname 104 bytes，含 NUL 为 105/107。service output=`jobroot/service`，cache=`service/cache`，native=`service/native`。
- 最后启动前采样（10:57:14.718+08）再次确认目标 8–15 无占用、18960 free、同一 D CID；目标 D 仅有 init `sleep`，其 host PID `585275` 的 `NSpid=585275 1`、cgroup 命中 D CID。

## 一次启动与就绪

- 约 10:57:15+08 使用 `docker exec -d ... bash .../base2_launch.sh` 启动一次；launcher 远端 SHA 与上述冻结值一致。无第二次启动、无重试、无参数变更。
- `run.log` 显示实际模型权重加载完成：8 rank 各约 `39.4728 GB`；rank 0–7 均有 `world_size=8`、`local_rank=rank` 与 BI enable 日志。DCP=1、TP=8、max model len 196608 等均与已核配方一致。
- GET `/v1/models` 成功，唯一 model id 为 `dsv41-operator-revalidation-tp8-20261013`。GET `/delivery/identity` 返回 service instance ID `3cbd844d9f684c43867b4b85c9c78ebe`，`engine_initialized=true`、`ASGI_startup_completed=true`，含 rank 0–7 的完整描述。
- 每 rank 描述均报告 source worker `/work/src_dspark_general_quality_v11/stack/parallel_tiny/dspark_formal_prep/worker.py`、TP=8、chips `8–15`、DCP=1、async=true、`draft_graph=true`、speculative tokens=5、BI=1、Engram device index=0。identity environment 也确认 `DRAFT_GRAPH=1`、`DSPARK_GRAPH_CAPTURE_METADATA=1`、`DSPARK_DRAFT_USE_CUDAGRAPH=1`、`VLLM_BATCH_INVARIANT=1`、正确 BI OPP 路径。
- GET `/delivery/metrics` 成功，SID 与 identity 一致；8 rank 均 pending=false、errors=[]，counters 为 0。这是无请求状态；整个本次操作没有 generation/客户端请求。
- identity 中的 runtime fingerprint `79513a0f20932cf696374573323b715aab0d04df774721fbee92f2f780da4319` 与 CPU plan fingerprint 相同，metadata 标 `simulated_hardware=true`；它是 plan/config fingerprint，不是实测硬件指纹。此次实际服务身份以 `service_instance_id` 和 rank 描述为准。

## 启动窗口 PID/NPU 归属

11:05:09+08 的时间戳快照用 `npu-smi`、目标 D 的 `docker top -eo pid,ppid,stat,etime,comm`，并在 worker 仍存活时读取 host `/proc/<pid>/status` 的 Name/PPid/NSpid 与 cgroup（没有读取 env/cmdline，也没有进入其他容器）。8 个 NPU worker PID 与 D 的 `docker top` 及 NSpid 完全交叉：

| physical | host PID | D/container PID | run.log rank | cgroup |
|---:|---:|---:|---:|---|
| 8 | 3494441 | 1627158 | 0 | 目标 D CID |
| 9 | 3494671 | 1627177 | 1 | 目标 D CID |
| 10 | 3495230 | 1627231 | 2 | 目标 D CID |
| 11 | 3495603 | 1627277 | 3 | 目标 D CID |
| 12 | 3495846 | 1627323 | 4 | 目标 D CID |
| 13 | 3496095 | 1627369 | 5 | 目标 D CID |
| 14 | 3496501 | 1627415 | 6 | 目标 D CID |
| 15 | 3496576 | 1627461 | 7 | 目标 D CID |

该样本时各 worker 约 46,784 MB，die HBM 约 49.6–49.9 GB / 65.5 GB；8–15 没有非项目 PID。相同快照有一个 physical 6 进程：host PID `3685432` / python3 / 16,099 MB，cgroup `/docker/f11a0b179a1f992c8821781c50cec5862bc6e7a90ef02ecf8bb2c0acad39f1e5`，不属于目标 D 且在本轮 8–15 分配范围之外；未进入或操作该外部容器。

## 边界与后续

启动日志有 `causal_conv1d_update_npu` 不可用、回退 PyTorch 并可能 stall decode-FULL graph capture 的 warning。rank identity 能证明 draft graph 配置为 true；但因本轮禁止生成请求，实际 decode graph capture/调用次数没有执行验证，metrics 的 draft_graph_calls=0 正是无请求结果。服务仍在线供 root 后续工作；本回执完成后不做 generation，不停止服务或外部 PID。

`base2_run.log.gz` 是 D `run.log` 快照的压缩归档；解压后为 179,181 bytes，SHA-256 `85935b1528a0dec0e8e70313d1f1d0b2fb9e5ae215de8e86194cfd1c0517513c`（D 原件在 11:08:45+08 的 SHA/size 查询一致）。没有 `run.exit` 文件，服务保持运行中。
