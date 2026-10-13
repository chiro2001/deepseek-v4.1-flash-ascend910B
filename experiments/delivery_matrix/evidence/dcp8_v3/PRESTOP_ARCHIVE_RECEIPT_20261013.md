# DCP8 v3 旧 job 停止前归档快照

本机采集：2026-10-13 04:40:57 CST；目标机 `a3-21` 采集：2026-10-13 04:32:58 CST。因两机时钟相差约 8 分钟，均单独记录。

## 停止前状态

- SSH 目标：`a3-21` → `l00886679@192.168.45.21`；服务容器 `dsv41-real-stack-tp8-20261010-a321`，完整 ID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`。
- 独立 `docker inspect` exit 0：`status=running`、`restarts=0`、container init host PID `585275`、`OOMKilled=false`、`dead=false`。服务未重启，`started=2026-10-12T17:15:46.71157919Z`。
- PID namespace / host PID 映射：runner `841623` / `1479511`；API `841625` / `1479521`；EngineCore host PID `1481233`；TP0–TP7 host PIDs `1481524,1481915,1481973,1482024,1482084,1482133,1482422,1482473`。容器 `/work/results/delivery_dcp8_v3/runner.pid` 内容为 `841623`。
- 18960 在目标机仍监听。GET `/metrics` 停止前为 `num_requests_running=0`、`num_requests_waiting=0`、`kv_cache_usage_perc=0.0`；检查没有额外 chat 请求。`run.exit` 在容器路径和其 `/work` bind 对应宿主机路径均不存在（两次 `test ! -e` exit 0）。
- 实际服务身份来自 `service_selection.json` / API identity：model=`dsv41-delivery-dcp8-v3`、form=`a3-tp8-dcp8`、service instance=`3d8c0140942c48d58d73a291b42059d2`、runtime fingerprint=`ee2b487b07fce2d9aac3a44e6c93bc2f1ff77dbaefe7ca6a2454a928f52e19e1`、candidate fingerprint=`8a7b052d6ab87cc5fed7ded6a1ce5903ffb36f018955d671477632356b005bdd`。原 `launched.json` 内 CPU-only planning fingerprint `0cbe...` 单独保留，不能当实际 runtime fingerprint。

## 已归档的停止前原件

- 本机副本 `service/run.log.prestop_20261013`：897404 bytes，SHA-256 `ee12598f80c6c718b0fec8bcba3f94cd261429b8766bebcc84d5861f98ae1220`。
- 本机副本 `service/runner.pid.prestop_20261013`：7 bytes，SHA-256 `4d9a44d24bd42f91a8049570a60d994eb52ac2b8c7d2ec5560cb5e63c3c47955`，内容 `841623`。
- 支撑身份材料：`launched.json` SHA `eda6818226e3a4192d7b328b83602ae1c47fff993769e802c4496b815db09a2d`；`service/plan.json` SHA `7f28398e47e8d829a2e3e87d51d64bc64dfa8a134468660b89a914fbd138534d`；`service/service_selection.json` SHA `ba3aadd5371c565b7df1f75e1baf18741f88d3ce29e34cd3b3b43307aba00499`。
- DCP v3 workload 已完成并归档；严格质量结果见 `quality/strict_quality_receipt_corrected.json`（GSM100=99/100，所以 strict overall=false；Vision23=23/23、147456-token needle=4/4）。不把这条质量结果改写成正式 PASS。

## 授权的停止动作

质量和性能阶段均已结束、无请求在飞。下一步只向容器 PID namespace 内 API PID `841625` 发送 SIGTERM，让原 runner 收到该子进程退出后产生 `run.exit`；保留容器 `b1aa...` 本身，不删除、不重建、不清理数据。随后另行记录 run.exit、listener 和容器/进程残留状态。这个停止动作不表示 DCP 服务故障，也不撤销已归档的测量/失败原件。
