# DCP8 v3 旧 job 停止后归档记录

停止动作：对经 `docker inspect` 与容器内 `ps` 双重确认的 API PID `841625` 执行 `docker exec dsv41-real-stack-tp8-20261010-a321 /bin/kill -TERM 841625`；远端命令 exit 0。没有 stop/restart/rm 容器，没有清理或改写模型/OPP/结果文件。

## 停止后核验

- 目标仍为 SSH `a3-21`（192.168.45.21）。容器完整 ID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0` 仍为 `running`、`restart_count=0`、host init PID `585275`、`OOMKilled=false`、`dead=false`。
- 容器内 API PID `841625` 已不存在；一次 `docker exec ps` 仍见 runner PID `841623` 为 `Zs [bash] <defunct>`（PPID 1），而宿主机 `docker top` 只显示容器 init `sleep infinity`，无活跃 runner/API/EngineCore/TP worker；宿主机 `ss -ltnp | grep :18960` exit 1，无 18960 listener。保留这个 zombie 观测，不为清它重启/停止项目容器。
- `/work/results/delivery_dcp8_v3/run.exit` 已生成，mtime `2026-10-13 04:34:13 +08`，内容 `0`，SHA-256 `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`。SIGTERM 后 runner正常返回0。
- final `run.log`：900824 bytes，SHA-256 `87103b479452f221c9ac5eed75bff8dff25fbc2e29b95cbfb68e7a8893b70756`；本机副本 `service/run.log.final_20261013` hash一致。`runner.pid` 原件为7 bytes、值`841623`，SHA `4d9a44d24bd42f91a8049570a60d994eb52ac2b8c7d2ec5560cb5e63c3c47955`；本机副本 `service/runner.pid.prestop_20261013` hash一致。停止前 run.log 快照保留为 `service/run.log.prestop_20261013`，没有覆盖。
- 停止后 `npu-smi info` 对 NPU 0–7（物理 chip 8–15）均报告 `No running processes found`，AICore%=0。硬件仍显示约2873–3128MB HBM/卡（设备/驱动占用），不把它解释为模型进程或逐层 cache-consumer 证据。

## 保留的 shutdown 原件与判读

最终 run.log 在 API SIGTERM teardown 阶段记录 EngineCore 退出、随后 `AsyncLLM output_handler failed` / `EngineDeadError: EngineCore encountered an issue`，并提示 resource_tracker 清理248个 semaphore 与11个 shared-memory对象。该记录原样保留在 final run.log；当时 metrics 已为 `num_requests_running=0` / `num_requests_waiting=0`，无请求在飞，endpoint 随后消失，runner exit 0。将其标记为停服/worker cleanup 路径的日志异常，不回写成 GSM/Vision/needle 推理失败，也不删掉该异常。

该服务生命周期已结束；项目容器仍运行且未重启。DCP 正式性能和质量原件均已归档，严格质量状态仍由 `quality/strict_quality_receipt_corrected.json` 决定（GSM100=99/100，overall strict false）。
