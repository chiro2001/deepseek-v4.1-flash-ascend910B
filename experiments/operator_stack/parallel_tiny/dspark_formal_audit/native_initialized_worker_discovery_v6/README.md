# SDK init-only 小门的自有 worker 终态清理 v6

本目录只修订 v5 driver 的失败收口。v5 的 boxing adapter、数学/tuple、强映射 ABI、lazy source getter、32代/static/keep和正式0.7显存预算保持；v1–v5冻包全部字节不变。NPU权限与本revision NPU作业为0，full43继续暂停。

实际 v5 在rank2 normal init显存门拒绝，36.64/61.27 GiB低于0.7要求的42.89 GiB，未到SDK/ABI/epoch。其他自有children等待TCPStore，SDK SIGTERM handler吞掉terminate，先前join未能收口。主线程已按验证的自有PID身份用pidfd结束它们；其他租户未被信号，其任务后来自然结束。旧原始exitcode/null范围及单独cleanup证据保持原样，不能改写为32代或boxing实际通过。

## 唯一生命周期修订

`owned_worker_cleanup.py`在worker创建后立即捕获其pidfd与PID/PPID/boot/start_ticks，并绑定driver PID/boot/start_ticks及实际cmdline SHA。失败收口只处理这个表，绝不按名字kill或枚举其他作业。发送SIGKILL前再次核对身份，读取自身pidfd的kernel fdinfo确认目标PID；不匹配、不可读或pidfd API/权限缺失均拒绝，没有数值PID回退。

正常成功先允许有界标准shutdown；失败直接对经过核验且仍存活的自有child发pidfd SIGKILL，避开SDK TERM handler。随后join并只记录确实reaped的exitcode。`owned_failure_cleanup.json`记录身份、信号、终态与pending。若任何child未终态/身份不可核验，结果明确失败，不把null作为最终exitcode；写完已知结果后避免进入multiprocessing隐式atexit join。没有device reset、上下文修复、缓存清理、资源预算下降或worker重试。

CPU实际Linux测试创建自有普通child，先验证它忽略SIGTERM，再以受保护pidfd SIGKILL回收。9门覆盖该负控、正常退出、start/PPID/boot/driver cmdline错配、缺birth identity、kernel pidfd目标错配及仅null signal的API预检，全部通过且没有skip/遗留测试child。原76 CPU门重跑通过，共85单测。ABI/数学与v5一致性另按文件字节核验。CPU没有导入torch_npu或运行SDK/NPU入口。

## 主线程新窗口

主线程重新完成fresh8–15资源/host门，以独立新snapshot/new worker执行同选项static_default；不得复用v5进程：

```bash
bash /work/NEW_PRIVATE_SNAPSHOT/stack/run_stack.sh \
  parallel_tiny/dspark_formal_audit/native_initialized_worker_discovery_v6/init_only_driver.py \
  --model /home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq \
  --physical-chips 8,9,10,11,12,13,14,15 \
  --output /work/NEW_JOB/gate --variant static_default --generations 32
```

`NEW_JOB/gate`须尚不存在，host仍提供相同ASCEND可见设备8–15。candidate源集13；root管理launcher/AA依赖与fresh门。本revision尚无实际NPU boxing/warmup/capture/32代/alias结论。所有device/formal/full-consumer/prepack receipt均false，没有正式性能结论。完整CPU日志、source SHA、旧包完整性与严格清单见 `evidence/`、`cpu_validation.json`、`freeze.json`、`SHA256SUMS`。
