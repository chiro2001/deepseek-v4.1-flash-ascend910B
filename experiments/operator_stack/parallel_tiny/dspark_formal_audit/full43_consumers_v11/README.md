# v11 完整视觉根 ownership 最小修复

首个实际 `formal_dspark_full43_diag_v1` 在载入目标与原生 DSpark 后，于 `session.py:66 before_graph` 访问 `self.target_root.model`，实际抛出 `AscendDeepseekV41ForConditionalGeneration has no attribute model`。rank5 原 traceback 先记录 39.1183GB 已载入。这是入口 ownership 失败；没有进入 43 层导出、数学、质量或性能门，旧 job 保持失败。

v11 仅修复该引用。它先严格核实际 conditional 根的 class/global symbol、原 `vl_model.py` 全字节 SHA `b7bdd54bc5080ea70bab0f27c3dc5d91dbc720e333df2933efb7df9855b71caf` 和原 constructor 的 language_model 所有权，再读取已注册的 `language_model.model`；核对应 source-bound 唯一 native backbone、实际 owner identity 和原 17 函数 SHA。draft 的原 `.model` 路径保持。原 conditional 根、vision、aligner、原参数及原 forward 数学与接口保留。此处无需猜测 wrapper，也不会绕过完整视觉配置。

入口相对 stack：

```text
parallel_tiny/dspark_formal_audit/full43_consumers_v11/formal_audit_driver.py
```

root 负责新的 snapshot/job、fresh physical8–15 检查和启动。argv 仍为 `--model <完整原checkpoint> --physical-chips 8,9,10,11,12,13,14,15 --output <新job>/gate --generations 32 --max-output-tokens 192 --mode diagnostic --execute`。没有 `--execute` 时仅 CPU preparation，不构造 SDK/NPU。graph/observer/prepack/native carrier/退出门继承原 v10；`diagnostic passed` 必须具有实际8 rank loaded43、至少32实际spec轮次的43输出与全部Tensor出口、正常shutdown和8个真实reaped0。它不授予正式通过。

新增 CPU 门用原 conditional constructor/forward AST 构造含 vision 的 CPU fixture，证明 v10 `.model` 假设确实失败，v11 精确 owned path 成立并保留 root/vision/aligner/参数 owner 与原 carrier；拒绝错误 class、改源字节、影子 owner、重复/缺失 backbone 和错误 role。20 CPU 契约、17 原函数 AST 与32 CPU轮编译出口在同版本 PyTorch2.10.0+cpu 全通过，SDK/NPU imports、动态库、skip、新NPU jobs 为0。CPU mock 与 AOT 仍不能替代真实消费者设备证明。

旧 cleanup 原2853B回执 SHA `54821a35bfb94fd8f288d002232860dc75c5c6fb7644fe4f10c278673cd95405` 记8次已核身份的 pidfd SIGKILL，但10秒内退出码均缺失、all_terminal=false。随后只读观察原8 PID/Core全部不可见、physical8–15无运行进程；这只证明当时释放状态，不能冒充原 Process reaped 退出码。旧 receipt、缺失 run.exit、旧失败状态都不改写。v11 保留原有界清理和 fail-closed 退出门；实际正常回收仍须新 job 证明。

本地新测试临时内容使用 `/home/chiro/tmp/dsv41-full43-v11-20261012`。同版本 CPU run 使用其 fresh `/work/offline_tools/.../runtime/tmp` 与 private torchinductor cache。原 v10/旧 NPU job 的实际临时路径保持只读。

完整依赖仍是 frozen v9 10、v8 70、IDS prepack18文本+2 metadata。所有17消费者、raw ND/borrowed stream/workspace/GMM2 alias/copy-before-pack order、accepted-prefix/frontend rollback、native AA math/logprob、一般质量和≤20ms整个main+DSpark step仍以 freeze 的 strict_uncovered_points 为准；全部相关 actual/formal/performance flags false。所有大 epoch raw 留远端，SSH 单文件严格<1MB。
