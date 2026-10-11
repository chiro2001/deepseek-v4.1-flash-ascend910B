# M1 event v2：CPU就绪，NPU权限0

v1的第8个72B HostArgs错误地填入BF16输出GM；v2改回v5实际成功路径的32B GM tensor-list descriptor。原数字门open/mapping/read/run/close代码在移除新增event owner字段后逐字节等于v5，原数学、固定objects、2032B tiling、E48真实参数和阈值未改。v1失败保留在相邻`m1_event_perf_v1_failure/`。

`verify_perf_bind_cpu.cpp`编译实际bridge bind/submit/finish，独立使用v5成功receipt的27组实际72B payload和5×32B descriptor，1801项检查通过。27个将第8参置为output GM的动态反例被拒绝；同一门编译冻结v1源码，在index7/offset56实际失败。fixture没有链接SDK runtime，也没有NPU调用。`CPU_ABI_contract.json`保存构建及结果，`CPU_alignment_audit.json`逐字段记录ABI、GM、owner、stream和workspace对齐；`CPU_native_workspace_clear_audit.json`保存原generated wrapper内部clearWorkspace及KFC实现证据。

一个Probe owner为各case绑定静态lists、workspace和持久args；case之间只在全部stop/finish完成后重新bind，in-flight bind会拒绝。SDK使用Torch自己的输出/workspace，raw和candidate顺序共用Probe scratch。每次调用后在事件外核对完整args、planner args和五份descriptor，防止被覆盖或提前释放。完整E48参数seal继续在case边界执行。native KFC清理保留在原kernel内部，steady samples没有host整workspace清零。

启动器将先对R1_e0执行一次同GM恢复后的raw numeric smoke，不创建Event，不计入性能样本；smoke核对v5输出、canary/unused rows、x/counts/token、args/descriptor和完整E48 seal。任何失败直接收口，candidate及所有事件样本均不执行。CPU直接执行实际smoke分支AST，submit、同步和数值失败的三条反例均阻止后续事件；当前smoke尚未上NPU。

smoke成功后延续9case×(3轮warmup＋12轮交错测量)三路原协议，候选planner计入两个事件区间。预期实际API接受SDK135、raw136（含smoke1）、candidate135、planner135，测量事件样本324。设备/队列事件可能包含host提交造成的空闲，不作为纯core时间、K5或完整TP8收益。准确新窗口命令见`CPU_ready_status.json`；当前无runtime attempt，须由root重新分配physical6，禁止重试v1。

CPU compact为33805B，SHA `24cc671349f1180760cfb3e3c6a0d78f4f472ccf2c6ef348ceccb3d30dd5da76`；原始构建和编译ABI日志保存在只读`evidence/CPU_build/`，逐文件SHA已通过。bridge 33496B，SHA `e1bf4769d2b07ef0b13670774d21929c4d79060f29faa3199b38e9a5fa65664b`。
