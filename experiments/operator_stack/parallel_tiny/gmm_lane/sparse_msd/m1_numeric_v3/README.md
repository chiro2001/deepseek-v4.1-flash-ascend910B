# M1 数值门 v3：GM字节通过，raw提交前对齐guard终态

依据[v2现有allocator证据](allocator_policy_evidence.json)，仅将workspace申请从显式NORMAL_ONLY改为公开HUGE_FIRST(enum0)，与安装Torch样例及同进程成功的大分配策略相同；另在新窗口记录aggregate HBM读数。见[精确diff](workspace_allocator_only.diff)。总allocation117440896B、有效workspace117440640B、前后canary、16/72B参数、原planner/mix对象、tiling与数学源保持不变。

[CPU就绪状态](CPU_ready_status.json)记录公开枚举编译assert、`--no-undefined`链接和本地/远端source SHA通过；新桥33216B，SHA `2addd4e88a3cfd07fa5df7bb0f5452395f773c831088024f152f9b490e425201`。CPU compact41943B记录运行前的CPU冻结。随后独立授权的v3窗口已运行；完整E48 NZ、scale、assist GM字节逐位通过，原SDK R1/e0一次数学成功，有效BF16全finite。raw入口在kernel提交前被冻结workspace对齐guard拒绝；[运行终态](evidence/runtime_v3_terminal/runtime_status.json)区分Python入口尝试与实际kernel启动：raw native/candidate/planner全0。

v3窗口执行`run_host_window.py`，其先检查fresh Health OK、physical6无进程、合法容器设备权限，再以双visible=6、count1/logical0→physical6启动唯一窗口。原生产backend设置、原loader转换、全E48 GM字节读取、同一真实SDK Tensor/stream、SDK-vs-raw及raw A/A门继续保持；随后候选需含planner、≤1ULP，覆盖R1/R3/R6、e0/e47、混合count与六槽。任何新device异常终止后续并释放，不reset、不清cache、不更换chip。

本门activation为确定性样本；实际SwigluQuant activation、route/cache/capture消费者覆盖、性能、正式TP8整网及K5 verifier均在本门之外。审批与运行命令见[CPU_ready_status.json](CPU_ready_status.json)。

本次probe把已对齐workspace allocation加128B作前canary，违反冻结validator的512B对齐要求（实际base%512=128）。guard未放宽，未启动raw kernel；清理全部0，fresh smi确认physical6无进程、Health OK，当前NPU权限回0。完整log42830163B留远端，SHA `1e317e44fccf3e44ee50ce72996131450af951ea7ff66c1afdc10ae0ab356d4c`。后续只能独立修正canary前缀布局、保留有效容量并通过CPU真实validator后再获新窗口；不能宣称A/A、候选精度或性能通过。
