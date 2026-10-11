# physical6 HostArgs R0 成功门

2026-10-11 07:29 UTC，独立进程双visible env为6，SDK可见count1、logical0→physical6。fresh smi确认启动前无process、Health OK，容器privileged且/dev/davinci6合法。Data注册与真实函数地址/大小配对后，以明确`aclrtLaunchKernelWithHostArgs`提交一次16B planner参数，block1/schedule0/zero placeholders。SDK记录`isNoNeedH2DCopy=0,argsSize=16`，实际任务在physical6/stream46执行。

R0同步、384B D2H成功；中间128B frame逐位等于CPU oracle，两侧128B canary均完整。planner_launches=1、planner_cases_passed=1、GMM_math_launches=0、weight_bytes=0。counts/frame/host参数源、stream、function/module/blob保持至同步和读回完成，全部cleanup返回0，owner仅在handle关闭后释放。没有reset、清cache、修改容器全局env或其他租户进程。

证据见[状态](evidence/runtime_r0_p6_20261011/runtime_gate_status.json)、[receipt](evidence/runtime_r0_p6_20261011/run_receipt.json)、[参数拷贝摘录](evidence/runtime_r0_p6_20261011/hostargs_copy_excerpt.log)和[释放后fresh smi](evidence/runtime_r0_p6_20261011/release_fresh_smi.json)。终态fresh smi确认physical6无进程、Health OK；容器全局visible仍为8–15。完整日志1557688B留远端，SHA256 `2cb346451e8e3645085c08cebe960a0e16b1b0d6f061b01022435ca411e62b53`，未传输大日志。

固定对象及loader/HostArgs共享库SHA与运行前后相同。历史普通host栈/no-copy=1失败仍在loader v2证据目录。此门只证明R0参数传递及planner可见性；不提供GMM数学、精度、性能、capture或多token verifier结论。后续已授权physical6的M1 capacity6数值门在独立目录开发，不属于本冻结交付。
