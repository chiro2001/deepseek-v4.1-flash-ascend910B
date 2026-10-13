# M1事件性能门：CPU准备，NPU权限0

数字门v5的同实例真实E48参数、原SDK Tensor/stream和fresh x生命周期继续作为前置。三路为实际`aclnnGroupedMatmulWeightNz`、raw native、planner→candidate；每轮轮换三路顺序，每个case独立warmup后交错采样，避免某路总是先执行。原9case profile、权重/数学/tiling/HostArgs保持。

[perf_bridge.cpp](perf_bridge.cpp)新增事件专用bind→submit→finish入口：静态lists和workspace/canary初始化在事件外，submit只提交原native，或按同stream提交planner后candidate；没有设备分配、整workspace清零、同步或D2H。Args为owner成员，GM/descriptor/stream/function/module/blob保持至stop事件及finish成功，失败后仍由owner收口。禁止直接对数字门的同步/D2H入口计时。

[event_protocol.py](event_protocol.py)三个事件单独报告fresh-input＋operator、operator的设备/队列duration；candidate的planner在两个区间内。另报告host提交与提交至stop同步的wall-clock，不能把它们当正式整网延迟。x在同GM地址恢复；stop完成后才做数值、frame、canary、输入及readonly seal，检查不进入事件区间。工作区预清零不在steady samples重复，native自身KFC清理仍在kernel内。

独立[runtime_event_gate.py](runtime_event_gate.py)及fresh资源包装器已完成，CPU syntax、协议事件顺序mock、balanced interleave、--no-undefined链接与source本地/远端SHA通过。每case3轮warmup和12轮测量，SDK/raw/candidate每轮轮换顺序；使用v5冻结有效输出作数值reference。每个调用stop后检查输出、frame/canary、实际x after SHA及counts/token，完整E48参数seal放在每case前后。submit成功路径无日志、分配、clear、sync或D2H；真实接受数由独立owner counters读取。参见[CPU就绪与精确命令](CPU_ready_status.json)。当前NPU权限0，该CPU准备未产生性能数字；K5/整网/capture及实际MoE activation消费者均另行覆盖。
