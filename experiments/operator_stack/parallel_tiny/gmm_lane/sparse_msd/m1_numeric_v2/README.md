# M1 数值门 v2：NZ29验证、workspace分配前终态

v2补齐生产backend的`allow_internal_format=True`，未删除format29断言或再次format-cast。实测单次INT8 NZ cast、INT32 dtype view、contiguous三阶段均format29，view/contiguous共用同一GM指针；最终shape`[48,2304,640]`、stride`[1474560,640,1]`、storage283115520B。详细阶段见[loader_layout_stages.json](evidence/runtime_v2_terminal/loader_layout_stages.json)。参数GM字节D2H尚未执行，不能把format元数据检查表述为物理字节校验。

Data注册、真实vector/mix地址大小、当前context及C2C地址查询成功。随后`aclrtMalloc(117440896B, ACL_MEM_MALLOC_NORMAL_ONLY)`返回207001，driver精确报drv6/out of memory/inner0x7020016。数学与planner启动均为0，立即终态清理，loader owner/blob全部关闭；fresh smi确认physical6无进程且Health OK。NPU权限回0，没有重试或迁移chip。

[allocator证据](evidence/runtime_v2_terminal/allocator_policy_evidence.json)显示既有日志中的aggregate HBM可用55940235264B，同进程此前两块285212672B的Torch policy0分配成功。公开枚举及安装Torch样例确认policy0为HUGE_FIRST；现有日志没有单独给出NORMAL_ONLY页面池余量，不能推断总HBM不足。SDK recent error里较早的allocator/stream查询文字不替代此次driver精确OOM。

见[终态](evidence/runtime_v2_terminal/runtime_status.json)和[compact manifest](v2_terminal_manifest.json)。完整log40044647B留远端，SHA `ec4a8ef76c921e51c8b7514d13ea741ec1b29ea99f123603d1de2fb890ba100a`；compact47027B。CPU修正独立在v3，原kernel/HostArgs/17数学体/tiling始终不变，无精度或性能结论。
