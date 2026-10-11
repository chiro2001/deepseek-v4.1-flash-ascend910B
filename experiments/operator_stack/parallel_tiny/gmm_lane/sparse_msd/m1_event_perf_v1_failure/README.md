# M1 event v1：失败冻结

2026-10-11 physical6唯一授权窗口在首个raw native warmup的stop事件同步报507015。实际API接受native1/candidate0/planner0，raw完成确认0，case完成0，测量事件样本0；没有性能收益结论。未重试。cleanup stream sync同样返回507015，其余释放及loader close成功，owner未保留。fresh释放确认physical6空闲、Health OK、容器全局env未变。

`evidence/`保存运行时和源文件原字节；`failure_file_manifest.json`逐文件SHA与`archive_receipt.json`的28309B compact SHA已验证，解包文件和compact均只读。错误地址文字不构成数学故障诊断。v2在独立目录进行CPU ABI对齐与新窗口准备。
