# M1数值门v5：九case全部逐位通过

真实层0/rank0的完整E48原W2 NZ、UINT64位模式scale与FP32 assist，固定M1/capacity6、K2304/N5120、counts48 type1，在physical6/logical0新窗口完成九组R1/R3/R6、expert0/47、混合count与六槽的几何及数值门。activation是确定性INT8/FP32样本。

全部SDK-vs-raw native、candidate-vs-raw native和candidate-vs-SDK的最大差异为0ULP；raw native A/A逐位一致，输出有效rows全finite。九次planner的128B frame逐位等oracle，workspace/y canary与unused rows完整。见[逐case状态](evidence/runtime_v5_success/numeric_gate_status.json)及各case结果。

本次只修probe输入生命周期。ND MSD前处理会原地重写x；每次SDK/rawA1/rawA2/candidate调用前在同一GM x指针恢复原始字节并D2H验证，调用后四条路径的x变换字节相同。每次真实调用后重新读取全部E48权重及完整scale/assist/counts/token seal，共36组seal全部保持。v4原始失败、输出、阈值保持冻结；完整CPU因果参考在独立诊断目录，未改kernel/math/tiling/HostArgs。

实际计数：SDK数学9、raw native提交接受18、candidate9、planner9、raw/candidate同步27。依据真实HostArgs返回0与kernel事件归档，入口attempt没有计作kernel启动。每case对应SDK1/raw2/candidate1/planner1，planner包含于候选路径。

cleanup全部0、owner/binary/blob0，fresh smi确认physical6无进程、Health OK，全局env未变，权限回0。无reset、clearcache或runtime重试。完整log44716886B留远端，SHA `de5b493c4f88d392e8d6bfb32f5247ed6749721ff27e6a17c043b3ebf96a2bc1`。原receipt785106B已完整取回。

这是固定M1容量6的数值结果。性能、真实SwigluQuant activation/route/cache/capture、正式TP8整网和K5 verifier覆盖尚属后续独立门。
