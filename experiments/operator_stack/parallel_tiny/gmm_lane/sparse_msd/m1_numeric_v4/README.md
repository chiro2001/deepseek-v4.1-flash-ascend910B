# M1数值门v4：原SDK/raw原生第一精度门终态失败

v4在同一个冻结validator基础上修正workspace的512B对齐：有效容量117440640B不变，自有backing117441408B仅多512B用于alignment slack及双128Bcanary。CPU fixture无ACL/runtime链接，复现旧布局拒绝，并通过全部512种allocation modulo、真实validator/list pack及边界反例。NPU窗口之前CPU compact44589B已冻结。

新physical6窗口依授权完成真实E48原loader NZ/scale/assist全GM字节校验，再执行R1/e0的原`torch_npu.npu_grouped_matmul`→实际`aclnnGroupedMatmulWeightNz`。SDK数学1次和raw native HostArgs接受1次均同步/读回成功；workspace/y canary与inactive rows完整，两个有效输出全finite。SDK/raw比较的原1ULP门失败：5119/5120值不同、5112超过1ULP、maxULP32253、maxabs1.009765625。未继续raw A2、candidate或planner，后者全0。无候选数学/精度/性能、整网或K5结论。

见[终态](evidence/runtime_v4_terminal/runtime_status.json)、[每case实际提交/同步计数](evidence/runtime_v4_terminal/actual_launches_per_case.json)、[首差异](evidence/runtime_v4_terminal/first_numeric_differences.json)和两个valid-output hex文本。入口attempt单列，不能计作kernel启动。实际SDK/raw kernel均physical6/stream47、block24/schedule1/MIX1:2、no-copy0，raw参数72B；SDK使用tilingKey69325233410。

首次比较失败后立即停后续，cleanup全部0、owner/binary/blob0；fresh smi确认physical6无进程、Health OK，全局env未变，NPU权限回0。完整runtime.log42844401B留远端，SHA `1d1df81e71430b84697761c37a810a1a7ac3291f42193e3f8f55d5ea61e3acea`。数学对象、13冻结源、17数学体、tiling及HostArgs保持不变；原始结果/阈值不覆盖。后续新活CPU诊断在独立目录，先定位私有raw原生控制与实际SDK的差异。
