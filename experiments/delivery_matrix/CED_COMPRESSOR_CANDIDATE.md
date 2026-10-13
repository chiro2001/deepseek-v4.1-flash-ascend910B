# CED D 压缩残余地址候选（尚未实机验证）

首CED code canary实际执行八rank KV transfer及128-token重放后，在末prompt
tail eager附近崩溃；rank3 plog指出`_pool_kernel+0x62c` MTE invalid-GM/
cross-device-timeout及SMMU故障，后续HCCL传播。未保存具体fault address，不能
凭kernel名和源码就裁定因果。原件见`evidence/ced_v3b_canary_failure_20261013/`。

## 窄改动

actual compressor源SHA为
`0d4d78ebb701c52364308f294dca59c8f94a0cdbf877011e60b31c91b8aa2b39`，与
a3-21 Ascend e43cf1e对应git blob相同。`_pooled_blocked`的四个masked projection
load用`seg_row_base+seg_off`构造地址；奇数ratio2起点可能出现seg_off=-1。
保留所有原mask、ring数据选择和数学，仅把这四个projection地址的seg_off下界
clamp到0，不改变真实in-segment lane。没有修改KV/mirror/Engram路由或归约。

`ced_compressor_20261013_v1`是D-only独立组件；bootstrap在SDK导入前选择，导入时
再次检查实际native SDK base SHA；worker ready描述必须报告实际加载文件SHA。
组件manifest SHA=`accf27fa871e62a78cb5e7aed3318c789f69feae14a5299db347795519e4e9f0`，
derived source SHA=`ea3ead30e43d1e11e84cfb410a0a2faeec54e03ead01e53478c30e75dca564bc`。
新CLI `--compressor-overlay`与启动前`DELIVERY_CED_COMPRESSOR`必须一致，manifest
纳入D runtime fingerprint；P不装此组件，P/D仍使用同一CED core与OPP。

## 验证与部署边界

6项CPU测试涵盖ratio2/4/8残余起点数学对照、旧负地址负控、四处address-only AST
差异/原mask保持、manifest/source漂移、D-only/preimport选择和实际source描述。
另有plan测试确认只改变组件身份，双流/草稿/模型kwargs不变；顶层100项CPU通过。
这是NumPy/Triton-spy，不是NPU编译/执行或fault根因证明。

新source-only CED v4包179474B：
`/home/chiro/tmp/dsv41-delivery-matrix/ced_delivery_20261013_v4.tgz`，SHA=
`3916d72f0fb798dac9ffe3dab243abd4999e2de42e6f0cc83a7b3785ef940aab`。
P4/D4将新建短job，model统一`dsv41-ced-pd-v4`，**保留best双流**，仅D加组件。
旧P已积累失败canary计数，不能与新D零计数混用，也不能重复delivery_begin叠wrap；
因此需新P/proxy测量窗口，不继承旧pair的SID或quality/performance结论。
若仍失败，保留此轮后再用单独stream-off候选；没有任何正式指标可从失败请求换算。
