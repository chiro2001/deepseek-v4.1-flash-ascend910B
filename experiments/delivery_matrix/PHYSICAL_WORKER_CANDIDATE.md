# 独立 TP8 芯片选择候选（0–7，未实机推理）

A2v3 的 8–15 启动门遇到非项目 xp_mk Qwen3 占 chip12/14，未启动、不动外部任务。
拟在用户授权的同台A3上使用空闲0–7，作为新A2v4。不同die组不称严格同卡A/B。

仅改可见设备环境不足以启动：actual best async worker SHA
`8dc2192a04f44e92081af8b85ae15df2cfcfbd837da38434612553dac2564d3d`
在load_model硬断言8–15。本组件只将这一处literal换为0–7，仍严格拒绝其他组；
其余完整40层、真实权重、MTP、Engram几何、SDK bias_vl、async gate、图与计数代码
逐字保持。原worker、原contract和candidate不改。派生类在原包的另一个模块名加载，
`.sdk_gate`仍来自原包；不临时伪造8–15环境来骗过断言。

`tp8_physical_0_7_20261013_v1` manifest SHA=
`df5461194e84c713352d85e3e92b03459c947c1541b3c045da1e676cb879c554`；
derived source SHA=
`7cabe528dc5df34d8851637deb1cd4ea2f13624ba5dbd427455d6d0fe5dc9ad0`。
CLI `--physical-chips`、`--physical-worker`写进实际runtime身份；四个physical/
visible/expected变量必须一致，实际原source与派生source均有SHA门，ready描述
还会再次核实际文件/manifest，不能继承旧质量。旧private API的8–15合同不复用；
原标准GSM/Vision评测器与delivery计步客户端不依赖该private合同。

新launcher先unset CED/DCP/旧geometry开关，再export新候选，避免Python/sitecustomize
在正确计划产生前就加载错误的半服务。A2显式DCP1；DCP8仍复用所选8ranks，不扩到64。

111项本机顶层CPU正负控通过；包含guard/body只一literal变化、原SDK/model/dummy/async
负控、原source/visibility漂移、derived真实文件检查、计划指纹与独立script生成，
以及实际DeliveryWorker描述中的manifest错绑负控（此最后新增测试未改部署包代码）。
这些是CPU stub，不是设备/精度/性能证明。

source-only generic v7包388779B：
`/home/chiro/tmp/dsv41-delivery-matrix/generic_delivery_20261013_v7.tgz`，SHA=
`827c8310684b615a815c813600589e9c3da4ab482087587219c197a25fd35873`。
v6冻结包及A2v3未启动的审查件原样保留；v7已通过36文件/source/plan/runner门、
完整IPC86/107及0–7空闲门，A2v4已init-only启动。尚未新ready或推理/精度/性能结果。
