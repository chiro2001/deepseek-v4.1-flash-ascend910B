# 20 ms目标后续：HC/norm融合筛选、路由舍入核对和chip6运行时诊断

日期：2026-10-10 · 分支：`feat/tiny-operator-opt-20261009`

**本阶段没有新增可采纳的端到端加速。已验证阶段最佳仍为22.674 ms/step，A=1，44.103 token/s；20 ms目标继续active。**
本阶段实现并筛选HC finish与全维RMS/RmsNormCast融合，修正对当前vendor路由输出的理解，
并定位chip6完整模型运行遇到的加载及同步问题。失败候选没有接入正常服务。

## 1. HC finish与norm融合的原理

原路径是project→finish→norm三个kernel。候选保留独立投影，把finish和5120维RMS放在一个Vector kernel：

```text
投影及原HC统计
 → 全部20次Sinkhorn、四路输入组合
 → HC输出舍入为BF16
 → 全5120维RMS与gamma
 → 输出BF16及对应路由FP32
```

必须保留HC的BF16中间边界；不能在归一化时直接使用未舍入的四路FP32组合。
为了避免4×8192 FP32矩阵归约的UB临时空间，候选顺序读取四行，比较pairwise与顺序相加，
并测试5120、8192两个tile。归一化使用完整5120维，不使用每tile的局部RMS。

## 2. 精度与性能筛选结果

所有配置先用signed输入、0/0.001/1/100尺度、有/无pre_mix验证，测试前固定门槛：
HC输出4e-3/1e-4、BF16 norm 1/64、FP32路由1e-4，全元素检查。没有按结果放宽阈值。

|版本|结果|处理|
|---|---|---|
|v1|编译选项`enable_auto_multi_buffer`不被当前Triton识别|保留失败，改用已验证的`multibuffer`参数|
|v2|HC四输出逐位一致；BF16 norm约1%独立图改善；FP32输出约66%元素不符|拒绝路由融合，核对原生语义|
|v3|按最终BF16扩展修正FP32边界，并匹配均值缩放顺序；剩2个元素不符|仍拒绝，没有放宽1e-4门槛|
|v4|进一步尝试64-lane归约布局，仍有2元素不符；源同步先后需补fingerprint确认|没有接入模型|

BF16路径较优的独立图约14.5→14.4 μs，收益很小；8192布局更慢。
这些是独立算子计时，不是模型ms/step，也不能据此宣称达到目标。
结果及源码在`hc_prenorm.py`、`probe_hc_prenorm.py`和`evidence/goal20/goal20_hc_prenorm_v*.json`。

## 3. 重要的语义修正：当前FP32路由输出是什么

当前vendor源码`csrc/moe/rms_norm_cast/op_kernel/rms_norm_cast.h`在最终BF16结果生成后，
明确再次Cast到FP32，并说明HashTopK应看到与原RMSNorm后Tensor.float相同的值。
上板signed随机输入与非恒定gamma检查得到：

```text
torch.equal(y_fp32, y_bf16.float()) == True
max_abs_difference == 0
```

因此，旧机会清单中“不能把BF16输出再转FP32替代独立未舍入FP32”的警告不描述当前vendor实现。
这个结论限定于本机当前BF16路径，不能推广到其他RMSNormCast版本。
候选必须重现实际的低精度舍入与归约顺序；简单输出未舍入的FP32 affine并不等价。

原生归约先按1/D缩放每个平方，再按64-lane重复向量折叠，最后WholeReduceSum。
Triton普通全向量sum与该顺序可能在BF16舍入边界产生差异。新候选虽然只差2个元素，
仍未通过预先固定的FP32门槛，不能以“差异很少”为由采纳。

## 4. chip6迁移与运行时定位

chip4出现其他租户占用后，保留其进程，迁移到单独容器`dsv41-tiny-goal20-20261010-c6`。
chip6先通过32元素计算冒烟，但完整模型尚无有效计时：

|尝试|证据与结果|
|---|---|
|static kernel模型重测|两次native调用栈停在_fx_func_run的stream.synchronize；编译输出目录为空；无计时|
|关闭static kernel|dummy param.uniform_报DSA随机任务80字节分配失败，207001/EL0019；退出1|
|串行化dummy初始化|沿用原NPU生成器和seed，仅每个参数初始化后同步；12451个参数完成，进入graph warmup|
|串行化后的graph warmup|调用栈仍在stream同步等待；无计时，仅停止自己的无结果实验|
|LAUNCH_BLOCKING诊断|框架明确拒绝ACL graph与ASCEND_LAUNCH_BLOCKING=1组合；未执行模型|
|操作超时诊断|改为进程内set_op_timeout_ms(30000)，LAUNCH_BLOCKING=0；用于定位，计时不作为正式性能|

临时调试工具py-spy安装在本次`runtime/debug_tools`，没有改全局Python依赖。
没有重置设备或操作其他租户进程。原始日志和调用栈保留在远程，关键证据已入分支。
共享SSH multiplex拥挤时，sync/fetch增加可选独立control socket，避免据观察超时误判远程任务结束。

## 5. 现成融合开关的适用性

已读当前框架的配置校验：qk norm/RoPE、qk norm/RoPE/cache以及MLA rope/cache融合开关
分别限定CUDA/ROCm/XPU，Ascend会被显式禁用。没有把设置为true当作融合生效或性能试验。
Ascend路径需要按实际算子与依赖实现并检查trace覆盖。

## 6. 后续工作与边界

继续定位chip6的具体同步算子；在可独占设备上重测候选。HC融合目前不采纳，保留失败证据。
后续还包括GMM1与激活epilogue融合、shared上投影真实布局、邻接norm/RoPE/cache操作以及metadata/图回放开销。
不减少Sinkhorn、KV窗口、稀疏indices或输出token来换取时延。

所有实验仍为TP1 tiny dummy BF16；真实checkpoint、TP8/W4A8、DSpark和CED-PD未验证。
此前报告：[22.674 ms阶段报告](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a3-21-tiny-goal20-operator-milestone-20261010-v1.html)。
本报告是工作与证据更新，目标尚未完成。
