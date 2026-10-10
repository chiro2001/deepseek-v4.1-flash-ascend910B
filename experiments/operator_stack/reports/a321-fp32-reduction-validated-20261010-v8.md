# 正式 TP8：FP32归约修复通过重复性与独立数值参考

2026-10-10 · a3-21 physical chip8–15 · 正式40层/384专家top6/W4A8权重

`formal_fp32_decode_reduction_v2` 正常退出0。三组同worker A/A的完整路由、生成token、Top5候选集合及logprob全部一致，最大logprob差0。首个decode步的80个隐状态归约，在8rank都与独立CPU FP64求和后转换的BF16参考逐位一致。当前通过的是eager下的精度验证；图模式叠加审计已经启动，尚没有有效正式性能或19ms验收。

## 为什么此前判断精度验证未通过

原对照双方是同一组worker、同一份正式checkpoint、同一prompt的两次生产基线请求，未安装本轮候选算子bank。最终token虽然相同，但decode路由与专家集合不同，Top5候选集合不同，共同候选logprob差曾达到约0.5–1.1，超过原`<1e-3`门槛。因此当时不能拿该基线直接验收候选优化。

后续16边界追踪在8rank均发现：归一化、Q/KV、可见KV cache、SparseFlashMla、wo_a输出和wo_b本rank局部乘积逐位一致，首个差异出现在wo_b的BF16 AllReduce之后。固定相同rank输入重复归约10次，后9次都与首次不同，max abs0.0234375–0.03125。详见[v7定位报告](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-allreduce-root-cause-20261010-v7.html)。

这定位了本轮测试的首个不稳定位置，不等同于证明所有模型形状或所有精度问题都只有该来源。

## 修复保留了哪些数值边界

实验选择world_size=8、5120个元素且末维5120的BF16隐状态向量。各rank局部矩阵乘法仍先产生原来的BF16结果，再无损转FP32，调用原生AllReduce，最后转回BF16供后续HC等模块使用。没有改权重或局部乘法公式，没有全局开启HCCL确定性开关。

BF16累加的不同顺序会产生不同舍入。FP32提高中间求和精度，在本轮数据中消除了最终BF16结果的差异。原始FP32和相对FP64最大绝对差仍有`5.960464477539063e-08`；转为模型所需BF16后，全部与参考逐位一致。因此结论是本轮验证通过，不宣称FP32归约在所有输入下数学上必然确定。

## 两道精度门的实际结果

|验证|范围|结果|
|---|---|---|
|整网同worker A/A|2K prompt、47输出token，三组reference/repeat|token、完整路由、专家集合、Top5与logprob全一致，delta0|
|独立归约参考|首decode步80个归约×8rank，共640份局部输入|修复后的BF16输出相对FP64参考全部逐位一致，max abs0|
|实际覆盖|40次attention＋40次MoE；46个decode步|每rank80个快照、3680次选择，8rank全通过|
|原BF16路径同批数学参考对照|在同一批已保存局部向量上调用原归约|约50.56%被比较元素不同，最大局部归约差0.25|

最后一行是局部归约输出相对于独立数值参考的差异比例，不是模型答案错误率。它与整网路由/logprob对照分别记录。

v1被错误覆盖预期81拦下，未跑完整精度审计；原来把Embedding也计入了5120元素守卫。实际每步80次，Embedding不在该守卫内。v2按实测覆盖修正为80，保留完整请求计数门。v1的warmup/reference事后单组对照也为delta0，只作为初步诊断，不替代v2三组与独立参考。

大原始请求与真实局部向量保留远端；8个rank向量文件可复用于独立归约试验，避免每个方案再次加载整模型。未通过SSH传送≥1MB数据。

## 下一步与交付边界

`formal_fp32_stack_graph_audit_v1`已启动正式权重的FULL_DECODE_ONLY图，capture `[1]`，同worker比较`tp8base/tp8core/tp8stack`。三套都安装同一已验证归约修复，继续检查原生A/A、候选捕获前后A/A、完整路由、Top5/logprob以及Indexer/cache消费者精度。正式W4A8激活候选覆盖0，未额外建立内容相同的act臂。

启动前核实8–15无占用、仍为已授权80C98001，保留其他chip租户。新证据守卫核验归约补丁源码sha、checkpoint config sha、物理chip、结果与数学审计文件sha，以及3组A/A和8rank/80记录通过标记，防止把未完成试验作为后续依据。

图审计通过后，关闭探针、路由/logprob审计和profiler，做同进程正式配对计时，再做客户端质量、服务归属与性能验收。需报告自洽`(ms/step,A,tok/s)`；当前形态无推测解码，A=1。FP32增加转换任务和通信字节，需要实测其代价。

独立归约代码已准备原BF16、FP32 AllReduce与固定顺序AllGather求和，包括普通/补偿求和及256/512/1024 block。尚未运行，不计收益；其事件计时不会替代模型端到端时延。

正式热点与去重数据见[v6微架构报告](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-attention-boundaries-microarch-20261010-v6.html)：82次实际AllReduce/step，chip8累计2.937ms；O投影小M的MTE2高，GMM Scalar高；没有UB容量occupancy与双缓冲收益证明。性能目标≤约19ms/step和最优正式TP8服务仍待验收，Goal保持active。

## 紧凑证据

`evidence/formal_a321/results/formal_fp32_decode_reduction_v2/`包含3组比较、覆盖、8rank独立数学审计、退出码与`validated_reduction_receipt.json`。已核验归约补丁sha为`586d70cf00f106002c019a20ccebc126f61c1c298f33ae9f70ab9da8a629bffd`，checkpoint config sha为`40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。
