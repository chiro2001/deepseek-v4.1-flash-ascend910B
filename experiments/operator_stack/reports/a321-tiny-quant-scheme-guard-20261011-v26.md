# tiny完成加载与图编译，修正量化包装器检查后重跑

2026-10-11 · 当前正式客户端目标 **≤17ms/step**，A=1时至少58.823529tok/s。

正式已验收结果仍为 `(19.376255ms/step,A=1,51.609559tok/s)`，GSM8K100/100、Vision23/23；17ms尚未达到。正式API已为tiny实验释放。本报告更新[v25](https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/a321-async-acceptance-tiny-alignment-20261011-v25.html)中尚待完成的运行状态。

8层真实权重tiny保留TP8/EP8、384/top6、本地48专家、W4A8和两张真实Engram表。缓存诊断适配的CPU字节布局/别名/null block与缺资源拒绝检查通过，生产规划器SHA不变。`tiny_tp8_alignment_audit_v2` 已完成模型加载、静态核编译、图准备和多模态预热，之后退出1；没有进入prefill路由比较或原生A/A，也没有性能结果。

失败来自新增的诊断检查：模块的 `quant_method` 外层是 `AscendFusedMoEMethod`，原检查只在外层类名中找W4A8。安装态 `quantization/method_adapters.py` 明确说明该包装器将权重创建、加载和执行委托给自己的 `quant_method`。已修正为逐层查看内部scheme，并要求八个 `AscendW4A8DynamicFusedMoEMethod`；包装器和内层的计数分别记录。修正后的包装器/内层收集回归通过，没有删除量化覆盖门。

此前控制器正确地因v2退出1拒绝启动profiling。当前新作业为 **`tiny_tp8_alignment_audit_v3`**，源码 **`/work/src_tiny_align_v7`**，沿用已复核的 `/work/tiny_tp8_8layer_v1`。再次确认8–15空闲、Alarm为已授权80C98001后启动，没有reset或停止其他租户。

新控制器 `tiny_profile_controller_v2` 每次轮询检查审计runner是否真实存活。仅当v3退出0、三组原生A/A和七次正式前四层prefill路由检查全通过，才启动 **`tiny_tp8_profile_v2`** 的六组关闭审计计时与七组×八rank微架构采集。结果仍仅用于缩层诊断，不能替代完整40层TP8、真实客户端或17ms目标验收。

v25已发布COS/links-server并下载核验SHA；第一批源码和紧凑证据已自检并双远端push `bff7c8f`。本轮修正与最新启动证据继续提交、从提交源码生成MANIFEST、自检并双push。完整17ms目标保持执行中。goal工具不允许覆盖未完成goal，旧goal仍active；最新17ms要求已写入 `ACTIVE_OBJECTIVE.md` 并作为客户端默认门槛。
