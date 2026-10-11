# DSpark context graph：独立CPU接线设计

已交付[原位adapter factory](context_adapter_factory_cpu.py)、[准入/lifetime协议](context_wiring_cpu.py)和[实际native AST CPU门](verify_context_wiring_cpu.py)。默认全native；最小候选只在原KV前置hook使用既有prepared entry，保留combine与MTP40/41/42串行路径。完整combine覆盖需另一个原位entry。[设计与DAG](DESIGN.md)记录两点间不能重排的native输入/metadata生成依赖。

[actual source guards](evidence/context_source_guards.json)保留11个完整源文件SHA及逐函数片段、V41/V4继承关系、store到MTP attention的同一cache引用链、全部8rank bank和step9 shape记录。当前正式draft graph关闭，未证明helper已capture；future outer merged-runnable graph会自然包含KV hook，应避免嵌套。profile辅助shape支持context6/query5，不能提供真实gid、slot值、live owner/alias分区或写后消费者证明。

CPU验证通过原helper的wkv/norm/rope/store三次顺序、串行MTP40→41→42及27个准入反例；adapter保留实际参数/entry引用到consumer完成，partial-write不自动native重试或运行下游MTP。检查使用CPU stubs，不包含浮点数值、device capture或rollback证明。source/trace大文件留远端，仅传小guard/compact。当前NPU调用/权限0，`ready_for_NPU_capture=false`，没有收益主张。

本目录独立于已冻结M1/K5/failure包；未修改真实model、SDK、so或其他代理文件。后续由root在SDK与actual consumer门后选择新的设备窗口。
