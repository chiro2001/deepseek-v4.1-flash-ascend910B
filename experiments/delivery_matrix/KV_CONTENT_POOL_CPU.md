# KV 内容唯一池：CPU 原语，尚未接入 CED

`kv_content_pool.py` 用一个 RAM-backed memfd arena 保存不可变payload；rank/group/
逻辑页的alias与payload分开。相同字节只占一个content span，hash相同仍比对
实际字节，不能把hash碰撞当相等。alias重绑定不覆盖旧read lease，最后一个
alias/pin释放后回收并合并span；池满保留原alias、不半提交。

进程间可导出同inode的只读FD；重新open O_RDONLY而非dup O_RDWR，阻止peer
创建写映射。memfd封住grow/shrink，避免truncate导致SIGBUS。不用磁盘文件
冒充DRAM，也不整池进行ACL host register；后续NPU接线需要有界pinned DMA
staging及真正的完成barrier，不能复用仍被设备读写的scratch。

当前只有**single-authority/thread-safe**元数据接口；还没有IPC/RPC coordinator、
CanonicalKVCacheRef mapping-runs适配、P2 group quota、P/D角色选择或NPU
D2H/H2D接线。两角色之间共享Engram是独立的`shared_engram.py`组件。
不得把这个CPU池称为已经完成的CED DRAM选项或TP8容量收益。

8项CPU正负控通过：8-rank命名alias一份payload、hash碰撞、池满回滚、
read lease跨remap、span合并/复用、只读同inodeFD、非法/空payload、
DMA source在hash/copy期间变化则拒绝提交。fixture仅使用很小的RAM池。

统计中的`allocated_span_bytes`是allocator账目，不是smaps物理resident/PSS；
capacity是memfd虚拟上限，不意味着已触页。metadata、peer mappings和DMA
staging尚未计入；实际容量/峰值/精度必须在完成接口后按同预算实机验证。
消费者必须持有lease直至最后一次CPU/H2D读取结束，不得在关闭lease后保留
其派生buffer。只读peer FD/mmap的生命周期也必须在最终coordinator中统一管理。
