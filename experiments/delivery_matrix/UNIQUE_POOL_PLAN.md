# CED 单侧内容唯一池：独立第二阶段实现边界

这是实现计划，不是已接线、容量或设备验收收据。三形态基线的冻结包不包含此方案。
`kv_content_pool.py` 目前仍只有进程内原语，不能用其 CPU 测试冒充 TP8 卸载。

## 选择及依据

优先使用实际本地页的 hash + 完整字节比较 + 一个 selected-side memfd 内容池。
不同 TP shard 保留各自原始字节；完全相同的页由多个 rank/ref alias 引用同一对象。
这样不用假设尚未接入 Ascend 的 CanonicalPageMapping，也不从几何相同推出数值相同。
首次 store 仍需各 rank D2H；去重省的是长期 DRAM 内容存储，不是首次传输。

只读 ABI 研究原件：
`/home/chiro/tmp/dsv41-offload-research/dram-single-side-tp8-wiring-review-20261013.md`，
SHA `57f99567cb7bda1881370fd35589081c1ca735e66bee5f94ac143b7c4f2c7a24`。
它绑定实际 vLLM `6e448d0…` / Ascend `e43cf1e…`；13 文件 manifest
`886fddfbde73f51c4aa65db0f30888e14719f7fba2766516bd1f4c4aa2920a3b`。

## 不能省略的接线

1. `off|prefill|decode` 严格选择一侧；非选中侧不创建或访问 KV pool，保持 Mooncake。
   P/D 的只读 Engram mmap 共文件是独立配置/状态面，不等于 KV 跨侧共享。
2. Ascend NPU spec/worker 不再创建全容量的每-rank private pinned tensor。
   每 rank 仅有界 pinned DMA staging；D2H event 完成后才读精确页字节入池，
   H2D 从已提交 alias 读入有界 staging 后调用原 `swap_blocks_batch`。
   staging 的所有存活期及字节计入容量，不能把它藏在“零复制”宣传中。
3. alias 包含 pool epoch、layout ID、逻辑 chunk 分配代际、group、稳定 ref ID、
   chunk 内 page slot、rank。manager 的一个 block 可含多个 NPU page，slot 不能省。
   单凭可复用的 CPU block ID 无法防止迟到 job 读写下一代内容。
4. store 先 provisional；只有选中 TP 组恰好 ranks0–7 全部成功才原子发布可命中键。
   completion 带 rank、job generation 与成功/失败；重复 completion 不计第二次。
   任一失败回滚 provisional aliases；不能只数八个任意输出。
5. load pins 与 eviction、abort、迟到 completion 分代绑定；提交的 DMA 不能假装可取消。
   abort 后目的 NPU pages 和 staging 必须等 end-event 再回收。
6. D-only hit 显式选来源，不让后置 Mooncake lookup 的 None 覆盖命中。
   Mooncake 保留 empty receive/control ACK，释放 P delayed-free；两端先注册 tracker，
   并幂等 latch early-done。P 实际 scheduler free 与 D H2D 完成分别验收。
7. 只 hash/copy 每 ref 的有效 `page_size_bytes`，不是底层 stride/alignment padding。
   staging/pool 对齐尾清零但不参加内容比较；ref 内未初始化尾须有有效范围契约。
   Engram token mirror、compressor state 和完整 group mask 的恢复另外验证。

## 容量陷阱与验收

实际 native CPU manager 的静态 logical-block 容量目前按所有 worker 副本预算。
即使 payload 已去重，只保留这个旧 logical-block 上限也不会提高可寻址 token 容量。
必须将逻辑槽配额与唯一 payload 配额分离：以实际唯一字节决定 admission/eviction，
加有界 alias/metadata 配额，pool 满时拒绝/回滚 store，不发布 partial hit。
不能把最理想的 8× 复制折叠率当成固定 block 容量承诺。

验收分别报告 unique payload、对齐 span、aliases/metadata、DMA staging、进程 PSS、
共享 Engram PSS、可命中 tokens 和失败/回滚统计。off/prefill/decode 保持相同总预算，
使用真实 D2H/H2D、重复命中续算、eviction/abort/fence 和独立精度测试证明收益。
CPU 原语或 init 成功均不足以通过这一阶段。
