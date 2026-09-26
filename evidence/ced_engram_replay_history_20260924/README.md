# CED D replay 的 Engram 历史页复用：CPU 对照

日期：2026-09-24。此目录只含离线模拟和源码数据流核查；没有改生产代码，没有启动 NPU，也没有模拟真实 A3 block allocator。

## 对照几何

本次几何取自 [`ced_p_mask_144k.json`](../../ced_p_mask_34fdf08/ced_p_mask_144k.json)：prompt 为 143,963 tokens，P 交接 143,962 个 token，D replay 128 个 token；同日 CED SWA 页映射原型使用 128-token KV block，见 [`static_results.json`](../../evidence/ced_swa_window_view_proto_20260924/static_results.json)。因此：

```text
P 已交接位置：[0, 143962)
D replay：    [143834, 143962)
replay_start：143834 % 128 = 90
```

replay 第一个 query `q=143834` 的 4-gram 历史需要读取四个绝对位置 `q, q-1, q-2, q-3`，即位置 `143834, 143833, 143832, 143831` 上的压缩 token IDs。本例四个位置都落在同一物理页；调度本批只会重写 `143834..143961`，不会重写前三个历史位置。

## 三臂结果

运行：

```bash
python evidence/ced_engram_replay_history_20260924/simulate_paged_ngram_reuse.py
```

脚本跑四次相同 prompt 和四次不同 prompt。每次复用同一个 replay 物理 block ID；对照缺页 barrier、异主旧页、从完整 prompt 正确 seed。还增加了“旧页恰好来自同 prompt”的控制臂。hash 数值使用小型合成表，只用于证明历史变化会改变 lookup ID，不是模型实际 Engram 表参数。

另有低层 CPU 对照跑了原 JIT kernel 和已有 repair helper：

```bash
python a2/patches/engram-true-tokens/tests/test_true_tokens_repair.py
```

结果全部通过。它分别复现了：页首缺页被 pad barrier 代替；页中同一行缺失 3 个历史槽位但 `miss_rows=0`、`err=-1`；present 旧页被静默读取且 `miss_rows=0`；repair mode=1 可计出 mismatch，mode=2 覆写后历史逐值正确。这个测试验证 JIT kernel/helper 语义，不是 CED 服务端 A/B。

相同 prompt（合成 seed 17）第一次 replay query 的结果：

| 臂 | query + 3 个前序 token | toy lookup IDs | 与 prompt 正确历史相同 |
|---|---|---|---|
| 缺页 / barrier | `[591, 0, 0, 0]` | `[56, 123, 265]` | 否 |
| 物理页留有另一 prompt 的旧值 | `[591, 532, 495, 458]` | `[78, 151, 278]` | 否 |
| 从完整 prompt seed 前序 token | `[591, 554, 517, 480]` | `[4, 126, 246]` | 是 |
| 旧页恰好属于同 prompt | `[591, 554, 517, 480]` | `[4, 126, 246]` | 是 |

同一 prompt 连续四次时，这些 hash ID 各自保持不变；若旧页来自同 prompt，页复用本身会**伪装成正确**，因为残留值刚好等于目标历史。不同 prompt 依次复用同一 ID 时，缺页臂和异主旧页臂的 hash ID 都随 query 变化，但均与各自正确历史不同；canonical seed 臂的 ID 与正确历史一致。`cpu_reuse_results.json` 保存了四次请求、前四个 replay query 的完整历史来源与 lookup IDs。

在两种 prompt 序列里，缺页和异主旧页臂都只有 replay 的前三个 query 错：`q=start` 还缺 3 个前序 token，`start+1` 缺 2 个，`start+2` 缺 1 个；`q=start+3` 的 4-gram 全部落在本批新写入区域，hash 恢复正确。相同 prompt 的前三个错值在四次重放中完全重复，因此“重复请求输出相同”本身不能排除它。不同 prompt 复用时，这两臂的 lookup ID 随 prompt 改变，但每次都偏离当前 prompt 正确 ID。

## 源码数据流

1. Model Runner 可取得完整 prompt 的 CPU token IDs。补丁版 runner 用 `positions_np` 与 `req_indices` 构造绝对 token 下标，再从 `input_batch.token_ids_cpu_tensor` 读取本批 scheduled IDs，见 [`model_runner_v1.patched.py`](../../a2/patches/engram-true-tokens/model_runner_v1.patched.py:1251)。因此它能够按同一 request row 读取 `replay_start-1 .. replay_start-3`。
2. 当前 Engram 调用只收到本批 scheduled `input_ids`、`positions`、request row 和 block table，见 [`model.py`](../../patches/files/model.py:1288)。没有把完整 `token_ids_cpu` 或 replay 起点之前的 token IDs 传给 `PagedNgramHistory.update()`。
3. Host 实现先将 scheduled IDs 映射为压缩 ID，再按物理 block ID 写页；只在 page ID 第一次出现时初始化 `-1`，见 [`engram_hash.py`](../../patches/files/engram_hash.py:322)。随后按当前位置与 lookback 从同一镜像读历史并算 hash，见同文件 351–399 行。
4. 缺镜像页会被建成全 `-1` barrier，后续 shift 补 pad，见 [`engram_hash.py`](../../patches/files/engram_hash.py:520)。镜像里已有的非负值则作为 token 使用。对于本例 offset=90，replay 本批首先把当前 token 写进该物理页；若页刚创建，整页初始化为 `-1` 后写入 replay 区间，于是同页的 `q-1..q-3` 是 barrier，但 page row 已存在。strict 缺页模式检测不到这些 barrier 槽，也检测不到 present 旧值。
5. JIT 镜像同样用物理页 ID 的 `page_present` 标记；只在首次见到时清页，后续复用 ID 不重置，见 [`engram_jit_kernel.py`](../../patches/files/engram_jit_kernel.py:108)。Engram lookup IDs 随后直接送到各层 embedding table，见 [`model.py`](../../patches/files/model.py:1310)。

完整 token 表已经有实验性交接实现：[`model_runner_v1.patched.py`](../../a2/patches/engram-true-tokens/model_runner_v1.patched.py:3007) 在 `prepare_engram_inputs()` 前发布 `input_batch.token_ids_cpu`；[`model.patched.py`](../../a2/patches/engram-true-tokens/model.patched.py:316) 保存这张表，`_engram_build_prev_tok()` 按位置构造历史并校验 runner 与 model 的行号对齐。它证明 runner→model 传递可行；这套 A2 offload repair 草案没有证明当前 CED 实验已启用。它会在 JIT kernel 前回填/覆写物理镜像页；本文推荐的 CED 诊断应先以只读比较和直接 canonical seed 为准，避免在未核清页共享语义前改写物理页。

源码表明：如果同一进程内 physical block ID 被重用、其 host mirror 仍有前一 owner 的值，replay_start 前的 3 个位置可能读到旧 token；若镜像缺页则会降级为 pad 历史。远端 KV load 本身不在这段逻辑里填充或校验 Engram 页镜像。

## 最小只读诊断

在 Engram hash 前增加门控、限量的 `V41_ENGRAM_REPLAY_HISTORY_TRACE=1` 诊断即可，不需要写入或修复 mirror：

- 每个 rank 只取一条 replay 请求的前四个 query；比较 mirror 4-gram 与从 `input_batch.token_ids_cpu` 按绝对位置取出并经相同 `token_map`/图像 mask 处理的 canonical 4-gram。
- 只记录 request row、query position、物理 page ID、页内 offset、每个 shift 的 `missing/barrier/present` 状态、逐 shift 是否相等、canonical 与 mirror 的 lookup ID；不记录 token 明文。
- 分开计数 missing 页、present-but-mismatched 槽位和 present-but-barrier 槽位。保留现有 strict 缺页开关作为整页缺失对照；同页先写当前 replay token 再读 lookback 时，`page_present` 已成立，strict 开关无法发现尚未 seed 的历史槽位。

若要做后续正确性干预，应只在计算历史时从完整 prompt seed `previous < replay_start` 的位置，不要把这些值写进可能共享或复用的物理页。这样才能避免污染 page mirror，并让正确 seed 臂与上述 CPU 对照一致。

## 边界

这个实验验证的是 `PagedNgramHistory` 镜像的代码语义，不证明 A3 allocator 本次一定复用了相同 ID，也不证明它导致了空回复或错误文本。实际模型的压缩 token map、素数、多头乘数、表内容，以及 A3 的逐请求 page ID 都需在真实服务上通过上述限量诊断采集。只有 canonical 与 mirror 的实际 hash 不同，才说明 Engram lookup 确实变化；要归因模型输出，还需单独做禁用/正确 seed 对照。
