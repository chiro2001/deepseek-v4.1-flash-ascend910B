# A3 CED-PD 暂停交接（2026-09-24）

> **2026-09-24 22:xx 勘误：本页写于根因定位之前。** 下面「D 图模式短针 2/2 乱码」
> 「关闭多流仍不稳定」等记录是**当时的观测**，不是原因。真正的原因是 D 的 replay
> 第一个 query 读到自己未持有的 SWA 页（行内 `0` ⇒ 物理块 0 ⇒ 池绕回后是别的请求
> 的数据），触发条件是物理块分配碎片化。修复已合入（`[CED-SWA-CLIP]`，
> `V41_CED_SWA_CLIP` 默认 `1`），离线判据与算子源码级核实见
> [`CED-D-1M-LAYOUT-BUG-20260924.md`](CED-D-1M-LAYOUT-BUG-20260924.md)。
> **8+8 真机 A/B 仍未执行**，所以本页「不要把任何 eager 或图消融配置当交付配置」
> 的要求继续有效。

用户已明确要求**交接并暂停**。恢复前先读本页、
[`A3-CED-PD-HANDOVER-20260924.md`](A3-CED-PD-HANDOVER-20260924.md)和
[`A3-CED-PD-WORKLOG.md`](A3-CED-PD-WORKLOG.md)。当前目标尚未完成；
不要把任何 eager 或图消融配置当成交付配置。

## 目标与资源

- 目标：A3-21 真实权重 TP8 P（层0–19及层20全局源）＋TP8 D（128-token
  SWA replay、完整40层生成），BF16 KV；仍需稳定1M、流式、多轮、缓存命中
  和同条件性能验收。复杂决策由主 Agent 作出，8+8 长运行交给子代理。
- A3-21 P 一直使用原实例，chip0–7，容器ID前缀 `0db3d626`、HTTP18990、
  KV19090。D 仅用chip8–15、HTTP18991、KV consumer19091；proxy HTTP18992。
  **恢复时必须重新只读确认**容器ID、health、端口与HBM，不能从本页推断
  当前 D/proxy 仍在或可重启。停D后chips8–15的HBM曾延迟释放约一分钟，
  需等到约2–4GiB/卡且无NPU进程/fuser holder，再启动下一实例；未做reset。
- A3-22 chip0/1由用户预留，不使用。本次检查宿主2.0TiB内存中约1.8TiB
  可用；NPU0–15被其他账号量化/评测占用。我们只有空闲tiny P容器
  `dsv41-ced-tiny-p-b6ca283` 在chip6，约8GiB宿主/5.2GiB HBM，
  运行及等待请求均为0；tiny D/proxy/profiler已停。1+1当前只做离线分析。

## 已验证与未通过

固定真实权重、`MAX_LEN=1048576`、BF16 KV、Engram host/index0、
`CPU_BIND=0`、`SPEC=PREFIX=0`。同一22-token请求SHA
`f670186097daa964d1eca4c436320e8dcee6dc93cda0ea41b1306224991aa881`；
同一1M D原始请求SHA
`f25d0b8b1a0ea3e4ff37131dc90182a2940ef075f7fc1ab8fcbb7c99cd9f9d8e`。

| D模式 | 短针/144K | 同SHA 1M D重复 | 结论 |
|---|---|---|---|
| 全 eager `GRAPH=0 EAGER=1` | 短针、144K A/B正确 | 4/4正确 | 仅正确性诊断基线；用户不接受作性能交付 |
| 原图 `GRAPH=1 EAGER=0` | 22-token短针2/2混杂乱码 | 未验 | 首个生成token已错误 |
| 图＋prompt-tail补丁 `777bc73` | 同SHA `max_tokens=1/2/32`、144K A/B正确 | 2/4正确 | 生成阶段实际有ACL graph replay，仍不稳定 |
| 上项＋`DSA_OVERLAP=0` | 短针、144K A正确 | 3/4正确，第四条`content=null` | 关闭DSA辅助流不足以修复 |
| 上项＋`MULTISTREAM=0` | 短针、144K A正确 | 3/4正确，第四条`content=null` | 两处模型多流都关仍不稳定 |
| 上项＋`V41_CED_METADATA_INLINE=1`（`d7953e5`） | 短针、144K A正确 | 前6次：对、对、对、空、对、对 | 元数据独立流也不是唯一原因；空回复没有持续锁死 |

图＋prompt-tail补丁的一轮标准1M A/B/C/D为A/B/C正确、D错误；不能写成
“1M四针通过”。上述图模式失败均HTTP200，无OOM/Traceback。`content=null`
的响应仅1 completion token，但原始响应和服务日志没有sampler token ID；
**不能推断它就是EOS**。原始证据和SHA清单在
[`evidence/ced_graph_ab_20260924/`](../evidence/ced_graph_ab_20260924/)。

暂停时第二组 `metadata-inline` **D-only** 序号实验已安全收尾：新D实例
没有先发短针/144K，直接对同一1M D请求串行8次；D1–D3、D5–D7返回
`RB9N-6014`，**D4和D8都返回 `content=null`、completion_tokens=1**。
8次均HTTP200、同一请求SHA、相同128-token replay、无U+FFFD；
没有第9条或在飞请求。这是明确的每4次重复现象，仍没有sampler token ID，
不能写成“每4次生成EOS”。原始逐条request/response/sidecar在A3-21：
`/home/l00886679/tmp/20260924/ced_numeric/pkg_d7953e5/src/results/ced_metadata_inline_repeat8_20260924_1450/probe/inline_repeat8/`。
**该组聚合SHA256SUMS和tar尚未生成**，恢复后先只读核对逐条sidecar，
再做聚合归档；不能假设已同步到本地或GitHub。

暂停时A3-21 P/D/proxy均保持running：P ID前缀`0db3d626`、chips0–7、
18990/KV19090；D ID前缀`ab521999`、chips8–15、18991/KV19091；proxy
ID前缀`e2a51073`、18992。D为`GRAPH=1 EAGER=0`、CED decode、
`V41_CED_METADATA_INLINE=1`、prompt-tail1、`MULTISTREAM=0`、
`DSA_OVERLAP=0`。暂停时全部health正常，未停服务/reset；恢复仍要重新核验。

## 代码、证据与分支

- GitHub实验分支 `feat/ced-pd-a3@688fe1e` 已推送并从HEAD通过
  `bash tools/selfcheck_pkg.sh`。主工作树随后本地提交了Engram离线证据
  `08e4cbe`，尚未推送；本页及最新工作日志还有未提交改动。
- 图prompt尾步诊断包 `fix/ced-graph-prompt-tail@777bc73`：
  COS私有key `share/xfer/ced_pkg_777bc73_20260924.tar.gz`，tar SHA-256
  `f096c822b4a5ec75fed019f55856c58ce7d101083715261590f07cbff72a2897`。
  它只把仍未算完的单token prompt尾步强制eager，生成decode继续走FULL图。
- 元数据主流诊断包 `fix/ced-metadata-inline@d7953e5`：
  COS私有key `share/xfer/ced_pkg_d7953e5_20260924.tar.gz`，tar SHA-256
  `da885de69ce53a5991894aad266e26ae57ee6f041cc1bdb123ed1bc29efc50f2`。
  该包在独立worktree `/home/chiro/projects/dsv41/ced-metadata-inline`，
  本地selfcheck已过；**尚未合入/推送主实验分支**。
- 1+1离线证据：
  [`ced_tiny_pos21_20260924/`](../evidence/ced_tiny_pos21_20260924/)记录eager/图
  profiler小型摘要，原始trace存私有COS；
  [`ced_swa_window_view_proto_20260924/`](../evidence/ced_swa_window_view_proto_20260924/)
  用CPU扫128种页对齐，当前lower SWA尾部传2页会漏左窗页，保留255-token
  跨度并传3页后静态覆盖；upper SWA在P未算，scratch不能恢复缺失历史。
  [`ced_engram_replay_history_20260924/`](../evidence/ced_engram_replay_history_20260924/)
  用真实144K replay几何做CPU模拟：D前3个replay query的Engram 4-gram
  可能读缺失/旧物理页而改变lookup；**尚未在A3真实请求上取证**。

## 恢复后顺序

1. 先只读确认A3-21现存P/D/proxy、D-only8逐条原始sidecar/请求SHA和健康
   状态；不要补发请求或重启。为该组补聚合SHA清单与只读归档，并检查
   本地`git status`及远端目录，避免重复测试。
2. D-only8恰在第4和第8条空回复。优先查图输入/metadata常驻buffer的
   跨请求生命周期、请求槽复用与生成首token；可加只读sampler token/logit
   和Engram canonical-vs-mirror探针。任何修复先单变量验证，不从
   `content=null`推断EOS，也不把8次结果当作根因证明。
3. 有稳定图模式正确性后再做标准1M四针、144K/1M流式与多轮、缓存命中，
   最后对照全40层PD做prefill、TTFT、TPOT、吞吐和资源统计。

恢复前不向用户推荐任何CED D生产启动命令；当前包装器显式标识诊断臂。
