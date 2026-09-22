# 053 — 档 C 在发布件 `94aeebb7…` 上的 ACLGraph 捕获：**单 die 判定 PASS**（三问全绿 + 三条反例臂）

> 2026-09-22 11:4x–12:0x CST。执行：子代理 **C1_graph1die**。
> 卡：**只有 c1（die 6）**，全程走 `tools/a3_chip.sh c1` 锁（**未出现退出码 75**；未碰 c0/c2、
> 未碰 Phy-ID 8–15、未碰别人的容器 / `dsv41-a3` / `mooncake-master`）；**未手设 `ASCEND_RT_VISIBLE_DEVICES`**；
> 起服前查过 `/dev/shm`（容器内 64 MiB，**Used 0**）；**未用 `/tmp`**；**未写 `upstream-v41/`**。
> 产物：`a2/agents/C1_graph1die/`（脚本）、`a2/logs/raw/053-c1-graph1die/`（原始输出）。
> 标记：**【实测】**= 有判别力的判据跑出来的数；**【推断】**；**【未确认】**。

## 0. 三条答案（先说结论）

| # | 问题 | 答案 | 判据出处 |
|---|---|---|---|
| **Q1** | 捕获期是否真的走了 bound 分支？ | **是**【实测】`capturing=True ... rows_bound=6`（= 1+num_spec），**不是** num_reqs=32；反例臂 `GRAPHSAFE=0` 时同一几何给出 `graph_safe=False rows_bound=None`（legacy） | `c1-tc-graph.ppr_final.txt`（12 行 True）/ `c1-tc-legacy.ppr_final` / `03-probe-fix.txt` |
| **Q2** | 图捕获成功了吗？ | **成功**【实测】`Graph capturing finished in 25 secs, took 1.48 GiB`；`EE1016=0 / Not_Supported=0 / capture failed=0`。**反例臂响亮地炸**：`EE1016=7 / Not_Supported=7 / capture failed=1`，起服死亡 | `c1-tc-graph.verdict.txt` / `c1-tc-legacy.verdict.txt` |
| **Q3** | 图模式的输出算得对吗？ | **对，逐字节相同**【实测】图臂 `fill=replay=b8794ee7ec13710b…a11aa`，**eager 参考臂逐字相同**；代码级另有 `replay(B)==eager(B)`、`replay(C)==eager(C)` 两组逐字节相等 | `c1-tc-graph.client.json` / `c1-tc-eager.client.json` / `03-probe-fix.txt` |

**★ 本条结论对应的发布件 md5（`md5sum` 现算，三处独立复算）**：

```
94aeebb757d6d5708268754481a05e0a  attention/dsa_v41.py
  (a) A3 宿主：agents/S_graphfix/pkgs/pkg-kv8pf/shadow/vllm_ascend/attention/dsa_v41.py
  (b) 容器内 overlay（本臂真挂的那一份，mkoverlay.sh 断言）
  (c) 探针运行期 mod.__file__ → md5（capture_probe.py 打印）
```
⇒ 与 `publish/kv8-graphsafe/README.md` §3 的成品**逐字节相同**；**不是** `22cbf20c…`（8 卡旧 PASS 的那份，
已丢失）也**不是** `1cc9e992…`（作废件）。

**★ 「1–2 卡 dummy + 截断 config 能否替代 8 卡」的答案：能替代这一类问题（【实测】）**，见 §3。

---

## 1. 臂与判据总表（全部【实测】）

单位：单 die（c1 / die 6）。模型 = `T_draftceiling/models/model-tiny-draft`
（= `L1_dummy` 的 tiny 瘦身 config **+ 还原 draft 两个键**：`num_nextn_predict_layers=3`、
`dspark_target_layer_ids=[37,38,39]`），`--load-format dummy`，**保留 DSpark 投机（5 draft tokens）**，
档 C = `VLLM_V41_KV8_SWA=1 VLLM_V41_RING_FP16=1 VLLM_V41_KV8=0 VLLM_V41_KV8_PREFILL=0`。

| 臂 | 图 | `VLLM_V41_KV8_GRAPH_SAFE` | 起服 | `Graph capturing finished` | `EE1016` / `capture failed` | `[SG-PPR]` 捕获期 | 输出 sha（fill=replay） |
|---|---|---:|---:|---:|---:|---|---|
| `c1-tc-graph` ★ 判决臂 | FULL_DECODE_ONLY | **1** | ✅ 141 s | ✅ **25 secs** | **0 / 0** | `capturing=True rows_bound=6`（num_reqs=32, query_rows=192） | `b8794ee7ec13710b…a11aa` |
| `c1-tc-eager` Q3 参考臂 | `--enforce-eager` | 1 | ✅ 40 s | —（正确：eager 不捕获） | 0 / 0 | 无捕获（压测期 `capturing=False rows_bound=6`） | **`b8794ee7ec13710b…a11aa`（与图臂逐字相同）** |
| `c1-tc-legacy` ❌ 反例臂 | FULL_DECODE_ONLY | **0** | ❌ **50 s 死** | 无 | **7 / 1** | `graph_safe=False rows_bound=None` | 服务没起来（无输出） |
| `probe --arm legacy_capture` ❌ | 代码级真捕获 | 1（但显式传 `rows_bound=None`） | — | — | **RAISED**（`EE1016`，见下） | — | 控制臂 `query_rows==num_reqs` 能捕获 ✅ |
| `probe --arm fix_capture` ★ | 代码级真捕获 | 1 | — | ✅ | 0 | `capturing=True graph_safe=True rows_bound=6` | `replay==eager` 逐字节 ✅ |
| `probe --arm degrade` ❌ | 不捕获 | 1 | — | — | — | API 抛异常 ⇒ `_sg_is_capturing()=False` ⇒ prefill 类批 **退回 legacy** | — |

三条反例臂分别打在**三个不同层**上：服务级（主开关关）、函数级（强制 legacy 分支）、API 降级级。**同一条判据在两臂上跑**，
两侧给出**不同的数**（0 vs 7 / `rows_bound=6` vs `None` / OK vs RAISED）⇒ 不是假阳性。

---

## 2. 三问的细节

### 2.1 Q1 —— 捕获期走的是 **bound 分支**，且 bound = `1+num_spec`（不是 num_reqs）

```
# c1-tc-graph（8 rank→1 rank，逐字复现 8 卡那一行的形状）
[SG-PPR] native_attention capturing=True  num_reqs=32 query_rows=192 num_prefills=0 max_query_len=6 swa_mcs=6 cmp_mcs∈{0,3,6} graph_safe=True rows_bound=6
[SG-PPR] native_attention capturing=True  num_reqs=16 query_rows=96  ...
[SG-PPR] native_attention capturing=True  num_reqs=8  query_rows=48  ...
[SG-PPR] native_attention capturing=True  num_reqs=7  query_rows=42  ...
  （24 行去重：capturing=True 12 行 + capturing=False 12 行；rows_bound=6 命中 24/24，rows_bound=32 命中 0）
```
* `query_rows(192) != num_reqs(32)` —— 这就是 436 的击穿形状（spec-decode 的 decode 批每请求 6 行）；
  补丁把它路由到上界分支，`rows_bound = max_query_len = 6 = 1 + num_spec`。
* **反例臂**（`GRAPHSAFE=0`，同一 tiny 模型 / 同一捕获桶）：
```
[SG-PPR] native_attention capturing=True  num_reqs=32 query_rows=192 ... graph_safe=False rows_bound=None   ← legacy 行为
```
  ★ 说明：任务书写的是"反例臂 `SHARED=0`"。我这里用的是**主开关 `VLLM_V41_KV8_GRAPH_SAFE=0`**——
  因为 `SHARED_PREFIX` 是**压测负载**开关（共享前缀），与图路由无关；本任务的 Q1 反例只能由主开关构造。
  （本臂压测未开共享前缀，见 §7 的边界。）
* **API 降级臂**（`is_current_stream_capturing` 抛异常）：
```
degrade: _sg_is_capturing() -> False
degrade: decode  类批 -> (True, 6)      ← 与降级无关（decode 形状本来就该走上界分支）
degrade: prefill 类批 -> (False, None)  ← ★ 退回 legacy，绝不猜
RESULT degrade_path=OK
```
  ⇒ 与 049 §7 的承诺一致：**API 不可用时 `_sg_is_capturing()` return False**，真 prefill 不会被误路由。

* **真 prefill 仍走旧支**（设计承诺，实盘确认）：`c1-tc-eager` 压测期的 prefill 批给出
```
[SG-PPR] native_attention capturing=False num_reqs=1 query_rows=256 num_prefills=1 max_query_len=256
         swa_mcs=256 cmp_mcs∈{0,128,256} graph_safe=False rows_bound=None    ← 旧支 + 它自己的融合 kernel
```

★ **一条额外证据（"冻结的常量语义正确"）**：同一进程里，**压测期的 decode 批**逐字给出
`capturing=False num_reqs=1 query_rows=6 max_query_len=6 swa_mcs=<变> graph_safe=True rows_bound=6`
（`c1-tc-eager.ppr_final.txt`：**18/18 行**都是 `graph_safe=True rows_bound=6`，而 `swa_mcs` 取到
**262 / 268 / 518 / 524 / 525 / 527** 六个不同值）。
⇒ 捕获期冻下来的 `rows_bound=6` 与 replay 期**逐字相同**；而**同一个函数里那个"值" `swa_mcs` 是会变的**
（518→527）。补丁刻意用"形状/配置常量"而不是"值"，这就是它与档 D 静默读错（捕获期把 `mcs=6` 冻成页数）的分野。

### 2.2 Q2 —— 捕获真的成功了（且反例臂真的炸）

```
# c1-tc-graph
Capturing CUDA graphs (decode, FULL): 100%|████| 9/9
(EngineCore) INFO [gpu_model_runner.py:6913] Graph capturing finished in 25 secs, took 1.48 GiB
EE1016 = 0   Not_Supported = 0   capture failed = 0

# c1-tc-legacy（★ 同样的 tiny 模型 / 同样的捕获桶，只把 graph_safe 置 0）
NPUGraph: ERROR — capture failed: RuntimeError: ... AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027
Not_Supported(EE1016): Synchronizing a stream failed. Reason: Stream (stream_id=34) during the capture stage is not supported.
→ NPUModelRunner init failed ⇒ 起服 50 s 死亡（ee1016_lines=7 / capture_failed_lines=1）
```
★ **反例臂的"死"同时证明了判决臂的"活"不是空判据**：`.item()`（host 同步）在 `kv8_ori_plane` 的
**int8 分支**里；若这条臂的 SWA 平面其实是 BF16，`kv8_ori_plane` 根本不会被调用，
**legacy 臂就不会炸**。它炸了 ⇒ 本臂的 **int8 SWA 重建路径是真在跑的**（与 8 卡 048/049 的失败签名同款）。

★ 判别力对照：`c1-tc-eager`（`--enforce-eager`）**没有** `Graph capturing finished` 行
⇒ "捕获完成"这行字确实只由真捕获打印，不是图臂碰巧捡到的。

### 2.3 Q3 —— 图模式输出 == eager 输出（**逐字节**），且图没把输入冻死

**端到端（两个独立进程，同 tiny-draft 模型、`--load-format dummy`、`--seed 0`、同 salt）**：

```
c1-tc-graph : fill=b8794ee7ec13710b038838ff9e5d8799ce3ec40684ed20b205e8ffda5b2a11aa
              replay=b8794ee7ec13710b038838ff9e5d8799ce3ec40684ed20b205e8ffda5b2a11aa  (match=True)
c1-tc-eager : fill=b8794ee7ec13710b038838ff9e5d8799ce3ec40684ed20b205e8ffda5b2a11aa
              replay=b8794ee7ec13710b038838ff9e5d8799ce3ec40684ed20b205e8ffda5b2a11aa  (match=True)
⇒ fill_equal=True / replay_equal=True
```
**代码级（同进程、真 ACLGraph、静态输入原地改写）**：
```
Q3 set B: eager scratch=1956fecf0a235c37 replay scratch=1956fecf0a235c37 same=True
          eager table  =83a6406a12d1af8a replay table  =83a6406a12d1af8a same=True
Q3 set C: eager scratch=48fc5d89d59ec469 replay scratch=48fc5d89d59ec469 same=True
          eager table  =876ed392708391e4 replay table  =876ed392708391e4 same=True
Q3 图未冻结输入（replay(B) != replay(C)）: {'scratch': True, 'table': True}   ← ★ 输入是活的
RESULT Q3 replay_eq_eager=True inputs_live=True
```
⇒ 三件事同时成立：① 图 replay 与 eager **逐字节相同**；② 换一组输入后**输出确实变了**（不是"读了冻结的值"）；
③ 表（block table）本身也逐字节一致（不是只对了数值、指针错了）。

★ **端到端 sha 的判别力**（照 049 §5 的口径）：该服务在 `temperature=0` 下同臂内都会抖（037 实测），
而这里**跨"图 vs eager"两个进程逐字相同** ⇒ 数值等价。（绝对 sha 值与本任务无关，逐字相同才是判据。）

---

## 3. ★ 「1–2 卡 dummy + 截断 config 能替代 8 卡吗？」——**能，就这一类问题**【实测】

### 3.1 几何与签名逐项对齐（8 卡 vs 1 卡 tiny）

| 量 | 8 卡 `sg-a-c-graph`（`049` §5.1，md5 `22cbf20c…`） | **本次 1 卡 tiny**（md5 `94aeebb7…`） | 一致？ |
|---|---|---|---|
| 捕获桶 | 192 行（32 请求 × (1+5)） | **192 行**（`cudagraph_capture_sizes=[…,192]`） | ✅ 逐字 |
| `num_reqs` / `query_rows` | 32 / 192 | **32 / 192** | ✅ 逐字 |
| `max_query_len` / `swa_mcs` | 6 / 6 | **6 / 6** | ✅ 逐字 |
| 捕获期 `[SG-PPR]` | `capturing=True … graph_safe=True rows_bound=6` | **同一行，逐字** | ✅ |
| `EE1016` / `capture failed` | 0 / 0 | **0 / 0** | ✅ |
| 捕获完成标志 | `Graph capturing finished in 324 secs` | `… in 25 secs` | ✅（时长不同，形状同） |
| 「图==eager」判据 | replay sha == eager 臂 sha 逐字 | **同判据，逐字相同** | ✅ |
| ❌ 关闭开关后 | 048 里 8 卡同栈炸 EE1016 | **本臂 `c1-tc-legacy` 同栈炸 EE1016（7 行）** | ✅ |
| 单臂代价 | **≈22 min/臂 + c0 排队（本次要等 ~1 h）** | **起服 141 s + 压测 ~5 s**（overlay 复用后 ≈2.5 min/臂） | ★ 快 **~9×**（不含排队） |

### 3.2 结论与**边界**（这条比"能"本身更重要）

* ✅ **能替代的**：本类问题的**全部三问**（图路由、图兼容性、图 vs eager 输出等价）——
  因为这三问**只依赖"代码路径 + 捕获形状"**，而这两者在 tiny 上**逐字复现**（上表）。
  这是 `T_draftceiling` 那条先例（tiny 上合成 draft 组能逐字复现 8 卡的 geometry 断言）的第二个实例。
* ⛔ **不能替代的（必须回 8 卡）**：
  1. **容量/性能类判据**：tiny 的 KV 容量、TTFT、命中率、`replay/fill` 倍率都不可外推；
  2. **真权重数值**：`--load-format dummy` 的 sha 只能证明"图==eager"，证明不了"算得对"（真权重下才算）；
  3. **多卡特有面**：TP=8 的通信 / EP / `enable_expert_parallel` 在单卡上**不存在**；
  4. **插件与卸载**：本臂**没挂 OffloadingConnector**（判据里 `kv_offload_*` 全 None）⇒ 卸载面不覆盖。
* ★ **一个必须记住的陷阱**：**本几何下"KV 容量"不是档 C 的判别量** ——
  8 卡实测档 C = 档 B = **427,643**（`050` §1.2：slot0–2 被 BF16 draft 平面顶死 131,072），
  本臂 tiny 同样给出 **20,826 = 档 B 的 20,826**。
  ⇒ 谁若拿"KV size 没变"当"int8 没生效"，就会得出**相反的结论**。判别 int8 是否生效要**看代码路径**（本臂用 EE1016 反例）。

---

## 4. 代码级探针（`capture_probe.py`，只占"能看见设备"这一件事）

不加载模型、不起服，直接在**真 `torch.npu.graph()` 捕获**里调**真函数**：

| 臂 | 做法 | 结果 |
|---|---|---|
| `legacy_capture` | ① 控制臂：`rows_bound=None` + `query_rows==num_reqs`（纯 decode 形状）② 目标臂：`rows_bound=None` + `query_rows=192` | ① **OK**（证明探针自身没把设备搞坏）② **RAISED `Not_Supported(EE1016)`**（`AclrtSynchronizeStreamWithTimeout … 107027`） |
| `fix_capture` | 捕获里打印 `_sg_is_capturing()` / `_kv8_graph_rows_bound()`，再跑 `kv8_ori_plane(..., rows_bound=6)`，replay 两组输入与 eager 逐字节比 | `capturing=True graph_safe=True rows_bound=6`；捕获 OK；**replay(B)==eager(B)、replay(C)==eager(C)、replay(B)!=replay(C)** |
| `degrade` | 把 `torch.npu.is_current_stream_capturing` 换成"抛异常" | `_sg_is_capturing()=False`；decode 类批 `(True,6)`；**prefill 类批 `(False,None)`** |

★ **探针纪律（`AGENTS §5b`）落实**：
1. **先装 hook 再 import** —— 探针在 import 目标模块**之前**就把 `PYTHONPATH` 指向 overlay，并断言
   `mod.__file__` 落在 overlay 里、`mod._kv8_graph_rows_bound.__code__.co_filename` 也指向它（不是镜像里的旧件）；
2. **打版本号/地址** —— 每次运行都打 `module md5 = 94aeebb7…` + 函数 `co_filename`；
3. **判据在反例臂上对称跑** —— 见 §1 的 6 条臂；
4. **不靠"已装载"下结论** —— `capture_probe` 都打印**热路径的返回值**（不是"已安装"横幅）。

### 4.1 踩到的坑（给后来人）

* **`sitecustomize` + `PYTHONPATH` ⇒ 循环导入**：首跑（raw `00-smoke.txt` / `01..03`）把
  `pkg/patch`、`pkg/patch_pgp` 一起放进 `PYTHONPATH` 时，两个 `sitecustomize` 会在**解释器启动阶段**
  import `vllm_ascend`，于是撞
  `ImportError: cannot import name 'DeviceOperator' from partially initialized module 'vllm_ascend.device.device_op'`
  （`device_op → vllm_ascend.ops.__init__ → … → moe_mlp → device_op`）。
  ⇒ **结论**：**代码级探针只用 `shadow` 进 `PYTHONPATH`**；要带补丁目录时**必须先 `import vllm` +
  `import vllm_ascend.ops`** 再 import 目标子模块（`capture_probe.py::setup()` 与 `smoke.sh` 已按此修）。
  （`vllm serve` 那条路不撞，是因为它先 import 了 `vllm`。）
* **`capturing=True` 与 `capturing=False` 会同时出现**：24 行去重里 **12 True + 12 False**，
  两组的 `num_reqs/query_rows` 完全相同（8 卡的 `sg-a-c-graph` 也是 96 + 96）。
  ⇒ 只看一行容易误判；**判据要按"有没有 `capturing=True` 且 `rows_bound=6`"来取**。

---

## 5. 复现（三步，全部可重放）

```bash
# 0) 本地→A3（走 COS，不用 ssh 管道）；A3 上取回
bash a2/agents/C1_graph1die/scripts/upload.sh
ssh A3-node1 'cd ~/tmp/20260922/c1_graph1die && coscli cp cos://uploads-new/share/xfer/c1_graph1die/c1_graph1die.tgz . && \
  tar xzf c1_graph1die.tgz -C ~/projects/dsv41-upstream-pr/agents/C1_graph1die --strip-components=1'

# 1) 容器内：造 overlay（断言 dsa_v41.py == 94aeebb7…）→ 跑三臂
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c1 --timeout 2400 --name c1graph1die -- \
  bash /work/agents/C1_graph1die/scripts/run_all.sh'

# 2) 代码级探针（不加载模型；PYTHONPATH 只放 shadow）
ssh A3-node1 'cd ~/projects/dsv41-upstream-pr && bash tools/a3_chip.sh c1 --timeout 1200 --name c1probes -- \
  bash /work/agents/C1_graph1die/scripts/probes_only.sh'

# 3) 汇总
python3 a2/agents/C1_graph1die/scripts/summarize_c1.py a2/logs/raw/053-c1-graph1die
```

---

## 6. 卡与环境纪律

| 项 | 状态 |
|---|---|
| 用卡 | **只有 c1（die 6）**，全程 `tools/a3_chip.sh c1`；**退出码 75 一次都没出现** |
| c0 / c2 | **未碰**（c0 整段时间被别人的 8 卡 `r8-e1-tierC-eager-cold` 持有；c2 未用） |
| Phy-ID 8–15 / `dsv41-a3` / `mooncake-master` | **未碰** |
| `ASCEND_RT_VISIBLE_DEVICES` | **未手设**（容器自带） |
| `/dev/shm` | 起服前查过：容器内 64 MiB / **Used 0**；三臂结束均正常 |
| `/tmp` | **未用**（本机临时区 `~/tmp/20260922/c1_graph1die/`） |
| `upstream-v41/` | **未写**（只读参考） |
| 共享容器 | 只 `docker exec`（经 `a3_chip.sh` 加锁）；**没改镜像里的任何源码**（overlay 只在我自己的 `/work/agents/C1_graph1die/pkg-gs`） |
| 每臂收尾 | `run_arm_c1.sh` 的 trap 杀自己起的 `vllm serve` 进程组；容器本体保留 |
| 交付件身份 | overlay 里 `attention/dsa_v41.py` = **`94aeebb757d6d5708268754481a05e0a`**（三处现算） |

---

## 7. 诚实边界（【未确认】的格子，不许用相邻数字顶替）

1. **本次没有挂 OffloadingConnector**：所以**卸载面的判据**（`BlockStored:CPU`、`CPU→GPU>0`、
   `external_prefix_cache_hits>0`、`replay≪fill`）**这轮一格都没有**——它们已经在 8 卡 `sg-a-c-graph`
   上过了（但那是在 **`22cbf20c…`** 上）；**在 `94aeebb7…` 上仍未在 8 卡复跑**（`S_graphfix` 的 `sg-c-*` 在跑）。
   ⇒ **本日志只回答"代码路径 + 图兼容性 + 图/eager 等价"，不回答容量与卸载收益。**
2. **档 D（long-KV 面）不在本任务范围**：本臂 `VLLM_V41_KV8=0 / KV8_PREFILL=0`
   ⇒ `_kv8_cmp_plane` 的 `graph_safe` 分支**未被执行**；它的 `index_select` 修复**本臂未验**。
3. **压测未开共享前缀**（`SHARED_PREFIX` 未设）⇒ 与 8 卡臂的 `PREFIX=1` 负载形状不完全同构；
   但 Q1–Q3 的判据都在**捕获形状**与**输出等价**上，不受此影响。
4. **tiny 的绝对 sha / TTFT / 容量**一律**不可**外推到 A2/A3 真权重（§3.2）。
5. **`capturing=False` 那 12 行的语义未逐行定案**（【推断】是 npugraph_ex 在 `torch.npu.graph()` 之外的
   warmup/编译段；两组的 `num_reqs/query_rows` 相同，不影响 Q1 的判据，但**没有**做进一步的栈级归因）。
6. 本日志的主体是**单 die**；**8 卡上对 `94aeebb7…` 的复跑**仍由 `S_graphfix` 的 `sg-c-graph-b`（判据⑧，同 sha 复跑）给出。
