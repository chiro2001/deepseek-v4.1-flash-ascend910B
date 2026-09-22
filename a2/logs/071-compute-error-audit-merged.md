# 071 — ★★★ **「计算错误隐患」四家合并盘点**（主代理汇总 + 逐条核实）

> 2026-09-22 19:5x。**执行**：主代理汇总四个子代理的独立盘点
> （`a3_p1_offload_draftgraph` / `c1_offload_x_draftgraph_1die` / `c2_int8_x_draftgraph_1die` /
> `engram_pool_contention_src`），并对**每一条关键声称做了独立核实**。
> 子代理原始盘点分别落在 `069`（A3 8 卡）、`067`（单卡卸载）、`066f`（单卡 int8）、`070`（源码侧）。
> 标记：**【实测】/【推断】/【未确认】**；★ = 主代理已独立复核。

---

## 0. ★★★ 一句话（四家独立收敛到同一件事）

> **四家从四个不同角度、用四组不同实验，独立地指向同一个空洞：
> 「**取回/读取的数据，其内容本身**」从来**没有被逐字节验证过**。**

| 谁 | 它看到的那一面 | 证据 |
|---|---|---|
| `a3_p1`（8 卡） | 全部证据是**计数器级** | `CPU→GPU=21,519,269,888` / `hits=901,120` —— 只证明"**搬了多少**"，没证明"**搬对没有**" |
| `c1`（单卡卸载） | sha 判据**只有 2 轮** + 每 prompt **1 token** | `sha_rounds=2`、`gen_tokens=32` ⇒ 判别力极弱 |
| `c2`（单卡 int8） | ★★ **所有单卡臂都在 `--load-format dummy` 下** ⇒ 被探的 KV 平面**全 0** | `out18` v2 探针：`ori_plane_flat/src_i8 (2253,128,1,512) n_nonzero=0 absmax=0`（阳性对照已过 ⇒ 探针可信） |
| `engram_src`（源码） | device-index 的探测**只证"注册被接受"** | `probe_host_mapping_capability` 只注册 **4 KiB 一页**；端到端可读性从未在 A2 验证 |

⇒ ★★ **这就是本晚最需要补的一格**（且它属于 **A 类：静默算错**）。

---

## 1. A 类 · 静默算错（按对 **A2 上线**的威胁排序）

### A1 ★★★ **「取回路径」的数值保真从未验证** —— 唯一的判据缺口（非已观测错误）

| 方向 | 现有证据 | 它**不能**证明什么 |
|---|---|---|
| 卸载取回（`a3_p1`） | 【实测】`p1b`：`CPU→GPU=21.5 GB`、`hits=901,120`、replay TTFT **1.42 s**（对照 `p2c` 池 1 MiB：`CPU→GPU=0`、replay **8.39 s**） | 只证"**量**在流动"；**没有比对过取回的字节** |
| 输出的"非回归"（`c1`） | 【实测】`fill/replay` sha × 2 轮 × 16 prompt × 1 token（`gen_tokens=32`） | ★ `062` 已证：**"热==冷"测的是路径，不是保真**；且 **sha 对"共同的错"零判别力** |
| int8 的"数值 clean"（`c2`） | 【实测】`档 B == 档 C` 三轮 8/8 逐字相同 | ⛔ **平凡** —— 被量化的 KV 平面**全是 0**（见 A2） |

★ **本仓已有先例**：`043` 发现 group 0 **每次只搬 1/8 个 block**（"读到未写过行 5352 → 0"）——
那**不是**靠任何常规判据发现的，是**靠自建探针**。
⇒ **判死的成本**：一次「写侧 GPU 页 vs 取回后 GPU 页」的**设备侧逐字节比对**（需重建 `transfer_async` 指针表，独占一轮）。

### A2 ★★ **`--load-format dummy` 让"数值 clean"变成平凡结论**（`c2` 发现，主代理已核实）

```
out18 v2 探针（阳性对照 C2_NAN_INJECT=1 已被准确捕获 ⇒ 探针可信）：
  tag                       call   shape                 n_nonzero  absmax
  ori_plane/deq              4     (12288, 512)              0        0      ← int8 反量化
  ori_plane_flat/src_i8      3     (2253,128,1,512)          0        0      ← ★ KV cache 本体
  cmp_plane_gs/scales       20     (192,512,1,4)             0        0
```
⇒ ★★ **零张量上永远不会有 NaN** ⇒ `066e` 的"NaN 不在反量化路径里"**已撤回**，
正确表述是「**未测到**；dummy 几何下该路径无法验证」。
⇒ ⚠️ **连带**：`066` 里"档 B == 档 C 三轮 8/8 逐字相同"这条**看起来最强**的数值判据**也是平凡的**
（int8 量化的是全零数据）。**真正成立的只有管路层**：不崩 / 不 NaN / 确定性 / 图兼容 / 容量指纹 `20,826`。
★ 补它的成本低：**非 dummy 权重 + 同一 overlay + 先确认 `n_nonzero>0`** → 再读 NaN 与 B-vs-C 一致性。

### A3 ★★★ **device-index 的"端到端可读性"在 A2 上从未验证**（`engram_src` 发现，主代理逐条核实）

```
① 探测只证"注册被接受"：probe_host_mapping_capability 只注册 4 KiB 一页
   它的 docstring 自己把两者分开，并指向 tools/probe_a2_hostmap.py
   ★ 主代理核实：该路径在 a2/ 工作区【不存在】—— 想验证的人按提示找不到工具
② 真探针存在：bench/probe_engram_hostmap.py（A3 上两个副本）
   ★ 它测的正是生产形态：writable file → mmap(MAP_SHARED) → host_register(MAPPED)
     → DLPack → device tensor → torch.index_select → 与 host 字节逐字节比
   ★ A3 结果（37-hostmap-ab-c1.log）：
       RESULT: SUPPORTED -- ... read back byte-for-byte in 3/3 production rows,
       ★ and host_mem_pool=1          ← ★★ 它自己的结论行绑定了 A3 的属性！
       exit code: 0（0=production 路径可用 / 3=不可用 / 1=探针错误）
③ ★★ 而 A2 是 host_mem_pool=0（logs/065 实测）⇒ 这个"SUPPORTED"不能外推
④ 一条 caveat 要判准确：那三份日志自带
     header check : MISMATCH between the probe constants and acl_rt.h -- do not trust the flag column above
   ★ 但有 [verdict] 说明「this query is informational only」⇒
     **MISMATCH 否定的是那列"信息性 flag"，不是字节比对测量本身**（两者要分开读）
```
⇒ ★★ **这是"静默算错在结构上可行"的一条**：lookup 拿到错的行 ⇒ Engram gate 加权错 ⇒ **输出错但不报错**。
⚠️ **缓解（已落地）**：主代理今日已把交付脚本的 `ENGRAM_DEVICE_INDEX` **默认改成 `0`**（与生产一致）
⇒ **A2 上这条路是关着的** ⇒ **A3 的 EH0012 崩溃不会在 A2 上重演**。
⇒ 但它意味着：**"device-index 加速"在 A2 上目前是**未启用**状态**（不是"已验证可用"）。

### A4 ★★ **`=1` + 两条早退 ⇒ 分片被裁 + 回退 host ⇒ 读从未加载的表**（`engram_src`，可达性低）

```
engram_hbm.py:272   _ENGRAM_DEVICE_INDEX = os.environ.get("V41_ENGRAM_DEVICE_INDEX","0") == "1"
engram_hbm.py:278   _ENGRAM_DEVICE_TRIM_SHARD = _ENGRAM_DEVICE_INDEX and (FALLBACK != "1")
engram_hbm.py:498   shard_rows = 1 if self.device_index else ...
                    self.weight = nn.Parameter(torch.empty(shard_rows, width, ...))   ← ★ 未初始化
engram_hbm.py:650   def load_checkpoint(...): if getattr(self,"device_index",False): return  ← ★ 永不加载
model.py:801        probe 只在 auto 下跑；"1" 走 elif ⇒ 免探测强开
model.py:766/788    load_format=dummy（未设 _ENGRAM_WITH_DUMMY）⇒ engram_history=None ⇒ setup 不调用
```
★ **诚实边界**：`dummy` / `nohost` 都是**诊断口径**，**生产可达性低**。
它的价值是证明 **"全有或全无"的设计意图被绕过**（裁了分片却仍可能走 host 路径）。

### A5 ★ **另一条"可能算错"的结构面（未确认）**

```
engram_device_index.py:635   _scatter_rows 的 except Exception: pass   ← ★ 静默吞设备 op 失败
engram_hbm.py:_metadata      设备路径对【越界 id】的保护【未见】（host 路径有 invalid 位）
```

---

## 2. B 类 · 响亮失败（可回滚）

| # | 隐患 | 证据 | 适用范围 |
|---|---|---|---|
| B1 | `207001`（大张量注册拿不到预算） | `069`：`ENGRAM=1` 下 97/128 成功、31 张（2.82/3.98/4.24 GiB）失败 | **A3 特有**（A2 实测 392 GiB 全过） |
| B2 | `EH0012` → `hdc disconnect` / `507901` | `069`：`pageable`（**零注册**）照样 ×9 ⇒ **与池后端无关** | **A3 特有**；★ **A2 已由 `ENGRAM_DEVICE_INDEX=0` 关掉这条路** |
| B3 | `507057`（cmp 面越界读表 → 未映射地址） | `049` §5.5.3；★ 但**垃圾页号落到已映射内存时是静默算错** | 档 D（已挡死） |
| B4 | `EE1016`（捕获期 `.item()` / 探针的急求值） | `048`/`049`（已修但**默认关**）；`066f` 发现**探针实参立即求值**也触发 | 需 `GRAPH_SAFE=1` |
| B5 | 1M + KV 预算不足 ⇒ `ValueError`（**自带需要的字节数**） | `066`：4 GiB 不够、6 GiB 通过；两读数给 **4,965.9 B/token** 逐字吻合 | A2 必须核 `--kv-cache-memory-bytes` |

---

## 3. C 类 · 判据失效 / 性能降级（"通过"变假象）

| # | 隐患 | 证据 | 状态 |
|---|---|---|---|
| **C1** | ★★ **同输入一致性判据只跑 2 轮 ⇒ 漏 40%** | `066f`：`fill!=replay` `[4,5,7]` vs **三轮任一不同** `[0,3,4,5,7]`；损坏**延迟到第 3 轮才显现** | ✅ 已定规则（`AGENTS §5b #16`）：**≥3 轮** |
| **C2** | ★★ **零张量上"clean"= 假 PASS** | `066f §3`（见 A2） | ✅ 已撤回 `066e` |
| **C3** | ★ **探针插在"没人调用的那一套实现"上** | `066c`：`grep -n "def kv8_ori_plane"` = **2 行**（`:505` / `:1656` import-time 替换） | ✅ 已定规则（§5b #10） |
| **C4** | ★ **trace 覆盖了错的流量**（720 行全来自 warmup/capture） | `066c` | ✅ 已识别 |
| **C5** | ★ **探针自身有观察者效应** | `067`：`C1_HITIDX` 每条 miss-scan 多调 32 次 `lookup()`；但**归零在 6 条无探针臂上同样出现** ⇒ 不是它造成的 | ✅ 已限定 |
| **C6** | ★★ **`collect_c1.sh` 在通过臂上打 `FAIL`**（指标名在本 build 不存在） | ★ 主代理核实：`agents/C1_offload_draftgraph/scripts/collect_c1.sh:36` 取 `kv_offload_block_stored_total`，而该名字在 `metrics_after.txt` 里 **0 命中** | ⚠️ **未修** ⇒ 谁照它读，会把**通过**读成**失败** |
| **C7** | ★ **`-> falling back to pinned` = 零报错慢 3.8×** | `069`：H2D 21→5.5 GB/s；★ 判据已给：`grep -c "aclrtHostRegister failed"` **必须=0** | ✅ 判据已固化 |
| **C8** | ★ **池越大 H2D 越慢**（A2 实测 20.5→5.0 GB/s） | A2 三期探针；机制【推断】= 页表项数量 | ⚠️ 窗口要实测（判据⑦） |
| **C9** | ★ `auto` 下**两份 DRAM 账同时付**（25.75 GB/rank/层照付 + 走设备路径） | `engram_src` 机械实测：9 个输入里 **8 个**两模块解析不一致 | ⚠️ 已由 `ENGRAM_DEVICE_INDEX=0` 缓解 |
| **C10** | ★ `bneck` 消融臂在设备路径下**静默失效** | `model.py:990` 只打一次 print 且只在 rank0 | ⚠️ 未修 |
| **C11** | 全部臂 `concurrency=1` ⇒ **P0-C（DG=1 + conc≥16）一次都没触及** | `069`/`067`/`066` 全臂 | ⚠️ A2 生产 `MAX_SEQS=4` 结构上够不到 |
| **C12** | `max_tokens=1` ⇒ 投机读数无效；`MAX_TOKENS=128` 才拿到有效 A | `069`：基线 `Accepted:1/Drafted:10` | ✅ 已固化 |
| **C13** | 引擎 `serve.log` ≠ 启动器 `.serve_a2.log`（后者 grep 恒空） | `068`/`069` 独立记同一坑 | ✅ 已固化 |

---

## 4. ★★ 对 A2 窗口的直接含义（三条必做）

| # | 动作 | 为什么 | 成本 |
|---|---|---|---|
| **①** | ★★ **在 A2 上跑 `bench/probe_engram_hostmap.py`**（端到端可读性探针） | 它直接回答"设备算子读 host 映射"是 (a) 读对 / (b) 崩 / (c) **静默读错**；★ A3 上它给 `SUPPORTED`，但**结论行自绑 `host_mem_pool=1`**，而 A2 是 0 | 一条命令、不占卡 |
| **②** | ★ **窗口判据加"取回内容"这一格** | 现在的判据（`CPU→GPU>0`/`hits>0`）只能判"有没有搬" | 需 KV 级逐字节（新探针）|
| **③** | ★ **保持 `ENGRAM_DEVICE_INDEX=0`** | 它是 A3 那条 `EH0012` 崩溃的**唯一开关**；`auto` 会在 A2 上把它打开 | 已在脚本默认里 |

### 关于 ① 的一条**重要限定**（主代理核实后的口径）

那三份 A3 日志自带的 `header check : MISMATCH …` **不是"探针坏了"** ——
`[verdict]` 明写那是「**informational only**」，它否定的是**信息性 flag 那一列**，
而**决定 verdict 的是字节比对测量本身**。⇒ **两者要分开读，别把这条当成"探针结果不可信"。**

---

## 5. 明确**不属于**隐患的（避免误读）

| 说法 | 判定 |
|---|---|
| "取回归零 = 取回了错数据" | ⛔ **不成立**：`CPU→GPU = 0.0`（零字节移动）+ replay 无加速（1.01×）⇒ 只能是**重算**（性能，不是算错）【推断】 |
| "`-> falling back to pinned` 会算错" | ⛔ **不是**：回落换的是 host 后端，不是数据；跨臂 fill sha 逐字相同 + A2 探针对 pageable/registered **都**做过逐字节往返⇒ 【推断】不是算错（证据是间接的） |
| "8 卡好、tiny 坏 ⇒ 差异在规模/几何" | ⚠️ **降级为【未确认】**：只证了 spec 逐字相同（排除"合成构造"），**没有测**是哪一项差异（`Σslot_pages`/块池规模/`MAX_SEQS`/请求长度全不同） |
| "`EH0012` 是注册争用更早一次表现" | ⛔ **已撤回**（`pageable` 零注册照样 ×9） |
