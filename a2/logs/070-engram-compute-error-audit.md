# 070 · Engram × 卸载 × device-index：**计算错误隐患盘点**（源码侧）

> **性质**：纯读码/读日志，**未占卡、未起容器、未跑 vLLM**。
> **读的是哪一份**：容器 `prbench-c0` 内 `/work/shadow-pkg/patches/files/`（md5 见 `068` §1.3）
> ＋ A3 上 `agents/N_ngram/out/37-hostmap-*.log`。
> ★ 口径确认：本卷沿用 `068` 的**已更正**口径 —— **Engram 先注册（339 行）→ 池后（985 行）**；
> **池的 207001 不是起服失败的原因**（它回落成 pinned 且成功），**起服的直接原因是 Engram 的 D2H 标量**。

---

## A. 静默算错（最危险）

| # | 隐患 | 依据（源码行 / 实验） | 已排除吗 | 证伪/证实需要什么 |
|---|---|---|---|---|
| **A1** | ★★★ **`V41_ENGRAM_DEVICE_INDEX=1` + 两条早退路径 ⇒ 分片被裁 + 回退 host ⇒ 读未加载的表** | 【实测·读码】见 §A.1 的四段代码 | ⛔ **未排除**（组合可达） | 反例：在**同一臂**里同时满足 `=1` + `load_format=dummy`（或 bneck `nohost`），看 host 路径是否被走到、输出是否为 1 行表的产物 |
| **A2** | ★★ **`auto`（发布默认）下 hbm 不裁分片 ⇒ 25.75 GB/rank/层 DRAM 白付** | 【实测·机械】§A.2 的对照表：`auto` ⇒ hbm `False` / model `True` | ⛔ 未排除（是**性能**项，不是算错） | 起服后读 `P2_WORKER_HOST_BYTES` / 进程 RSS，与 `=1` 臂比 |
| **A3** | ★★★ **探测只证"注册被接受"，端到端可读性在 A2 上从未验证** | 【实测·读码+文件】`engram_device_index.py:208-250` 注释；结果文件只有 `37-hostmap-{c0,c1,c2}.log`（**A3 槽位**） | ⛔ **未排除**，且**这是 A2 的问题** | 在 **A2** 上跑 `bench/probe_engram_hostmap.py`（★ 见下：docstring 指的工具名不存在） |
| **A4** | ★ **`_scatter_rows` 的 `except Exception: pass` 会静默吞掉设备 op 失败** | 【实测·读码】`engram_device_index.py:635-643` | ⛔ 未排除 | 强制让 `npu_scatter_nd_update_` 失败，看是否静默走 `index_put_` 并给出不同结果 |
| **A5** | ★ **`lookup` 的 `invalid = bool(flat.min() < 0 or flat.max() >= self.rows)` 依赖 host 标量**；`_empty_metadata[-1]` 用 `int(invalid)` 传"越界"标志 | 【实测·读码】`engram_hbm.py:_metadata` | ⛔ 未确认（host 路径的越界保护；设备路径有无等价保护**未见**） | 设备路径造一个越界 id，看是崩、静默 clamp 还是正确屏蔽 |

### A.1 四段代码（A1 的证据链，全部【实测·读码】）

**(1) 分片被裁的判据 —— 只看 env，不看探测结果**

```python
# engram_hbm.py:272
_ENGRAM_DEVICE_INDEX = os.environ.get("V41_ENGRAM_DEVICE_INDEX", "0") == "1"
# engram_hbm.py:278
_ENGRAM_DEVICE_TRIM_SHARD = _ENGRAM_DEVICE_INDEX and (
    os.environ.get("V41_ENGRAM_DEVICE_FALLBACK", "0") != "1")
# engram_hbm.py:497
self.device_index = bool(_ENGRAM_DEVICE_TRIM_SHARD and storage_format == "int8")
# engram_hbm.py:498
shard_rows = 1 if self.device_index else self.end - self.start
self.weight = nn.Parameter(torch.empty(shard_rows, width, dtype=torch.int8 ...))
```

★ 注意 `torch.empty` —— **未初始化**，不是零。

**(2) 分片**永不加载****

```python
# engram_hbm.py:650-654
def load_checkpoint(self, model_path, key, chunk_rows=65536):
    if getattr(self, "device_index", False):
        # the loader would fill a per-rank torch shard that the lookup never reads.
        return
```

**(3) 探测**只在 `auto` 下跑** —— `=1` 是"免探测强开"**

```python
# model.py:801-814
if _ENGRAM_DEVICE_INDEX_MODE == "auto":
    ok, detail = probe_host_mapping_capability()
    if not ok:
        print("[DEVICE-INDEX] 本机不支持 host 内存设备直索，自动回退到 host 路径" ...)
        return                      # ← 只有 auto 才会在这里早退
elif _ENGRAM_DEVICE_INDEX_MODE not in ("1", "true", "on", "yes"):
    return
# mode == "1" ⇒ 两个分支都不早退 ⇒ 直接建表，探测被跳过
```

**(4) 还有一条早退：`engram_history is None` ⇒ setup 根本不调用**

```python
# model.py:766-772
if ascend_config.enable_engram and (
        vllm_config.load_config.load_format != "dummy" or _ENGRAM_WITH_DUMMY):
    self.engram_history = PagedNgramHistory(config, tokenizer)
# model.py:788
if _ENGRAM_DEVICE_INDEX and ascend_config.enable_engram and self.engram_history is not None:
    self._engram_device_setup()      # ← load_format=dummy 且未设 _ENGRAM_WITH_DUMMY 时**不会调用**
```

**(5) 于是回退到 host 路径**

```python
# model.py:986-990
if (_ENGRAM_DEVICE_INDEX and self._engram_dev_ready      # ← 仍是 False
        and _BP_STATE.refresh().mode != "nohost"):
    ...
return self._prepare_engram_host(input_ids, positions)   # ← 读那个 1 行、未加载的表
```

⇒ **组合**：`V41_ENGRAM_DEVICE_INDEX=1` **且**（`load_format=dummy` 或 bneck `nohost`）
⇒ 分片被裁、从未加载、却被 host 路径读。**唯一的响声是那句"自动回退到 host 路径"（且只在 auto 下打）**。

★ **诚实边界**：`load_format=dummy` 与 `nohost` **都是诊断/冒烟口径，不是生产口径**
（dummy 权重本身就是垃圾）。所以 A1 是**真隐患但生产可达性低** —— 它的价值在于
**它同时也是"这条设计意图被绕过"的证明**：作者的原话是"全有或全无"，
而 `=1` + 早退把"全无"变成了"静默错"。

### A.2 两模块对**同一个 env** 的解析（机械实测，9 个输入 8 个不一致）

```python
# engram_hbm.py:272
os.environ.get("V41_ENGRAM_DEVICE_INDEX", "0") == "1"
# model.py:103-106
m = (os.environ.get("V41_ENGRAM_DEVICE_INDEX", "auto") or "auto").strip().lower()
m not in ("0", "false", "off", "no", "")
```

实测（在容器里跑的两段表达式，非推演）：

| env | hbm 裁分片 | model 走设备路径 | |
|---|---|---|---|
| （未设） | False | **True** | ⚠ MISMATCH |
| `auto` | False | **True** | ⚠ MISMATCH ← **发布默认** |
| `AUTO` | False | True | ⚠ MISMATCH |
| `1` | **True** | True | 一致 |
| `true` / `True` / `on` / `yes` | False | True | ⚠ MISMATCH ×4 |

★ **没有任何输入产生"裁分片 + 不走设备路径"的解析组合** ⇒ 解析不匹配**本身**不直接致错；
真正的 A1 来自 §A.1(3)(4) 的**早退**。
★ 但解析不匹配解释了 **A2**：发布默认 `auto` 下 hbm 认为是关的 ⇒ **照旧分配并加载 25.75 GB/rank/层**，
而 model 认为是开的 ⇒ 走设备路径。**两份 DRAM 账同时在付。**

---

## B. 响亮失败（会崩/报错，但**不静默**）

| # | 现象 | 依据 |
|---|---|---|
| B1 | `aclrtHostRegister` 拒绝只读 VMA ⇒ **`ret=507899`** | 【实测·读码】`engram_device_index.py:418` 的异常文案自己列了这条 |
| B2 | 无 device context ⇒ **`ret=107002`**（★ 源码注释：*"which reads like a permission problem"* —— **会被误读成权限问题**） | 【实测·读码】`engram_device_index.py:190-193` |
| B3 | 池注册失败 ⇒ `ret=207001` + **回落 pinned**（★ **不崩**，只慢） | 【实测】`068` §E3 |
| B4 | Engram `repeat_interleave` 拿不到 D2H 暂存 ⇒ `LocalScalarDenseNpu.cpp:23` + `copy_stream` ⇒ **起服失败** | 【实测】`068` §E5 |
| B5 | `rtsHostRegister ... reason=driver error:out of memory` | 【实测】`068` §E5 |
| B6 | `nohost` 模式以外的 `bneck` 取值在设备路径下**没有对应物** | 【实测·读码】`model.py:990-1000`（见 C2） |

★ **B2 的副作用是判据失效**：`107002` 的真实含义是"没 context"，
但文案像权限 ⇒ 排查方向会被带偏（`052` 已经踩过一次：探针缺 `aclrtSetDevice` ⇒ 一律 107002）。

---

## C. 性能降级 / 判据失效

| # | 隐患 | 依据 |
|---|---|---|
| **C1** | ★ `auto` 下 **25.75 GB/rank/层 的 DRAM 照付**（`068` A2） | 【实测·机械】§A.2 表 |
| **C2** | ★★ **`bneck` 消融臂在设备路径下静默失效** —— `model.py:990-1000` 明说其它取值"**在设备路径上没有对应物**…该臂失效"，但**只打一次 print**（还只在 rank0）。没看日志的人会以为自己在做对照实验 | 【实测·读码】 |
| **C3** | 池有 **31 张回落 pinned** ⇒ 那部分 H2D 从 16–21 掉到 **5.3 GB/s** | 【实测】`068` §E3 + `065` §3c |
| **C4** | ★ docstring 指的端到端工具 **路径不存在**：`engram_device_index.py:248` 写 `tools/probe_a2_hostmap.py`，实仓只有 `bench/probe_engram_hostmap.py` ⇒ **想验证的人按提示找不到工具** | 【实测·文件】`find` 结果 |
| **C5** | 那三个端到端探测日志**自己**带着：`header check : MISMATCH between the probe constants and acl_rt.h -- do not trust the flag column above` | 【实测·文件】`37-hostmap-conc-c0.log` |

---

## D. 一句话：上线前必须先排除哪一条？

**先排除 A3（探测缺口）** —— 因为它是**生产真实路径**、且**在 A2 上从未验证过**：

```
在 A2 上跑端到端可读性探针（独立进程、index_select + 逐字节对账），
看"设备算子读 host 映射"是 (a) 读对、(b) 崩、还是 (c) 静默读错。
```

依据：`logs/065` §3c 只证明了 **注册被接受 + H2D/D2H 逐字节对账**（那是 `torch.copy_`，
**不是"设备算子直接索引 host 内存"**）；`probe_host_mapping_capability` 的 docstring 自己把两者分开，
而**能证 (c) 的那个探针在 A2 上一次都没跑过**（`37-hostmap-*.log` 全是 A3 槽位）。

★ 而且 **A2 的发布默认是 `auto`** ⇒ 探测通过就**自动走设备路径**
（`logs/065` 已证 A2 能注册 ⇒ probe 会返回 ok）⇒ **A2 会在没有端到端证据的情况下打开它。**

**A1 排第二**：它是真隐患，但只在 `=1` +（`dummy`|`nohost`）下可达，属于"顺手堵上"级别
（§E 的两条修法都是**响亮化**，不改变生产行为）。

---

## E. Q3 答：两条修法本身会不会引入算错？（**不会，已实测**）

### E.1 `output_size=` 给错 ⇒ **响亮报错，不是静默截断**

在容器里实测（CPU torch `2.10.0+cpu`，ATen 层行为共用）：

```
sum(counts) = 10
  output_size= 10 -> OK, out_len= 10
  output_size=  8 -> RuntimeError: allocated size does not match required size
  output_size=  4 -> RuntimeError: allocated size does not match required size
  output_size= 12 -> RuntimeError: allocated size does not match required size
  output_size=  0 -> RuntimeError: allocated size does not match required size
```

⇒ ★ **给错就抛**（ATen 的 `allocated size does not match required size`），**不会静默截断/留空**。
而且这里的 `output_size` 是**免费的**（`input_ids` 的第 0 维就是总行数）⇒ **无风险**。

### E.2 `searchsorted` 版与 `repeat_interleave` 版**逐元素等价**（6/6 案例）

```
b=[0, 5, 8]        size=8   eq=True
b=[0, 1, 2, 3]     size=3   eq=True
b=[0, 6, 6, 9]     size=9   eq=True      ← ★ 等值边界（空请求）
b=[0, 10]          size=10  eq=True      ← 单请求
b=[0, 0, 4]        size=4   eq=True      ← ★ 首请求为空
b=[0, 1, 1, 1, 5]  size=5   eq=True      ← ★ 连续空请求
ALL EQUAL = True
```

⇒ ★ 含**空请求 / 等值边界 / 单请求**三种边界全等。**两条修法都不引入新的算错风险。**

★ 唯一残留：`output_size` 依赖调用方**正确传**总行数。若调用方从别处传（不是 `input_ids.numel()`），
就会退化成"抛异常"而不是"正确" ⇒ **建议草案里直接写成 `input_ids.numel()`，不新增参数面**
（见 `agents/Engram_pool_src/CANDIDATES.md` 候选 1 的注释）。

---

## F. 【未确认】清单（本卷新增）

1. **A2 上"设备算子直读 host 映射"的端到端结果** —— 全卷最该补的一条；
2. 设备路径（`device_engram_lookup`）对**越界 id** 有无等价于 host 路径的保护（`_metadata` 的 `invalid` 位）；
3. `_scatter_rows` 走 `except` 分支时，`index_put_` 与 `npu_scatter_nd_update_` 的结果是否**逐元素相同**
   （若不同，回落就是静默算错）；
4. 那三个 `37-hostmap` 日志里 `header check : MISMATCH` 的影响面（探针自己的常量表 vs `acl_rt.h`）。
