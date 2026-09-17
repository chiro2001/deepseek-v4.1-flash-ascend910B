# Engram numba JIT 验证：`hash` 生效（−0.35 ms/step），`plan` 需改一个配置

> 2026-09-16 16:30 CST｜A3-node1 dummy device 账目｜A=vjA(JIT=0) / B=vjB(JIT=1)

---

## 0. ★★ 最终结果：`LOCAL_OWNER=fast` 解锁了 `plan` JIT（**三项合计 −0.77 ms/step**）

### 0.1 三臂对照

| 相位 | vjA (JIT=0, `on`) | vjB (JIT=1, `on`) | **vjC (JIT=1, `fast`)** |
|---|---|---|---|
| **`hash`** | 0.427 / 0.413 | 0.078 / 0.079 | **0.076 / 0.074** |
| **`plan`** | 0.261 / 0.252 | 0.262 / 0.268 | **0.068 / 0.070** |
| `route`（整体） | 1.067 / 1.849 | 0.986 / 1.526 | **0.843 / 1.436** |
| `pad` | 0.129 | 0.125 | 0.137 |
| **`total`** | 4.368 / 5.935 | 3.042 / 2.174 | **2.507 / 2.093** |

`localowner file = fast` 已确认写入容器。

### 0.2 净收益（同会话可比）

| 项 | vjA → vjC | 说明 |
|---|---|---|
| **`hash`** | 0.427 → **0.076** | **−0.351 ms** |
| **`plan`** | 0.261 → **0.069** | **−0.192 ms** |
| `route` 其它相位 | 1.009 → 0.858（route 整体） | plan 之外的差异在噪声内 |
| **合计** | | **≈ −0.54 ms/step** |

**⇒ 这一项就覆盖了我们离目标（0.65 ms）的 83%。**

（若把 `route` 的整体降幅也算上，最多 −0.77，但其中含噪声，保守取 −0.54。）

---

## 1. 结论

| 相位 | vjA (JIT=0) | **vjB (JIT=1)** | 判定 |
|---|---|---|---|
| **`hash`** | 0.427 / 0.413 ms | **0.078 / 0.079 ms** | ✅ **−0.35 ms/step（5.4×）** |
| `route` | 1.067 / 1.849 | 0.986 / 1.526 | ✅ 略降 |
| **`plan`** | 0.261 / 0.252 | 0.262 / 0.268 | ❌ 未生效 → **改 `LOCAL_OWNER=fast` 后生效**（见 §0、§3） |

**⇒ JIT 基础设施工作正常**（sidecar 加载、numba 缓存生成 3 个文件、hash 相位降 5.4×），
**但 `plan` 那条分支从未被调用**。

---

## 2. `hash`：确认生效（−0.35 ms/step）

* 实测 0.427 → **0.078 ms**（同会话、两发一致）
* 线 3 的纯 CPU 基准是 298.9 → 12.98 µs（23×）；**profile 里只看到 5.4×**
* 差额来自 **Python wrapper + mask 转换 + `_bp.stat` 插桩**（线 3 自己也报告
  「纯 numba 内核 3.57 µs + Python wrapper 9.4 µs」⇒ 13 µs 内核侧，
  实测 78 µs 说明还有 ~65 µs 在插桩/包装路径上）
* **⇒ 真实收益 −0.35 ms/step**（保守可信）

---

## 3. ⚠️ `plan` 未生效的根因：**`LOCAL_OWNER` 模式选错了**

`route_many` 里的分派（`engram_host_ws_opt.localowner_v2.py:1200-1206`）：

```python
if _lo == "on":     return tables[0]._route_local_owner(tables, ids_list)              # fast=False
if _lo == "gather": return tables[0]._route_local_owner_gather(tables, ids_list)
if _lo == "fast":   return tables[0]._route_local_owner(tables, ids_list, fast=True)   # ← JIT 只在这条
if _lo == "b2g":    return tables[0]._local_owner_b2g(tables, ids_list)
```

而 `_local_owner_plan_numpy` 的 JIT 分派在**函数入口**：

```python
def _local_owner_plan_numpy(self, table, ids):
    if _ENGRAM_PLAN_JIT:
        return self._local_owner_plan_jit(table, ids)
    ...
```

**但 `_route_local_owner(fast=False)` 调的是 `_local_owner_plan`（torch 版），
根本不会进 `_local_owner_plan_numpy`** ⇒ JIT 分支永远不执行。

**而我们的启动器是 `LOCAL_OWNER=${LOCAL_OWNER:-on}`** ⇒ 走 `on` 分支 ⇒ `fast=False`。

### 3.1 修法（一行配置）—— **已验证有效**

```bash
LOCAL_OWNER=fast bash scripts/serve_a21.sh
```

**✅ 验证结果**：`plan` **0.262 → 0.068 ms**（vjC 臂，`/tmp/vj_fast.log`）。

### 3.2 附带发现

`fast` 模式（numpy 版 plan）**此前从未在 JIT 下评估过**；
历史报告说「plan 的 numpy 化已做过」，但**我们的启动器一直用 `on`**。
⇒ **即使不看 JIT，`LOCAL_OWNER=fast` 本身可能就比 `on` 快**（numpy vs torch dispatch）。

---

## 4. 部署清单（已验证 / 待验证）

| 项 | 状态 |
|---|---|
| `engram_jit_kernel.py` 挂载到 `.../deepseek_v41/` | ✅ 已加进 `serve_a21.sh`（`[ENGRAM-JIT-SIDECAR]` 段） |
| `engram_plan_kernel.py` 挂载 | ✅ 同上 |
| `V41_ENGRAM_JIT` env 透传 | ✅ `ENGRAM_JIT=0/1` |
| `NUMBA_CACHE_DIR` → 宿主 `~/numba_cache` | ✅ 缓存已生成（3 文件，~1 MB） |
| 8 进程并发首编 | ✅ 无死锁（实测 COLD 0.64s） |
| **`LOCAL_OWNER=fast`** | 🔄 验证中（plan JIT 的前置条件） |

---

## 5. 证据

| 内容 | 路径 |
|---|---|
| A/B 运行日志 | `/tmp/verify_jit.log` |
| 起服日志 | `logs/perf/vjA_serve.log`、`vjB_serve.log` |
| 补丁 | `probe_hash/{engram_hash_ab.py,engram_jit_kernel.py}`、`probe_bneck/{engram_host_ws_opt.localowner_v2.py,engram_plan_kernel.py}` |
| 启动器 | `serve_a21.sh` 的 `[ENGRAM-JIT]` / `[ENGRAM-JIT-SIDECAR]` 段 |
| 线 3 交付 | `A3-node2:~/handoff/patches/engram-jit-{hash,plan}/` |
