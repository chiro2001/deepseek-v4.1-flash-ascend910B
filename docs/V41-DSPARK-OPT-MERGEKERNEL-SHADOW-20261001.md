# ★ AscendC 融合算子被 `.so`/`.py` 同名遮蔽 —— 从未真正启用过（2026-10-01）

## 0. 症状

`V41_DCP_MERGE_KERNEL=1` 起服正常、精度正常（A=3.15），但**性能与关闭时无差异**。
深挖发现 kernel **根本没跑**。

```
$ docker exec dsv41-dspark8 python3 -c "
    from vllm_ascend.attention import v41_merge_kernel as mk; print(mk.available())"
ImportError: dynamic module does not define module export function (PyInit_v41_merge_kernel)
```

## 1. 根因：Python 导入优先级

`importlib._bootstrap_external._get_supported_file_loaders()` 的顺序是：

1. `ExtensionFileLoader`（`.so`）
2. `SourceFileLoader`（`.py`）
3. `SourcelessFileLoader`（`.pyc`）

而 `vllm_ascend/attention/` 下：

| 文件 | 角色 |
|---|---|
| `v41_merge_kernel.py` | ctypes wrapper（`available()` / `merge_pre` / `merge_post`） |
| `v41_merge_kernel.so` | **ctypes 加载的目标库**（AscendC 编译产物） |

两者**同名同目录** ⇒ `from vllm_ascend.attention import v41_merge_kernel`
实际加载的是 **.so**，并被 Python 当作扩展模块解析
⇒ `ImportError: dynamic module does not define module export function (PyInit_...)`。

## 2. 而调用方把这个异常**吞掉了**

`dsa_v41.py::_v41_merge_kernel_on()`：

```python
try:
    from vllm_ascend.attention import v41_merge_kernel as _mk
    _MERGE_KD_CACHE = bool(_mk.available())
except Exception:          # noqa: BLE001
    _MERGE_KD_CACHE = False        # ← 静默退回 Python 路径
return _MERGE_KD_CACHE
```

⇒ **融合算子永远无法启用，且不报错、不崩溃、不告警。**

## 3. 这是「静默退回」链条的最后一环

| 环节 | 状态 |
|---|---|
| `serve_a2.sh` 只挂 `*.py`、`.so` 进不了容器 | ✅ 已修（2026-10-01，改为同时挂 `.so`） |
| **`.so` 与 `.py` 同名 ⇒ 导入命中 .so ⇒ ImportError ⇒ 静默 False** | ❌ **本文发现的最后一环** |

前两次修复都以为"挂上就能用"，但真正的判据是 `available() == True`，
而不是"文件在容器里"。

## 4. 修法

把 `.so` 改名为不与 `.py` 冲突的名称，并同步 wrapper 里的 `_LIB_NAME`：

| 文件 | 变更 |
|---|---|
| `v41_merge_kernel.so` | → **`libv41merge_ops.so`** |
| `v41_merge_kernel.py` | `_LIB_NAME = "v41_merge_kernel.so"` → `"libv41merge_ops.so"` |

**判据**（必须实测）：

```bash
docker exec <ctr> python3 -c "
from vllm_ascend.attention import v41_merge_kernel as mk
print('available() =', mk.available())"     # 必须 True
```

## 5. 影响与修复后的验证

| 项 | 值 |
|---|---|
| 修复前 `available()` | **False**（静默） |
| 修复后 `available()` | 见下节实测 |
| 单卡 kernel 实测（先前） | pre 7.08 µs + post 6.61 µs = 13.68 µs/层 ⇒ ×38 = **0.520 ms/step** |
| 加上消掉的图节点收益（邻居预测） | 潜在 **5.5–7.5 ms/step**（需 before/after gap 判据核实） |

## 6. 教训

**"配置项已设置" ≠ "功能已生效"。**
本项目已经因为这类"静默退回"踩坑三次：

1. `serve_a2.sh` 只挂 `.py`（`.so` 不在容器里）
2. 文件驱动开关在 ACL graph 捕获后失效
3. **本文：`.so`/`.py` 同名遮蔽**

⇒ 任何"开关 + 可选路径"的设计，都必须有一个**运行时判据**（如 `available()`）
把"真的生效了"暴露出来，而不是靠"文件存在"推断。

---

## 7. 修复后的实测（tiny，TP2+DCP2）

### 7.1 修复共两处

| # | 缺陷 | 修法 | md5 变化 |
|---|---|---|---|
| **1** | `.so` 与 `.py` 同名 ⇒ 导入命中 `.so` ⇒ ImportError ⇒ 被吞 ⇒ 静默 False | `.so` → `libv41merge_ops.so` + 同步 `_LIB_NAME` | `_LIB_NAME` 改 |
| **2** | `_merge_kd` 下 `weights=None`，但多处诊断仍用 `weights.*`，而守卫只有 `not _is_capturing()` | 4 处诊断加 `weights is not None` | `f2958736` → `c21d4afa` |

### 7.2 缺陷 2 的精确证据（tiny 与 TP8 同源复现）

```
File ".../dsa_v41.py", line 3728, in _native_attention
File ".../dsa_v41.py", line 609, in _v41_dcp_merge_attention
AttributeError: 'NoneType' object has no attribute 'sum'
RuntimeError: NPUModelRunner failed, error is 'NoneType' object has no attribute 'sum'
```

`weights` 在 `_merge_kd` 下被置 `None`（改由 kernel 内部算）：
```python
weights = (None if _merge_kd else torch.nan_to_num(torch.exp(_delta.clamp(max=60.0))))
```
而诊断守卫是 `not _is_capturing()` —— 崩溃发生在
`_warmup_and_capture → _dummy_run` 的**预热阶段**（此时**还没进 capture**）
⇒ 守卫失效 ⇒ 诊断跑 ⇒ 崩。

**⇒ `not _is_capturing()` 不能作为 `_merge_kd` 的保护。**

### 7.3 修复效果（`available()` 判据 + 性能）

```
$ docker exec dsv41-tinyspark bash -lc "V41_DCP_MERGE_KERNEL=1 python3 -c \
    'from vllm_ascend.attention import v41_merge_kernel as mk; print(mk.available())'"
available() = True                      ← 首次为 True（此前恒 False）
lib = .../vllm_ascend/attention/libv41merge_ops.so
```

| 臂（tiny，5 轮中位） | ms/step | A | vs 基线 |
|---|---:|---:|---:|
| `V41_DCP_MERGE_KERNEL=0`（基线） | **36.24** | 2.00 | — |
| 修复前 `=1` | **起服失败** | — | — |
| **修复后 `=1`** | **34.12** | 2.00 | **−5.9%（−2.12 ms/step）** |

* 5 轮：35.05 / 34.12 / 34.08 / 33.95 / 34.24（离散仅 ±1.6%）
* `A = 2.00` 与基线**完全一致** ⇒ 正确性未变

**⇒ AscendC 融合算子首次真正生效，且显著。**

### 7.4 为什么 tiny（DCP=2）也会有 5.9%

融合算子砍的是 **merge 的逐元素后处理**（每层 ~15 个小算子 ⇒ 2 个 kernel），
这部分与 DCP 度无关 —— **每层都要做**。所以 DCP=2 上就能看到收益。

⇒ **DCP=8 上预期收益更大**（merge 的通信部分也更贵）。

## 8. 教训（更新）

**"开关已设置" ≠ "功能已生效"** —— 本项目已因这类静默退回踩坑**三次**：

1. `serve_a2.sh` 只挂 `*.py` ⇒ `.so` 不在容器里
2. 文件驱动开关在 ACL graph 捕获后失效
3. **`.so`/`.py` 同名遮蔽** ⇒ ImportError 被 `except: False` 吞掉

而且这次修复暴露出**第 4 类**：
4. **"新路径下的 None 值撞上旧诊断的守卫"** —— 新路径（`_merge_kd`）引入
   `weights=None`，而老诊断的守卫（`not _is_capturing()`）对它无效。

⇒ 任何"开关 + 可选路径"的设计都必须有**运行时判据**（如 `available()`），
且新路径引入的 None/sentinel 值必须与所有既有消费者的守卫对齐。
