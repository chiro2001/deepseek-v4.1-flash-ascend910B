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
