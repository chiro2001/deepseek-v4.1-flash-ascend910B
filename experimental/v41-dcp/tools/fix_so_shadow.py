#!/usr/bin/env python3
"""修复 AscendC 融合算子被 .so/.py 同名遮蔽的问题。

## 症状
`V41_DCP_MERGE_KERNEL=1` 时 `available()` 恒为 False（静默退回 Python 路径），
实测报错：
```
ImportError: dynamic module does not define module export function (PyInit_v41_merge_kernel)
```

## 根因
Python 的导入优先级是 **扩展模块（.so）> 源文件（.py）**
（`importlib._bootstrap_external._get_supported_file_loaders()` 的顺序：
 ExtensionFileLoader → SourceFileLoader → SourcelessFileLoader）。

而 `v41_merge_kernel.py`（wrapper）与 `v41_merge_kernel.so`（ctypes 目标）
**同名且同目录** ⇒ `from vllm_ascend.attention import v41_merge_kernel`
实际加载的是 .so 并被当作 Python 扩展模块解析 ⇒ ImportError。

而调用方的 `_v41_merge_kernel_on()` 把这个异常**吞掉**了：
```python
try:
    from vllm_ascend.attention import v41_merge_kernel as _mk
    _MERGE_KD_CACHE = bool(_mk.available())
except Exception:
    _MERGE_KD_CACHE = False       # ← 静默退回
```
⇒ **融合算子永远无法启用**，且不报错。

## 修法
把 .so 改名成不与 .py 冲突的名字（`libv41merge_ops.so`），并同步 `_LIB_NAME`。

## 影响
这是"静默退回"的最后一环 —— 之前修过 serve_a2.sh 只挂 `*.py` 不挂 `*.so`
（已修），但仍因同名遮蔽而不可用。
"""
import hashlib
import pathlib
import shutil
import sys

D = pathlib.Path.home() / "dcpw/vllm_ascend/attention"
PY = D / "v41_merge_kernel.py"
OLD_SO = D / "v41_merge_kernel.so"
NEW_SO = D / "libv41merge_ops.so"
NEW_NAME = "libv41merge_ops.so"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    if NEW_SO.exists() and not OLD_SO.exists():
        print("[ok] 已改名为 %s" % NEW_NAME)
    else:
        if not OLD_SO.exists():
            raise SystemExit("[FAIL] 缺 %s" % OLD_SO)
        shutil.move(str(OLD_SO), str(NEW_SO))
        print("[mv] %s -> %s" % (OLD_SO.name, NEW_SO.name))

    print("[before] py md5=%s" % md5(PY))
    s = PY.read_text()
    if '_LIB_NAME = "libv41merge_ops.so"' in s:
        print("[ok] _LIB_NAME 已更新")
    else:
        n = s.count('_LIB_NAME = "v41_merge_kernel.so"')
        if n != 1:
            raise SystemExit("[FAIL] _LIB_NAME 锚点 %d 次" % n)
        s = s.replace('_LIB_NAME = "v41_merge_kernel.so"',
                      '_LIB_NAME = "libv41merge_ops.so"', 1)
        # 顺带把注释里的旧名提一下
        s = s.replace('"""`.so` 是否存在且能加载。"""',
                      '"""`.so` 是否存在且能加载。\n\n'
                      '★ 文件名故意与 .py 不同（`libv41merge_ops.so`）：\n'
                      '  Python 导入优先级是 扩展模块(.so) > 源文件(.py)，\n'
                      '  同名会让 `import v41_merge_kernel` 命中 .so 并报\n'
                      '  ImportError: dynamic module does not define module export function\n'
                      '  ⇒ 而调用方吞掉异常 ⇒ 融合算子静默失效（实测踩过）。\n'
                      '"""', 1)
        PY.write_text(s)
        print("[after ] py md5=%s" % md5(PY))
    import ast
    ast.parse(PY.read_text())
    print("[ok] 语法通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
