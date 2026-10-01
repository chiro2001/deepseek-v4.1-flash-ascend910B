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

---

## 9. ★★★ 第二个根因：kernel 走裸指针，收到**非连续输入** ⇒ 输出乱码

### 9.1 症状（TP8+DCP8 真权重）

修复 §7 的两个缺陷后，kernel 终于能启动，但**精度全崩**：

| 用例 | 输出 | 判定 |
|---|---|---|
| `17×23 → 391` | `'6. false;  fight; 0;'` | ❌ |
| 长针 T=904 → `Q7` | `'([ potentially (可能需要关于'` | ❌ |
| T=2000 | `'Upper ( .mod., (or (or'` | ❌ |
| T=8000 | `"derive 'HMTconcerning_"` | ❌ |
| T=16000 | `'iac megaf`&#&)&amp&#&#'` | ❌ |

**5/5 全错**。而 tiny（DCP=2）只错 1/6 —— 因为 tiny 是 dummy 权重、
logits 量级仅 ~1e-4，看不出系统性错误。

### 9.2 根因

`v41_merge_kernel.py` 把张量**裸指针**交给 kernel，**不检查 strides**：

```python
rc = lib.v41_merge_post_launch(
    _GRID, ctypes.c_void_p(s),
    ctypes.c_void_p(pack.data_ptr()),
    ctypes.c_void_p(ori_out.data_ptr()),      # ← 非连续！
    ctypes.c_void_p(out.data_ptr()),
    ctypes.c_void_p(tt.data_ptr()),
)
```

而调用方传进来的是**切片**：

```python
_oi = ori_out[:, head_slice[0]:head_slice[1], :]      # [T,8,512]
# stride = (64*512, 512, 1) 而非连续的 (8*512, 512, 1)
```

kernel 按**连续布局** `[T, Hout, D]` 读 ⇒ 行 stride 用 `Hout*D`（实际 `H*D`）
⇒ 从第 1 行起全部错位。

**同一文件的 `V41-SUBALPHA-ABORT` 注释精确写过这个失败模式**：
> 真实路径与微基准的差别：两个输入都是 **strided 切片**
> （`_pack[..., :D][:, h0:h1]` 与 `ori_out[:, h0:h1]`）…
> ⇒ **本平台上"把多个逐元素算子融合/改写"的写法一律不可信，无论离线微基准是否逐位一致。**

差别在于：注释里那三次是 **torch 逐元素算子**（至少还看 strides）；
这次是 **ctypes 直调 kernel**，错得更彻底。

### 9.3 修法

在 wrapper 里对**所有**交给 kernel 的输入做连续性防御：

```python
def _as_contig(t, tag):
    """kernel 只认连续布局（走 data_ptr 裸指针）⇒ 非连续输入必须先拷成连续。"""
    if t is None:
        return t
    return t if t.is_contiguous() else t.contiguous()
```

`merge_pre`（output / lse / ori_lse / pack）与
`merge_post`（pack / ori_out / out）**两处都加**。
连续时 `.contiguous()` 是 no-op ⇒ 零开销。

### 9.4 tiny 验证（修复前后对比，同一 prompt 集）

| 臂 | ms/step | A | 与 MK=0 逐字对比 |
|---|---:|---:|---|
| MK=0（基线） | 36.24 | 2.00 | — |
| MK=1（**仅修 §7 两缺陷**） | 34.12 | 2.00 | **5/6**（`列出三种颜色。` DIFF） |
| MK=1（**+ 连续性修复**） | **34.05** | 2.00 | **6/6 PASS** ✅ |

* 修复前那条 DIFF 是**确定性**的（MK=0 下同 prompt 连打 5 次完全一致）
  ⇒ 不是 flaky，是真实的数值差异。
* 加连续性修复后 **6/6 逐字一致**。

⇒ **连续性假设被 tiny 验证成立。** 真正的判据仍是 TP8 真权重（见 §10）。

## 10. 三个修复的汇总

| # | 缺陷 | 症状 | 修法 |
|---|---|---|---|
| 1 | `.so`/`.py` 同名遮蔽 | `available()` 恒 False（静默） | `.so` → `libv41merge_ops.so` |
| 2 | `_merge_kd` 下 `weights=None` 撞老诊断守卫 | **起服失败** | 4 处诊断加 `weights is not None` |
| 3 | kernel 收到非连续输入 | **真权重下 5/5 乱码** | wrapper 加 `_as_contig()` 防御 |

**三个都必须修**，缺任一个这条路径都不可用（且前两个是"静默/崩溃"，
第三个才是"算错"—— 最危险的那类）。
