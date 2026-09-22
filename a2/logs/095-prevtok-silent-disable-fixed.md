# 095 — 修掉"精确修补**静默停用**"：`_engram_build_prev_tok` 的 `except: return None` 现在会**计数 + 打印**

> 2026-09-23 00:5x CST。执行：**主代理**（在等 `fit` 臂起服的间隙做）。
> 前置：`090`（覆盖率 0.5%，★ 该数已被 `097` 撤回重推）、`094`（预注册 + "无论如何都该修"那条）。
> 标记：**【实测】/【推断】**。

---

## 0. 一句话

`model.py` 的 `_engram_build_prev_tok()` 里原本是这样结束的：

```python
except Exception:   # noqa: BLE001  ★ 任何异常都退回"今天的行为"，绝不把异常带进 forward
    return None
```

⇒ 一旦 `build_prev_tok` 抛异常（**行越界 / 行宽不符 / numpy 不连续 / 任何类型的错**），
**整条"精确修补"就此静默停用**，而日志里**一个字都没有** ——
**从外面看，和"根本没开 `TRUE_TOKENS`"完全一样**。

这正是本仓反复出现的失败模式（`079 §3` 同族：**沉默 ≠ 没跑**），而且它专门坑"修复类"功能：
**你无法区分「修复生效了但没事可修」与「修复压根没跑」**。

---

## 1. 改法（已落 `patches/engram-true-tokens/model.patched.py`，md5 `67fad329bb135bff745326daba51fa3c`）

```python
except Exception as _exc:
    _PREVTOK_ERR["n"] += 1
    if _PREVTOK_ERR["n"] <= 3:            # 每 rank 只打前 3 次，不刷屏
        _fr = traceback.extract_tb(_exc.__traceback__)[-1]
        print("[ENGRAM-PREVTOK-ERR] #%d 发布/构造 prev_tok 失败 ⇒ **精确修补静默停用**"
              "（退化为 pad 兜底）：%s: %s @ %s:%s" % (...), flush=True)
    return None
```

★ **为什么不硬 raise**：A2 生产是"宁可降级也不许挂"的场景；
但它**绝不静默** —— 与 `engram_hash` 里 pageless 提示、`TRUE_TOKENS` 的累计打印**同一套处理原则**。

★ 新增模块级 `_PREVTOK_ERR = {"n": 0}`（在 `_ENGRAM_ROW_TOKENS` 旁边）。

### 1.1 本地验证

```
python3 -m py_compile  patches/model.patched.py     → ✓ 通过
bash a2/patches/engram-true-tokens/tests/run_all.sh → ✓ 全部通过（门 0 py_compile + 调用点契约 + 两个语义测试）
```

---

## 2. ★ 为什么**现在**不能更新 A3 的挂载源（重要）

`fit` 臂（`r8-4axis-fit`，**正在跑**）把 `shadow-pkg/patches/files/model.py` 以 **`:rw`** 挂进了容器。
**此刻覆盖它 = 改动一个正在运行的服务所挂的文件**（虽然 Python 已 import 过、大概率无害，
但这是"绝不在臂运行中改它的挂载源"这条纪律该覆盖的情形）。
⇒ **等 `fit` 跑完再 `cp`**：

```bash
# fit 跑完之后
cp -a <shadow>/patches/files/model.py <shadow>/patches/files/model.py.before-prevtok-err-$(date +%H%M%S)
cp -f a2/patches/engram-true-tokens/model.patched.py <shadow>/patches/files/model.py
md5sum <shadow>/patches/files/model.py      # 期望 67fad329bb135bff745326daba51fa3c
```
★ **注意**：`mode2` 臂起服前**必须**做这一步，否则它带的是"静默版"。
★ 并且 `logs/081` 的**合并件新鲜度门**会因此失效 —— 合并件的 `--prod` 变了，
⇒ **必须重新合并**（`check_merged_fresh.sh` 会当场报出来，这正是那道门的价值）。

---

## 3. 这条与 `094` 预注册的关系

`094 §5` 把"修法"分了三支，其中**第三支**是：
> 一行都不打 ⇒ 候选 C：`except` 吞了异常 ⇒ **把 `except` 改成计数 + 首次打印**（现在是静默 `return None`）

⇒ 本条**把那一支先修掉了**，于是 `mode2` 臂上会出现**两种可能的新日志**，都能立刻判读：

| 看到什么 | 含义 |
|---|---|
| `[ENGRAM-PREVTOK-ERR]` ≥ 1 | ★ 构造 `prev_tok` **失败过** ⇒ 精确修补**静默停用** ⇒ 必须先修这个（而不是去调 mode） |
| 只有 `[ENGRAM-PREVTOK-DIAG]` 的 reason 分布 | 构造成功、只是**真 token 拿不到** ⇒ 按 `094 §5` 的前两支修 |

★ **两者都没有** ⇒ 说明这一臂**根本没走到**那条路径（那就要回去查 `TRUE_TOKENS`/`ROW_IDS` 是否真进容器）。
