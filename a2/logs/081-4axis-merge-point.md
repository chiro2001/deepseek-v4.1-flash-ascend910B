# 081 — ★★★ 四轴同开的**唯一集成点**是 `model.py`，而旧的合并件会**静默丢掉 Engram 接线**

> 2026-09-22 21:4x CST。执行：**主代理**（本机 + A3-node1 只读核查；不占卡）。
> 目的：为「`ENGRAM=1` × 卸载 × int8 档 C × `DRAFT_GRAPH=1`」这一臂扫清**挂载/合并**层面的坑。
> 标记：**【实测】/【推断】**。

---

## 0. 一句话

四轴同开要同时改 **`models/deepseek_v41/model.py`** 的两拨东西
（**Engram 的 `set_engram_row_tokens` 接线** ＋ **KV8 档 C 的 3 处 hunk**），
而 8 卡 runner 在 `TIER != B` 时会用 `MERGEed_MODEL` **顶替**生产那份
⇒ 若合并件是拿**旧的生产版**做的，**Engram 那段会一声不响地消失**。
本轮发现现存的 `patched/model_merged.py`（09:54 生成）**正是这种旧的**，已按当前生产版重新合并并验证。

---

## 1. 【实测】为什么会撞：三份 `model.py` 指向同一个容器路径

| 来源 | 挂载方式 | 目标路径 |
|---|---|---|
| 影子包（生产版，`PATCH_MODE=mount`） | `-v $PKG/patches/files/model.py:…:rw` | `models/deepseek_v41/model.py` |
| int8 档 C（`pkg-ring`） | ★ **不直接挂**，改由 runner 传 `R8_MERGED_MODEL` | 同上 |
| runner 的合并件 | `-v $R8_MERGED_MODEL:…:ro`（`serve_a2.sh:1001-1003`） | 同上 |

★ **`merge_model.py` 的头注释早就记过这个坑**：直接挂两份会 **`Duplicate mount point`**
（`logs/048` 的第一次起服就是这么死的）。所以正解是**合并成一份**。

---

## 2. 【实测】现存的合并件缺 Engram 接线

```
$ grep -c set_engram_row_tokens <R8>/patched/model_merged.py
0                                   ← ★ 缺！
$ md5sum <R8>/patched/model_merged.py
6a1b78853103e57bacecb571694010a2    （生成时间 09:54）
```
而**当前**生产版（`shadow-pkg/patches/files/model.py`，md5 `33c05aaf2ff3378f386cd9734a2d62ab`）
里 `set_engram_row_tokens` 计数 = **1**（我 21:3x 加的 Engram row-tokens 钩子）。

⇒ **结论**：那份合并件是拿 **09:54 的生产版**做的 ⇒ 若直接用它起四轴同开的臂，
**Engram 的 row-tokens 接线会在 tier C 路径上整段消失**，而**没有任何报错** ——
`grep -c "ENGRAM-ROW-TOKENS"` 只会是 0，人很容易读成"没触发"（`080 §3.1` 刚记过这个"沉默≠没跑"的口径）。

---

## 3. ✅ 修法：用当前生产版重新合并（已做完，带自证）

```bash
# ① 从镜像抽原版
docker run --rm --entrypoint cat $IMG /vllm-workspace/.../models/deepseek_v41/model.py > model.img.py
#   md5 = e5d2490ef541e2f044ab639565c3fd7c

# ② 三方合并（runner 自带的合并器）
python3 $R8/patch/merge_model.py \
  --img  model.img.py \
  --kv8  $X/pkg-kv8pf/shadow/vllm_ascend/models/deepseek_v41/model.py   # md5 fd7ff753…
  --prod $PKG/patches/files/model.py                                    # md5 33c05aaf… ← 当前这份
  --out  $R8/patched/model_merged.py
```

**合并器自带的三道自证全过**（原文）：
```
[merge] 从 (镜像 → pkg-kv8pf) 现算出 3 个 hunk
[merge] ★ 自证①：diff 回放到镜像版 ⇒ 与 pkg-kv8pf 版逐字节相同 ✅
[merge] ★ 生产版 3 个锚点全部唯一命中 ✅
  OK   import 了 swa_plane_kwargs / long_kv_plane_kwargs
  OK   get_kv_cache_spec 用了 swa_plane_kwargs() / SWASpec 带 scale_dim / long-KV 用 long_kv_plane_kwargs()
[merge] AST OK：8102 个节点，1393 行
[merge] 生产版 → 产物的差异行数 = 9（期望 = 3 个 hunk 的 ± 行数之和）
```

**安装后核对**（MD5 + 三条 grep + 编译）：
| 项 | 值 |
|---|---|
| `patched/model_merged.py` | md5 **`5567746663c9ddb054fb41f567bc9f22`** |
| `set_engram_row_tokens` | **1** ✅（Engram 接线在） |
| `swa_plane_kwargs` / `long_kv_plane_kwargs` | 2 / 2 ✅（档 C/D 的几何在） |
| `py_compile` | **OK** ✅ |
| 旧的备份 | `patched/model_merged.py.pre4axis-214152`（md5 `6a1b7885…`） |

---

## 4. 【实测】顺带扫掉另外三处**疑似**重复挂载点（都**不是**冲突）

| 文件 | int8 包里有吗 | 四件套（`DRAFT_GRAPH=1`）要挂吗 | 结论 |
|---|---|---|---|
| `attention/dsa_v1.py` | pkg-ring / pkg-kv8pf **都有** | 挂（→ `attention/dsa_v1.py`） | ⚠️ 名字像，但 R8 int8 块**没挂**这个文件（只挂 `dsa_v41.py`）⇒ **不冲突** |
| `spec_decode/dspark_proposer.py` | 都有 | 挂 | R8 int8 块**没挂** ⇒ **不冲突** |
| `spec_decode/llm_base_proposer.py` | 都有 | 挂 | R8 int8 块**没挂** ⇒ **不冲突** |

★ R8 int8 块实际挂的**只有 5 个**（`serve_a2.sh:900-965` 实测展开）：
`core/deepseek_v41.py`、`core/kv_cache_interface.py`、`models/deepseek_v41/compressor.py`、
`ops/triton/compressor/compressor_triton.py`、`attention/dsa_v41.py`（+ 档 D 的 `attention/kv8_prefill_triton.py`）。
★ 注意 **`dsa_v1.py`（v1）≠ `dsa_v41.py`（v41）** —— 一字之差，我用 `find` 逐个确认过，别靠眼睛。

⇒ **四轴同开的唯一集成点 = `model.py`**，已经修好。

---

## 4b. ★★★ 同一格**当天第二次**踩到：合并件又（这次是反方向的）过期了

修完 §3 之后我紧接着修了另一个 bug（`engram_repair` 的调用点契约，见 `logs/082` 与
`patches/engram-true-tokens/README.md §0d`）：`model.py` 的调用点从
`return build_prev_tok(...)` 改成 `return build_prev_tok(...)[0]`。
⇒ **`--prod` 一变，刚合并好的那份立刻过期**：

```
$ grep -n "return build_prev_tok(" <R8>/patched/model_merged.py
358:        return build_prev_tok(ntok, pos_np, req_np, lookback, ids)     ← ★ 没有 [0]
$ md5sum <R8>/patched/model_merged.py
5567746663c9ddb054fb41f567bc9f22      （= 基于旧 prod 33c05aaf 的那份）
$ md5sum <影子包>/patches/files/model.py
0f9feba129a8f28c9707b40d4109e92b      （prod 已经换成带 [0] 的了）
```

⇒ ★★ **`--prod` 换版本 ⇒ 合并件过期 ⇒ 又变回"第一个请求抛 `TypeError`"**。
这正是 §0 那句话的镜像版本：**不只是"用旧的 prod 会丢功能"，而是"prod 一更新，合并件就不新鲜"**。

### 处置：① 重新合并（用当前 prod）② **加一道新鲜度门**

```
$ python3 merge_model.py --img model.img.py --kv8 pkg-kv8pf/…/model.py \
        --prod shadow-pkg/patches/files/model.py --out patched/model_merged.py
[merge] 输入 md5: img=e5d2490e kv8=fd7ff753 prod=0f9feba1
[merge] ★ 产物 … md5=5c990b04b7c3306920d837483b0dbd11 行数=1404
```
| 项 | 值 |
|---|---|
| 新合并件 md5 | ★ **`5c990b04b7c3306920d837483b0dbd11`** |
| 调用点 | `return build_prev_tok(...)[0]` ✅ |
| `set_engram_row_tokens` / `swa_plane_kwargs` / `long_kv_plane_kwargs` | 1 / 2 / 2 ✅ |
| `py_compile` | OK ✅ |
| 旧的（stale）备份 | `patched/model_merged.py.stale-214613`（md5 `5567746663…`） |

### 新门：`a2/scripts/check_merged_fresh.sh`（不占卡、不启容器）

**做法**：用**同一套输入**重新合并到临时文件，与已安装的那份逐字节比对；
因为 `merge_model.py` 自带自证（diff 回放镜像版必须逐字节等于 kv8 版 + 锚点唯一命中），
所以"重新合并"本身是可信操作，不是猜。

**双向实测**：
| 场景 | 结果 |
|---|---|
| 已安装 = 新鲜 | `✓ 新鲜：合并件 == 用当前输入重新合并的结果` ⇒ **rc=0** |
| 已安装 = 旧的 stale 件 | `⛔ 合并件已过期` + `差异行数 = 13` ⇒ **rc=1**，并打印处置命令 |

★ **一个自己踩的弱判据**：第一版标志位对比只比 `grep -c "return build_prev_tok("` ⇒
**带不带 `[0]` 都是 1 行** ⇒ 抓不到本次事故。已改成**按行内容判**（`case ... in *")[0]"*`）。
⇒ 教训：**"计数相等"经常不等于"内容相同"**（与 `043` 的"单向判据失效"同族）。

---

## 5. 给下一臂的检查清单（进 `logs/080` 那道"先看 dmesg"门之后的第二步）

```
① sudo -n dmesg -T | grep -iE "killed process" | tail -3      # 有没有新 OOM（先看 dmesg）
② grep -c "merged model\|合并版" <arm>.serve_a2.log            # ★ model.py 必须是"合并版"，不是生产那份
③ grep -m1 档位 <serve.log>                                   # 必须是 C
④ 起服后：grep -ac 'KeyError:' / 'EE1016' / '507057' / 'EH0012' 全 0
```
