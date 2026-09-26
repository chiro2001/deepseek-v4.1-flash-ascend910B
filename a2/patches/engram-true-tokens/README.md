# 补丁与挂载说明（Engram × 卸载的精确修复）

> 设计见 `a2/docs/ENGRAM-OFFLOAD-EXACT-FIX.md`。**全部默认关**，门控关闭时行为与今天逐字相同。
> 本目录**只读交付物**：不改 `patches/files/`、不改影子包、不改任何生产脚本。

## 0. 文件清单（含 md5，便于复制到发布包时校验）

| 文件 | 作用 | 挂到容器里的位置 |
|---|---|---|
| `engram_repair.py` | **出货模块**：`plan_repair_slots` / `apply_repairs` / `build_prev_tok` | `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_repair.py:ro` |
| `engram_hash.patched.py` | 打完补丁的 `engram_hash.py`（完整文件） | `…/models/deepseek_v41/engram_hash.py:rw`（**替换生产那份挂载**） |
| `engram_hash.true_tokens.diff` | 同上，unified diff（+38 行） | — |
| `model.patched.py` / `model.host_rows.diff` | `model.py`：交接点 + 行序校验 + 传参（+60 行） | `…/models/deepseek_v41/model.py:rw` |
| `model_runner_v1.patched.py` / `model_runner_v1.publish.diff` | runner：发布 host token 表（+30 行；基线 md5 `9d84d9b073aeece3fd3bbc143ba20567`）⇒ patched md5 ★ **`a94887de05bb63370a6604260b358101`** | `/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro` |

★ `engram_repair.py` 既是**出货件**又是**单测的被测对象**（`tests/_loader.py` 直接 import 本文件），
避免"测的是一份、跑的是另一份"。

## 0b. ★ 基线对齐（2026-09-22 21:2x 主代理重做，**必读**）

| 项 | 值 |
|---|---|
| `engram_hash.py` **当前基线** | md5 **`dc63b40b1ec739ddc467edceebe85478`** |
| `engram_hash.patched.py` | md5 `d18d53357fd925abdb3cf9ca0c84f0bd`（已**按当前基线重新生成**） |
| `engram_hash.true_tokens.diff` | 已重新生成，头是标准的 `a/` `b/` ⇒ **`patch -p1` 干净应用**（实测 `--dry-run` 通过） |

**为什么重做**：21:3x 发现 075 的提示文案会**污染判据**（文本里含裸 `KeyError` 子串，
而判据是 `grep -c KeyError`）⇒ 改了文案 ⇒ 基线 md5 从 `240c5a04…` 变成 `dc63b40b…`。
原 `.diff` 因此带 8 行上下文偏移（仍能应用），但 `.patched.py` 是**整文件**，
沿用旧的会把旧文案带回容器 ⇒ 这里统一按新基线重生成。
★ 应用补丁后请**再跑一次** `bash tests/run_all.sh`（已重跑，全绿）。

## 0d. ★★★ 2026-09-22 22:1x **必修**：`model.patched.py` 的**调用点契约** bug（第一版会把元组当 `prev_tok` 传下去）

| 项 | 内容 |
|---|---|
| 现象 | A3 实测（臂 `p3b-true1-rowids1`，`True_Tokens=1`）：**起服成功、第一个请求就崩** |
| 报错 | `TypeError: tuple indices must be integers or slices, not tuple` @ `engram_repair.py:160`（`tok = int(prev_tok[row, sh])`），栈经 `engram_hash.py:509` → `apply_repairs`（`serve.log:2143`） |
| 根因 | `engram_repair.build_prev_tok()` 返回 **`(prev_tok, stats)` 二元组**，而 `model.patched.py` 的调用点写成 `return build_prev_tok(...)` ⇒ **整个元组**被当 `prev_tok` 传进 `apply_repairs` |
| ★ 为什么单测抓不到 | `test_true_tokens_repair.py` 是**隔离**测 `apply_repairs`（自造规范的 2-D 数组喂进去），**从不经过调用点** ⇒ **单测全绿、真实链路一跑就死** |
| ★★ 是什么抓到的 | **A/B 臂本身**：`True_Tokens=0`（pad 版）**不调用这条路径** ⇒ 全绿；只有 `=1` 的臂才踩上 ⇒ 这条纪律（"新轴必须单变量上、并留反例臂"）当场兑现 |
| 修法 | 调用点改为 `return build_prev_tok(...)[0]`（`+ [0]`，一行） |
| 新 md5 | `model.patched.py` → ★ **`0f9feba129a8f28c9707b40d4109e92b`**；`model.host_rows.diff` 已按同基线重生成（94 行，`patch -p1` 应用后与 patched **逐字节相同** 且编译通过） |
| 新防线 | ★ `tests/test_callsite_contract.py`（**调用点契约测试**）：① 断言出货模块返回的是 `(arr, stats)` 且 `arr` 可按 `[row, sh]` 索引；② 断言调用点那行**以 `[0]` 结尾**；③ **反向验证**——把元组直接喂 `apply_repairs` **必须抛 TypeError**；④ 对照——正确的 2-D 数组必须正常返回。已并入 `tests/run_all.sh` |

## 0e. ★★★ 2026-09-22 23:1x **必修**：计数上报口径 —— **一次性提示不能承载判据**

| 项 | 内容 |
|---|---|
| 现象 | A3 实测（`p3b2-true1-rowids1`）：`[ENGRAM-TRUE-TOKENS]` 只落了 **`计数={'unavailable': 6}`**，**`absent/filled/mismatch` 累计值一次都没落盘** |
| 根因 | ① `_engram_true_tokens_note()` 与 pageless 提示**共用** `_PAGELESS_WARNED[0]` 的"只打一次"旗标；② 第一次调用是 **warm-up decode（n=6）** ⇒ 那一次必然全是 `unavailable` ⇒ 之后所有更有信息量的调用都被"只打一次"吞掉 |
| ★ 后果（很具体） | **`mismatch > 0`（"镜像里是别人的 token" = 陈旧页静默算错）至今没有真机读数** ⇒ **无法判定要不要升 `mode=2`**（`ENGRAM-OFFLOAD-EXACT-FIX.md` §2 的残余静默点②） |
| 修法 | ① **独立旗标**；② **按键累计** `_TT_CUM`（六键）；③ 打印规则 = 首次 + 累计四元组变化后每 `V41_ENGRAM_TRUE_TOKENS_LOG_EVERY`（默认 200）次再打 + ★ **`mismatch` 一出现立刻打**；④ `self.engram_true_token_stats_cum` **按键累计**（原来 `engram_true_token_total` 只留总数，**丢掉分解**） |
| 新防线 | `tests/test_callsite_contract.py` 增加"计数上报口径"检查：用 **AST** 断言 `_engram_true_tokens_note` 函数体**不引用** `_PAGELESS_WARNED`，并断言存在 `_TT_CUM` / `LOG_EVERY` |
| ★ 写检查时又踩一次 | 第一版用 `grep 字符串` 判"不引用"，被**我自己的注释**判成假失败（与 `a2/logs/079 §3` 同一天第二次）⇒ **判代码要用 AST；字符串搜索只适合判"存在性"，不适合判"不存在"** |

★ 与 `0d` 的关系：两者都是**"模块各自都对、接起来/上报口径不对"**这一类；
`0d` 让第一个请求**崩**（响亮），`0e` 让判据**永远拿不到数**（静默 —— 更危险）。

## 0c. ★★ 2026-09-22 21:3x **必修**：`model_runner_v1.patched.py` 曾 `py_compile` 不过

| 项 | 内容 |
|---|---|
| 现象 | `SyntaxError: name '_ENGRAM_ROW_TOKENS_DISABLED' is used prior to global declaration`（第 3011 行） |
| 根因 | `_model_forward()` 里第 2999 行**先读**该全局，而 `global` 声明写在下面的 `except` 里 ⇒ Python 要求 `global` 出现在该作用域**任何使用之前** |
| 后果 | **import 期就崩、起服必挂**（不是运行期才出问题） |
| ★ 陷阱 | `ast.parse()` **能过**（所以只做 AST 检查会漏）；只有 `py_compile` / `compile()` 的 **symtable** 阶段才报 |
| 修法 | 把 `global _ENGRAM_ROW_TOKENS_DISABLED` **提到 `_model_forward()` 函数体顶部**（`assert forward_context is not None` 之后），删掉 `except` 里那一行 —— **纯 hoist，无语义变化** |
| 新 md5 | `a94887de05bb63370a6604260b358101`（= A3 上正在跑的那份，**逐字节对齐**） |
| 已加的防线 | `tests/run_all.sh` 新增**门 0：py_compile 全部交付件**（`model_runner_v1.patched.py` / `model.patched.py` / `engram_hash.patched.py` / `engram_repair.py`），并说明"AST 检查会漏" |
| 自洽性 | `model_runner_v1.publish.diff` 已按同一基线重生成；实测 `patch -p1` 应用后与 `model_runner_v1.patched.py` **逐字节相同**且 `py_compile` 通过 |

## 1. 环境变量（两层名字，方向单向）

| 宿主（影子包读） | 容器（生产代码读） | 含义 |
|---|---|---|
| `VLLM_V41_ENGRAM_TRUE_TOKENS` | `VLLM_V41_ENGRAM_TRUE_TOKENS` | `0` 关 / `1` 缺页回填+计数 / `2` 覆写陈旧槽位 |
| `V41_ENGRAM_ROW_IDS` | `V41_ENGRAM_ROW_IDS` | `1` = runner 允许发布 host token 表 |

★ 与本项目既有教训一致（`logs/065 §3`）：**只认一套名字**；影子包在拼装时把宿主值写成
`inner.sh` 里的字面量 export（做法见 `a2/scripts/make_shadow_pkg.sh:208-230`），
并在起服前打印生效值。

## 2. 影子包：追加的挂载块（插在 `make_shadow_pkg.sh` 的 `MOUNT_BLOCK` 里）

```python
        '# ---------- [A2-ENGRAM-TRUE-TOKENS] Engram × 卸载的精确修补（默认关） ----------',
        'if [ "${VLLM_V41_ENGRAM_TRUE_TOKENS:-0}" != "0" ]; then',
        '  _ER="${A2_ENGRAM_REPAIR_FILE:-$_A2F/engram_repair.py}"',
        '  [ -f "$_ER" ] || die "VLLM_V41_ENGRAM_TRUE_TOKENS=$VLLM_V41_ENGRAM_TRUE_TOKENS 但缺 $_ER"',
        '  MOUNTS+=(-v "$_ER:/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_repair.py:ro")',
        '  echo "[serve_a2] [A2-ENGRAM-TRUE-TOKENS] engram_repair.py <- $_ER"',
        'fi',
        'if [ "${V41_ENGRAM_ROW_IDS:-0}" != "0" ]; then',
        '  _MR="${A2_ENGRAM_RUNNER_FILE:-$_A2F/model_runner_v1.patched.py}"',
        '  [ -f "$_MR" ] || die "V41_ENGRAM_ROW_IDS=1 但缺 $_MR"',
        '  _want=$(md5sum "$_MR" | cut -d" " -f1)',
        '  MOUNTS+=(-v "$_MR:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro")',
        '  echo "[serve_a2] [A2-ENGRAM-ROW-IDS] model_runner_v1.py <- $_MR (md5=$_want)"',
        'fi',
```

★ 注意：**别**把它做成 `A2_*` 那种"内层名"（`serve_a2_offload.sh:70-79` 已有一次同类事故），
本方案的两个 env 在宿主与容器里**同名字面量**，由影子包直接透传。

## 3. 三道 fail-closed 门（缺一不可）

| 门 | 判据 | 落在哪 | 失败后果 |
|---|---|---|---|
| **G1 文件门** | 门控开 ⇒ 补丁文件必须存在，否则 `die` | 影子包组装期（上面那段） | 拒绝起服 |
| **G2 生效门** | 容器内 `md5sum /…/worker/model_runner_v1.py` 必须 == 我们钉死的**patched 文件 md5**（在 `inner.sh` 里、起 vllm 之前） | 容器内 | 拒绝起服（防"挂了但没生效"= 静默退回 pad） |
| **G3 运行门** | `VLLM_V41_ENGRAM_TRUE_TOKENS>=1` 时，第一次 `update(n>0)` 必须已经看到发布（`_ENGRAM_ROW_TOKENS["ids"] is not None`）；否则**打印一次 ERROR** 并把 `engram_true_token_miss=1` 计入 | `model.py` | 不中断服务，但**日志立刻可见**（今天的行为 = pad） |

★ G3 为什么不是硬 raise：A2 生产是"宁可降级也不许挂"的场景；但它**绝不静默**
（一次 ERROR + 计数器），并能与 §4 的读数一起判读。

## 4. 上车后的读数（就是本方案的验证证据）

```
grep -c 'ENGRAM-TRUE-TOKENS'            <serve.log>   # 首次修补的那一行（含计数）
grep -a 'engram_true_token'             <serve.log>   # 累计统计
grep -c 'ENGRAM-PAGELESS'               <serve.log>   # 期望 0（兜底路径不该被触发）
```

| 计数器 | 含义 | 怎么用 |
|---|---|---|
| `absent` | 本会 KeyError / 变 pad 的槽位（= 缺页） | `>0` ⇒ **命中**了 `logs/073` 那条病 |
| `filled` | 实际回填的槽位（mode≥1） | 应 ≈ `absent` |
| `mismatch` | 镜像**有值但 ≠ 真值**（陈旧页） | ★ `>0` ⇒ 今天存在**静默错** ⇒ 必须升 mode 2 |
| `unavailable` | 真值不可得（超 `num_tokens_no_spec` / 未发布） | `>0` 说明还有兜底在跑，需查 §8 清单 |
| `oob` | 修补遇到越界页（已扩容重跑） | 应与 kernel 的 `oob` 计数同量级 |

## 5. 回滚（一条命令级）

```bash
# 1) 关精确修补（保留 runner 挂载也无害：发布没人读）
VLLM_V41_ENGRAM_TRUE_TOKENS=0 …            # 行为 = 今天（pad 兜底）
# 2) 想连 pad 兜底一起回到"crash 而不是静默错"（排查用）
VLLM_V41_ENGRAM_PAGELESS_STRICT=1 …
# 3) 彻底回退：不设 V41_ENGRAM_ROW_IDS（runner 覆盖也不会挂）
```

★ 本方案**没有**动 `engram_jit_kernel.py`：`21e7d99` 的 pageless 修复与它**互不干扰**
（我们的回填发生在 kernel 之前；kernel 只会对"我们没修到的槽位"走 barrier）。
