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
| `model_runner_v1.patched.py` / `model_runner_v1.publish.diff` | runner：发布 host token 表（+30 行；基线 md5 `9d84d9b073aeece3fd3bbc143ba20567`） | `/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro` |

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
