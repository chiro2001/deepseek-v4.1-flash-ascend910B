# 一类"哑开关"：`serve_v2.sh` 会读、`serve_a2.sh` **从不透传**（2026-10-05）

> 起因：我在做 `enable_force_eplb` 的 A/B 时，只验证了**启动器 env**（`FORCE_EPLB=1` 确实在
> `~/tmp/launch_armFE.sh` 里），**没有验证它有没有进到容器的 `--additional-config`**。
> 后来复核发现：那一轮的 `--additional-config` 里**根本没有** `enable_force_eplb`
> ⇒ **那次"A/B"是同配置对同配置**，负结果无效。
> 顺着这条线做了一次系统审计，发现**同类漏洞共 5 个**。
> 标注：【实测】。

## 1. 机制：两层脚本，只有一层会透传

```
启动器（launch_*.sh，宿主）
   └─ serve_a2.sh（宿主）→ 生成 inner.sh（容器内）→ exec serve_v2.sh（容器内）
```

`serve_v2.sh` 用 `${VAR:-默认}` 读环境变量；而环境变量要进容器，
必须由 `serve_a2.sh` 在**生成 inner.sh 的那个 heredoc 里**显式写一行 `export VAR=...`。

**漏写 = 静默无效**：启动器里 `export FORCE_EPLB=1` 看起来完全正常，
容器里 `serve_v2.sh` 却永远取到默认 0，**没有任何警告**。

## 2. 审计方法（可复算）

```bash
# ① serve_v2.sh 读取的全部变量（形如 VAR=${VAR:-...}）
python3 - <<'PY'
import re, io
v2 = io.open("scripts/serve_v2.sh", encoding="utf-8").read()
print("\n".join(sorted(set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})=\$\{\1:-", v2)))))
PY
# ② 生成的 inner.sh 里实际 export 的变量
grep -oE '^\s*export\s+[A-Z_]+' results/<run>/inner.sh
# ③ 差集 = 哑开关
```

## 3. 结果：**5 个哑开关**（本次已全部接线）

| 开关 | 作用 | 之前的状态 |
|---|---|---|
| **`FORCE_EPLB`** | 强制专家打散（`enable_force_eplb`） | ❌ 静默无效 —— **本研究中的一个 A/B 因此作废** |
| **`DSA_CP`** | DSA context parallel（`enable_dsa_cp`） | ❌ 静默无效 |
| **`MC2_ALG`** | MC2 通信算法（如 hierarchy） | ❌ 静默无效 |
| **`WEIGHT_NZ`** | 权重 NZ 模式（`weight_nz_mode`） | ❌ 静默无效 |
| **`ENGRAM_HOST_RESTORE`** | engram 宿主态恢复 | ❌ 静默无效 |

（另有 `CHAT_TEMPLATE`/`SPEC_EAGER`/`HCCL_*`/`PYTORCH_NPU_ALLOC_CONF`/`TASK_QUEUE_ENABLE` 等
由别的路径设置，不在本表。）

**修复**（`scripts/serve_a2.sh`，紧跟既有的 `export MULTISTREAM=...` 行）：

```bash
export FORCE_EPLB=${FORCE_EPLB:-0} DSA_CP=${DSA_CP:-0} ENGRAM_HOST_RESTORE=${ENGRAM_HOST_RESTORE:-0}
export MC2_ALG=${MC2_ALG:-} WEIGHT_NZ=${WEIGHT_NZ:-}
```

默认值与 `serve_v2.sh` 的内建默认一致 ⇒ **不改变现有交付行为**，
只是让这 5 个开关**从启动器可达**。

## 4. 顺带修掉一个"改脚本会毁掉 inner.sh"的地雷

`serve_a2.sh` 生成 inner.sh 的 heredoc 是**未加引号**的（`<<INNER_EOF`），
因此**插入文本里的反引号/`$()` 会被当命令替换执行** ——
`set -e` 下一旦命令失败，生成的 `inner.sh` 就是 **0 字节**
（本仓早前记录过"往该 heredoc 插块 ⇒ inner.sh 变 0 字节，原因未查明"，
现在有了机制解释）。

我这次插入的注释里就带了两个反引号，**已自查并修掉**
（`tools/fix_dumbknobs_comment.py`，改完 `inner.sh` 3025 B、以 `exec` 正常结尾）。

⇒ **纪律**：往这个 heredoc 里加内容时，**只用纯文本**，不要出现反引号、`$(`、`;
`、`!` 等会被展开的形式。

## 5. 处置

1. **`FORCE-EPLB-NEGATIVE-20261005.md` 作废**（那次 A/B 无效）——
   已在该文档顶部标注，并**重做**真正的 `FORCE_EPLB=1` A/B；
2. 5 个开关已接线（默认不变），后续任何相关实验都能从启动器开启；
3. **纪律**：任何"从启动器设 `X=1`"的实验，**必须**在容器内验证一次
   （`nvidia-smi`/`serve.log` 的 `--additional-config` 行，或 `docker exec` 读 env），
   不能只看启动器脚本里有没有那一行 —— 这与之前"`V41_*` 开关不可达"是**同一类错误**，
   只是发生在启动器的下一层。

## 6. 复现

```bash
bash ~/tmp/launch_armFE.sh                       # 起一个 FORCE_EPLB=1 的臂
grep -o "additional-config {[^}]*}" results/armFE_forceeplb/serve.log | head -1
#   ⇒ 修复前：里面没有 enable_force_eplb（哑开关）
#   ⇒ 修复后：应有 "enable_force_eplb":true
```
