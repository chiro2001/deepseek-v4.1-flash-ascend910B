# 112 — ★ 一个 bash 陷阱：注释行把 `nohup env` 的参数表截断，导致 11 个变量静默丢失

> 2026-09-23 04:3x–04:5x CST。发现：子代理 **`PROF_WIRE`**（文件 diff + 离线复现）；主代理复核现场日志。
> 标记：**【实测】/【推断】**。

---

## 0. 一句话

我给 `run_arm_r8.sh` 加 `V41_PROFILE` 透传时，把**注释块插在了 `nohup env \` 的参数表中间**。
bash 的 `\`+换行把物理行粘成**一条逻辑行**，而 `#` **把这条逻辑行从中间截断**
⇒ 原本一条 `nohup env <11 个变量> bash serve_a2.sh` **裂成两条命令**：

1. `nohup env MODEL=… PATCH_MODE=mount PYTHON_PGO=0` —— **没有 command operand**
   ⇒ `env` 只把环境 **dump 到 stdout**、exit 0、**什么都没起**；
2. `V41_PROFILE=… bash serve_a2.sh &` —— **只剩后半段前缀**，
   前 11 个变量（`MODEL/IMAGE/DEVS/CHIPS/NAME/PORT/SERVED_NAME/CPU_BIND/DROPCACHE/PATCH_MODE/PYTHON_PGO`）**全靠继承**。

⇒ **"从交互 shell 起臂能成功"是巧合**（那些变量恰好在 shell 里 export 过）；**独立进程必然失败**。

---

## 1. 【实测】症状全集（都是同一个根因的不同面）

| 症状 | 直接原因 |
|---|---|
| `[serve_a2][FAIL] 必须设置 MODEL=<模型目录>` | `MODEL` 没继承到，且 `run_arm_r8.sh:36` 的 `MODEL=${MODEL:-…}` **不 export** ⇒ 在断开的第 2 条逻辑行上只是普通 shell 变量 |
| `镜像 dsv41-a2:v8 不存在` | `IMAGE` 落到 `serve_a2.sh:37` 的 **A2 默认值** |
| `起容器 dsv41-a2` / `devs=0 1 2 3 4 5 6 7` | `NAME`/`DEVS` 同样落到 `serve_a2.sh:38`/`:48` 的 **A2 默认值** |
| **`DRAFT_GRAPH=1 但 draft 版文件安装失败`** | `PATCH_MODE` 落到 `serve_a2.sh:234` 的 **`baked`**（A2 默认）⇒ 走了"从镜像内安装"的分支（镜像是 A3 的，没有那个路径） |
| `tee: …/driver.log: No such file or directory` | 该 RID 的输出目录还没建（`run_arm_r8.sh` 前半段没跑） |
| `serve.log` **0 字节** | 起服的引擎与 runner 写的日志不是同一个进程 |

★ 两条失败臂的 `serve_a2.log` 铁证：**成功臂 `PATCH_MODE=mount`** vs **失败臂 `PATCH_MODE=baked`**。

---

## 2. 【实测】离线复现（`PROF_WIRE` 的 `repro_env_break.sh`，不需要卡）

用**那段真实文本** + 一个 stub 替掉 `serve_a2.sh`，`bash -x` 实录：
```
+ nohup env MODEL=… PATCH_MODE=mount PYTHON_PGO=0     ← 第 1 条：无 command operand
+ V41_PROFILE=0 …                                      ← 第 2 条：从下一段起
+ bash <stub>
```
- **变体 A（父进程没 export）** ⇒ stub 收到 `MODEL=<UNSET> IMAGE=<UNSET> NAME=<UNSET> DEVS=<UNSET> PATCH_MODE=<UNSET>`
- **变体 B（父进程 export 过）** ⇒ 全部收到
⇒ **这就是"成功是巧合"的机制**，比"`env` 只增量设置"这个表述深一层。

---

## 3. 处置

| # | 动作 | 状态 |
|---:|---|---|
| 1 | **绕过（已在用）**：在 `run_perf_fix_arms.sh` 里**显式 export 全套**（`MODEL/IMAGE/PATCH_MODE/DEVS/CHIPS/…`）⇒ 即使参数表被截断，第 2 条命令也能从环境拿到 | ✅ **实测有效**：`r8-prof` 臂的 `serve_a2.log` 已打 `PATCH_MODE=mount`（首次越过该坑） |
| 2 | **根治**：把注释块**整块搬到 `nohup env \` 那行之上**（不能留在参数表里）⇒ `PROF_WIRE` 已给 patch | ⏳ 待应用（`patches/run_arm_r8.fix-env-comment-and-profile-gate.patch`） |
| 3 | **静态门**：`PROF_WIRE` 的 `check_env_chain.sh` 会在起臂前查 ①`nohup env` 参数表里有无裸注释行 ②`env` 列表是否含 `V41_PROFILE=` ③`run_4axis_arm.sh` 的 CMD 是否含 `MODEL=/IMAGE=/V41_PROFILE=` | ✅ 就绪（补丁前 **4 FAIL**、补丁后 **0 FAIL**） |

---

## 4. 教训（与本仓红线第 10 条同族）

> **"参数表里能不能插注释"这件事本身，是需要被验证的前提。**

★ 这与本卷已记的多次事故是同一族：
- `logs/099` 端口门被别人的 `health=200` 满足；
- `logs/100` `pgrep -f` 匹配到自己的 ssh 命令行；
- 今天的 `N_MNT_R8` 按**路径字符串**计数、G2b 按**路径**判 graphsafe、`docker exec "$TAG"` 用错容器名；
- 本条：**判据（`env` 参数表）被自己的注释截断**。

**可操作做法**：① 在 `nohup env \`／`docker run \` 这类长参数表里**绝不插入注释**（要写就写在 `\` 链的**上方**）；
② 起臂链**任何一环都不要依赖"调用方恰好 export 过"** —— 显式传，或加静态检查门。
