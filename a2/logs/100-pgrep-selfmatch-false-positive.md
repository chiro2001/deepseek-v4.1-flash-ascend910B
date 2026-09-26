# 100 — ★★★ 我自己的巡检命令出错：`pgrep -f kv_offload_client` **匹配到了我自己的 ssh 命令行**（"进程在跑"是假的）

> 2026-09-23 01:2x CST。执行：**主代理**（自纠）。
> 性质：**判据卫生**（不是新功能）。标记：**【实测】**。

---

## 0. 一句话

我用 `ssh <host> '... pgrep -c -f kv_offload_client ...'` 巡检"压测客户端在不在跑"，
得到的 `bench=1` **是假的** —— `pgrep -f` 匹配的是**整条命令行**，
而**我这条 ssh 命令自己的命令行里就含 `kv_offload_client` 这个字符串**。

⇒ 于是我把"**臂还在起服**"误读成"**已经进压测**"，并据此抱怨"容器不见了但客户端还在"。
**两处判据同时错**（进程计数 + 容器列表），差点又一次做出错误归因。

---

## 1. 【实测】证据

```
$ ssh A3-node1 '... pgrep -af "run_arm_r8|...|kv_offload_client" | head -5 ...'
3562551 bash -c cd $HOME && mkdir -p .../final2 && nohup env TAG=...      ← 我自己的驱动命令
3562554 bash .../run_4axis_arm.sh
3562602 bash .../run_arm_r8.sh
★ 3622612 bash -c echo "=== 驱动/客户端进程 ==="; pgrep -af "... kv_offload_client" ...   ← ★ 这就是"匹配到的那个进程"，是我自己
$ ssh A3-node1 'pgrep -af "[b]ench/kv_offload_client" | head -2'
★ 空（那会儿真的没有客户端）
```
★ 判据差别：`pgrep -f kv_offload_client`（**会自匹配**） vs `pgrep -af "[b]ench/kv_offload_client"`（用方括号打断字面量 ⇒ 不自匹配）。

---

## 2. 【实测】同一时刻的另一个错判：`docker ps | grep | head -5`

我同时用
```
docker ps --format "{{.Names}}\t{{.Status}}" | grep -E "r8-|4axis|prbench" | head -5
```
判断"容器在不在"，那一刻**只看到 `prbench-c0/c1/c2`** ⇒ 我判"容器不见了"。
而随后精确查：
```
docker ps -a --filter name=r8-r8-4axis-final --format "..." 
  → r8-r8-4axis-final   Up 6 minutes   2026-09-23 00:09:01
```
⇒ ★ **容器一直在**。`head -5` 是这次错判的帮凶（截断），
但**根子**是"我用一个**可能被其它对象满足**的宽松判据做了存在性判断"。

---

## 3. 根因与今天其它条目的关系（**第 9 次同类**）

| 出处 | 判据被谁"冒充" |
|---|---|
| `079 §3` | 判据被**自己的提示文本**满足（字符串空间重叠） |
| `093` | 前置 env 没带全 ⇒ 跑的是**另一份文件** |
| `099` | 判据被**另一个容器**的服务满足（端口空间重叠） |
| ★ `100` | 判据被**我自己的巡检命令**满足（进程名字符串重叠） |

⇒ 一句话同族：**判据必须绑定到唯一的对象，而且判据自身不能出现在被测集合里。**

---

## 4. 修法（写进巡检习惯）

| 要判的东西 | ❌ 不要用 | ✅ 用 |
|---|---|---|
| 压测客户端在不在 | `pgrep -f kv_offload_client` | ★ `pgrep -af "[b]ench/kv_offload_client"`（方括号断字面量）或 `pgrep -x python3` + 精确路径 |
| 臂的容器在不在 | `docker ps \| grep r8 \| head -N` | ★ `docker ps -a --filter name=<精确名> --format ...`（**加 `-a`**、**用 filter**、**不截断**） |
| 端口是否被占 | `curl .../health` 返回 200 | ★ `ss -ltn \| grep ":$PORT"`（这是 `099` 的 G4 门采纳的判据） |
| 服务是不是**我的** | 本机 `curl 127.0.0.1:$PORT/health` | ★ `docker exec <NAME> curl ...`（只认自己容器内的 200） |

★ 并且：**凡"计数>0"的判据，都要问一句"这个 0/1 有没有可能来自判据自身"**。
