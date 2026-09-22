# 085 — ★★★★ `ENGRAM_DEVICE_INDEX` 一开，**`075`/`077` 修的整条 host 路径被绕过**（"四轴通过"会变成假通过）

> 2026-09-22 23:3x CST。执行：**主代理**（本机读 `model.py` + A3-node1 只读核查；不占卡）。
> 触发：`r8-4axis`（第一次起的那条）报 `DEVICE-INDEX=1` 而 `ENGRAM-TRUE-TOKENS=0`，
> 与同日 `p3b2`（`DEVICE-INDEX=0` / `TRUE-TOKENS=8`）**恰好相反**。
> 标记：**【实测】/【推断】**。

---

## 0. 一句话

`EngramModel.prepare_engram()` 在 device-index 可用时走 `_prepare_engram_device()`，
而**那个函数里 `engram_history` 出现 0 次** —— `self.engram_history.update()` **只在
`_prepare_engram_host()` 里**（`model.py:1108`）。

⇒ 因此 **`ENGRAM_DEVICE_INDEX=auto`（A3 上会被打开）会让下面这些东西全部不跑**：
Engram 的 page 镜像、`075`/`077` 的 pageless 修复、`TRUE_TOKENS` 的精确修补、`mismatch` 计数。
**"四轴同时跑通"如果在这种配置下通过，它证明的不是我们要验收的那条路。**

---

## 1. 【实测】静态判据（本机，代码级）

对 `patches/files/model.py` 按"函数体"切分后统计：

| 函数 | 起始行 | `engram_history` 出现次数 | `.update(` 出现次数 | 函数体行数 |
|---|---:|---:|---:|---:|
| `_prepare_engram_device` | **892** | ★ **0** | ★ **0** | 106 |
| `_prepare_engram_host` | **1041** | 2 | **1**（= 1108 行那个调用） | 98 |

⇒ ★ 这是**结构性**的，不是"某个分支偶尔跳过"：device 路径**根本没有**访问镜像的那个对象。

## 2. 【实测】运行期对照（同一天的两条臂，同一个 runner）

| 臂 | 容器内 `V41_ENGRAM_DEVICE_INDEX` | `grep -ac 'DEVICE-INDEX' serve.log` | `grep -ac 'ENGRAM-TRUE-TOKENS' serve.log` |
|---|---|---:|---:|
| `p3b2-true1-rowids1` | `0`（显式设过） | **0** | ★ **8** |
| `r8-4axis`（第一次起，`22:13:19`） | ★ **`auto`** | ★ **1** | ★ **0** |
| `r8-4axis`（重启后，`22:26:56`） | `0` | **0** | （起服中，尚为 0） |

运行期那条 `[DEVICE-INDEX] 能力探测通过，Engram 算子入图已启用：host mapping registered`
就是它真的走了 device 路径的**代码痕迹**。

★ **判据口径**（今天第三次用到同一条方法论）：**不要凭 env 名字猜，要看代码痕迹** ——
`grep -ac 'DEVICE-INDEX'`：`1` = device 路径（要避免），`0` = host 路径（要的）。

---

## 3. 为什么这一格对**目标**是致命的（三条理由，按严重性）

| # | 理由 |
|---|---|
| 1 | ★★★ **判据没覆盖需求**：我们要验收的是"`ENGRAM=1` + 卸载 + int8 + draft 入图 **同时跑通**"，而 Engram 与卸载的**交互点**正是 `engram_history` 的 page 镜像（`073` 的 KeyError 就是它）。device 路径把这一格**整个拿掉** ⇒ 通过也不代表通过。 |
| 2 | ★★ **不能外推到 A2**：A2 生产是 `ENGRAM_DEVICE_INDEX=0`（用户 09-20/09-21 的启动命令；`a2/scripts/serve_a2_offload.sh` 默认也是 0，并且**显式给 auto 时会打印响亮警告**）。 |
| 3 | ★ **A3 上还有已知的崩溃史**：`logs/069` 实测 device-index 打开 ⇒ Engram 表注册 **183 GiB** ⇒ `EH0012` + 池拿不到注册预算 ⇒ 起服失败/推理崩。 |

★ 为什么 runner 会漏掉：`run_arm_r8.sh` 的 `env` 列表里**没有** `ENGRAM_DEVICE_INDEX`
（`grep -n ENGRAM_DEVICE_INDEX run_arm_r8.sh` = 空），而 shadow 的 `serve_a2.sh` 读
`${ENGRAM_DEVICE_INDEX:-auto}` ⇒ **不显式传就是 `auto`**。
（好消息：`nohup env A=1 … bash serve_a2.sh` 里的 `env` 会**继承父环境** ⇒
在调用方 `export ENGRAM_DEVICE_INDEX=0` 或写进那条 `env` 列表都能生效 —— 后者更显式，已采纳。）

---

## 4. 修法（已落 `a2/scripts/run_4axis_arm.sh`）

1. **默认并显式传** `ENGRAM_DEVICE_INDEX=0`（写进 `env` 列表，不依赖调用方环境）；
2. 顶部横幅**打印**该值（`★ ENGRAM_DEVICE_INDEX=0 （必须 0：否则 host 路径整段被绕过）`）；
3. 起服期判据从四条加到**五条**：
```
5) docker exec <ctr> sh -c 'env | grep ENGRAM_DEVICE'      # 期望 0
   grep -ac 'DEVICE-INDEX'      <serve.log>                # ★ 必须 0
   grep -ac 'ENGRAM-TRUE-TOKENS' <serve.log>               # ★ 应 >0（证明修补代码在跑）
```

---

## 5. 这条发现的一般化价值

它属于**"配置把被验收的代码路径绕过去了"**这一类 —— 与今天另外两条同族：

| 同族 | 表现 | 出处 |
|---|---|---|
| `081` | 合并件过期 ⇒ **Engram 接线静默消失** | 派生件不新鲜 |
| `079 §2` | 镜像 tag 三处不一致 ⇒ **静默用旧镜像** | 修复没上车 |
| ★ `085` | `ENGRAM_DEVICE_INDEX=auto` ⇒ **整条 host 路径被绕过** | 配置选错路径 |

⇒ 共同教训：**"跑通了"必须先问"跑的是哪条路"**；
而回答这个问题**只能靠代码痕迹**（日志/计数/md5），不能靠配置项的名字。
