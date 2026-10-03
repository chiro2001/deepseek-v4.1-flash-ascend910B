# 内网中断的逐跳排查（2026-10-03）

> 用户：「VPN 断开，排查网络问题，看是路径上哪里断掉了」+「10.8.0.* 内网能否访问到平板 / 平板的端口转发还在不在」。
> 本文按**逐跳实测**给结论，全部标注 **【实测】/【推断】**。

---

## 0. 结论（一页）

**两个独立的故障叠加**，不是一个：

| # | 断点 | 证据 | 处置 |
|---|---|---|---|
| **A** | **GamePC 的 UniVPN 掉了** ⇒ 第二跳（到 `192.168.45.0/24`）整段不通 | GamePC 上 `TAP-Windows Adapter V9 #2` 状态 **Disconnected**；无 `192.168.61.86`；无 `192.168.45.*` 路由；`Test-NetConnection 192.168.45.21:22` = **False** | 等 UniVPN 自行重连（**已恢复**：TAP 变 **Up**、Test-NetConnection = **True**） |
| **B** | **平板上 `relay-34500` 死了，且自愈机制永远失败** | pidfile 指向**已死**的 pid 30525；实际占端口的是一批**陈旧 ncat**；三处探测全 CLOSED；`ncat -v -z 127.0.0.1 34500` = **TIMEOUT**（不是 refused） | **已修**（见 §3） |

**另外回答用户的两个问题**：

1. **10.8.0.\* 内网能否访问平板** → **能**【实测】。从 `10.8.0.10`(server-mini) 与 `10.8.0.1`(VPS) 两侧：`ping 10.8.0.21` 通（RTT 30–150 ms）、`8022/tcp`(Termux sshd) **OPEN**。
2. **平板的端口转发还在不在** → **当时已经死了**，**现在已恢复**：三处（平板自测 / server-mini / VPS）`34500` 全部 **OPEN**。

---

## 1. 权威链路（`lcvo/docs/HOST_TOPOLOGY.md`）

```text
server-mini (192.168.101.7 / tun1 10.8.0.10)
  └─ ssh GamePC 192.168.101.5
       └─ GamePC 的 UniVPN (192.168.61.86)
            └─ 192.168.45.21 (a3-21) / .22 (a3-22)

另一条（平板咽喉，relay-healthcheck 监控的）：
  … → 平板(10.8.0.21) → ncat:34500 → 120.46.228.53:34500 → 公司内网 → a3-21/a3-22
```

---

## 2. 逐跳实测（故障当时）

| 跳 | 测试 | 结果 | 判定 |
|---|---|---|---|
| server-mini → GamePC | `ssh chiro@192.168.101.5` | **通**（认证成功） | 第一跳 OK |
| GamePC → a3-21 | `Test-NetConnection 192.168.45.21 -Port 22` | **False**（connect failed） | **★ 断在这里** |
| GamePC 的 VPN 地址 | `Get-NetIPAddress` | **无 `192.168.61.86`**；`OpenVPN TAP-Windows6` = 169.254.x（链路本地、Disconnected） | UniVPN 未建立 |
| GamePC 路由 | `Get-NetRoute` | **无 `192.168.45.0/24`** | 无路可走 |
| server-mini → a3-21 | `ssh a3-21`（ProxyJump GamePC） | `Connection timed out during banner exchange`（在**认证之后**、`a3-21` 那一跳超时） | 印证第二跳 |
| VPS → a3-21 | VPS 路由表只有 `10.8.0.0/24` 与 `192.168.101.0/24` | **无 `192.168.45.0/24`**，ping 100% 丢包 | 旁路不成立 |
| 平板 → a3-21 | 平板自己 `ping 192.168.45.21` | **100% 丢包**，22 端口 CLOSED | 平板侧同样不是旁路 |
| a3-21 → VPS 反向隧道 | VPS 上 `2223/2224` | **未监听** | 备用路也断 |

**⇒ 结论**：从 server-mini 出发的三条可能通路（GamePC-UniVPN / VPS / 平板）**全部**在"到 `192.168.45.0/24`"这一步断掉，而**共同的上游**是那张内网 VPN。用户判断的"中间某个地方链路中断"成立。

---

## 3. 平板 relay-34500 的真实故障机理（★ 值得记的一条）

### 3.1 现象

```bash
$ ~/.local/bin/relay-34500.sh status
未在运行                      # ← pidfile 里是已死的 pid 30525
$ ncat -v -z 127.0.0.1 34500
Ncat: TIMEOUT.                # ← 不是 "Connection refused"
$ tail ~/.relay-34500.log
Ncat: bind to 0.0.0.0:34500: Address already in use. QUITTING.   # × 几十条
```

### 3.2 两个叠加的缺陷

1. **`start()` 只按 pidfile 判"在不在运行"** —— Android 回收进程后 pidfile 变成陈旧值，
   于是 `status` 说"未在运行"、`start` 直接 `ncat -lk`，**必然**撞 `Address already in use`
   （真正占端口的是**更早那批** ncat）。
2. **`ncat -lk` 的子进程会继承监听 socket**（实测 pid 31364 的三个子进程命令行与父进程完全相同）。
   只杀父进程 ⇒ 子进程仍持有端口 ⇒ **"端口在 LISTEN 但没人 accept"**，
   连接表现为 **TIMEOUT** 而不是 refused —— 这正是 `relay-healthcheck` 探测到的"34500 不可达"。

**后果**：server-mini 上的 `relay-healthcheck.timer`（每 5 分钟）调
`relay-34500.sh restart` —— 因为缺陷 1，**必然失败**，日志里留下成片的
`Address already in use` + `远程重启命令执行失败`。**自愈机制在场，但从来没有生效过。**

### 3.3 处置（**改在 server-mini 侧，不动平板脚本**，风险最低）

`relay-healthcheck.sh` 的情况 B 从

```bash
ssh … '~/.local/bin/relay-34500.sh restart'
```

改为**硬重置**：

```bash
ssh … 'pkill -f "[n]cat -lk"; sleep 2; rm -f ~/.relay-34500.pid; ~/.local/bin/relay-34500.sh start'
```

- `pkill -f "[n]cat -lk"` 的方括号写法可避免**匹配到自己的 ssh 命令行**（踩过）；
- 先清掉**父子两代**持有者，再清 pidfile，最后 `start`；
- 原脚本已备份为 `~/.local/bin/relay-34500.sh.bak-20261003`。

### 3.4 验证（正控 + 负控）

| 用例 | 结果 |
|---|---|
| 正控：relay 健康时跑 healthcheck | `健康：relay 34500 在监听`（静默退出 0） |
| **负控**：先 `pkill ncat -lk` 把 relay 打死（端口 CLOSED），再跑 healthcheck | **`已自动修复 relay`** ✅（改造前这里必然失败） |
| 修复后三处探测 | 平板自测 **OPEN** / server-mini **OPEN** / VPS **OPEN** |

---

## 4. 诚实边界 / 未修项

1. **平板脚本本身没改**（只在 server-mini 侧做了硬重置）。平板上的 `relay-34500.sh`
   仍有"只看 pidfile"的缺陷；如果哪天不经过 healthcheck 直接手工 `start`，还会踩。
   我写过一版改进（`~/tmp/tunnel/relay-34500.fixed.sh`）但**实测会把 relay 弄挂**
   （pidfile 记到 `setsid` 那道），**所以没有部署**，只作为素材留着。
2. **进程数会显示 3 而不是 1**：`ncat -lk` 的父进程 + 每来一个连接 fork 的子进程
   命令行相同。**这不是泄漏**（实测父子关系明确），判"健康"应以**端口探测**为准。
3. **`termux-job-scheduler -p` 会挂住**，不要用它查调度状态。
4. GamePC 的 UniVPN 是**单点**（拓扑文档 §7 已列）：它一断，`192.168.45.0/24` 就整段不可达，
   且**没有任何自动恢复**——这次是靠它自己重连的。若要做冗余，需要第二条独立内网通路。
