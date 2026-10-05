# 内网链路手册（拓扑 / 排查 / 已知故障）

> **这是查网络问题的第一入口。** 本文自包含：拓扑、每跳的探测命令、
> 已知故障的**特征签名**与处置、以及所有组件的位置清单。
> 事故复盘见 [`NETWORK-OUTAGE-RCA-20261003.md`](NETWORK-OUTAGE-RCA-20261003.md)。
> 权威的机器级拓扑（CPU/内存/Quartus 等）在 `lcvo/docs/HOST_TOPOLOGY.md`，
> 本文只覆盖**网络**这一面。

**最后核实：2026-10-05**（当晚 GamePC UniVPN 卡死已**远程修复**，见 §4.5；含可执行的恢复脚本）

---

## 0. 三十秒排查（先跑这段）

```bash
# 四跳一次性体检 —— 任何一跳 CLOSED 就是断点
for hp in 192.168.101.5:22 10.8.0.21:8022 10.8.0.21:34500 117.72.247.67:22; do
  h=${hp%:*}; p=${hp#*:}
  printf '%-22s %-6s -> ' "$h" "$p"
  timeout 8 bash -c "cat </dev/null >/dev/tcp/$h/$p" 2>/dev/null && echo OPEN || echo CLOSED
done

# 最终目的：能不能到 a3-21 / a3-22
timeout 20 ssh -o ConnectTimeout=10 a3-21 'echo A21_OK'
timeout 20 ssh -o ConnectTimeout=10 a3-22 'echo A22_OK'
```

| 看到什么 | 断在哪 | 去看 |
|---|---|---|
| 全 OPEN 但 `ssh a3-21` timeout | 第二跳（到 `192.168.45.0/24`）——**先跑 §4.1 的"对端是否真断"判据**：TLS 能握手 ⇒ 是客户端卡死 | §4.5（有远程恢复脚本）；对端真断才看 §4.1 |
| `10.8.0.21:8022` OPEN、`:34500` CLOSED | 平板端口转发 | §4.2 平板 relay |
| `192.168.101.5:22` CLOSED | 第一跳 / GamePC 本身 | §3 第 ① 跳 |
| 全 CLOSED | 本地网络或 VPN 全断 | §3 逐跳 |

---

## 1. 拓扑（全部通路）

```text
                        ┌──────────────── server-mini (192.168.101.7 / tun1 10.8.0.10)
                        │
        【主通路】        │
        └─ ssh GamePC 192.168.101.5
                             └─ GamePC 上的 UniVPN（TAP-Windows Adapter V9 #2）
                                  └─ 192.168.45.21 (a3-21) / .22 (a3-22)
                                     ※ 这是唯一常态化通路，GamePC 是单点

        【咽喉】         任意 10.8.0.* 主机
                             └─ 平板 10.8.0.21 (Termux sshd :8022)
                                  └─ ncat :34500 → 120.46.228.53:34500 → 公司内网
                                     ※ 由 server-mini 的 relay-healthcheck 监控 + 自愈

        【反向隧道】     a3-21 ─(出网 443)─> VPS 117.72.247.67
                             └─ VPS 上开 2223 → a3-21:22
                                VPS 上开 2224 → 192.168.45.22:22
                                     ※ 绕开 GamePC 的备用入口；由 a3-21 侧脚本保活

        【外层 VPN】      server-mini tun1 10.8.0.10 / GamePC 10.8.0.11 / VPS 10.8.0.1 / 平板 10.8.0.21
                             ※ 这一层（10.8.0.0/24）与内网层（192.168.45.0/24）是**两张不同的网**
```

**★ 最容易搞混的一点**：`10.8.0.0/24` 通 **不代表** `192.168.45.0/24` 通。
它们由不同的 VPN 承载；本手册把前者叫"外层 VPN"、后者叫"内网"。
2026-10-03 那次故障就是**外层通、内网断**。

---

## 2. 地址与端口清单

| 角色 | 地址 | 端口 | 备注 |
|---|---|---|---|
| server-mini | `192.168.101.7`，tun1 `10.8.0.10` | — | 编排机，本手册的"本机" |
| **GamePC** | `192.168.101.5` | 22 | **通内网的唯一跳板**；同时是 WSL 宿主（WSL 2222） |
| GamePC 的内网身份 | UniVPN 的 TAP 地址，**会变**（见过 `192.168.61.86` → `192.168.59.100`） | — | ⚠ **别拿这个地址当判据**；判据见 §4.1 |
| GamePC 通往内网的路由 | **`192.168.0.0/16` 直连（on-link）** 落在 TAP-Windows Adapter V9 #2 上 | — | ★ 不是 `192.168.45.0/24`，见 §4.1 |
| **平板** | `10.8.0.21` | **8022**=Termux sshd，**34500**=端口转发 | 用户 `chiro` |
| 平板转发的目标 | `120.46.228.53` | 34500 | 公司内网的落地端 |
| VPS | `117.72.247.67`，tun0 `10.8.0.1` | 22，2223→a3-21:22，2224→a3-22:22 | 反向隧道的公网落点 |
| a3-21 | `192.168.45.21` | 22 | 主算力机 |
| a3-22 | `192.168.45.22` | 22 | 备用算力机 |
| a3-21 的服务 | — | `19210`(TP8) / `19310`(tiny) | 见 README |

### SSH 别名（`~/.ssh/config`）

| 别名 | 实际 | 走法 |
|---|---|---|
| `a3-21` | `l00886679@192.168.45.21` | `ProxyJump chiro@192.168.101.5` |
| `a3-22` | `l00886679@192.168.45.22` | 同上 |
| `gpu5080` / `wsl5080` | GamePC / 其 WSL | 直连 |
| `ysy21` / `ysy22` | `spark.zhujihezi.com:39921/39922` | 另一套机器，**独立于外层 VPN**。2026-10-03 实测：TCP `39921` **OPEN**（网络可达），但 `ssh ysy21` 被拒（`Permission denied (publickey,password)`）⇒ **是认证问题不是网络问题**，用之前先确认密钥/密码 |

---

## 3. 逐跳探测（按顺序，定位断点）

```bash
# ① 本机 → GamePC（第一跳）
timeout 20 ssh -o ConnectTimeout=10 chiro@192.168.101.5 'echo JUMP_OK'

# ② GamePC → a3-21（第二跳，★ 最常断的一跳）
timeout 60 ssh chiro@192.168.101.5 \
  'powershell -NoProfile -Command "Test-NetConnection -ComputerName 192.168.45.21 -Port 22 -InformationLevel Quiet"'

# ③ GamePC 的 VPN 身份与路由（判断 UniVPN 起没起）—— **用 InterfaceDescription 定位，不靠名字**
cat > /tmp/ps_vpn.ps1 <<'PS1'
$tap = Get-NetAdapter | Where-Object { $_.InterfaceDescription -eq 'TAP-Windows Adapter V9 #2' }
Write-Output ("TAP ifIndex={0} Status={1}" -f $tap.ifIndex, $tap.Status)
Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $tap.ifIndex -ErrorAction SilentlyContinue |
  Select-Object IPAddress, PrefixLength | Format-Table -AutoSize | Out-String -Width 120
Write-Output "=== 通往内网的路由（应为 192.168.0.0/16 且 ifIndex 与上面一致）==="
Get-NetRoute -AddressFamily IPv4 |
  Where-Object { $_.DestinationPrefix -eq '192.168.0.0/16' } |
  Select-Object DestinationPrefix, NextHop, ifIndex | Format-Table -AutoSize | Out-String -Width 120
Write-Output "=== 决策路径 ==="
Find-NetRoute -RemoteIPAddress 192.168.45.21 -ErrorAction SilentlyContinue |
  Format-List IPAddress,InterfaceIndex,DestinationPrefix,NextHop | Out-String -Width 160
Write-Output "=== 实测（主判据）==="
Test-NetConnection -ComputerName 192.168.45.21 -Port 22 -InformationLevel Quiet
PS1
scp -q /tmp/ps_vpn.ps1 chiro@192.168.101.5:ps_vpn.ps1
timeout 90 ssh chiro@192.168.101.5 'powershell -NoProfile -ExecutionPolicy Bypass -File ps_vpn.ps1'

# ④ 外层 VPN 各点（10.8.0.*）是否互通
timeout 10 ssh -o ConnectTimeout=8 chiro@117.72.247.67 'ping -c 2 -W 2 10.8.0.21'

# ⑤ 平板（外层可达）
timeout 40 ssh -o BatchMode=yes -p 8022 chiro@10.8.0.21 'hostname; ~/.local/bin/relay-34500.sh status'

# ⑥ 最终目的
timeout 20 ssh -o ConnectTimeout=10 a3-21 'hostname'
```

> **Windows 上跑 PowerShell 的坑**：多层 ssh 嵌套时引号会被 shell 吃掉
> （本手册实测踩过）。**统一用"写 .ps1 → scp 过去 → `-File` 执行"**，
> 不要拼 `powershell -Command "...\"...\"..."`。

---

## 4. 已知故障模式（按出现频率）

### 4.1 ★ GamePC 的 UniVPN 掉线 ⇒ 内网 `192.168.0.0/16` 整段不通

**特征签名**（四条同时成立即可确诊）：

```
ssh chiro@192.168.101.5                          → 通     （第一跳没问题）
Test-NetConnection 192.168.45.21 -Port 22 …Quiet → False  （第二跳 connect failed）★主判据
Get-NetAdapter（TAP-Windows Adapter V9 #2）      → Disconnected
Get-NetRoute                                     → 没有 192.168.0.0/16 这条路由
```

> ⚠ **两条容易写错、我自己先写错过的判据**（2026-10-03 实测纠正）：
> 1. **路由是 `192.168.0.0/16`，不是 `192.168.45.0/24`。**
>    UniVPN 下发的是整段 `192.168.0.0/16` 直连（on-link，NextHop `0.0.0.0`），
>    `192.168.45.21` 只是落在这段里。**正常工作时也查不到 `192.168.45.0/24`** ——
>    拿它当判据会把好的链路误判成断的。
> 2. **UniVPN 的 IP 会变**（见过 `192.168.61.86` → `192.168.59.100`），
>    **不要拿某个固定 IP 当"在线"判据**。接口一律用
>    `InterfaceDescription -eq 'TAP-Windows Adapter V9 #2'` 定位
>    （Windows 侧接口别名是中文，在 ssh 里会显示成乱码，**按名字匹配不可靠**）。

**健康时的样子**（2026-10-03 修复后实测，可作对照）：

```
TAP ifIndex=17 Status=Up
该接口 IPv4：192.168.59.100/32
路由：DestinationPrefix=192.168.0.0/16  NextHop=0.0.0.0  ifIndex=17
Find-NetRoute 192.168.45.21 → IPAddress=192.168.59.100, InterfaceIndex=17,
                               DestinationPrefix=192.168.0.0/16, NextHop=0.0.0.0
Test-NetConnection 192.168.45.21 -Port 22 -InformationLevel Quiet → True
```

另外两个可佐证的观察：
- `openvpn.exe` / `UniVPN.exe` **进程还在**（只是隧道没建立）——所以别只看进程；
- `ssh a3-21` 的报错是 `Connection timed out during banner exchange`
  （**已经在认证之后**才超时，说明卡在第二跳而不是第一跳）。

**处置**：
1. **先判"对端是否真的断"**（★ 2026-10-05 新增，本次就是靠它翻案的）：从任意 10.8.0.x 主机跑
   `openssl s_client -connect 10.8.0.21:34500 -brief` ——
   若 **TLS 握手成功**（证书 `CN=LOCAL-101930058418`），说明**链路与网关正常**，
   故障在 **UniVPN 客户端自身**（见 §4.5），不要再等；
2. **对端真断时**：等它自己重连（2026-10-03 那次就是自行恢复的，无需人工）；
3. 需要人工干预时：**先按 §4.5.2 的远程流程恢复**（已实战验证，附带脚本
   `tools/gamepc_univpn_recover.ps1`）；只有连 GamePC 的 SSH 都进不去时，才需要到机器旁；
4. 恢复判据：`Get-NetAdapter` 里 `TAP-Windows Adapter V9 #2` = **Up**
   且 `Test-NetConnection 192.168.45.21` = **True**。

**为什么没有自动恢复**：GamePC 是单点，拓扑文档 §7 已把它列为已知单点。
目前**没有**任何针对 UniVPN 的保活脚本 —— 系统里唯一的 tunnel unit
`ssh-tunnel-8888.service` 保的是 `ysy21:8888` 那条转发，**与内网无关**，
不要以为它在保 a3 通路。

### 4.5 ★ UniVPN 客户端卡死 —— **根因已查明，可远程恢复**（2026-10-05 实战修复）

**重要性**：这一档和 §4.1 的"对端真断"**外部表现完全一样**（TAP Down / 第二跳不通），
但处置不同——**等它不会自愈**。2026-10-05 这次卡了 2 h+，最后**远程修好**（不再需要到机器旁）。

#### 4.5.0 架构：三件套 + 一个服务（先搞清这个，否则会像上一版手册那样误判"远程修不好"）

```text
UniVPNService（Windows 服务 = promote\UniVPNPromoteService.exe，LocalSystem）
   └─ 监听 RPC 127.0.0.1:29190
        ├─ UI（UniVPN.exe，跑在**登录用户的交互会话**里）
        │    当 UI 连上 RPC 并请求 ⇒ 服务执行 `Start CSDK`
        └─ CSDK（serviceclient\UniVPNCS.exe，由服务用 CreateProcessAsUser 拉起）
             负责真正的 SSL 隧道（连 10.8.0.21:34500 = 平板 relay）→ 建立 TAP 网卡
```

* 三者**缺一不可**；`UniVPNCS.exe` 直接手工启动会秒退（它需要服务用用户的 token/环境块拉起）。
* `ClientAutoBoot=0`：开机不会自动拉起客户端；服务是 `Auto` 启动的。

#### 4.5.1 ★ 根因：服务启动时**已有 UI 进程** ⇒ RPC accept 线程不会被创建

服务的启动日志有**两条互斥路径**（实测对比 9/30 正常启动 vs 10/5 两次坏启动）：

```
正常（9/30 03:16:41，启动时没有 UI）:      坏（10/5 16:35:35 / 18:07:41，启动时有 UI）:
  [RPC Listen socket][632]                   [Kill UI process][process already exists 1, need kill it]
  [RPC Listen][port 29190][socket 632]       [RPC Listen socket][640]
  [RPC Send Thread][start]     ← ★ accept    [RPC Listen][port 29190][socket 640]
                                              [Start UI process][begin/command line/success]
                                              （**缺 `RPC Send Thread][start]`** ← 病根）
```

⇒ **判据（一条就够）**：`UniVPN_UniVPNPromoteService_0.log` 里本次启动块**有没有
`[RPC Send Thread][start]`**。没有 ⇒ 内核会完成 TCP 握手（netstat 显示 ESTABLISHED），
但**服务永远不 accept**：UI 连上后 40 s 超时报
`[Process get remote port][uniVPN ui get remote port packet send failed!]` →
`[Process main socket connect RPC][ui get remote port failed!]` → 最后
`[UI and RPC connection failed!]` 并退出；**CSDK 永远起不来**。

> ⚠️ 这条也是"**不要随便重启 UniVPNService**"的原因：只要当时有 UI 在跑，
> 重启出来的新实例**必然**落进坏分支（我 10/5 两次重启都因此把状态越弄越坏）。

**两条前置事实**（2026-10-05 实测）：
* UniVPN 客户端配置（`%APPDATA%\UniVPN\config\内蒙蓝区.ini`）写的是
  **`GatewayAddress = 10.8.0.21` / `GatewayPort = 34500`** ⇒
  所谓的"公司网关"**就是平板的 relay**（→ `120.46.228.53:34500`）。
  所以 §4.2 的 relay 挂掉会同时打断 UniVPN，**两者是同一条链路**。
* 链路健康时，`openssl s_client -connect 10.8.0.21:34500 -brief` 必成功（TLS1.3）；
  从 **GamePC 本机**跑同样成功，且**可以稳定保持 45 s**（`.NET SslStream` 长连接实测）。

**特征签名**（判据按 ①→④ 递进）：

```
① openssl s_client -connect 10.8.0.21:34500   → 握手成功（链路 OK）
② promote 服务日志：本次启动块**缺 `[RPC Send Thread][start]`**（★ 本条是病根）
③ UI 日志 %APPDATA%\UniVPN\log\UniVPN_UniVPNUI_0.log 尾部（相隔约 40 s）：
     [UI ERROR][1][Process get remote port][uniVPN ui get remote port packet send failed!]
     [UI ERROR][1][Process main socket connect RPC][ui get remote port failed!]
     [UI ERROR][1][UI and RPC connection failed!]
④ GamePC 上 UniVPNCS.exe 不存在；TAP-Windows Adapter V9 #2 = Disconnected；
   无 192.168.0.0/16 路由；Test-NetConnection 192.168.45.21 -Port 22 → False
```

> ⚠️ 两个常见误读：`[CADM WARN][Route recovery item is Empty]` 是**现象不是原因**；
> `netstat` 里看到 `127.0.0.1:29190 ESTABLISHED` **不代表服务健康**（内核握手完成 ≠ 应用 accept）。

#### 4.5.2 ★ 恢复流程（2026-10-05 实战验证，全程远程）

顺序**不能变**（关键在"起服务时必须没有 UI"）：

```bash
# ① 杀 UI（必须在重启服务之前）
ssh chiro@192.168.101.5 'powershell -NoProfile -Command "Get-Process UniVPN -ErrorAction SilentlyContinue | Stop-Process -Force"'
# ② 重启服务，并在日志里确认出现 [RPC Send Thread][start]
ssh chiro@192.168.101.5 'powershell -NoProfile -Command "Restart-Service UniVPNService -Force"'
# ③ 在**交互会话**里起 UI（ssh 落在会话 0，直接起 GUI 不可见；用计划任务 + Interactive/Highest）
# ④ UI 会自动尝试登录，弹出「安全告警：不可信的VPN服务器证书！」对话框 ⇒ 用 UIAutomation
#    点「确定 Enter」（会话 1 内执行；按钮名按 ASCII 特征 '*Enter*' 匹配，中文匹配会被 GBK 破坏）
# ⑤ 判据：TAP=Up + 有 192.168.0.0/16 路由 + Test-NetConnection 192.168.45.21:22 = True
```

**已封装成脚本**（以上 5 步全在里头，含日志断言与证书对话框自动点击）：

```bash
scp tools/gamepc_univpn_recover.ps1 chiro@192.168.101.5:
ssh chiro@192.168.101.5 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\chiro\gamepc_univpn_recover.ps1'
```

2026-10-05 实测输出（摘要）：`accept_thread_created=True` → `[Start CSDK][success]` →
证书告警点击成功 → `TAP=Up / route16=1 / Test-NetConnection=True`。

> 附注：证书告警每次登录都会弹（服务器证书不在信任链上）。UI 上「更改设置」可以**永久忽略**
> 该告警——是否点由使用者决定（涉及安全提示），本手册不自动改。

#### 4.5.3 上一版手册里"全部无效"的尝试（保留为负面记录，避免重走）

| 尝试 | 结果 |
|---|---|
| 只重启 `UniVPNService`（UI 还在） | 新实例落进坏分支（无 accept 线程）⇒ 无效，且**掩盖真因**（看似"服务重启过了"） |
| 会话 1 起 GUI | GUI 起来但连不上 RPC ⇒ 弹 `QMessageBox「警告」`，点确定后 `UI and RPC connection failed!` 退出（当时不知道要先清 UI 再重启服务） |
| 直接启动 `serviceclient\UniVPNCS.exe`（会话 0/1） | **秒退**，无日志（必须由服务用用户 token 拉起） |
| 尝试 `UniVPNUserConsole.exe` 当 CLI | 它是 MFC GUI，无命令行入口 |

**可复用的诊断命令**（GamePC 有 SSH + 管理员权限：`chiro@192.168.101.5`，实测 `elevated_admin=True`）：

```bash
# 决定性的链路判据（本题两台机器都该跑一遍）
ssh -o ControlMaster=no chiro@192.168.101.5 \
  'powershell -NoProfile -Command "(Test-NetConnection 10.8.0.21 -Port 34500 -InformationLevel Quiet)"'
timeout 12 bash -c 'echo Q | openssl s_client -connect 10.8.0.21:34500 -brief -no_ign_eof'   # 期望 TLS1.3 + CN=LOCAL-…

# 客户端现场
ssh -o ControlMaster=no chiro@192.168.101.5 \
  'powershell -NoProfile -Command "Get-NetAdapter -InterfaceDescription \"TAP-Windows Adapter V9 #2\" | Select Status; Get-Process UniVPN,UniVPNCS -ErrorAction SilentlyContinue"'
```

### 4.2 平板 `relay-34500` 死掉 ⇒ 咽喉断（外层仍通）

**特征签名**：

```
10.8.0.21:8022   → OPEN      （Termux 活着）
10.8.0.21:34500  → CLOSED
ncat -v -z 127.0.0.1 34500（在平板上） → TIMEOUT   ← ★ 不是 "Connection refused"
~/.relay-34500.sh status → "未在运行"               ← 但端口其实被占
~/.relay-34500.log       → "Ncat: bind to 0.0.0.0:34500: Address already in use" ×N
ps -ef | grep "ncat -lk" → 多个命令行完全相同的进程
```

**TIMEOUT 而不是 refused 的含义**：端口在 LISTEN 但**没人 accept**
（`ncat -lk` 的**子进程继承监听 socket**，父进程死了子进程还在占）。

**处置**（在 server-mini 上执行，**不用登平板**）：

```bash
timeout 60 ssh -o BatchMode=yes -p 8022 chiro@10.8.0.21 \
  'pkill -f "[n]cat -lk"; sleep 2; rm -f ~/.relay-34500.pid; ~/.local/bin/relay-34500.sh start'
```

- `pkill -f "[n]cat -lk"` 的**方括号**写法能避免匹配到**自己这条 ssh 命令行**（踩过）；
- 必须清掉**父子两代**持有者，只杀父进程会留下黑洞。

**自动自愈**：`relay-healthcheck.timer`（user systemd，每 5 分钟）已经修好会做上面这套
**硬重置**。历史上它调的 `relay-34500.sh restart` **必然失败**（`start()` 只看陈旧 pidfile
就去 bind），所以日志里全是 `远程重启命令执行失败`——**自愈在场但从未生效过**。

### 4.3 a3-21 的反向隧道没起（备用入口缺失）

**特征**（2026-10-03 实测确认：当时确实是这个状态）：

```bash
# VPS 上应看到 2223/2224 两条监听；一条都没有 = 没起
ssh chiro@117.72.247.67 'ss -ltn | grep -E ":222[34]"'

# a3-21 上数进程 —— **必须用方括号写法**，否则会匹配到你自己这条 ssh 命令
ssh a3-21 'ps -eo cmd | grep -c "[a]3-vps-tunnel"'    # 0 = 没在跑

# 另一个旁证：日志文件根本不存在
ssh a3-21 'ls -la ~/.a3-vps-tunnel.log'    # No such file
```

> ⚠️ **不要用 `pgrep -c -f a3-vps-tunnel`**：远端那条 `bash -c "…pgrep…a3-vps-tunnel…"`
> 的命令行里**本身就含这个字符串**，会把包装 shell 也数进去
> （2026-10-03 实测：`pgrep` 数出 **2**，而真实进程数是 **0**）。

**处置**：在 a3-21 上

```bash
setsid nohup ~/.a3-vps-tunnel.sh >/dev/null 2>&1 </dev/null &
```

（脚本与说明见 §5；它的日志是 `~/.a3-vps-tunnel.log`。）

> **注意**：这条隧道与 §4.1 **互不替代**——它只提供"从 VPS 进 a3-21:22"，
> 而 a3-21 自己仍然需要能出网（走 443）。2026-10-03 故障期间它也是断的。

---

## 5. 组件清单（出问题先看这里）

### server-mini

| 路径 | 作用 |
|---|---|
| `~/tmp/tunnel/relay-healthcheck.sh` | 咽喉健康检查 + 自愈（**已被 timer 调用**） |
| `~/tmp/tunnel/relay-health.log` | 检查日志（**每 5 分钟一行**，看它就能知道咽喉史） |
| `~/tmp/tunnel/a3-vps-tunnel.sh` | 反向隧道脚本（**部署在 a3-21，不是本机**） |
| `~/.config/systemd/user/relay-healthcheck.{timer,service}` | 5 分钟定时器（`Persistent=true`） |
| `~/tmp/tunnel/relay-34500-watchdog.sh` | Termux 侧看门狗（**部署在平板**） |
| `~/.ssh/config` | `a3-21`/`a3-22` 的 `ProxyJump` |

### 平板（`10.8.0.21`，Termux）

| 路径 | 作用 |
|---|---|
| `~/.local/bin/relay-34500.sh` | 端口转发控制（start/stop/restart/status/log） |
| `~/.local/bin/relay-34500-target.sh` | 每条连接的目标（`ncat 120.46.228.53 34500`） |
| `~/.relay-34500.pid` / `~/.relay-34500.log` | pid / 运行日志 |
| `~/.local/bin/relay-34500-watchdog.sh` | 由 `termux-job-scheduler` 周期调用 |
| `~/.termux/boot/10-relay-34500.sh` | 开机自启（需 Termux:Boot） |

### a3-21

| 路径 | 作用 |
|---|---|
| `~/.a3-vps-tunnel.sh` | 反向隧道保活（supervisor loop） |
| `~/.a3-vps-tunnel.log` | 其日志 |

---

## 6. 维护要点 / 陷阱

1. **别用 `termux-job-scheduler -p`** 查调度状态 —— 实测会**挂住**（本手册踩过）。
2. **`pgrep -f <名字>` / `grep <名字>` 会匹配到「你自己这条 ssh 命令」** ——
   远端 `bash -c` 的命令行里含那个字符串就会被一起数进去
   （实测把「没在跑」数成 **2**）。**一律用方括号写法**：
   `grep -c "[a]3-vps-tunnel"`、`grep "[n]cat -lk"`。
3. **`ps -ef | grep -c "ncat -lk"` 会数出 3 而不是 1**：父进程 + 每来一个连接 fork 的子进程，
   命令行相同。**这不是泄漏**；判健康一律以**端口探测**为准。
4. **Android 不让普通应用读 `/proc/net/tcp`**（SELinux），`ss`/`netstat` 看不到连接是正常的
   —— 用 `ncat -z 127.0.0.1 34500` 代替。
5. **`grep -rl ... ~/` 这类全盘搜索会炸**（家目录里有巨大的 `.config/Token Monitor/*.json`）。
   加 `--include` 与 `--exclude-dir`，或限定目录。
6. **多层 ssh 里的引号**：Windows PowerShell 一律走「写 `.ps1` + `-File`」，别拼 `-Command`。
7. **外层通 ≠ 内网通**：`ping 10.8.0.21` 通只证明 §1 里的「外层 VPN」好，**不证明** a3 可达。

---

## 7. 相关文档

| 文档 | 内容 |
|---|---|
| [`NETWORK-OUTAGE-RCA-20261003.md`](NETWORK-OUTAGE-RCA-20261003.md) | 2026-10-03 那次中断的完整复盘（逐跳证据） |
| `lcvo/docs/HOST_TOPOLOGY.md` | 权威机器拓扑、CPU 集合规则、传输配额、remote 执行协议 |
| `~/tmp/tunnel/relay-health.log` | 咽喉的**历史**记录（自愈是否在生效，看它最快） |
