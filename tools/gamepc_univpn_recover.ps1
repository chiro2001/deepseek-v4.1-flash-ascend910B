# GamePC (192.168.101.5) UniVPN 客户端恢复脚本 —— 2026-10-05 实战验证
#
# 背景（详见 docs/NETWORK-RUNBOOK.md §4.5）：
#   UniVPN 的架构是「promote 服务（UniVPNPromoteService，Windows 服务，监听 RPC 127.0.0.1:29190）
#   + UI（UniVPN.exe）+ CSDK（serviceclient\UniVPNCS.exe）」。
#   **只有 UI 先连上 RPC，服务才会 `Start CSDK`**；CSDK 负责真正的 SSL 隧道。
#
#   已确认的坑：若 UniVPNService 启动时**已有 UI 进程存在**，服务会走 "Kill UI process / Start UI process"
#   分支，而**该分支不会创建 RPC accept 线程**（日志里缺少 `[RPC Send Thread][start]`）⇒
#   UI 连上去 40 s 后报 `ui get remote port packet send failed` / `UI and RPC connection failed!`，
#   CSDK 永远拉不起来，TAP 一直 Disconnected。**此时重启服务也无用（同样条件会复现）。**
#
# 正确顺序（本脚本实现）：先杀 UI → 再重启服务（无 UI ⇒ 会建 accept 线程）→ 再在**交互会话**里起 UI
#   → UI 自动尝试登录 → 弹「不可信 VPN 服务器证书」告警对话框 → 点「确定 Enter」→ 隧道建立。
#
# 用法（在 server-mini 上）：
#   scp tools/gamepc_univpn_recover.ps1 chiro@192.168.101.5:
#   ssh chiro@192.168.101.5 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\chiro\gamepc_univpn_recover.ps1'
#
# 判据：日志出现 `[RPC Send Thread][start]` + `[Start CSDK][success]`；
#       Get-NetAdapter 'TAP-Windows Adapter V9 #2' = Up；Test-NetConnection 192.168.45.21 -Port 22 = True

$log = 'C:\Program Files (x86)\UniVPN\log\UniVPN_UniVPNPromoteService_0.log'
$exe = 'C:\Program Files (x86)\UniVPN\UniVPN.exe'
$user = 'chiro'
$tmp = 'C:\Users\chiro'

function Say($m) { Write-Output ("[recover] " + $m) }

# ---------- 1) 杀 UI（关键：服务启动时必须没有 UI） ----------
$ui = Get-Process UniVPN -ErrorAction SilentlyContinue
if ($ui) { $ui | ForEach-Object { Say ("kill UI pid=" + $_.Id); Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } }
Start-Sleep -Seconds 3
Say ("ui_proc_count=" + ((Get-Process UniVPN -ErrorAction SilentlyContinue | Measure-Object).Count))

# ---------- 2) 重启服务，并断言 accept 线程真的建了 ----------
$before = (Get-Content $log -ErrorAction SilentlyContinue).Count
Restart-Service UniVPNService -Force -ErrorAction Continue
Start-Sleep -Seconds 12
$new = Get-Content $log | Select-Object -Skip $before
$hasSendThread = ($new -join "`n") -match 'RPC Send Thread'
$hasStartUI = ($new -join "`n") -match 'Start UI process'
$new | ForEach-Object { Say ("log: " + $_) }
Say ("accept_thread_created=" + $hasSendThread + " start_ui_branch_taken=" + $hasStartUI)
if (-not $hasSendThread) { Say "ABORT: 服务仍未创建 RPC accept 线程（先确认没有 UI 进程再重试）"; exit 2 }

# ---------- 3) 在交互会话里起 UI ----------
$uiStart = Join-Path $tmp 'gpu_ui_start.ps1'
Set-Content -Path $uiStart -Encoding ASCII -Value @'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
# 等证书告警对话框，点 “确定 Enter”（用 ASCII 特征匹配：按钮名以 ' Enter' 结尾）
$root = [System.Windows.Automation.AutomationElement]::RootElement
$deadline = (Get-Date).AddSeconds(90)
$clicked = $false
while ((Get-Date) -lt $deadline -and -not $clicked) {
  Start-Sleep -Seconds 3
  $p = Get-Process UniVPN -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $p) { continue }
  $cond = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ProcessIdProperty, [int]$p.Id)
  $wins = $root.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)
  foreach ($w in $wins) {
    if ($w.Current.ClassName -ne 'certificateDialog') { continue }
    $btnCond = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
    foreach ($b in $w.FindAll([System.Windows.Automation.TreeScope]::Descendants, $btnCond)) {
      if ($b.Current.Name -like '*Enter*') {
        try { $b.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke(); $clicked = $true } catch {}
      }
    }
  }
}
$clicked | Out-File 'C:\Users\chiro\gpu_cert_clicked.txt' -Encoding ascii
'@
Unregister-ScheduledTask -TaskName 'CodexUniVPNRecover' -Confirm:$false -ErrorAction SilentlyContinue
$a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ("-NoProfile -ExecutionPolicy Bypass -File " + $uiStart)
$p = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName 'CodexUniVPNRecover' -Action $a -Principal $p -Force | Out-Null
Start-ScheduledTask -TaskName 'CodexUniVPNRecover'

# 起 UI（同一任务机制：会话 1 + 最高权限）
Unregister-ScheduledTask -TaskName 'CodexUniVPNUI' -Confirm:$false -ErrorAction SilentlyContinue
$b = New-ScheduledTaskAction -Execute $exe
$q = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName 'CodexUniVPNUI' -Action $b -Principal $q -Force | Out-Null
Start-ScheduledTask -TaskName 'CodexUniVPNUI'
Say "UI task started, waiting up to 120s for login ..."

# ---------- 4) 等结果 ----------
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 3
  $tap = (Get-NetAdapter -InterfaceDescription 'TAP-Windows Adapter V9 #2' -ErrorAction SilentlyContinue).Status
  $route = (Get-NetRoute -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.DestinationPrefix -eq '192.168.0.0/16' } | Measure-Object).Count
  if ($tap -eq 'Up' -and $route -ge 1) { $ok = $true; break }
}
Say ("TAP=" + (Get-NetAdapter -InterfaceDescription 'TAP-Windows Adapter V9 #2' -ErrorAction SilentlyContinue).Status + " route16=" + (Get-NetRoute -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.DestinationPrefix -eq '192.168.0.0/16' } | Measure-Object).Count + " ok=" + $ok)
Say ("Test-NetConnection 192.168.45.21:22 = " + (Test-NetConnection -ComputerName 192.168.45.21 -Port 22 -WarningAction SilentlyContinue).TcpTestSucceeded)
Get-Content 'C:\Users\chiro\gpu_cert_clicked.txt' -ErrorAction SilentlyContinue | ForEach-Object { Say ("cert_dialog_clicked=" + $_) }
Say "cleanup: removing scheduled tasks"
foreach ($t in @('CodexUniVPNRecover','CodexUniVPNUI')) { Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction SilentlyContinue }
Remove-Item (Join-Path $tmp 'gpu_ui_start.ps1'),(Join-Path $tmp 'gpu_cert_clicked.txt') -Force -ErrorAction SilentlyContinue
Say "done"
