# DCP8 v3 现场误报更正记录

采集与更正时间：2026-10-13 03:58 CST。此记录更正一次错误的“目标容器/服务已消失”报告；它不是服务故障记录。

## 结论

最初检查在本地 `server-mini` 上执行，却把本机进程表、监听表和本机 Docker daemon 当成了 A3 目标现场。该检查不能证明远端容器或服务消失。随后经 SSH 指定目标 `a3-21` 复核，容器仍在运行，API 和 runner 进程仍在容器 PID namespace 中；147456-token Dihuo 输出 JSON 已在目标主机生成。主代理独立复核也确认目标 `127.0.0.1:18960` listener 存在。没有证据表明 DCP 服务发生故障；本次检查未重启服务、未重发请求、未修改或清理目标数据。

## 现场身份与证据

- 本地采集机：`server-mini`，x86_64；Docker context 为 `default`，endpoint `unix:///var/run/docker.sock`，daemon 名称 `server-mini`、ID `01188969-5a13-4a5a-9cec-88907563a104`。该 daemon 的 `docker ps -a` 不包含 DCP 目标容器。
- 目标 SSH：`a3-21` → 用户 `l00886679`、HostName `192.168.45.21`，ProxyJump `chiro@[192.168.101.5]`。目标主机报告 `localhost.localdomain`、aarch64；目标 Docker 为旧版本接口，不支持 `docker context` 子命令。
- 目标容器：`dsv41-real-stack-tp8-20261010-a321`，ID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0`。经 `a3-21` 执行 `docker ps -a --no-trunc --filter id=...` 返回 exit 0，状态 `Up 3 hours`。`docker inspect` 返回 exit 0：`running=true`、`paused=false`、`restarting=false`、`OOMKilled=false`、`dead=false`、`restart_count=0`、容器 init 的宿主机 PID 为 `585275`。
- PID namespace：经 `a3-21` 执行 `docker exec <container> ps -p 841623 -p 841625 -p 1506117 ...` 返回 exit 0；输出包含 runner PID `841623` 和 API PID `841625`（均约 34:26 elapsed，API 采样 `%CPU=3.8`），未包含旧快照中的客户端 PID `1506117`。这只说明该 PID 在采样时不在容器进程表中，不推断其退出码。容器内 PID 不应直接用本机或远端宿主机 `ps -p` 判断。
- 数据：容器 `/work` bind 到目标宿主机 `/home/l00886679/projects/dsv41-real-stack-tp8-20261010`。目标宿主机文件列表显示 `results/delivery_dcp8_v3/run.log` 为 549006 bytes、mtime 03:47:34；`/home/l00886679/tmp/dsv41-workload-bench/dihuo_147456_dcp8_v3.json` 为 16642155 bytes、mtime 03:47:18。主代理在 03:48 CST 独立确认 147456 请求已正常生成输出并且 18960 listener 存在。该结论以 SSH 目标现场为准，不以本地 daemon 的空列表为准。

## 误报成因与命令结果

1. 初始 `ps -p 1506117,841625`、`ss -ltnp | rg ':18960'`、`docker ps ...` 在本机 `server-mini` 运行；本机无匹配 PID/listener、且本机 Docker 列表无目标容器，均与 A3 现场无关。最初的命令组最终 exit 0（末尾 `find` 成功），不能把总 exit 0 或空匹配解释为目标服务状态。
2. 首次 SSH Docker 查询使用 `docker context show && docker info ... && docker ps -a`。远端 stderr 为 `docker: 'context' is not a docker command.` / `See 'docker --help'`，命令 exit 1；由于 `&&` 短路，远端 `docker info` 与 `docker ps -a` 根本没有执行。它不能作为“远端无容器”的证据。随后单独执行的远端 `docker inspect` 和 `docker ps` 均成功，结果见上文。
3. 尝试在容器内用 `ss -ltnp` 查 listener，`docker exec` exit 126，stderr 为 `OCI runtime exec failed: exec failed: unable to start container process: exec: "ss": executable file not found in $PATH: unknown`。这是容器缺少 `ss` 命令，不是无 listener；listener 由主代理在目标现场独立验证。
4. 一次组合式 `run_status` SSH 查询 exit 1：SSH 输出 `mux_client_request_session: session request failed: Session open refused by peer`，随后报告现存 ControlSocket、禁用 multiplexing；查询还因嵌套引号/变量作用域把日志路径解析成 `/run.log`，stderr 包含 `stat: cannot statx '/run.log': No such file or directory` 与 `tail: cannot open '/run.log' for reading: No such file or directory`。该输出是错误构造的诊断命令结果，不代表真实 job 文件缺失；不用于状态判断。
5. 宿主机 `find` 扫描 job 子树时 exit 1，stderr 来自活跃 `tmp/pymp-*` 目录的 `Permission denied`；在报错前已列出 runner、run.log、service receipts 和 launched.json。另一个不递归 job tmp 的目标结果文件扫描 exit 0，并找到上述 Dihuo JSON。权限报错不作服务故障解释。

## 后续采集约束

所有现场命令必须显式绑定 SSH 目标 `a3-21`（192.168.45.21），并在记录中标明命令执行于本机、目标宿主机还是容器 PID namespace。分别记录容器宿主机 PID `585275` 与容器内 runner/API PID `841623`/`841625`；不得用本机 Docker daemon 空列表或宿主机 `ps` 为空推断目标容器/容器进程不存在。若容器内缺少诊断工具，明确记录工具不可用并用目标宿主机/容器替代只读检查，不能将检查失败等同于服务异常。
