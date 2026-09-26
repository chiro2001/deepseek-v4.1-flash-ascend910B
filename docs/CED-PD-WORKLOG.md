## 环境坑：git push 被全局代理卡死（2026-09-25）

现象：`git push` / `git ls-remote` 全部报
`TLS connect error: error:0A000126:SSL routines::unexpected eof while reading`，
但 `curl https://github.com` 返回 200（0.7 s）。

原因：`~/.gitconfig`（**全局**）里配了 `http.proxy = https.proxy = http://127.0.0.1:14514`，
而这个本地代理已经死了（用 `curl -x http://127.0.0.1:14514` 也连不上）。

处置：**没有改全局配置**，只在仓库里加了空值覆盖：

```bash
cd <repo>
git config --local http.proxy ""
git config --local https.proxy ""
```

之后 `git ls-remote origin` / `git push` 直接可用（验证：`ls-remote` 返回
`5f8082b … refs/heads/feat/ced-pd-a3`）。

临时替代：`git -c http.proxy= -c https.proxy= push origin <branch>`。

⚠️ 只在本仓库生效；其它仓库若同样报 TLS eof，用同法或找网络管理员确认
`127.0.0.1:14514` 是否应该恢复。
