# DCP 运维存档

本目录是**防丢失**存档：DCP overlay 挂载机制从未提交进 git，只存在于
a3-21 的工作区，任何 `git reset --hard` / `checkout` / `stash` 都会静默抹掉它
（2026-10-01 踩过一次，代价一轮 10 分钟起服）。详见
[`../../docs/V41-DCP-OPS-PITFALLS-20261001.md`](../../docs/V41-DCP-OPS-PITFALLS-20261001.md)。

| 文件 | 说明 |
|---|---|
| `serve_a2.sh.dcp-mount-20261001` | 完整文件快照（`origin/main` 的 serve_a2.sh + DCP 挂载 + `*.so` 白名单） |
| `serve_a2.sh.dcp-mount-20261001.sha256` | 校验和 |
| `dcp-mount.patch` | 相对 `origin/main` 的增量补丁（`git apply --check` + 逐字节比对已验证） |
| `restore_serve_a2_dcp.sh` | a3-21 侧幂等恢复脚本（校验 sha256 后才覆盖） |

## 恢复方式（a3-21）

```bash
# ① 从 GitHub 取存档
scp <本目录>/serve_a2.sh.dcp-mount-20261001 a3-21:~/dcp_durable/
scp <本目录>/serve_a2.sh.dcp-mount-20261001.sha256 a3-21:~/dcp_durable/
bash ~/restore_serve_a2_dcp.sh

# ② 或者：以 origin/main 为基线打补丁
cd ~/cedpd-repo && git checkout -- scripts/serve_a2.sh
git apply -p1 <本目录>/dcp-mount.patch
```

## 起服后必查（不要看"起服成功"）

```bash
docker inspect <name> --format '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' \
  | grep -c vllm_ascend                # >= 15
docker exec <name> md5sum \
  /vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py
grep -a "PP/DCP/PCP guard bypassed" <run>/serve.log
```
