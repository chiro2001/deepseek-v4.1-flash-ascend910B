# CED wire wrapper 准备件（未启动）

- 独立副本 `ced_wire_audit.py`：6,870 B，SHA-256
  `82873d5b2c454d50580b80c39a46970f69e18c7b15118fa8f7dcec78d098c0d1`；与源文件
  字节一致。按 root 授权，该副本也以同样字节复制至 P 容器的可写准备目录
  `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/proxy5/ced_wire_audit.py`，
  P 内 SHA/size 复核一致。
- 固定 proxy 源仅做 SHA 核验：工作树 `ced_proxy.py` 与 hash-verified CED v4 提取包
  中的 `ced_proxy.py` 均为 SHA-256
  `813c5a0e1a8b7e7988ee9f02fd3a83d8fa38ede9a88713a9c65e5cf0cf99eb33`。
- `ced_wire_audit_help.txt` 是本地 `python3 ced_wire_audit.py --help` 输出；
  `p_container_help.txt` 是 P 内实际副本的 `python3 ... --help` 输出与 SHA 校验。均未
  调用启动参数。`p_container_postcheck.txt` 记录 P 内 `18992` 返回 `111`、wrapper 文件
  存在且 audit output
  `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/proxy5/wire` 仍不存在。
- root 允许创建 P `jobs/proxy5` 仅作为 wrapper 源码准备目录。没有创建 `wire` audit
  output 目录，没有启动 proxy、建立 listener、发送 HTTP 或 generation 请求。
