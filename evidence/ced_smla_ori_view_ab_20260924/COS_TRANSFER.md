# 私有 COS 归档回执

- COS key：`share/xfer/ced_smla_ori_view_ab_20260924.tar.gz`（私有对象）
- 远端生成与本机下载的压缩包 SHA-256 均为：
  `f52829be0b8828b12e08ffd26a0239c2f9e44cf782ba8b9740a19d722ba3d4eb`
- 包大小：43,140 bytes；tar 清单 35 项，包含 31 个由包内 `SHA256SUMS`
  覆盖的文件。解压后在本目录执行 `sha256sum -c SHA256SUMS`，31/31 通过。

此回执是下载解压后在本机追加的传输记录，不属于 COS tar 包；`SHA256SUMS`
保持远端原样，用于验证包内归档文件。
