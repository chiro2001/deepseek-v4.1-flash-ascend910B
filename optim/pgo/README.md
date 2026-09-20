# optim/pgo/ —— PGO 产物目录（**本目录不随包分发二进制**）

## 为什么这里是空的

PGO 产物（`python3` + `libpython3.12.so.1.0`）是**与目标机器绑定**的构建产物：

* 跟 **CPU 微架构**、**gcc 版本**、**glibc（发行版）** 相关；
* 把它打包进 git 仓库等于把 30 MB 的二进制塞进版本历史，且**在别的机器上未必合适**
  （实测 A2 镜像与 A3 镜像的 libpython md5 就不同）。

所以：**二进制不进包，改为在目标机器上由脚本编译一次。**

## 怎么生成

在**目标机**（要跑服务的机器）上执行：

```bash
bash build_scripts/00_ensure_pgo.sh
```

它会：

1. 算**构建指纹**（`CPU part` + `gcc` + `glibc` + CPython 源码 sha256 + `mtune` 开关）；
2. 与 `optim/pgo/.build_marker` 比对 —— **一致就秒退**（"只编译一次"）；
3. 不一致 / 无 marker 时，起一个**编译容器**（不占用服务容器、不挂 NPU）跑
   `01_setup_build_env.sh → 06_package.sh` 六步；
4. 产物落 `optim/pgo/{python3,libpython3.12.so.1.0}`，并写 marker；
5. 探测容器内 libpython 落点，写 `optim/pgo/TARGET_PATH.txt`。

编译通常在 30–40 分钟量级（首次），之后增量。

## 中断/失败后怎么续编（v8 起）

编译容器**固定名字**（默认 `pgo-build-a2`）、**不再 `--rm`**：30–40 分钟的编译一旦中途
失败，容器与"已装好的构建依赖 + 已完成进度"都留着，直接重跑同一命令即可续编：

```bash
bash build_scripts/00_ensure_pgo.sh     # 参数没变 ⇒ 自动 docker start -ai 复用，从断点继续
```

规则：

* 复用判据 = `PGO_JOBS / PGO_MTUNE / PGO_SKIP_INSTALL / PGO_FORCE` 与容器创建时一致；
  任一变化（或 `PGO_FORCE=1`）会删掉旧容器、**全新编译**；
* 代理变量与 `scripts/openEuler.repo` 只在**容器创建时**生效 —— 改了它们要 `PGO_FORCE=1`；
* 想进现场看：`docker start -ai pgo-build-a2`；编译**成功**后容器默认仍保留，
  要清掉加 `PGO_RM_CONTAINER=1`；
* **防假成功**：如果编译链非零退出，第 5 步只采纳 **mtime 晚于本次开工**的产物；
  上次残留的产物会被显式忽略并报 FAIL（不写 marker）—— 否则会写出一份"指纹正确但产物陈旧"
  的 marker，之后每次都被"秒钟退出"骗过去。

## 内网环境（自签证书 / TLS 中间盒）

内网常见自签证书：`curl` 会因证书链校验失败而拒绝下载 CPython 源码，yum/dnf 也会卡住。
这类机器才需要显式降级：

```bash
INSECURE_TLS=1 bash build_scripts/00_ensure_pgo.sh
```

它会（**默认关**，public 网络别开）：给 `02_fetch_source.sh` 的 curl 加 `-k`；
往容器内 `/etc/yum.conf` 与 `/etc/dnf/dnf.conf` 写 `sslverify=False`（幂等，重复跑不叠加）。
**完整性并不依赖 TLS**：`02_fetch_source.sh` 仍会做华为云/阿里云**双源 sha256 交叉校验**，
`tools/fetch_corpus.sh` 也仍逐个文件比对 sha256 —— `-k` 只影响传输层。

内网自建源（可选）：把 repo 文件放成 `scripts/openEuler.repo`，存在时才会挂进容器
（不存在时不挂 —— 否则 docker 会在宿主上创建一个同名**目录**，把容器里的 repo 顶坏）。
本包**不含**任何内网地址/凭据。

## 镜像自带 libpython 的隔离（A2 真机踩到的坑）

构建镜像里本来就装了一份**非 PGO** 的 CPython（prefix `/usr/local/python3.12.13`）。
新编出的 `./python` 带 `DT_RPATH=/usr/local/python3.12.13/lib`，而 ld.so 的搜索顺序是
**RPATH 先于 `LD_LIBRARY_PATH`** ⇒ 它会抓到那份旧库，报
`undefined symbol: __gcov_indirect_call`（缺 gcov 运行库符号）。

`04_make.sh` 因此在**构建容器内**把旧库移到 `/work/logs/image-libpython-quarantine/`
（= 宿主 `optim/pgo/build/logs/image-libpython-quarantine/`，属 .gitignore 覆盖范围）。
只影响构建容器：打包出去的 `python3`/`libpython` 来自 `/staging`（新编的那份），
`05_install.sh` / `06_package.sh` 都从 `/staging` 取产物，不受影响。

## 产物怎么被用上

```bash
# ① 把产物放进镜像（会同时写 TARGET_PATH.txt / 做 python 版本校验）
bash scripts/build_image.sh

# ② 起服时由脚本挂载覆盖 libpython（可回滚）
PYTHON_PGO=1 bash scripts/serve_a2.sh     # 默认就是 1；没有产物会自动降级并告警
```

## 不想要 PGO？

完全可以跳过 —— 收益本身是**预期 0~3%、且必须实测**（构建参数默认不含 `-march/-mtune`，
代码生成是通用 aarch64）：

```bash
SKIP_PGO=1 bash scripts/build_image.sh    # 镜像小 ~30 MB
PYTHON_PGO=0 bash scripts/serve_a2.sh     # 起服不挂 PGO
```

## 想针对本机核调优（可选实验）

```bash
PGO_MTUNE=1 bash build_scripts/00_ensure_pgo.sh   # CFLAGS 加 -mtune=native
```

安全（不引入新指令集），但**收益需自己 A/B**。

## 其它开关

| 变量 | 作用 |
|---|---|
| `PGO_FORCE=1` | 忽略指纹，强制重编 |
| `PGO_BUILD_IMAGE=<image>` | 指定编译容器镜像（默认复用本包的基础镜像） |
| `PGO_JOBS=<n>` | 并行度（默认 `min(48, nproc)`） |
| `PGO_CPUSET=<0-3,…>` | 编译容器可用的 CPU（默认 node0 的 cpulist） |
| `PGO_CONTAINER_NAME=<name>` | 编译容器名（默认 `pgo-build-a2`） |
| `PGO_RM_CONTAINER=1` | 编译结束后删掉编译容器（默认保留，便于续编/排错） |
| `INSECURE_TLS=1` | 跳过 TLS 证书校验（**仅内网自签证书**；默认关） |
| `PGO_SKIP_INSTALL=1` | ⚠️ **预留未接线**：当前会被透传进容器，但六个步骤都不读它（等价于无效）。要"少做事"请用 `PGO_JOBS` 调并行度 |

失败时看 `optim/pgo/build/logs/`。

> `optim/pgo/build/` 是宿主侧编译工作区（源码树 + `.o`，可增量），
> 体积可达数百 MB，**不应提交到 git**（见仓库根的 `.gitignore`）。
