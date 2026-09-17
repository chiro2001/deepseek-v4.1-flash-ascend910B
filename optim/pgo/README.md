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
3. 不一致 / 无 marker 时，起一个**一次性容器**编译（不占用服务容器、不挂 NPU）；
4. 产物落 `optim/pgo/{python3,libpython3.12.so.1.0}`，并写 marker；
5. 探测容器内 libpython 落点，写 `optim/pgo/TARGET_PATH.txt`。

编译通常在 30–40 分钟量级（首次），之后增量。

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
| `PGO_SKIP_INSTALL=1` | 只编译，不做自检 |

失败时看 `optim/pgo/build/logs/`。

> `optim/pgo/build/` 是宿主侧编译工作区（源码树 + `.o`，可增量），
> 体积可达数百 MB，**不应提交到 git**（见仓库根的 `.gitignore`）。
