# PGO 编译工作区（**不打进 tar**）

这个目录是 `build_scripts/00_ensure_pgo.sh` 的**宿主侧编译工作区**：

```
optim/pgo/build/
├── src/Python-3.12.13/     ← CPython 源码树（保留 .o，支持增量）
├── out/                    ← 06_package.sh 的产出（python3 / libpython...）
└── logs/                   ← configure / make / install / verify 日志
```

## 为什么放在容器外

用户要求"**在容器外有编译缓存，只编译一次**"：

* 编译在**一次性容器**里跑（不占服务容器、不挂 `/dev/davinci*`）
* 但工作区落在**宿主**，所以第二次可以增量、产物持久
* `.build_marker` 记录指纹（`cpu_part + gcc + glibc + 源码 sha256 + mtune 开关`）：
  **一致就秒退**，不一致自动重编

## 为什么不出现在 tar 里

这里会有 ~650 MB（源码树 + 目标文件），而且**是与机器绑定的中间产物**。
打包时用 `tar --exclude='a2_pkg_v6/optim/pgo/build/*'` 排除，只留本 README。

到 A2 上第一次跑 `bash build_scripts/00_ensure_pgo.sh` 时会重新生成。

## 手动清理

```bash
# 只清源码树，保留 marker（下次会重新解包源码）
rm -rf optim/pgo/build/src

# 完全重来（连 marker 一起）—— 下次会重新编译
rm -rf optim/pgo/build optim/pgo/.build_marker
```
