# CPython 3.12.13 PGO+LTO：独立验证通过（可交付）

> 2026-09-16 16:10 CST｜A3-node1｜**在真实服务镜像里实测**（非构建容器）
> 构建方：子代理 `cpython_pgo_build`（A3-node2 独立容器）｜验证方：主 Agent（A3-node1）

---

## 0. 结论：**可用，采纳为 host 侧优化项**

| 验证项 | 结果 |
|---|---|
| 产物 md5 跨机一致 | ✅ `python3` `7f7d616878fc6c9c23e75f40d936e3ee`、`libpython3.12.so.1.0` `f1ebbee1405d0e31136aa4480b57b3dc`（A3-node2 与 A3-node1 相同） |
| 版本正确 | ✅ 新解释器 `Python 3.12.13 (main, Sep 16 2026, 07:53:13) [GCC 11.4.0]`（旧的是 `Aug 3 2026`） |
| **`import torch/torch_npu/vllm`** | ✅ **带 NPU 设备时全部成功**，版本与旧完全一致（`2.10.0+cpu` / `2.10.0.post4` / `0.27.1`） |
| 纯 Python 性能 | ✅ **−16% ~ −23%**（见 §2） |
| 只替换 libpython 即可 | ✅ 二进制保持 `--enable-shared`，ABI 未变 |

---

## 1. 验证方法（**在同一镜像里 A/B，最小侵入**）

不改任何文件、不重启服务，用 `-v` 挂载把 PGO 的 `libpython3.12.so.1.0` 覆盖进去：

```bash
IMG=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3
P=/usr/local/python3.12.13

# 旧
sudo docker run --rm --entrypoint $P/bin/python3 $IMG /tmp/cmp.py

# 新（只挂 libpython）
sudo docker run --rm \
  -v ~/cpython_pgo/libpython3.12.so.1.0:$P/lib/libpython3.12.so.1.0:ro \
  --entrypoint $P/bin/python3 $IMG /tmp/cmp.py
```

**为什么只挂 libpython 就够**：镜像里的 `python3` 是**符号链接**指向 `python3.12`，
而 `python3.12` 只是 10 KB 的启动器，**真正的字节码解释器在 `libpython3.12.so.1.0`（30 MB）里**
—— 那正是 PGO+LTO 生效的地方。

---

## 2. 性能实测（同一容器、同一 `--cpuset-cpus 300-303`、3 轮取最小）

| 基准 | OLD | **NEW (PGO+LTO)** | 提升 |
|---|---|---|---|
| `tiny_call`（6 次算术循环） | 0.465 µs | **0.390 µs** | **−16.1%** |
| `dict_loop`（6 次字典写） | 0.719 µs | **0.556 µs** | **−22.7%** |
| `list_append`（6 次 list append） | 0.536 µs | **0.429 µs** | **−20.0%** |

**⇒ 与子代理在构建机的测量一致（−15~18%），且是在真实服务镜像里复现的。**

---

## 3. 对我们的实际意义

Engram host 侧的**纯 CPU 相位**（`hash` 0.401 + `plan` 0.241 + `lookup` 0.205 = **0.847 ms/step**）
以及 vLLM 其它 Python 胶水，都会按 **−16~23%** 缩放。

**与 numba 的关系（互补，不冲突）**：
* numba 把 `hash`/`plan` 从 **~0.64 ms → ~0.03 ms**（22–117×）
* PGO 对**剩下的** Python 代码（`lookup` 的 wrapper、其它 host 逻辑）给 −16~23%
* 两者可叠加

**粗估总收益**：
| 项 | 收益 |
|---|---|
| numba `hash` + `plan` | −0.57 ms |
| numba `lookup`（进行中） | −0.17 ms |
| **PGO（对剩余 host 代码 −20%）** | **≈ −0.05 ms**（Engram 剩余）+ vLLM 其它胶水的额外收益 |

⇒ **PGO 本身不是大头，但它是"免费"的全局改善**，且对小张量 dispatch 密集的负载有效。

---

## 4. 部署方式（三选一）

| 方式 | 做法 | 适用 |
|---|---|---|
| **A. 起服时挂载**（推荐，可回滚） | `docker run -v <host>/libpython3.12.so.1.0:$P/lib/libpython3.12.so.1.0:ro` | 我们的 A/B 实验 |
| **B. `docker cp` 进容器** | `sudo docker cp libpython3.12.so.1.0 dsv41-a21-perf:$P/lib/libpython3.12.so.1.0`（先备份 `.orig`） | 已跑容器 |
| **C. 完整树覆盖** | 用 `install_staging.tar.gz`（128 MB）覆盖整个 `$P` | 需要 pip 编译 C 扩展带 PGO flags 时 |

**⚠️ 注意**：只换 libpython 时 `_sysconfigdata` 仍是旧值 —— 运行时无影响，
但**以后 pip 编译 C 扩展不会带 PGO flags**（需方式 C 才彻底）。

---

## 5. 未验证 / 风险

| 项 | 状态 |
|---|---|
| 真实 NPU 端到端推理 | **未测**（需起服务；当前主线服务在跑别的实验） |
| 服务侧实际 ms/step 收益 | **未测**（预期小，因为 host 只占一部分） |
| 75 个扩展模块逐一回归 | 子代理已验 `lib-dynload` 75 个 **IDENTICAL** |
| 多进程长期稳定性 | 未测 |
| 热替换对已跑进程 | **无效**，必须重启进程 |
| 构建容器内 apt 升级过 `libssl3`/`libbz2-1.0` | 子代理已在服务同镜像内验证无符号缺失；回滚预案要留 |

---

## 6. 证据

| 内容 | 路径 |
|---|---|
| 产物 | `A3-node1:~/cpython_pgo/{python3,libpython3.12.so.1.0}`；`A3-node2:~/cpython_pgo/out/` |
| 构建报告 | `A3-node2:~/cpython_pgo/REPORT.md`、`BUILD_INFO.md` |
| 应用/回滚 | `A3-node2:~/cpython_pgo/out/apply.sh`、`rollback.sh`（含 `/tmp` 假前缀演练） |
| 性能对比脚本 | `A3-node1:/tmp/py_perf_cmp.py` |
| 镜像基线核实 | `A3-node2:~/cpython_pgo/REPORT.md` §1（`CONFIG_ARGS`、75 个扩展清单） |
