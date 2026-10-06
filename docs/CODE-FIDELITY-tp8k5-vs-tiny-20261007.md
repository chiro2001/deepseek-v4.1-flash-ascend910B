# ★ 代码保真度问题：tp8k5 与 tiny 跑的是**两套差异极大的代码**（2026-10-07）

> 在追 TP=8 非确定性时发现的**结构性事实**，它**更正**了上一轮"非确定性是 TP=8 特有"的结论，
> 也解释了我为何无法在 tp8k5 上使用仓库里的诊断探针。
> 全部为【实测】。

## 0. 一页纸

| 事实 | 证据 |
|---|---|
| **tp8k5 的 `attention/dsa_v41.py` 是镜像烘焙版**（1092 行、**0 探针**、mtime **Sep 11**） | `docker exec … wc -l` + `docker cp` 后看 mtime |
| tiny 的同一文件是 **4636 行、71 处探针**（我们开发版） | `~/dcpw/vllm_ascend/attention/dsa_v41.py` |
| **两者 model.py 都 `from vllm_ascend.attention.dsa_v41 import …`** | `patches/files/model.py:80` |
| `dsa_v41.py` **只在 CED / PROBE 模式才被挂载** | `serve_a2.sh:1266-1270`（`_ced_dsa="$PKG/experimental/ced/dsa_v41.py"`） |
| **原始 tp8k5 run 也没有挂载它** ⇒ 我的恢复是忠实的 | 原始 `driver.log` 无 CED/PROBE 标记 |
| ⇒ 仓库里的 `_perf_flags` 探针（`idxdet`/`kvfp`/`blkfp`/`det_reduce`…）**在 tp8k5 上不存在** | 容器内 `grep -c _perf_flags` = **0** |

## 1. 文件对照

| 文件 | tp8k5 容器内 | tiny（`~/dcpw/…`） | `patches/files/` | 备注 |
|---|---:|---:|---:|---|
| `attention/dsa_v41.py` | **1092 行 / 0 探针** | 4636 行 / 71 探针 | **不存在** | tp8k5 用镜像版 |
| `models/deepseek_v41/indexer.py` | **244 行 / 0 探针** | 439 行 / 9 探针 | 273 行 | tp8k5 用镜像版 |
| `attention/dsa_v1.py` | 2172 行 | — | 2172 行（md5 相同 `58e1d6b6…`） | 由 `patches/files` 挂载 |
| `models/deepseek_v41/model.py` | 1650 行 | 1650+ | 1650 行 | 由 `patches/files` 挂载 |

tp8k5 一共挂载 **18 个文件**（`engram_*` / `dsa_v1.py` / `model.py` / `block_table.py` /
`ascend_forward_context.py` / `token_dispatcher.py` / `device_metadata.py` /
`patch_cudagraph.py` / `rope_dsv4.py`），**不含 `dsa_v41.py` 与 `indexer.py`**。

镜像版 `dsa_v41.py` 的 mtime 是 **Sep 11 09:19**（`docker cp` 保留容器内 mtime），
而 tiny 版是与我们 DCP/探针工作同步演进的 4636 行版本。
（注：镜像版本里只有 1 处 `dcp|merge_kernel` 命中，说明它**不含** DCP 合并与定序归约那套实现。）

## 2. 挂载条件（`serve_a2.sh`）

```bash
1266:    [ "${PROBE:-0}" != "1" ] || die "CED decode 实验不能与 PROBE=1 同时覆盖 dsa_v41.py"
1267:    _ced_dsa="$PKG/experimental/ced/dsa_v41.py"
1270:    MOUNTS+=(-v "$_ced_dsa:.../attention/dsa_v41.py:ro")
1321:  [ "${PROBE:-0}" != "1" ] || die "CED cache snapshot 不能与 PROBE=1 同时覆盖 dsa_v41.py"
1324:  _ced_dsa="$PKG/experimental/ced/dsa_v41.py"
1328: # PROBE=1 时用只读挂载覆盖 dsa_v41.py 并注入 sparse_capture.py
```

⇒ **只有开 CED decode 实验或 `PROBE=1` 时**，才会用 `experimental/ced/dsa_v41.py`（**1455 行、0 探针**）
覆盖镜像版。**普通交付启动不会覆盖它。**

## 3. 对我上一轮结论的更正

上一轮我写："**非确定性是 TP=8 特有**（tiny TP=2 逐位为 0）"。

**这个对比被两个变量同时污染**：

| 维度 | tp8k5 | tiny |
|---|---|---|
| TP | **8** | **2** |
| `dsa_v41.py` | **1092 行（镜像版）** | **4636 行（开发版）** |
| `indexer.py` | 244 行（镜像版） | 439 行（开发版） |

⇒ 正确表述应为：
> 【实测】tp8k5（TP=8 + 镜像版 attention）decode 非确定；
> tiny（TP=2 + 开发版 attention）decode 确定。
> 【未确认】差异究竟来自 **TP 规模**、还是**两套 attention 实现**，或两者共同作用。

另外，我此前引用的"本仓已有 HCCL allreduce 定序记录"（`_v41_ordered_allreduce`、
`det_reduce` 等）**属于 tiny 的开发版**，**不在 tp8k5 的运行代码里** ——
所以那条线索**不能直接用来解释 tp8k5 的抖动**，只能作为"同类问题在我们代码里出现过"的参考。

## 4. 为什么 `idxdet` 探针没触发（顺带记录）

我在 tp8k5（eager）里写了 `/tmp/v41_perf_flags` 并置 `idxdet=1`，然后发了一条 3216-token 的请求：

```
容器内 grep -c _perf_flags .../models/deepseek_v41/indexer.py  ⇒ 0
容器内 grep -c idxdet       .../models/deepseek_v41/indexer.py  ⇒ 0
```

⇒ **探针代码根本不在 tp8k5 的运行文件里**，所以标志位写了也不会打印。
这不是探针失效，而是**探针不可用**。

## 5. 这对项目意味着什么（需要决策）

1. **交付件到底是哪一版 attention？** 如果 tp8k5 是"交付基线"，那它跑的是
   **Sep 11 的镜像版 `dsa_v41.py`**，而不是我们仓库里持续演进的 4636 行版本。
   两者相差 **3658 行 diff**。
2. **这意味着我们近期在 `dsa_v41.py` 上做的所有工作（DCP 合并、定序归约、各种探针）
   默认并不在交付里生效** —— 只有开 CED 模式才挂载（而且挂的是 1455 行的另一份）。
3. **诊断能力受限**：仓库里那套 `_perf_flags` 探针体系无法直接用于 tp8k5 的排障。
   要么把探针移植到镜像版/`experimental/ced` 版，要么用 CED 模式起服再测。
4. 【建议】先明确"哪一份 `dsa_v41.py` 是交付件"，再决定非确定性的修法落在哪份代码里。
   否则可能出现"在 A 文件里改了、交付跑的是 B 文件"的静默失效 —— 本仓历史上已经踩过
   同类坑（`patches/files/draft/dsa_v1.py` vs `patches/files/dsa_v1.py`）。

## 6. 复现

```bash
# 容器内文件规模与探针数
ssh a3-21 'docker exec dsv41-tp8k5 bash -lc "for p in attention/dsa_v41.py   models/deepseek_v41/indexer.py attention/dsa_v1.py; do   printf \"%-40s lines=%-7s probes=%s\\n\" \$p   \$(wc -l < /vllm-workspace/vllm-ascend/vllm_ascend/\$p)   \$(grep -c _perf_flags /vllm-workspace/vllm-ascend/vllm_ascend/\$p); done"'
# 挂载清单
ssh a3-21 'docker inspect --format "{{json .Mounts}}" dsv41-tp8k5 | python3 -c   "import sys,json;[print(m[chr(34)+chr(83)+chr(111)+chr(117)+chr(114)+chr(99)+chr(101)+chr(34)],chr(45)+chr(62),m[chr(34)+"Destination"+chr(34)]) for m in json.load(sys.stdin)]"'
# 镜像版存档
ssh a3-21 'docker cp dsv41-tp8k5:/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py   ~/tmp/imgcode/dsa_v41_image.py'
```
