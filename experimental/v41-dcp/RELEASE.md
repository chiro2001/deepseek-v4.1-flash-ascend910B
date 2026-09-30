# V4.1 Flash — TP8 + DCP8 overlay（a3-21 / 8×910B 单机）

> **状态（2026-09-30 19:30）：仅供内部使用，未对外发布。**
> ① 容量与 ③ 性能达标；**② 正确性受 `npu_sparse_flash_mla` 算子缺陷阻塞**
> （`T ≥ 560` 的 prompt 存在概率性乱码）。算子侧已拿到 20 秒可复现的同源用例。
> 详见 §2 与 `../../docs/V41-DCP-RCA-20260930.md`。

本包通过 `V41_DCP_MOUNT=<dir>` 把 18 个 `.py` 整树挂进官方镜像
（`scripts/serve_a2.sh` 带 md5 守门），**不改镜像**。

---

## 0. 一句话结果

| | DCP1 基线 | **DCP8（本包，正确配置）** | 比值 |
|---|---|---|---|
| **KV 容量**（5 GiB 池，1M 上下文） | 1,242,687 token | **6,082,458 token** | **4.90×** |
| **ms/step**（同会话配对，BAT=2048，SPEC=0） | 26.12 | **33.77** | **1.29×** |
| **tok/s**（单流 API 口径） | 38.28 | **29.61** | 0.77× |
| **A**（平均接受长度；SPEC 关闭） | 1.0 | 1.0 | — |
| **短问答 / 同 prompt 逐字对拍** | 112/112 | **112/112 逐字相同** | — |
| **长上下文多选针** | 6/6 | **T ≤ 450 通过；T ≥ 560 失败（算子缺陷）** | — |

口径：TP8/DP1、`--decode-context-parallel-size 8`、`BAT_TOKENS=2048`、
`--no-async-scheduling`、`PREFIX=0`、`MAX_LEN=1M`、5 GiB 池、设备 8–15；
性能取流式相邻 token 间隔**中位数**（丢弃前 8 个，5 次取中位）。

> 未开 indexer 复制时容量是 **8,634,871 token（6.95×）**，但那条路径
> `top-k` 不成立（分片 indexer 上 QLI 的因果掩码无法表达）⇒ **不可用于正确性**。
> 本包的 4.90× 是**正确配置下的容量**。

---

## 1. 启动

```bash
# ① 同步 overlay 到宿主目录（本包解包后即为该目录）
tar -I zstd -xf v41-dcp-overlay-<commit>-<md5>.tar.zst -C ~/dcpw

# ② 起服（chip 8-15、端口 19210）
cd ~ && setsid env DCPMOUNT=$HOME/dcpw AUTO_CHIPS=0 \
  CHIPS="8 9 10 11 12 13 14 15" PREFIX=0 BAT_TOKENS=2048 EAGER=1 ENGRAM=0 \
  EXTRA_KV_ARGS="--no-async-scheduling" \
  DCP_EXTRA_ENV="V41_DCP_ALLOW_CAPACITY_PROBE=1 V41_DCP_REPLICATE_INDEXER=1" \
  nohup bash launch/dcp_stage_capacity.sh > ~/dcp_x.nohup.log 2>&1 < /dev/null & disown

# ③ 就绪判据（冷启动 5–20 分钟）
curl -s -o /dev/null -w '%{http_code}\n' --noproxy '*' http://127.0.0.1:19210/health
```

**两个环境变量是必需的**：
* `V41_DCP_ALLOW_CAPACITY_PROBE=1` —— 放开 V4.1 的 `PP=DCP=PCP=1` 门；
* `V41_DCP_REPLICATE_INDEXER=1` —— **indexer K 每 rank 一份全量副本**，
  这是全局 top-k 成立、长上下文正确的前提（代价是容量 6.95× → 4.90×）。

---

## 2. 已知限制：② 正确性（外部算子缺陷）

`npu_sparse_flash_mla`（SMLA）在 **`compress_ratio=1` 的稀疏索引路径**上**非确定**：
同一次 forward 内、**逐位相同**的输入连调两次，`lse`/`out` 不同，并出现 NaN。

* 【实测】生产 8-chip DCP8：`T ≤ 450` 完全确定（`lse_bit_identical=True`）；
  `T ≥ 560` 起非确定，且 **NaN 在两次调用间随机出现** ⇒ 读未初始化内存。
* 【实测】单卡隔离复现（无 DCP、无服务栈）：值域跨 ≥3 页必现；生产同源 dump
  在单卡上 100% 复现（`ALL`/`STEADY` 均 `bit_identical=False`）。

**复现包（公开可读）**：
```
cos://uploads-new/share/dsv41-dcp8-smla-nondeterminism-repro-20260930.tar.zst
```
内含生产 dump（2 个 rank）、单卡重放脚本、README（复现命令 + 实测表 + 已排除假设）。

**本轮已穷尽的规避尝试（全部失败）**：降 topk（算子拒绝非 512/1024）、
匀均重复填索引槽（`floor`/`ceil`）、索引升序重排、dense 替换稀疏、
SMLA 前强制同步、分块调用、块表补列、换新 metadata 副本。

---

## 3. 包内结构

| 路径 | 说明 |
|---|---|
| `overlay/vllm_ascend/` | 18 个 `.py`：`attention/dsa_v41.py`（DCP impl/builder/LSE 合并/top-k remap）、`core/deepseek_v41.py`（槽位规划/state ring/复制面）、`worker/block_table.py`、`patch/platform/*` 等 |
| `launch/dcp_stage_capacity.sh` | 起服脚本（含设备安全检查、md5 守门、结果归档） |
| `tools/dcp_correctness.py` | 正确性回归（短问答 + 长文多选针） |
| `tools/dcp_ab.py` | 性能 A/B（三元组口径） |
| `tools/dcp_sync.sh` | overlay 同步（md5 守门） |
| `probes/` | 算子缺陷复现包：单卡最小复现、生产 dump 重放、输入变异、触发条件刻画 |
| `package_release.sh` | 本包的构建/校验/自检器（`build` / `verify` / `selftest`） |

## 4. 校验本包

```bash
sha256sum -c v41-dcp-overlay-<commit>-<md5>.tar.zst.sha256   # 归档完整性
bash package_release.sh verify v41-dcp-overlay-<commit>-<md5>.tar.zst  # 逐文件 + 篡改负控
```

打包参数固定（`owner/group=0`、`--mtime` 取提交时间、`--sort=name`）
⇒ **同一份输入必然得到同一个 sha256**，可当"这就是我跑的那版"的身份。
