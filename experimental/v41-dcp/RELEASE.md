# V4.1 Flash — TP8 + DCP8 overlay 发布包

> DeepSeek-V4.1 在 **8×910B（单机 8 芯，A2 形态仿真）** 上启用
> `--decode-context-parallel-size 8` 的一套可挂载实现。
> 目标：把 KV cache 容量做到 DCP1 的 **~8 倍**，同时保持正确性、decode 性能不出现数量级退化。

---

## 0. 一句话结果

| | DCP1 基线 | **DCP8（本包）** | 比值 |
|---|---|---|---|
| **KV 容量**（5 GiB 池，1M 上下文） | 1,242,687 token | **8,634,871 token** | **6.95×** |
| **ms/step**（单流，BAT=2048，SPEC=0） | 30.83 | **35.00 / 35.18 / 35.35**（三次独立会话，中位 **35.18**） | 1.141× |
| **tok/s**（单流，API 口径） | 31.1 – 31.4 | **28.57 / 28.43 / 28.29** | 0.90× |
| **A**（平均接受长度） | 1.0 | 1.0 | — |
| **长上下文多选针** 2K/8K/16K × 2 种 | — | **6/6** | — |
| **短问答** | 4/6 | 4/6（逐项一致） | — |

容量换来了 **1M 上下文下 1 路 → 8 路**并发（或 128K 下 9 路 → 65 路）。

口径：TP8/DP1、`BAT_TOKENS=2048`、`--no-async-scheduling`、`PREFIX=0`、`MAX_LEN=1M`、
5 GiB 池、设备 8-15；性能取流式相邻 token 间隔**中位数**（丢弃前 8 个）。

---

## 1. 这个包是什么

**不是**镜像，是 **12 个 `.py` 的整树挂载（overlay）**。`scripts/serve_a2.sh` 支持
`V41_DCP_MOUNT=<dir>`：起服时把它们 `-v` 挂进容器的
`/vllm-workspace/vllm-ascend/vllm_ascend/...`，并有**三重保险**：

1. 起服前打印挂载清单；
2. 起服后**逐文件 md5 比对**容器内 vs 宿主，不一致直接 die；
3. 与 `patches/files/*` 的重复目的地自动去重（overlay 优先）。

⇒ 所以这个包**不需要重打镜像**，也不需要 `docker commit`。

```
overlay/vllm_ascend/                        # ← V41_DCP_MOUNT 指向这里
├── attention/dsa_v41.py                    # 主体：SMLA 调用 + 跨 rank LSE 合并
├── attention/context_parallel/v41_dcp.py   # 纯函数：坐标映射 / top-k remap / 本地长度
├── worker/block_table.py                   # DCP 槽位映射（写侧）
├── core/kv_cache_interface.py              # 四个 cache 平面的 DCP 语义
├── core/deepseek_v41.py                    # 容量规划（分片 vs 复制）
├── models/deepseek_v41/{model,indexer,engram_hbm}.py
├── patch/platform/{patch_v41_dcp,patch_kv_cache_coordinator,__init__}.py
└── platform.py
launch/dcp_stage_capacity.sh                # 一键起服（选卡 + 重试 + 抓容量行）
tools/dcp_sync.sh                           # 本地 overlay → 远端，镜像式同步 + md5 守门
tools/dcp_correctness.py                    # 正确性回归（短问答 + 长上下文多选针）
tools/dcp_ab.py                             # 性能 A/B（(ms/step, A, tok/s) 三元组）
tools/v41_capacity_sweep.py                 # 纯解析容量模型（秒级，无需设备）
```

---

## 2. 怎么用

```bash
# ① 解包
tar -xf v41-dcp-overlay-<commit>-<指纹>.tar.zst -C /some/dir
cd /some/dir && sha256sum -c MANIFEST.sha256      # 逐文件校验

# ② 起服（8 芯，DCP8）
cd <你的 dsv41 发布包>            # 需要 scripts/serve_a2.sh / serve_a3.sh / serve_v2.sh
setsid env \
  DCPMOUNT=/some/dir/overlay \
  AUTO_CHIPS=0 CHIPS="8 9 10 11 12 13 14 15" \
  PREFIX=0 BAT_TOKENS=2048 \
  EXTRA_KV_ARGS="--no-async-scheduling" \
  DCP_EXTRA_ENV="V41_DCP_ALLOW_CAPACITY_PROBE=1" \
  nohup bash launch/dcp_stage_capacity.sh > ~/dcp.nohup.log 2>&1 < /dev/null &
# 冷启动 12–20 分钟（8 rank 加载 490 GB + 编译 150~176 个 static kernel）

# ③ 等就绪
curl -s --noproxy '*' http://127.0.0.1:19210/health     # 期望 200

# ④ 正确性 + 性能
python3 tools/dcp_correctness.py     # 期望：长上下文多选针 6/6
python3 tools/dcp_ab.py ab --reps 5 --arm base:          # 期望：~35 ms/step，A=1
```

**关键开关**（都有默认值，通常不用改）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `V41_DCP_MOUNT` | — | **必给**，指向 `overlay/` |
| `V41_DCP_ALLOW_CAPACITY_PROBE` | — | **必给 `1`**，否则 DCP 路径不激活 |
| `EXTRA_KV_ARGS` | — | 建议 `--no-async-scheduling`（容量第一杠杆） |
| `BAT_TOKENS` | 8192 | 2048 可把容量从 4.08× 提到 6.95× |
| `CED_MAX_NUM_BLOCKS` / `CED_BYTES_PER_BLOCK` | 29076 / 540928 | 防 32 位页偏移回绕的上界 |

---

## 3. 实现要点（改了什么）

V4.1 有**四个 cache 平面**，DCP 下语义**不同**，这是整套实现的核心：

| 平面 | DCP 语义 | 为什么 |
|---|---|---|
| `long_kv`（压缩态 MLA） | **分片** 1/dcp | 每 rank 只存 1/8 序列 |
| `indexer.k_cache` | **分片**（配 remap） | 同上 |
| `swa`（滑窗 128） | **复制**（每 rank 全量） | A3 上 `ori` 路径被硬绑：`ori_win_left` 必须 127、`ori_mask_mode` 必须 4、`ori_sparse_indices` 是 A5-only ⇒ **滑窗只能表达成"以本地 KV 末端为右沿的连续带"**；而且它**不贵**（滚动窗口，与序列长度无关） |
| `compressor.state_cache` | 复制 | 环状缓冲，非 full |

**跨 rank 合并（`_v41_dcp_merge_attention`）** —— 这是最容易做错的地方：

```
每个 rank 上报 (O_r, L_r)，全局 = Σ_r e^{L_r}·O_r / Σ_r e^{L_r}
```

三个必须遵守的细节（都踩过）：

1. **必须沿 head 维 all-gather q**。TP 切的是 head，rank r 只持 `[8r, 8r+8)`；
   `all_reduce` 是**逐元素**求和 —— 不 gather 就等于把不同 head 相加。
2. **归一化参考点必须各 rank 共享**。取"跨 rank 最大值"需要一次 all_gather
   （实测 46 µs/层）；改用**第二次纯 ori 调用的 LSE** 作参考点 ⇒ **零通信**
   （各 rank 逐位相同，且 `L_r ≥ L_ori` 恒成立，指数不上溢）。
3. **`(1 − 1/dcp)` 要折进 ori 项**：`all_reduce` 线性 ⇒
   `Σ_r(_onum_r·k) = k·_n_all`，把 k 先乘到 `[T,H,1]` 上，
   归约后的重算子从 2 个降到 1 个。

**已知的坑（写在这里省你一轮起服）**：

* `_prepare_q_for_dcp` 必须与 `_native_attention` 里的
  `dcp_active = _v41_dcp_on() and compress_ratio in (1,2)` **同条件**。
  ratio=0 的层（纯滑窗）不做跨 rank 归约，q 必须保持本地 8 head；
  无条件 gather 会让算子报
  `Invalid_Argument_Tensor_Shape(EZ0009): ... Sinks's dimension(8) should be equal to
  the head num of query(64)`。
* **capture 区内绝不能做 host 同步**：`int(device_tensor)` / `.item()` / 布尔索引
  会让 8 个 worker 全部报 `Not_Supported(EE1016): stream is captured`。
  凡是要读 device 标量的地方，一律走纯 device 侧构造。
* 改文件驱动的消融开关（`/tmp/v41_perf_flags`）**对 decode 阶段无效** ——
  decode 走 `FULL_DECODE_ONLY` 整图捕获，Python 分支只在捕获那一刻求值。
  要测 decode 增量只能用**设备侧 profile** 或**离线图级夹具**。

---

## 4. 性能：+4.5 ms/step 花在哪

DCP8 相对 DCP1 慢 **+4.52 ms/step**。按 38 层（层 0/1 是纯滑窗，不走合并路径）
折算是 **+119 µs/层**。2-chip 图级**累积式**分解（可加，自证 122.1 + 14.2 = 136.3）：

| 组件 | µs/层 | 通信? |
|---|---|---|
| 后处理（`sub`/`slice`/`clamp_min`/`div`/`cast`） | 42.2 | 否 |
| 第一次 SMLA（ori⊕cmp，64 head） | 27.4 | 否 |
| q 的 head all_gather | 20.6 | ★ |
| 打包 all_reduce | 18.0 | ★ |
| 第二次纯 ori SMLA | 14.8 | 否 |
| 残差 / 布局 | 16.4 | — |
| **合计** | **136.3** | |

⇒ **集合通信只占约 1/3（两个 collective 一起去掉只省 36.5 µs/层）**。
优化前是 79%（q gather 48.3 + LSE gather 46.1 + all_reduce 57.4 = 151.8），
本轮把 LSE gather **完全消掉**、q gather 提到 indexer 之前与计算重叠。

**两项"别再优化"的硬结论**（实测）：

* q 的 head all_gather **已在地板上**：空 collective（16 B）46 µs vs 真实形状（8 KB）
  46 µs ⇒ **纯每-call 延迟，0% 是数据量**；要到 8 MB 才爬到 74。
* **all-to-all 替代不可行**：合并需要"固定 head 切片、对全部 KV 分片求和"，
  只能整行（要 gather q）或整列（要读全部 8 份 KV）；"只算 1/8 head"
  等于只算 8×8 块矩阵的对角块。

**聚合吞吐不会下降**【推断】：collective 是**纯延迟**（与 batch 无关），
并发越高摊得越薄；而容量 6.95× 直接把 1M 上下文的并发上限从 **1 路提到 8 路**。
⚠️ 这条是推断 —— 没有 C>1 的 DCP8 实测数据。

---

## 5. 结论标注

* 【实测】容量 8,634,871 token（三次逐位验证）、长针 6/6、短问答 4/6、
  35.00/35.18/35.35 ms/step、28.57/28.43/28.29 tok/s、A=1、collective 地板值、
  组件表全部来自真机或 2-chip 图级夹具。
* 【推断】聚合吞吐不会下降（依据：collective 纯延迟 + 容量比）。
* 【未确认】C>1 的 DCP8 实测；8-chip 上 collective 的真实边际（2-chip 夹具低估）。
