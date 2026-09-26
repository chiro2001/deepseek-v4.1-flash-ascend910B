# dsv41-a3-ced-pd —— **A3 单机 8+8 PD 分离 + CED 形态** 的镜像层 patch 包

本包把 A3 上**已验证可跑**的「PD 分离 + CED」工作形态，拆成：

```
官方基础镜像（内网已有，18 层，24.9 GB）
        +   我们的 14 个“工作层”（本包携带，压缩后 1.2 MiB）
= local/dsv41-a3-ced-pd:v1
```

**不需要 skopeo、不需要联网、不需要 NPU** 就能重建；
`bash rebuild.sh` 会做逐步验收（见 §3）。

> 与 GitHub 侧启动器的关系：两者装的是**同一批文件**，
> 清单见 `deploy/a3-ced-pd/PAYLOAD.md`，逐文件 md5 比对由
> `deploy/a3-ced-pd/verify_consistency.sh` 完成（本包实测 **30/30 一致**）。

---

## 一、这是什么部署形态

| | |
|---|---|
| 硬件 | **单台 A3，8 块 910C 卡 = 16 个 die**（P 用 die 0–7 = 卡 0–3，D 用 die 8–15 = 卡 4–7） |
| 拓扑 | P（prefill）/ D（decode）/ proxy 三个容器，**PD 分离** |
| P 侧 | 只跑 **layer 0–19 + layer-20 全局源投影**（不是全 40 层） |
| D 侧 | **128-token 有界重放 + 全 40 层** decode，开 **DSpark** 推测解码 |
| KV 精度 | **BF16** |
| 上下文 | 1M（`MAX_LEN=1048576`） |

**为什么 P 只跑 20 层**：D 需要的全局 KV 只来自层 2/8/14/20（`kv_source_layer_ids`），
且层 ≥20 的 SWA 只依赖最近 128 个 token ⇒ P 算完前 20 层 + 层 20 的源投影即可，
其余交给 D 在重放步里补齐。prefill 计算量按层数比下降，实测 **2.07×**。

**为什么 DSpark 只在 D 侧**：DSpark 的草稿层从 target 的 **37/38/39 层**残差取隐状态，
而 P 在第 20 层就 break —— 这三层在 P 上物理不存在。这是拓扑问题，不是配置问题。

## 二、怎么用（三选一）

```bash
cd images/dsv41-a3-ced-pd

# ① 标准：校验基底 → 重建 → 文件系统验收 → 冒烟（推荐）
bash rebuild.sh

# ② 你们的内部镜像源名不同
BASE_IMAGE=<你们内网的基底名> bash rebuild.sh

# ③ 只要镜像，不做验收（快）
SKIP_SELFTEST=1 bash rebuild.sh
```

重建出来的镜像默认 tag `local/dsv41-a3-ced-pd:v1-rebuilt`；也可以：

```bash
NEW_TAG=dsv41-a3-ced-pd:v1 bash rebuild.sh
docker save dsv41-a3-ced-pd:v1 | ssh <内网A3> 'docker load'
```

### 起服

```bash
export MODEL=<DeepSeek-V4.1-Flash W4A8+DSpark 模型目录>   # 273 GB，需现场准备
export PATCH_MODE=baked                                    # ★ 用镜像里烘好的文件

# 两种写法都行：
bash deploy/a3-ced-pd/launch/serve_p.sh    &&  bash deploy/a3-ced-pd/launch/serve_d.sh
# 或者直接用镜像内的脚本：
#   P: DEVS="0 1 2 3 4 5 6 7"  PORT=18990 KV_PORT=19090 V41_CED_ROLE=prefill bash /opt/dsv41/scripts/serve_a3_ced_pd.sh prefill
#   D: DEVS="8 9 10 11 12 13 14 15" PORT=18991 KV_PORT=19091 V41_CED_ROLE=decode  bash /opt/dsv41/scripts/serve_a3_ced_pd.sh decode
# 再起官方 load_balance_proxy（端口 18992）
```

## 三、`rebuild.sh` 的验收标准（**不是"能启动就算过"**）

| 判据 | 标准 |
|---|---|
| ① 基础镜像 | `docker inspect` 的 **18 个 layer diffID 与打包时逐条相同**；不同 ⇒ 直接 FATAL（说明不是同一基底） |
| ② 文件系统 | 重建镜像的根文件系统清单 **305,787 条，与原镜像逐条相同**（覆盖全部路径与类型） |
| ③ 工作层文件 | **90 个 payload 文件**逐个 sha256 一致 |
| ④ 冒烟 L1（不需 NPU） | payload md5 清单逐条、补丁件 `py_compile`、起服脚本在位 |
| ⑤ 冒烟 L2（需 NPU） | `import vllm`（挂了设备才跑，无卡自动跳过） |

原始清单由打包工具在 a3-21 上**从原镜像现场生成**，随包携带：
`fs-manifest.original.txt.gz`、`payload-sha256.original.txt.gz`、`checks/payload-md5.txt`。

## 四、起服后的**硬门**（不看这些就别相信结果）

按"漏了会怎样"排序：

```bash
# ① 池钳位：漏了 → 1M 静默空答（HTTP 200 + 1 token EOS）
grep -a "CED-32BIT-GUARD" d/serve.log | head -1
#   期望 num_blocks=29076 max_page_stride=147712

# ② 图模式前提：漏了 → 144K 多轮乱码（打满 max_tokens + 含 <｜box｜>）
grep -ac "one-token prompt tail forced eager" d/serve.log      # 期望 > 0

# ③ D 侧必须关多流：否则长上下文静默算错（实测 0/4）
docker exec <d> bash -lc 'echo $MULTISTREAM $DSA_OVERLAP'      # 期望 "0 0"

# ④ draft 图 metadata：漏了 → A≈1.0（草稿白跑），**ms/step 看不出问题**
grep -a "dspark-graph-capture" d/serve.log | head -1
#   期望 "... built draft attention metadata (groups=1 layers=3 ...)"
#   出现 "LEGACY metadata-less capture" 就是坏的

# ⑤ 接受长度（唯一能证明草稿在干活的判据）
grep -a "SpecDecoding metrics" d/serve.log | tail -1
#   期望 Mean acceptance length ≈2.4–3.4；**A≈1.0 就是草稿没产出**

# ⑥ 组拓扑：P 12 组、D 13 组
grep -a "CED decode: upper SWA groups" d/serve.log
#   期望 upper SWA groups=(7,8,9,10,11) draft(g12) groups=(12,) total_groups=13
```

⚠️ **判静态核不要用 `compile start`**：编译缓存命中时它是 0，会给**假阴性**。
真判据是 `static shape kernel will be used`（本镜像已把 `STATIC_KERNEL=1` 的
收益验过：**−4.4 ms/step，−9.6%**）。

## 五、本形态的**已验证**结果（A3-21 真权重）

| 项 | 144K | 1M |
|---|---|---|
| 四针 A/B/C/D | **4/4** | **4/4** |
| 流式 TTFT | 10.57 s | 100.04 s |
| 多轮（三轮） | **3/3** | **3/3** |
| 缓存命中（`PREFIX=0` 口径） | ✅ | ✅ |

**总计 21/21 通过**；四针答案与 `SPEC=0` 交付口径**逐字节相同**
（`ZQ7K-3341` / `VX2M-8890` / `HT4P-5527` / `RB9N-6014`）。

**性能**：prefill 相对全 40 层基线 **2.07×（144K）**；
decode 并发 4 时 **41.07 ms/step**（`STATIC_KERNEL=1`）；
单流（2048 prompt / 256 输出，《地火》）**99.7 tok/s**，A=2.76。

⚠️ **DSpark 的收益只在低并发成立**：并发 4 时它把 ms/step 从 28.2 抬到 41.1（**1.45×**），
换来 A≈2.4。收益集中在接受长度高的请求上、成本由全批承担。
高并发吞吐场景应保持 `SPEC=0`，或按并发自适应切换。

## 六、本包的来源（可复算）

| 项 | 值 |
|---|---|
| 基底 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（18 层，id `sha256:1f2c08195c5b…`） |
| 源镜像（本层从它切出） | `local/dsv41-a3-ced-pd:v1`（32 层，id `sha256:f9a66cf61e58…`） |
| 工作层 | **14 层**，压缩后 **1.2 MiB** |
| 文件系统条目 | 305,787 |
| payload 文件 | 90 |
| **代码基线** | 本仓 `feat/ced-pd-a3` @ `ffb75ca`（镜像内 `/opt/dsv41/BUILD_INFO.txt` 的 `repo_rev`） |
| 基底内的 vLLM 基线 | `vllm 6e448d0`（+1 dirty = admission gate）/ `vllm-ascend e43cf1e9f`（+24 dirty = 我们的补丁件 + `.cedpdorig` 备份） |
| 生成工具 | `scripts/make_image_patch_kit.py`（本仓 `main`），`--job 'dsv41-a3-ced-pd\|local/dsv41-a3-ced-pd:v1\|v41'` |
| 生成时间/机器 | 见 `MANIFEST.json`（a3-21） |

**镜像内自带可核对的东西**：

```bash
docker run --rm --entrypoint cat local/dsv41-a3-ced-pd:v1 /opt/dsv41/BUILD_INFO.txt
docker run --rm --entrypoint cat local/dsv41-a3-ced-pd:v1 /opt/dsv41/deploy/PAYLOAD.sha256
```

`PAYLOAD.sha256` 的 md5 与本仓 `deploy/a3-ced-pd/payload/PAYLOAD.sha256`
**逐字节相同**（实测三处 md5 全等：本仓重新生成 = a3-21 payload = 镜像内），
⇒ `仓库 → payload → 镜像` 这条链是闭合的。

## 七、这个包里到底装了什么（90 个 payload 文件）

| 组 | 数量 | 装到哪 |
|---|---:|---|
| vllm-ascend 补丁件 | 13 | `…/vllm_ascend/{models/deepseek_v41,ops,worker,…}/*.py` |
| **CED 件** | 2 | `…/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`、`…/attention/dsa_v41.py` |
| draft 版暂存 | 3 | `/opt/dsv41/patches/draft/`（`PATCH_MODE=baked` + `DRAFT_GRAPH=1` 时装到 live tree） |
| vLLM core 补丁 | 4 | `/opt/dsv41/ced_*.patch`、`/opt/dsv41/admission_gate.patch`（后者构建期已 apply） |
| 起服脚本 | 8 | `/opt/dsv41/scripts/`（含 `serve_a3_ced_pd.sh`） |
| 溯源 | 3 | `/opt/dsv41/BUILD_INFO.txt`、`/opt/dsv41/deploy/{PAYLOAD.sha256,build_smoke.py}` |
| 备份 | 13 | 每个被覆盖的 vLLM 文件旁边留 `.cedpdorig`（可回滚） |
| 其它 | 44 | `__pycache__` 等构建产物 |

> ⚠️ **三个 CED core 补丁故意留到运行期打**：`core_scheduler_replay.patch` /
> `_prefill_hit.patch` / `_runner_prompt_tail.patch` 各自带**基线 sha256 硬门**
> （scheduler.py / model_runner_v1.py 的固定哈希）。留到运行期 ⇒ 基底版本不同会
> **拒绝起服**，而不是静默跑错版本。烘进镜像反而绕过这层保护。

## 八、共用机注意

* **`CPU_BIND=0` 是必需逃生口**：目标 NUMA 节点被占满时 `migratepages` 会
  100% CPU 无限自旋、服务永不就绪、连 `docker stop` 都拿不到 exit event。
* **`DROPCACHE=0`**：A3 是共用机，默认不清整机 page cache（会打到别人）。

## 九、更细的文档

本包只保证"镜像可重建、内容可核对"。设计取舍、故障史与判据细节在仓库里：

| 想了解 | 看 |
|---|---|
| 整体交接 | `docs/CED-PD-HANDOVER-20260926.md` |
| 验收矩阵与判据 | `docs/CED-PD-ACCEPTANCE.md` |
| 性能 | `docs/CED-PD-PERF-20260925.md` |
| DSpark × CED 架构关系 | `docs/CED-PD-DSPARK-CED-RELATION-20260926.md` |
| decode 时延拆解（+17ms 从哪来） | `docs/CED-PD-DSPARK-LATENCY-BREAKDOWN-20260926.md` |
| **精度/乱码措施总账** | `docs/CED-PD-ACCURACY-MEASURES-20260926.md` |
| 32 位页步长上界 | `docs/CED-PD-BLOCK-BOUND-20260925.md` |
| 部署形态与 payload 清单 | `deploy/a3-ced-pd/{README,PAYLOAD}.md` |
