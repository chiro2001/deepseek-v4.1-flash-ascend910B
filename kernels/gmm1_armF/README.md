# gmm1 armF 自定义 kernel（交付内核，最小补丁集）

**这是什么**：`GroupedMatmulSwigluQuantV2`（MoE 第一跳，40 次/步）的重编内核。
改动是 **减少 AIC↔AIV 的跨核同步轮次**（armF：SyncAll 4→2 / Reset 5→2，单段直线路径）。

| 项 | 值 |
|---|---|
| 算子 | `_C_ascend::grouped_matmul_swiglu_quant_v2` |
| 内核文件 | `GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.o` |
| `.o` md5 | **`01f600d2e49959145089c165839c9579`** |
| `.json` md5 | `4d6795143a4cf62568e476306b3e1ede` |
| 资源指纹 | `_2_mix_aic = 37328`（判定"内核真的在跑"，见 `docs/KERNEL-CACHE-STALE-20261004.md`） |
| 实测收益 | 算子 **−7.56 µs/次**；端到端 **−0.67 ~ −1.09 ms/步**（≈3%），同内核对照漂移仅 −0.03 |
| 正确性 | 144K 四针 + 流式 + 多轮 + prefix **11/11 PASS**；`regress2` 5/6（与基线同） |
| 来源 | gmm1 子代理线交付的完整 vendor 包 `~/tmp/gmm1/pkg_armF`（本目录是**它的最小补丁集**） |

## 为什么是"最小补丁集"

完整包（`~/tmp/gmm1/pkg_armF`，64 MB）里是**整棵 vendor 树**（1136 个文件），
但逐文件 md5 对比镜像原生 vendor 后，**只有 2 个文件内容不同**（本文这两个）：

```bash
# 复算（在 a3-21 上；需要 docker 读镜像原生 vendor）
docker run --rm --entrypoint bash <BASE> -lc \
  'cd /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer && \
   find . -type f -print0 | xargs -0 md5sum' | sort -k2 > /tmp/stock_md5.txt
sudo bash -c 'cd <PKG>/vendors/custom_transformer && find . -type f ! -name "*.bak*" -print0 | xargs -0 md5sum' \
  | sort -k2 > /tmp/pkg_md5.txt
# 差集 = 本目录的 2 个文件
```

⇒ **115 KB 就能完整复现这个 kernel**，不必把 64 MB 塞进版本库。

## 怎么用（两条路，都能落地）

### 路 A：作为 `V41_HC_OPP_PKG`（开发/挂载模式，当前交付实例用的就是它）
`patches/opp_override_block.sh` 会在容器启动时把
`$V41_HC_OPP_PKG/vendors/custom_transformer/.` 覆盖到镜像 vendor，
所以只要**保持目录层级**指向本目录即可（本目录已按 `vendors/custom_transformer/...` 组织）：

```bash
export V41_HC_OPP_PKG=$PKG/kernels/gmm1_armF
```

⚠️ **必须同时清 static kernel 缓存**（否则运行时可能复用旧内核，改动静默不生效）：

```bash
export V41_OPP_CLEAR_SKCACHE=1     # 起服会多花 ~8–18 min 重编译
```

### 路 B：烘进镜像层（部署形态，推荐给内网新机器）
`tools/build_a3_tp8_image.sh` 已经会**自动**把本目录并入工作层
（落在 `…/vendors/custom_transformer/…` 真实路径）并在自检里逐字节核对：

```bash
bash tools/build_a3_tp8_image.sh          # ⇒ local/dsv41-a3-tp8:<时间戳>（19 层）
```

实测（2026-10-05）：`local/dsv41-a3-tp8:20261005-0633`

| 镜像 | 该 `.o` 的 md5 |
|---|---|
| 基础镜像 | `2e4a834a21a2b104bc8a7703c5fc65a4`（stock） |
| 工作层镜像 | **`01f600d2e49959145089c165839c9579`**（= armF） |

⇒ **即使不传 `V41_HC_OPP_PKG`，部署方拿到的也是带该 kernel 的镜像**。

## 怎么验（三道，缺一不可）

```bash
# ① 文件：容器内 md5 == 上面那个值
docker exec <name> md5sum <容器内路径>/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.o

# ② 执行：资源指纹（内核名不可用于区分版本！文件名哈希由 op 定义决定）
#    采一份 profile，看该算子的 _2_mix_aic（stock 与 armF 不同）——见 docs/KERNEL-CACHE-STALE-20261004.md

# ③ 端到端：144K 四针 + 并发 2 不同针（tools/ced_pd_acceptance.py --mode all）
```
