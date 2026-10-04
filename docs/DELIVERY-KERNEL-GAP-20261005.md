# 交付物缺口修复：验证过的自定义 kernel 当时**不在任何发布件里**（2026-10-05）

> 起因：本轮顺手核对"发布物与实测配置是否一致"时发现 ——
> 我们在交付实例上用的 gmm1 armF 内核（≈3% 收益、144K 11/11 验证过）
> 只存在于 **a3-21 的 `~/tmp/gmm1/pkg_armF`（64 MB 临时目录）**，
> **既不在 git 仓库里，也不在已发布的镜像补丁包里**。
> 后果很隐蔽：别人按 README 部署，镜像能起来、功能正常，但**性能少 3%**，
> 而且没有任何报错提示。

## 1. 问题定性

| 项 | 状态（修复前） |
|---|---|
| 交付内核 | `GroupedMatmulSwigluQuantV2`（armF：SyncAll 4→2 / Reset 5→2） |
| 存放位置 | `~/tmp/gmm1/pkg_armF/`（完整 vendor 树，64 MB，**临时目录**） |
| git 仓库 | ❌ 无 |
| 镜像补丁包 `dsv41-a3-tp8-imagekit-v1` | ❌ 无（它只烘 `-v` 挂载清单；OPP 包是**容器启动时**才覆盖的，不在清单里） |
| 实测收益 | 算子 −7.56 µs/次 ⇒ 端到端 **−0.67 ~ −1.09 ms/步** |
| 证据 | `docs/FINAL-R5-20261005.md`、`docs/KERNEL-CACHE-STALE-20261004.md` |

**根因**：OPP 覆盖是"运行期行为"（`patches/opp_override_block.sh` 在容器启动时 `cp -a`），
所以**挂载清单里看不到它** ⇒ 镜像补丁包会静默漏掉。

## 2. 修复（三件套）

### 2.1 把内核纳入仓库（最小补丁集，115 KB）

逐文件 md5 对比"完整包 vs 镜像原生 vendor"（1136 个文件）后，**只有 2 个文件不同**：

| 文件 | md5 |
|---|---|
| `…/grouped_matmul_swiglu_quant_v2/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.o` | **`01f600d2e49959145089c165839c9579`** |
| 同名 `.json` | `4d6795143a4cf62568e476306b3e1ede` |

⇒ 仓库新增 `kernels/gmm1_armF/`（按 `vendors/custom_transformer/…` 层级组织，
可直接当 `V41_HC_OPP_PKG` 用），并写清两道用法与三道验收（见该目录 `README.md`）。
`.gitattributes` 把 `.o/.json` 标为 **binary**。

### 2.2 镜像补丁包构建器：自动并入 + 逐字节自检

`tools/build_a3_tp8_image.sh` 新增 **③b 步**：

* 把 `kernels/gmm1_armF/vendors/custom_transformer/.` 复制进 build context 的
  **最终容器路径**（`/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/…`）；
* 自检阶段**额外**逐字节核对这批 OPP 文件（原先只核对挂载件）；
* 若缺 `kernels/gmm1_armF`，打 **WARN 并明说"gmm1 armF 的 ≈3% 会回退"**（不再静默）。

### 2.3 产出并验证新镜像与发布包

```
bash tools/build_a3_tp8_image.sh
  ⇒ local/dsv41-a3-tp8:20261005-0633（base 18 层 + 工作层 1 层 = 19 层）
     · 挂载件 89 个文件全部逐字节一致
     · OPP 内核 2 个文件逐字节一致
     · 反例对照：engram_hash.py base a4287dd5 → new 56dea733（这层确实改了东西）
```

**内核级的反例对照（本轮新增判据）**：

| 镜像 | `…fa3d6d3d….o` 的 md5 |
|---|---|
| 基础镜像 | `2e4a834a21a2b104bc8a7703c5fc65a4`（stock） |
| 工作层镜像 | **`01f600d2e49959145089c165839c9579`**（= armF） |

⇒ **即使不传 `V41_HC_OPP_PKG`，部署方拿到的也是带该 kernel 的镜像**。

发布包（层补丁，给内网新机器）：

```
sudo python3 tools/make_image_patch_kit.py --out ~/dsv41-kit-v2 \
     --base quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
     --job 'dsv41-a3-tp8|local/dsv41-a3-tp8:20261005-0633|both'
  ⇒ 1 个工作层、1.2 MiB 压缩、89 个 payload 文件
  ⇒ payload-paths.txt / payload-sha256.original.txt.gz 里**明确含**那 2 个内核文件
```

## 2.4 ★ 发布镜像的端到端验证（不传任何 OPP 环境变量）

只做文件级 md5 还不够（镜像可能"文件对了但起不来"）。本轮的收口实验：
**用新镜像起一个完整服务，完全不设 `V41_HC_OPP_PKG`**，看三件事。

| 判据 | 结果 |
|---|---|
| 能否起服 | ✅ `health=200`、`/v1/models 200`（约 15 min，含 static kernel 冷编译） |
| 容器内该内核的 md5 | ✅ **`01f600d2e49959145089c165839c9579`**（= armF，不是 base 的 `2e4a834a`） |
| 是否有运行期 OPP 覆盖 | ✅ 驱动日志里 **`OPP-OVERRIDE` 出现 0 次** ⇒ 内核**来自镜像层**，不是启动时拷进去的 |
| 正确性 | ✅ `ced_pd_acceptance --mode all`（144K）**11/11 通过，失败 0** |

```bash
# 复现（关键：**不要**设 V41_HC_OPP_PKG）
sed -e 's|^export IMAGE=.*|export IMAGE=local/dsv41-a3-tp8:20261005-0633|' \
    -e '/V41_HC_OPP_PKG/d' ~/tmp/launch_armF.sh > ~/tmp/launch_armIMG.sh
bash tools/run_arm_suite.sh ~/tmp/launch_armIMG.sh armIMG_v2 0 0 1
docker exec <name> md5sum <容器内 .o 路径>          # 期望 01f600d2…
grep -c "OPP-OVERRIDE" results/armIMG_v2/driver.log  # 期望 0
```

⇒ **"仓库 → 镜像层 → 部署"这条链已经端到端打通**：部署方拿到镜像就能获得
与我们在 a3-21 上实测**同一份**内核，不需要任何额外环境变量或临时目录。

## 3. 遗留（明确列出，不含糊）

| 项 | 状态 |
|---|---|
| `deploy/a3-ced-pd/`（CED-PD 16-die kit） | **仍未含**该内核。原因：gmm1 armF 是在 TP8 形态上验证的；CED-PD 的 D 侧虽然也有 MoE，但没有实测过 ⇒ 要纳入必须先跑一次 CED-PD 的 A/B |
| 运行期 `V41_HC_OPP_PKG` 路径 | 保留（开发/挂载模式仍用它），但它**不是**部署路径；新部署一律走镜像层 |
| 静态内核缓存 | 任何"换 kernel"的场景**都必须** `V41_OPP_CLEAR_SKCACHE=1`（`docs/KERNEL-CACHE-STALE-20261004.md`）；镜像层方式在新机器上是冷缓存，不受影响 |

## 4. 复现

```bash
# ① 复算"最小补丁集"（确认只有 2 个文件不同）
bash tools/build_a3_tp8_image.sh    # 自检里会打印 OPP 文件数与逐字节结果
# ② 造层补丁发布包
sudo python3 tools/make_image_patch_kit.py --out ~/dsv41-kit-v2 --base <BASE> \
     --job 'dsv41-a3-tp8|<新镜像tag>|both'
# ③ 内核级反例对照（base vs new）
docker run --rm --entrypoint md5sum <BASE> <容器内 .o 路径>
docker run --rm --entrypoint md5sum <新镜像> <容器内 .o 路径>
```
