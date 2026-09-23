# A3「工作层」镜像补丁包 —— 把 TP8 工作形态固化成一层

> 面向：把 A3 上已验证可跑的那套形态，**只用一层**（0.2 MiB）搬到内网新 A3。
> 已发布：**`dsv41-a3-tp8-imagekit-v1`**（2.73 MB）—— 见 links-server。

---

## 0. 为什么不传完整镜像

| | 完整镜像 | 本方案的"工作层" |
|---|---|---|
| 体积 | **25 GB** | **0.2 MiB**（层）+ 2.7 MB（含 30 万条清单的包） |
| 内网需要什么 | 什么都不要 | **官方基础镜像**（内网本来就有）+ `docker` |
| 需要 skopeo/联网 | 否 | 否 |

基础镜像 = `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（18 层）。
我们只加了 **1 层**：原先靠 `-v` 挂进去的 19 个文件。

### 为什么不能用 `skopeo copy` 做到"逐字节同 digest"

这一层是**本地 `docker build` 出来的、从未 push 过 registry** ⇒ docker 只保留了
**解压后的 diff 目录**，原始压缩 blob 不存在 ⇒ 没有"我们的 blob"可以拼进 manifest。
**能复现的是文件系统**，所以本包的验收就是按这个目标做的（见 §3）。

---

## 1. 三层结构（谁在哪一层）

```
官方基础镜像（内网已有，18 层，25 GB）
  └── + 我们的工作层（1 层，0.2 MiB，31 条目 = 19 文件 + 12 目录）
        = 重建镜像（= 我们 A3 上跑通的那个形态）
```

工作层里的 19 个文件**就是原先 `-v` 挂载的那 19 个**（每个都落在容器内真实路径）：

| 容器内路径 | 个数 | 作用 |
|---|---|---|
| `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/*.py` | 8 | Engram/模型层补丁件（含 **ENGRAM×卸载 P0 修复**：`engram_hash.py` / `engram_jit_kernel.py`） |
| `/vllm-workspace/vllm-ascend/vllm_ascend/{ascend_forward_context,ops/rope_dsv4,worker/block_table,attention/dsa_v1,ops/fused_moe/token_dispatcher}.py` | 5 | 前端/rope/块表/attention/MoE dispatch 优化件 |
| `/opt/dsv41/scripts/*.sh` | 5 | `serve_a2.sh` / `serve_a3.sh` / `serve_v2.sh` / `run_test.sh` / `build_image.sh` |
| `/opt/dsv41/admission_gate.patch` | 1 | 准入闸补丁 |

---

## 2. 怎么造这个包（4 步，全部可复算）

```bash
# ① 造"基镜像 + 工作层"的派生镜像（**挂载清单不手抄**，从 A3 入口的 DRY_RUN 取）
bash tools/build_a3_tp8_image.sh                 # ⇒ local/dsv41-a3-tp8:v1（19 层）

# ② 把它切成"官方基础镜像 + 工作层"
sudo python3 tools/make_image_patch_kit.py --out <kit 目录> \
     --base quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
     --job 'dsv41-a3-tp8|local/dsv41-a3-tp8:v1|v41'

# ③ 验收（**这一步不能跳**）
cd <kit 目录>/images/dsv41-a3-tp8 && sudo bash rebuild.sh

# ④ 打包 + 上传
bash tools/package_image_kits.sh <kit 目录> <输出目录>       # ⇒ .tar.zst + .sha256
coscli cp <kit>.tar.zst cos://uploads-new/share/<name>.tar.zst
coscli object-acl --method put cos://uploads-new/share/<name>.tar.zst --acl public-read
```

★ 关键设计（都是为了防"看起来装上了"）：
* **挂载清单不手抄**：`build_a3_tp8_image.sh` 从 `serve_a3.sh` 的 `DRY_RUN` 输出里解析
  （用 `tools/mount_list_pairs.py`，它会**成对解析**并检查"目标唯一"——`MOUNTS` 里每条
  挂载占**两个**元素，只删一半会让 docker 报 `invalid reference format`，本仓踩过）；
* **派生镜像自检**：镜像内每个文件的 md5 必须等于宿主源文件（19/19）；
* **反例对照**：确认基础镜像里的 `engram_hash.py` 是**另一个** md5（`a4287dd51da0…`，
  未修版）—— 若相同，说明这一层什么都没改。

---

## 3. `rebuild.sh` 的验收标准（**不是"能启动就算过"**）

| # | 判据 | 标准 |
|---|---|---|
| ① | 基础镜像 | 18 个 layer 的 diffID **与打包时逐条相同**（不同 ⇒ 直接 FATAL） |
| ② | 文件系统 | 重建镜像的根文件系统 **305,702 条目录项，与原镜像逐条相同**（**全部**路径与类型） |
| ③ | 工作层文件 | **19 个文件 sha256 逐字节相同** |
| ④ | 冒烟 L1（**不需 NPU**） | 补丁件 md5 19/19、语法编译 13/13、起服脚本 4/4 在位 |
| ⑤ | 冒烟 L2（需 NPU） | `import vllm` 通过（无卡自动跳过） |

原始清单由打包工具在 a3-21 上**从原镜像现场生成**、随包携带：
`fs-manifest.original.txt.gz`（30 万条）、`payload-sha256.original.txt.gz`（19 条）、
`checks/payload-md5.txt`（19 条 md5）。

### 已实测（a3-21，2026-09-23）

```
OK: 18 layers, diffIDs match
OK: 305702 filesystem entries identical to the original image
OK: 19 working-layer files byte-identical
v41 checks: PASS            # L1：md5 19/19、py_compile 13/13、脚本 4/4
import vllm OK 0.27.1       # L2：NPU 冒烟
v41 checks: PASS
```

★ 还做过**往返验证**：把 `.tar.zst` 解开到全新目录、在**解包目录**里跑 `rebuild.sh`
⇒ 上面五条全部再次通过（证明"上传的东西 == 能用的东西"）。

---

## 4. 内网 A3 怎么用

```bash
# 下载（或用 links-server 页面上的 get.sh）
curl -fL -O https://uploads-new-1254016670.cos.ap-shanghai.myqcloud.com/share/dsv41-a3-tp8-imagekit-v1.tar.zst
sha256sum -c <(echo "fbd4bdcc6a1bde89e49cd9042e9284fd925eaaace4532ccaea5b9f9fb476d890  dsv41-a3-tp8-imagekit-v1.tar.zst")
tar -x -I zstd -f dsv41-a3-tp8-imagekit-v1.tar.zst

# 重建（只需 docker；不需要 skopeo / 联网 / NPU）
cd dsv41-kits-20260923/images/dsv41-a3-tp8
bash rebuild.sh                       # 默认 tag: local/dsv41-a3-tp8:v1-rebuilt
NEW_TAG=dsv41-a3-tp8:v1 bash rebuild.sh
BASE_IMAGE=<你们内网的基础镜像名> bash rebuild.sh    # 镜像名不同时

# 起服（**不需要任何 -v**）
DEVS="8 9 10 11 12 13 14 15" MODEL=<模型目录> \
CPU_BIND=0 DROPCACHE=0 SERVED_NAME=deepseek-v41 \
bash /opt/dsv41/scripts/serve_a3.sh
```

两个 A3 上**必须**的参数（都是实测出来的，不是风格问题）：
* `CPU_BIND=0` —— 关掉内部 NUMA 绑核/迁移。A3 共用机上目标节点常被占满 ⇒
  `migratepages` 会 100% CPU 无限自旋，服务永远不就绪，连 `docker stop` 都拿不到
  exit event（解法：`sudo pkill -9 -x migratepages`）。
* `DROPCACHE=0` —— A3 是共用机，默认不清整机 page cache（那会打到别人的任务）。

---

## 5. 重建后**必须**核对的三件事

1. **基础镜像 diffID 一致** —— `rebuild.sh` 第 0 步会 diff，不一致直接停。
2. **三个关键文件指纹**（`rebuild.sh` 会打印，用于跨机核对"是不是同一版"）：

   | 文件 | md5 |
   |---|---|
   | `models/deepseek_v41/engram_hash.py` | `3a842bbb6d0dd783c65087ccef347370` |
   | `models/deepseek_v41/model.py` | `0a7dfb21a5e676b3ccb0a4eb9551087f` |
   | `attention/dsa_v1.py` | `9a36e709b0937589eab05c5316a62591` |

   **对照**：基础镜像里的 `engram_hash.py` 是 `a4287dd51da0…`（未修版）。
3. **起服后**在容器内再核一次（判据落在"实际生效后的可观测痕迹"上，而不是"我打了包"）：

   ```bash
   docker exec <容器> md5sum /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py
   ```

---

## 6. 来源与工具出处

| 项 | 值 |
|---|---|
| 基础镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（18 层，id `sha256:1f2c08195c5b…`） |
| 源镜像 | `local/dsv41-a3-tp8:v1`（19 层，id `sha256:fafb1665853e…`） |
| 代码基线 | `vllm 6e448d0`（+0 dirty）/ `vllm-ascend e43cf1e9f`（**+13 dirty** = 13 个补丁件） |
| 工作层 | diffID `sha256:f6b1d9ebb7d9…`，31 条目，压缩 0.2 MiB |
| 包 sha256 | `fbd4bdcc6a1bde89e49cd9042e9284fd925eaaace4532ccaea5b9f9fb476d890` |

**工具出处**：`tools/make_image_patch_kit.py` + `tools/kit_tools/fs_manifest.py` +
`tools/package_image_kits.sh` 来自本工作区的另一个项目
（`~/projects/vllm-0.26.0-release-inst/vllm-ascend-stub-repro-text-v1/scripts/`，
那套已交付过 4 个套件）。本仓做的改动只有三处，都可复算：
1. 新增 `v41` 一档（`SELFTEST_KIND["v41"]` + `V41_CHECK`：文件级判据 + 可选 NPU 冒烟）；
2. 为 `v41` 档生成 `checks/payload-md5.txt`（19 条，取自**源镜像**内真值）；
3. 全部套件都是 `v41` 档时，README 用**本仓自己的** `V41_README`（默认 prose 是那个项目的 va26 措辞）。
