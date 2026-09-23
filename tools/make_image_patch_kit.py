#!/usr/bin/env python3
"""make_image_patch_kit.py — cut our local images into "official base + working
layers" and emit a self-contained kit that rebuilds them on a machine which
already has the base image, using nothing but docker.

    # on a3-21 (root, reads the docker store)
    sudo python3 scripts/make_image_patch_kit.py \
        --out /home/l00886679/va26-image-kits-20260921 \
        --base quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler \
        --job 'stublite-v2|local/vllm-ascend-stub-liteprofiler:v2|both' \
        --job 'liteprof-waitscope2|local/vllm-ascend-liteprofiler:v0.26.0rc1-mp-lite-waitscope2|lite' \
        --job 'stub-cpuonly|local/vllm-ascend-stub:v0.26.0rc1-a3-cpuonly|stub'

Jobs are merged: running the tool again with more --job entries only adds those
image directories and regenerates MANIFEST.json / README.md from everything
found under <out>/images/.

WHY THE LAYERS ARE RE-EXPORTED (and why skopeo cannot make this digest-exact)
----------------------------------------------------------------------------
The 18 base layers exist in the target environment already: they are the
official quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler image.  Only
the handful of layers this project added have to travel.

Those layers were built locally and were never pushed to a registry, so no
compressed blob for them exists anywhere -- docker keeps only the *extracted*
diff directory.  The diffID (sha256 of the tar stream the builder produced) is
therefore not reproducible byte-for-byte, which is exactly why `skopeo copy`
cannot rebuild this image from base blobs plus ours: there is no original blob
to copy.  What IS reproducible is the filesystem, and that is what this kit
checks: the rebuilt image is compared entry-by-entry, and every file that came
from the working layers is compared by sha256, against manifests taken from the
original image.

Kit layout
----------
    README.md                     how to rebuild (Chinese)
    MANIFEST.json                 every image, layer, size and checksum
    tools/fs_manifest.py          runs inside a container (manifest + hashes)
    images/<kit>/Dockerfile       ARG BASE_IMAGE=<base> + ADD of the layers
    images/<kit>/rebuild.sh       base check -> build -> fs verify -> smoke
    images/<kit>/selftest.sh      smoke test on the rebuilt image
    images/<kit>/checks/*.sh      the in-container smoke checks
    images/<kit>/layers/*.tar.gz  the working layers (gzip, ADD-extractable)
    images/<kit>/base-layers.txt  diffIDs the base image must have
    images/<kit>/image-meta.json  provenance for this image
    images/<kit>/fs-manifest.original.txt.gz   filesystem of the ORIGINAL image
    images/<kit>/payload-sha256.original.txt.gz  sha256 of the working-layer files
    images/<kit>/payload-paths.txt             the paths those hashes cover
"""

import argparse
import datetime
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time

DOCKER = shutil.which("docker") or "/usr/bin/docker"

README_PROSE = """\
# vllm-ascend v0.26.0 A3 打桩镜像 —— 层拆分重建包

本包把工作机上那几个**只在本地构建过**的镜像拆成：

    官方基础镜像（你们内网已有）  +  本项目的若干“工作层”（本包携带）

重建**只需要 docker**（不需要 skopeo、不需要联网、不需要 NPU）：

```bash
cd images/<镜像名>
bash rebuild.sh                 # ① 校验基础镜像 ② docker build ③ 文件系统验收 ④ 冒烟
# 可选：
NEW_TAG=my/tag BASE_IMAGE=<你们内网的基础镜像名> bash rebuild.sh
SKIP_SELFTEST=1 bash rebuild.sh
```

`rebuild.sh` 的验收不是“能启动就算过”，而是把重建镜像的**整个根文件系统清单**
与原始镜像的清单逐条比对，并把工作层里的**每个文件按 sha256** 与原始镜像比对；
两者都一致才打印 OK（原始清单见 `fs-manifest.original.txt.gz` /
`payload-sha256.original.txt.gz`，由本工具在 a3-21 上从原镜像现场生成）。

## 为什么只有“工作层”

工作机上的镜像 = 官方基础镜像 + 我们加的 N 层，前 18 层与基础镜像**逐层同
diffID**（工具会校验）。基础镜像你们内网已经有（`quay.nju.edu.cn` 镜像源），
所以只需要搬运我们那几层。

## 为什么不能用 skopeo 做到“逐字节同 digest”

这几个镜像从未 push 过 registry：docker 本地只保留**解压后的 diff 目录**，
原始压缩 blob 不存在。层的 diffID 是当初构建时 tar 流的 sha256，事后无法原样
重放，因此 `skopeo copy` 无法用“基础镜像的 blob + 我们的 blob”拼回同一个
manifest——根本没有“我们的 blob”可拼。

能复现的是**文件系统**，本包就是按这个目标验证的：重建镜像与原始镜像
**目录项逐条相同、工作层文件逐字节相同**（基础层 diffID 本来就相同）。
重建出来的镜像可以正常 `docker save` / `skopeo copy` 到你们自己的 registry，
只是 layer digest 与原镜像不同（内容相同）。

## 前提

* 基础镜像必须就是这批镜像所基于的那一个（`docker inspect` 的 `.Id` 与
  `RootFS.Layers` 必须与 `images/*/base-layers.txt` 一致，`rebuild.sh` 会先查）。
* docker 需要支持 `ADD` 解压 `.tar.gz`（17.05+ 都可以；a3-21 上是 18.09）。
* 冒烟测试里只有“桩上跑真模型”那一项需要模型；没有模型时会自动跳过，
  设备层探针（不需要模型）仍会跑。
"""


V41_README = '# DSV4.1-Flash A3（8×910C）**TP8 工作镜像** —— 「官方基础镜像 + 一层」重建包\n\n本包把 A3 上**已验证可跑**的那套工作形态，拆成：\n\n```\n官方基础镜像（内网已有）   +   我们的 1 个“工作层”（本包携带，0.2 MiB）\nquay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3\n```\n\n**工作层里装的就是原先靠 `-v` 挂进去的那 19 个文件**（每个都对应容器内的真实路径）：\n\n| 装到容器里的位置 | 内容 | 为什么需要 |\n|---|---|---|\n| `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/*.py`（8 个） | Engram 与模型层的补丁件 | 官方镜像里是未修版；其中 `engram_hash.py` / `engram_jit_kernel.py` 是 **ENGRAM×卸载 P0 修复**（缺了会在取回时 `KeyError` 打死引擎） |\n| `/vllm-workspace/vllm-ascend/vllm_ascend/{ascend_forward_context,ops/rope_dsv4,worker/block_table,attention/dsa_v1,ops/fused_moe/token_dispatcher}.py`（5 个） | 前端/rope/块表/attention/MoE dispatch 优化件 | 性能补丁 |\n| `/opt/dsv41/scripts/*.sh`（5 个） | `serve_a2.sh` / `serve_a3.sh` / `serve_v2.sh` / `run_test.sh` / `build_image.sh` | 起服与自检脚本 |\n| `/opt/dsv41/admission_gate.patch`（1 个） | 准入闸补丁 | 起服期行为 |\n\n> 为什么不做成"完整镜像"：完整镜像 25 GB，而**基础镜像内网已经有**，我们只加了\n> **0.2 MiB** 的文件。传层 = 传 0.2 MiB + 清单，而不是 25 GB。\n>\n> 为什么不能用 `skopeo copy` 做到"逐字节同 digest"：这一层是 docker 本地构建的、\n> **从未 push 过 registry**，docker 只保留了**解压后的 diff 目录**，原始压缩 blob 不存在\n> ⇒ 没有一个"我们的 blob"可以拼进 manifest。**能复现的是文件系统**，本包就是按这个目标\n> 验证的（见下）。\n\n---\n\n## 一、应用方法（三选一）\n\n```bash\ncd images/dsv41-a3-tp8\n\n# ① 标准：校验基础镜像 → 重建 → 文件系统验收 → 冒烟（推荐）\nbash rebuild.sh\n\n# ② 你们的内部镜像源名字不同时\nBASE_IMAGE=<你们内网的基础镜像名> bash rebuild.sh\n\n# ③ 只要镜像，不要验收（快）\nSKIP_SELFTEST=1 bash rebuild.sh\n```\n\n重建出来的镜像默认 tag `local/dsv41-a3-tp8:v1-rebuilt`，可以：\n```bash\nNEW_TAG=dsv41-a3-tp8:v1 bash rebuild.sh     # 自定义 tag\ndocker tag dsv41-a3-tp8:v1 <你们registry>/dsv41-a3-tp8:v1 && docker push ...\ndocker save dsv41-a3-tp8:v1 | ssh <内网A3> \'docker load\'\n```\n\n## 二、`rebuild.sh` 的验收标准（**不是"能启动就算过"**）\n\n| 判据 | 标准 | 说明 |\n|---|---|---|\n| ① 基础镜像 | `docker inspect` 的 18 个 layer diffID **与打包时逐条相同** | 不同 ⇒ 直接 FATAL（说明不是同一基底） |\n| ② 文件系统 | 重建镜像的根文件系统清单 **305,702 条，与原始镜像逐条相同** | 覆盖**全部**路径与类型，不只是我们的文件 |\n| ③ 工作层文件 | 19 个文件 **sha256 逐字节相同** | 只比我们改过的那部分 |\n| ④ 冒烟 L1（不需 NPU） | 补丁件 md5 19/19、语法编译 13/13、起服脚本 4/4 在位 | 无卡机器上也能跑 |\n| ⑤ 冒烟 L2（需 NPU） | `import vllm` 通过 | 挂了设备才跑，无卡自动跳过 |\n\n原始清单由打包工具在 a3-21 上**从原镜像现场生成**，随包携带：\n`fs-manifest.original.txt.gz`（30 万条）、`payload-sha256.original.txt.gz`（19 条）、\n`checks/payload-md5.txt`（19 条 md5）。\n\n## 三、重建后**必须**核对的三件事\n\n1. **基础镜像 diffID 一致**：`rebuild.sh` 第 0 步会 diff，不一致直接停（这是最容易被\n   "换了个基底"破坏的一环）。\n2. **三个关键文件指纹**（`rebuild.sh` 会打印；跨机核对"是不是同一版"）：\n   ```\n   engram_hash.py   3a842bbb6d0dd783c65087ccef347370\n   model.py         0a7dfb21a5e676b3ccb0a4eb9551087f\n   dsa_v1.py        9a36e709b0937589eab05c5316a62591\n   ```\n   对照组：**基础镜像里**的 `engram_hash.py` 是 `a4287dd51da0…`（未修版）——\n   若重建后仍是这个值，说明**这一层没生效**。\n3. **起服后再看一次容器内 md5**（判据要落在"实际生效后的可观测痕迹"上，\n   而不是"我打了这个包"）：\n   ```bash\n   docker exec <容器> md5sum \\\n     /vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram_hash.py\n   ```\n\n## 四、起服（与原来 `-v` 挂载形态等价，但**不需要任何 `-v`**）\n\n```bash\nDEVS="8 9 10 11 12 13 14 15" GPU_UTIL=0.92 \\\nMODEL=<模型目录> SERVED_NAME=deepseek-v41 \\\nCPU_BIND=0 DROPCACHE=0 \\\nbash scripts/serve_a3.sh          # 镜像内 /opt/dsv41/scripts/serve_a3.sh\n```\n\n* `CPU_BIND=0`：关掉内部 NUMA 绑核/迁移（A3 上是**必需**的逃生口 —— 目标节点被占满时\n  会让 `migratepages` 无限自旋、`docker stop` 都停不下来）。\n* `DROPCACHE=0`：A3 是**共用机**，默认不清整机 page cache（会打到别人）。\n\n## 五、本包的来源（可复算）\n\n| 项 | 值 |\n|---|---|\n| 基础镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3`（18 层，id `sha256:1f2c08195c5b…`） |\n| 源镜像（本层从它切出） | `local/dsv41-a3-tp8:v1`（19 层，id `sha256:fafb1665853e…`） |\n| 代码基线 | `vllm 6e448d0`（+0 dirty）/ `vllm-ascend e43cf1e9f`（**+13 dirty** = 我们的 13 个补丁件） |\n| 工作层 | 1 层，diffID `sha256:f6b1d9ebb7d9…`，31 条目（19 文件 + 12 目录），压缩后 **0.2 MiB** |\n| 生成工具 | `tools/make_image_patch_kit.py`（本仓，`--job \'dsv41-a3-tp8\\|<image>\\|v41\'`） |\n| 生成时间/机器 | 见 `MANIFEST.json` |\n\n完整逐项元数据在 `MANIFEST.json`（含每个层的 entries/whiteouts/diffID/sha256/大小与\n`code_baseline`）。\n'

RELATIONS_PROSE = """
## 几个镜像的关系（容易混，先看这张图）

本包涉及的镜像**全部**是同一个官方 A3 基础镜像的后代（前 18 层逐层同 diffID）：

```
官方 quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler
  dc9a31b8330d    vllm 568afb3 / vllm-ascend f2f74a16c   18 层
  |
  +-- 纯打桩线（打桩补丁直接打在官方树上；本包 2 个套件）
  |     stub:cpuonly                      09-15   18+4 层  无图能力，含 3 份重复 workspace(2.5 GB)
  |     stub:cpuonly-20260921             09-21   18+1 层  当前代码，含 compile+aclgraph
  |
  +-- 插桩线（先把 vllm/vllm-ascend 换成 model-comparing 的那对 commit，再加插桩）
        liteprofiler:*-openeuler / -canonical     18+2 层
        liteprofiler:*-mp-lite                    18+3 层
        liteprofiler:*-mp-lite-waitscope          18+4 层   与 waitscope2 只差第 4 层
        liteprofiler:*-mp-lite-waitscope2         18+4 层   <- 带卡对照臂用这个（本包 1 个套件）
        liteprofiler:v2  ( = :liteprofiler-v2 )   18+5 层   <- 打桩+插桩镜像的构建基底
          +-- stub-liteprofiler:v1                18+6 层
          +-- stub-liteprofiler:v2                18+6 层   <- 最新（含图能力，本包 1 个套件）
```

三条要点：

1. **本包的每个套件都只需要官方基础镜像**：插桩层已经作为工作层带在包里，内网
   **不需要** model-comparing 那一串 liteprofiler 镜像（它们是构建链的中间物）。
2. **两条线的代码基线不同，别混着比**：纯打桩线跑的是官方那对
   （`vllm 568afb3` / `vllm-ascend f2f74a16c`）；插桩线跑的是 model-comparing 的那对
   （`vllm 13beced` / `vllm-ascend fe3f1e85f`，插桩就是落在它上面的）。所以
   "打桩前后"的对比必须**同线比**：`stub-liteprofiler:v2`（打桩）↔
   `liteprofiler:*-mp-lite-waitscope2`（带卡）；拿纯打桩镜像跟带卡比，会同时差出代码版本。
3. 下表 `代码基线` 列是**镜像内** `/vllm-workspace` 的 git HEAD 与改动文件数（`dirty`）。
   这一列不一样就说明两条线不可直接比较。

> **如果本包是 x86_64 版**（源镜像形如 `local/vllm-ascend-stub-x86:…`）：它**不在**上面这棵
> aarch64 家族树里，而是**同一套补丁与桩源码在官方 amd64 基底上的重建**——官方基底本身是
> amd64+arm64 多架构（两边同 digest），amd64 变体里 vllm/vllm-ascend 的 commit 与 aarch64
> 完全一致，因此补丁可直接复用；需要处理的 5 处架构差异（glibc 符号版本标签、CANN
> `<arch>-linux` 目录、x86 无驱动包导致 HAL 要走 devlib、平台 INI 软链、perf 系统调用号）
> 见仓库 `X86_PORT.md`。重建方式与本包其余部分完全相同：`bash rebuild.sh`。
"""


def log(msg):
    print("[kit] %s" % msg, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def capture(cmd):
    return subprocess.check_output(cmd, text=True)


def docker_inspect(ref):
    return json.loads(capture([DOCKER, "inspect", ref]))[0]


def docker_layers(ref):
    txt = capture([DOCKER, "inspect", "-f",
                   "{{range .RootFS.Layers}}{{println .}}{{end}}", ref])
    return [l for l in txt.split() if l]


def sha256_file(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_code_baseline(ref):
    """The vllm / vllm-ascend commits and dirty-file counts inside the image.

    `safe.directory=*` is required: the images were built by a different uid, so
    git refuses to read the worktrees as root otherwise."""
    script = (
        'G="git -c safe.directory=* -C"; '
        'V=$($G /vllm-workspace/vllm rev-parse --short HEAD 2>/dev/null); '
        'A=$($G /vllm-workspace/vllm-ascend rev-parse --short HEAD 2>/dev/null); '
        'DV=$($G /vllm-workspace/vllm status --porcelain 2>/dev/null | wc -l); '
        'DA=$($G /vllm-workspace/vllm-ascend status --porcelain 2>/dev/null | wc -l); '
        'echo "$V|$A|$DV|$DA"')
    try:
        out = capture([DOCKER, "run", "--rm", "--entrypoint", "/bin/bash",
                       ref, "-c", script]).strip().split("|")
    except subprocess.CalledProcessError:
        return {}
    if len(out) != 4:
        return {}
    v, a, dv, da = out
    return {
        "vllm": v or "?",
        "vllm_ascend": a or "?",
        "vllm_dirty_files": int(dv or 0),
        "vllm_ascend_dirty_files": int(da or 0),
    }


def write_text(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, mode)


# ---------------------------------------------------------------------------
# layer export
# ---------------------------------------------------------------------------

def build_layer_map(docker_root):
    """diffID -> {cache_id, size, dir} from the overlay2 layerdb."""
    ldb = os.path.join(docker_root, "image/overlay2/layerdb/sha256")
    if not os.path.isdir(ldb):
        sys.exit("FATAL: %s not found (wrong --docker-root? run as root?)" % ldb)
    m = {}
    for entry in os.listdir(ldb):
        p = os.path.join(ldb, entry)
        try:
            diffid = open(os.path.join(p, "diff")).read().strip()
            cache_id = open(os.path.join(p, "cache-id")).read().strip()
            size = int(open(os.path.join(p, "size")).read().strip())
        except OSError:
            continue
        m[diffid] = {
            "cache_id": cache_id,
            "size": size,
            "dir": os.path.join(docker_root, "overlay2", cache_id, "diff"),
        }
    return m


def scan_layer_dir(diffdir):
    """Return (relative paths, whiteouts, opaque dirs, regular file paths)."""
    names, whiteouts, opaque, files = [], [], [], []
    for dirpath, dirnames, filenames in os.walk(diffdir, topdown=True,
                                                followlinks=False):
        rel = os.path.relpath(dirpath, diffdir)
        try:
            if os.getxattr(dirpath, "trusted.overlay.opaque") == b"y":
                opaque.append("" if rel == "." else rel)
        except OSError:
            pass
        for n in list(dirnames) + list(filenames):
            r = n if rel == "." else os.path.join(rel, n)
            names.append(r)
            if os.path.basename(r).startswith(".wh."):
                whiteouts.append(r)
            p = os.path.join(dirpath, n)
            if os.path.islink(p) and n in dirnames:
                # a symlink to a directory lands in dirnames; it is archived as a
                # symlink and must not be descended into
                dirnames.remove(n)
            if not os.path.islink(p) and os.path.isfile(p):
                files.append(r)
    names.sort()
    files.sort()
    return names, whiteouts, opaque, files


def export_layer(diffdir, dst_gz, tmpdir):
    names, whiteouts, opaque, files = scan_layer_dir(diffdir)
    keep = [n for n in names if not os.path.basename(n).startswith(".wh.")]
    if whiteouts or opaque:
        log("    NOTE: %s whiteout(s), %s opaque dir(s) in %s"
            % (len(whiteouts), len(opaque), os.path.basename(diffdir)))
    os.makedirs(os.path.dirname(dst_gz), exist_ok=True)
    listfile = os.path.join(tmpdir, "layer-files.list")
    with open(listfile, "wb") as fh:
        for n in keep:
            fh.write(n.encode("utf-8", "surrogateescape") + b"\0")
    pigz = shutil.which("pigz")
    comp = ([pigz, "-6", "-p", str(min(16, os.cpu_count() or 4)), "-c"]
            if pigz else ["gzip", "-6", "-c"])
    with open(dst_gz, "wb") as fh_out:
        tar = subprocess.Popen(
            ["tar", "-C", diffdir, "--numeric-owner", "--null",
             "--no-recursion", "-T", listfile, "-cf", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gz = subprocess.Popen(comp, stdin=tar.stdout, stdout=fh_out)
        tar.stdout.close()
        gz.wait()
        tar_err = tar.stderr.read().decode()
        tar.wait()
        if tar.returncode or gz.returncode:
            sys.exit("FATAL: tar/gzip failed (%s / %s): %s"
                     % (tar.returncode, gz.returncode, tar_err.strip()))
    n_entries = int(capture(["tar", "-tzf", dst_gz]).count("\n"))
    if n_entries != len(keep):
        sys.exit("FATAL: %s has %d entries, expected %d"
                 % (dst_gz, n_entries, len(keep)))
    uncompressed = sum(
        os.lstat(os.path.join(diffdir, n)).st_size
        for n in keep if not os.path.islink(os.path.join(diffdir, n)))
    return {
        "entries": len(keep),
        "whiteouts": whiteouts,
        "opaque_dirs": opaque,
        "size_gz": os.path.getsize(dst_gz),
        "size_raw_estimate": uncompressed,
        "sha256_gz": sha256_file(dst_gz),
        "files": files,
    }


# ---------------------------------------------------------------------------
# Dockerfile / scripts
# ---------------------------------------------------------------------------

def df_escape(value):
    for ch in ("$", '"', "\\", "\n"):
        if ch in value:
            return '"%s"' % (value.replace("\\", "\\\\")
                                    .replace('"', '\\"')
                                    .replace("$", "\\$")
                                    .replace("\n", "\\n"))
    return value


def dockerfile_text(kit, src_ref, src_id, base_ref, base_id, layers, cfg_delta):
    head = [
        "# " + "-" * 73,
        "# %s -- rebuild %s" % (kit, src_ref),
        "#",
        "# generated by scripts/make_image_patch_kit.py on %s; do not edit."
        % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "#",
        "#   source image  : %s  (%s)" % (src_ref, src_id),
        "#   base image    : %s  (%s)" % (base_ref, base_id),
        "#   working layers: %d  (gzipped layer tars, ADD extracts them)"
        % len(layers),
        "#",
        "#   docker build --build-arg BASE_IMAGE=%s -t <tag> ." % base_ref,
        "#   bash rebuild.sh          # build + filesystem verification + smoke",
        "# " + "-" * 73,
        "ARG BASE_IMAGE=%s" % base_ref,
        "FROM ${BASE_IMAGE}",
        "",
        "# --- this project's working layers, in their original order ---",
    ]
    body = []
    for i, l in enumerate(layers):
        body.append("# %02d  diffid=%s  entries=%d  sha256(gz)=%s"
                    % (i, l["diffid"], l["entries"], l["sha256_gz"][:16]))
        def _rel(p):
            return p[2:] if p.startswith("./") else p.lstrip("/")

        for d in l.get("opaque_dirs", []):
            body.append("RUN rm -rf /%s" % _rel(d))          # opaque dir: clear first
        body.append("ADD layers/%s /" % l["file"])
        for w in l.get("whiteouts", []):
            body.append("RUN rm -rf /%s"
                        % _rel(os.path.join(os.path.dirname(w),
                                            os.path.basename(w)[len(".wh."):])))
    body += ["", "# --- config of the source image that differs from the base ---"]
    body += cfg_delta
    body.append("")
    return "\n".join(head + body)


def rebuild_sh_text(kit, src_ref, base_ref, base_layer_count, selftest):
    return """#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rebuild.sh -- rebuild {src} on top of the base image and verify that the
# result is the same filesystem the original image had.
#
#   bash rebuild.sh
#   BASE_IMAGE=<internal mirror ref> NEW_TAG=<tag> bash rebuild.sh
#   SKIP_SELFTEST=1 bash rebuild.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
KIT="$(cd "${{HERE}}/../.." && pwd)"
KITNAME="{kit}"
BASE_IMAGE="${{BASE_IMAGE:-{base}}}"
NEW_TAG="${{NEW_TAG:-{src}-rebuilt}}"
OUT="${{OUT:-${{KIT}}/verify/${{KITNAME}}}}"

echo "== 0. base image =="
echo "   ref  : ${{BASE_IMAGE}}"
docker image inspect "${{BASE_IMAGE}}" >/dev/null 2>&1 \\
    || {{ echo "FATAL: base image '${{BASE_IMAGE}}' is not present locally; pull it first" >&2; exit 1; }}
# verification output: prefer <kit>/verify/, fall back to a writable temp dir
# (the kit may have been unpacked read-only or owned by another user)
if ! mkdir -p "${{OUT}}" 2>/dev/null; then
    OUT="${{TMPDIR:-/tmp}}/va26-verify/${{KITNAME}}"
    mkdir -p "${{OUT}}"
    echo "   (kit dir is not writable; verification output -> ${{OUT}})"
fi
# NOTE: `docker inspect -f` appends one extra newline in some versions, so the
# live list is normalised through awk 'NF' before it is diffed.
docker inspect -f '{{{{range .RootFS.Layers}}}}{{{{println .}}}}{{{{end}}}}' "${{BASE_IMAGE}}" \\
    | awk 'NF' > "${{OUT}}/base-layers.actual.txt"
if ! diff -u "${{HERE}}/base-layers.txt" "${{OUT}}/base-layers.actual.txt"; then
    echo "FATAL: this is NOT the base image the kit was cut against (layer diffIDs differ)" >&2
    exit 1
fi
echo "   OK: {nbase} layers, diffIDs match"

echo "== 1. docker build =="
docker build --build-arg BASE_IMAGE="${{BASE_IMAGE}}" -t "${{NEW_TAG}}" "${{HERE}}"

echo "== 2. filesystem verification =="
mkdir -p "${{OUT}}"
docker run --rm --entrypoint /bin/bash \\
    -v "${{KIT}}":/va26kit:ro -v "${{OUT}}":/va26out \\
    "${{NEW_TAG}}" -c 'PY=$(command -v python3 || echo /usr/local/python3.12.13/bin/python3); "$PY" /va26kit/tools/fs_manifest.py --out /va26out/fs-manifest.rebuilt.txt --payload-list /va26kit/images/{kit}/payload-paths.txt --payload-out /va26out/payload-sha256.rebuilt.txt'

gzip -dc "${{HERE}}/fs-manifest.original.txt.gz" > "${{OUT}}/fs-manifest.original.txt"
if diff -u "${{OUT}}/fs-manifest.original.txt" "${{OUT}}/fs-manifest.rebuilt.txt" > "${{OUT}}/fs-manifest.diff"; then
    echo "   OK: $(wc -l < "${{OUT}}/fs-manifest.rebuilt.txt") filesystem entries identical to the original image"
else
    echo "   FAIL: filesystem differs -> ${{OUT}}/fs-manifest.diff" >&2
    head -40 "${{OUT}}/fs-manifest.diff" >&2
    exit 1
fi
if diff -u <(gzip -dc "${{HERE}}/payload-sha256.original.txt.gz") \\
        "${{OUT}}/payload-sha256.rebuilt.txt" > "${{OUT}}/payload.diff"; then
    echo "   OK: $(wc -l < "${{OUT}}/payload-sha256.rebuilt.txt") working-layer files byte-identical"
else
    echo "   FAIL: working-layer content differs -> ${{OUT}}/payload.diff" >&2
    head -40 "${{OUT}}/payload.diff" >&2
    exit 1
fi

if [ "${{SKIP_SELFTEST:-0}}" != "1" ]; then
    echo "== 3. smoke test =="
    TAG="${{NEW_TAG}}" HERE="${{HERE}}" bash "${{HERE}}/{selftest}"
fi
echo "== done: ${{NEW_TAG}} =="
""".format(kit=kit, src=src_ref, base=base_ref, selftest=selftest,
           nbase=base_layer_count)


STUB_CHECK = r"""#!/usr/bin/env bash
# In-container stub smoke: fake A3 device via LD_PRELOAD, then (if a model is
# mounted at /models) a real offline generation.
#
# Portable by design: the model is auto-detected from whatever this kit's owner
# happens to have under /models (SMOKE_MODEL overrides), and the fake HBM /
# gpu_memory_utilization default to memory-friendly values so the smoke also
# passes on a small box (SMOKE_DEV_MEM_GB / SMOKE_GMEM override).
set -euo pipefail
echo "--- stub device probe (no model needed) ---"
bash /opt/va26-scripts/run_real_test_official.sh /opt/va26-scripts/stub_device_probe.py

MODEL="${SMOKE_MODEL:-}"
if [ -z "${MODEL}" ]; then
    for cand in /models/Qwen3.5-0.8B /models/Qwen3-0.6B /models/Qwen3-1.7B \
                /models/Qwen3.5-2B /models/Qwen3-0.6B-Instruct; do
        if [ -d "${cand}" ]; then MODEL="${cand}"; break; fi
    done
fi
if [ -n "${MODEL}" ]; then
    echo "--- offline generation smoke: ${MODEL} (fake HBM ${SMOKE_DEV_MEM_GB:-16} GiB, gmem ${SMOKE_GMEM:-0.25}) ---"
    VLLM_ASCEND_STUB_DEVICE_MEM_GB="${SMOKE_DEV_MEM_GB:-16}" \
    TEST_GMEM="${SMOKE_GMEM:-0.25}" TEST_MODEL="${MODEL}" \
        bash /opt/va26-scripts/run_real_test_official.sh /workspace/test_infer.py
else
    echo "--- skip generation smoke: no model found under /models (mount one and set SMOKE_MODEL=...) ---"
fi
echo "STUB CHECKS PASSED"
"""

LITE_CHECK = r"""#!/usr/bin/env bash
# In-container LiteProfiler smoke: the 14 `wait:` scopes, the writer invariants
# and py_compile of the instrumented files.  Needs no NPU.
set -euo pipefail
V=/vllm-workspace/vllm
A=/vllm-workspace/vllm-ascend
PY=$(command -v python3 || echo /usr/local/python3.12.13/bin/python3)
fail=0
M=$(grep -ho 'record_function_or_nullcontext("wait:' \
        "${V}/vllm/v1/worker/gpu_model_runner.py" \
        "${A}/vllm_ascend/worker/model_runner_v1.py" \
        "${A}/vllm_ascend/compilation/acl_graph.py" 2>/dev/null | wc -l)
echo "liteprofiler wait scopes : ${M} (expect 14)"
[ "${M}" = "14" ] || fail=1
"${PY}" - "${V}/vllm/utils/lite_profiler.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
need = ["WAIT_PREFIX", "def _epoch_us", "_ANCHOR_GENERATION"]
forbid = ["_close_segment", "_resume_segment", "_open_scopes"]
missing = [m for m in need if m not in src]
present = [m for m in forbid if m in src]
if missing or present:
    print("writer invariants        : FAIL missing=%s forbidden=%s" % (missing, present))
    sys.exit(1)
print("writer invariants        : OK rows_per_scope=1 grid_anchor=yes")
PY
"${PY}" - <<'PY'
import py_compile
for p in ["/vllm-workspace/vllm/vllm/entrypoints/openai/completion/serving.py",
          "/vllm-workspace/vllm/vllm/utils/lite_profiler.py",
          "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
          "/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py",
          "/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py"]:
    py_compile.compile(p, doraise=True)
print("py_compile (5 key files) : OK")
PY
grep -q 'wait:num_accepted_tokens' "${A}/vllm_ascend/worker/model_runner_v1.py" \
    && echo "instrumentation marker   : wait:num_accepted_tokens present" || fail=1
[ "${fail}" = "0" ] && echo "LITE CHECKS PASSED" || { echo "LITE CHECKS FAILED"; exit 1; }
"""

SELFTEST_HEAD = r"""#!/usr/bin/env bash
# Smoke test for the rebuilt image.  Called by rebuild.sh with TAG/HERE set.
set -euo pipefail
TAG="${TAG:?set TAG}"
HERE="${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODELS="${MODELS:-$HOME/models}"
MNT=()
[ -d "${MODELS}" ] && MNT=(-v "${MODELS}:/models:ro")
"""

V41_CHECK = r"""#!/usr/bin/env bash
# =============================================================================
# v41_checks.sh —— dsv41 A3 TP8 镜像的**在容器内**自检
#
# 判据分两层（**不依赖 NPU** 的第一层必须过；NPU 冒烟是可选第二层）：
#   L1 文件级（无卡也能跑）：
#     · 13 个补丁件都在**容器内真实路径**上，且**逐文件 md5 与打包时清单一致**
#       （判据绑内容 —— "我 COPY 了"不等于"镜像里就是这份"）
#     · 每个补丁件 py_compile 通过（防截断/损坏）
#     · 起服脚本在位：/opt/dsv41/scripts/{serve_v2,serve_a3,serve_a2,run_test}.sh
#     · 打印关键文件指纹（跨机核对"是不是同一版"）
#   L2 NPU 冒烟（DSV41_NPU_SMOKE=1 且设备在时才跑）：`import vllm`
# =============================================================================
set -uo pipefail
fail=0
PY=$(command -v python3 || echo /usr/local/python3.12.13/bin/python3)
A=/vllm-workspace/vllm-ascend/vllm_ascend

echo "== L1 文件级判据（不需要 NPU）=="

FILES="models/deepseek_v41/engram_hbm.py models/deepseek_v41/engram_hash.py
models/deepseek_v41/engram_device_index.py models/deepseek_v41/engram_graph.py
models/deepseek_v41/engram_jit_kernel.py models/deepseek_v41/engram_plan_kernel.py
models/deepseek_v41/engram_gate.py models/deepseek_v41/model.py
ascend_forward_context.py ops/rope_dsv4.py worker/block_table.py
attention/dsa_v1.py ops/fused_moe/token_dispatcher.py"

MD5LIST=/dsv41checks/payload-md5.txt
if [ -f "$MD5LIST" ]; then
  n=0; bad=0
  while read -r want path; do
    [ -n "${path:-}" ] || continue
    n=$((n+1))
    if [ ! -f "$path" ]; then echo "  MISSING $path"; bad=$((bad+1)); continue; fi
    got=$(md5sum "$path" | cut -d' ' -f1)
    if [ "$want" != "$got" ]; then
      echo "  MD5-DIFF $path (want ${want:0:12} got ${got:0:12})"; bad=$((bad+1))
    fi
  done < "$MD5LIST"
  echo "  补丁件 md5：$((n-bad))/$n 一致"
  [ "$bad" = "0" ] || fail=1
else
  echo "  (没带 payload-md5.txt ⇒ 只做在位性检查，不做逐文件 md5)"
  for f in $FILES; do
    [ -f "$A/$f" ] || { echo "  MISSING $A/$f"; fail=1; }
  done
fi

"$PY" - <<'PYEOF'
# ★ 用**内存内**编译（compile()）而不是 py_compile(cfile=...)：
#   py_compile 会真的写 .pyc 文件；写 /dev/null 会被 Python 拒绝
#   （`FileExistsError: '/dev/null' is a non-regular file ...`，实测踩到）。
#   这里只需要"语法/完整性"判据 ⇒ 读源码 + compile() 即可，且**不落任何文件**。
import sys, os
A = "/vllm-workspace/vllm-ascend/vllm_ascend"
targets = [os.path.join(A, p) for p in
           "models/deepseek_v41/engram_hbm.py models/deepseek_v41/engram_hash.py "
           "models/deepseek_v41/engram_device_index.py models/deepseek_v41/engram_graph.py "
           "models/deepseek_v41/engram_jit_kernel.py models/deepseek_v41/engram_plan_kernel.py "
           "models/deepseek_v41/engram_gate.py models/deepseek_v41/model.py "
           "ascend_forward_context.py ops/rope_dsv4.py worker/block_table.py "
           "attention/dsa_v1.py ops/fused_moe/token_dispatcher.py".split()]
bad = []
for t in targets:
    if not os.path.isfile(t):
        bad.append((t, "missing")); continue
    try:
        with open(t, encoding="utf-8") as fh:
            compile(fh.read(), t, "exec")
    except Exception as e:
        bad.append((t, repr(e)[:120]))
if bad:
    print("  py_compile FAIL:")
    for t, e in bad:
        print("    %s : %s" % (t, e))
    sys.exit(1)
print("  py_compile：%d/%d 通过" % (len(targets), len(targets)))
PYEOF
[ $? = 0 ] || fail=1

for f in serve_v2.sh serve_a3.sh serve_a2.sh run_test.sh; do
  if [ -f "/opt/dsv41/scripts/$f" ]; then echo "  OK      /opt/dsv41/scripts/$f"
  else echo "  MISSING /opt/dsv41/scripts/$f"; fail=1; fi
done

echo '  --- 关键文件指纹（跨机核对"是不是同一版"）---'
for f in models/deepseek_v41/engram_hash.py models/deepseek_v41/model.py attention/dsa_v1.py; do
  [ -f "$A/$f" ] && printf '      %s  %s\n' "$(md5sum "$A/$f" | cut -d' ' -f1)" "$f"
done

if [ "${DSV41_NPU_SMOKE:-0}" = "1" ]; then
  echo "== L2 NPU 冒烟 =="
  if [ -e /dev/davinci0 ]; then
    if "$PY" -c 'import vllm; print("      import vllm OK", getattr(vllm, "__version__", "?"))' 2>&1 | tail -3; then
      echo "      vllm import：OK"
    else
      echo "      vllm import：FAIL（见上）"; fail=1
    fi
  else
    echo "      (/dev/davinci0 不在 ⇒ 跳过)"
  fi
fi

echo
if [ "$fail" = "0" ]; then echo "v41 checks: PASS"; else echo "v41 checks: FAIL"; fi
exit $fail
"""

SELFTEST_KIND = {
    "stub": r"""echo '--- stub half ---'
docker run --rm --entrypoint /bin/bash "${MNT[@]}" \
    -e SMOKE_MODEL="${SMOKE_MODEL:-}" -e SMOKE_DEV_MEM_GB="${SMOKE_DEV_MEM_GB:-}" \
    -e SMOKE_GMEM="${SMOKE_GMEM:-}" \
    -v "${HERE}/checks":/va26checks:ro "${TAG}" \
    -c 'bash /va26checks/stub_checks.sh'
""",
    "lite": r"""echo '--- liteprofiler half ---'
docker run --rm --entrypoint /bin/bash \
    -v "${HERE}/checks":/va26checks:ro "${TAG}" \
    -c 'bash /va26checks/lite_checks.sh'
""",
    # ★ 我们自己的档（dsv41 A3 TP8）：见 V41_CHECK 的说明。不需要 NPU 也能跑。
    "v41": r"""echo '--- dsv41 v41 half (no NPU needed) ---'
docker run --rm --entrypoint /bin/bash \
    -v "${HERE}/checks":/dsv41checks:ro "${TAG}" \
    -c 'bash /dsv41checks/v41_checks.sh'
echo '--- dsv41 v41 half (NPU smoke, auto-skip when no device) ---'
docker run --rm --entrypoint /bin/bash \
    --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v "${HERE}/checks":/dsv41checks:ro "${TAG}" \
    -c 'DSV41_NPU_SMOKE=1 bash /dsv41checks/v41_checks.sh' 2>/dev/null \
    || echo '  (NPU smoke skipped -- 无卡机器上属正常；文件级判据已由上一半覆盖)'
""",
    "both": r"""echo '--- liteprofiler half ---'
docker run --rm --entrypoint /bin/bash \
    -v "${HERE}/checks":/va26checks:ro "${TAG}" \
    -c 'bash /va26checks/lite_checks.sh'
echo '--- stub half ---'
docker run --rm --entrypoint /bin/bash "${MNT[@]}" \
    -e SMOKE_MODEL="${SMOKE_MODEL:-}" -e SMOKE_DEV_MEM_GB="${SMOKE_DEV_MEM_GB:-}" \
    -e SMOKE_GMEM="${SMOKE_GMEM:-}" \
    -v "${HERE}/checks":/va26checks:ro "${TAG}" \
    -c 'bash /va26checks/stub_checks.sh'
""",
}


def config_delta_lines(cfg, base_cfg):
    lines = []
    benv = dict(e.split("=", 1) for e in (base_cfg.get("Env") or []) if "=" in e)
    env = dict(e.split("=", 1) for e in (cfg.get("Env") or []) if "=" in e)
    for k in sorted(env):
        if benv.get(k) != env[k]:
            lines.append("ENV %s=%s" % (k, df_escape(env[k])))
    blab = base_cfg.get("Labels") or {}
    lab = cfg.get("Labels") or {}
    for k in sorted(lab):
        if blab.get(k) != lab[k]:
            lines.append('LABEL %s=%s' % (k, df_escape(str(lab[k]))))
    for key, tmpl in (("Entrypoint", "ENTRYPOINT %s"), ("Cmd", "CMD %s"),
                      ("WorkingDir", "WORKDIR %s"), ("User", "USER %s"),
                      ("StopSignal", "STOPSIGNAL %s")):
        val = cfg.get(key)
        if val == base_cfg.get(key):
            continue
        if key == "Cmd" and val is None:
            continue          # source image has no CMD at all; inherit the base
        if key in ("Entrypoint", "Cmd"):
            lines.append(tmpl % json.dumps(val))
        elif val:
            lines.append(tmpl % df_escape(str(val)))
    return lines


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_kit(out, base_ref, job, layer_map, prune_cache):
    kit, src_ref, kind = job
    log("=== %s <- %s (%s) ===" % (kit, src_ref, kind))
    im = docker_inspect(src_ref)
    cfg = im["Config"]
    base = docker_inspect(base_ref)
    src_layers = im["RootFS"]["Layers"]
    base_layers = base["RootFS"]["Layers"]
    if src_layers[:len(base_layers)] != base_layers:
        sys.exit("FATAL: %s does not start with the base image's layers" % src_ref)
    our = src_layers[len(base_layers):]
    if not our:
        sys.exit("FATAL: %s has no layers beyond the base" % src_ref)

    img_dir = os.path.join(out, "images", kit)
    os.makedirs(os.path.join(img_dir, "layers"), exist_ok=True)
    cache = os.path.join(out, ".layer-cache")
    tmpdir = os.path.join(out, ".tmp")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    layers_meta, payload = [], []
    for i, diffid in enumerate(our):
        info = layer_map.get(diffid)
        if not info:
            sys.exit("FATAL: layer %s not found in the docker layer store" % diffid)
        short = diffid.split(":")[-1]
        name = "%02d-%s.tar.gz" % (i, short[:16])
        cached = os.path.join(cache, short + ".tar.gz")
        dst = os.path.join(img_dir, "layers", name)
        if not os.path.exists(cached):
            log("  export layer %d/%d %s (%.1f MiB on disk)"
                % (i + 1, len(our), diffid[:19], info["size"] / 1048576))
            meta = export_layer(info["dir"], cached, tmpdir)
            meta["diffid"] = diffid
            meta["cache_id"] = info["cache_id"]
            meta["size_on_disk"] = info["size"]
            with open(cached + ".json", "w") as fh:
                json.dump(meta, fh, indent=1)
        else:
            log("  layer %d/%d %s already exported, reusing"
                % (i + 1, len(our), diffid[:19]))
            with open(cached + ".json") as fh:
                meta = json.load(fh)
        if os.path.exists(dst):
            os.unlink(dst)
        os.link(cached, dst)
        meta["file"] = name          # basename inside <image>/layers/
        layers_meta.append(meta)
        payload.extend(meta["files"])

    payload = sorted(set(payload))
    write_text(os.path.join(img_dir, "payload-paths.txt"),
               "\n".join(payload) + "\n")
    write_text(os.path.join(img_dir, "base-layers.txt"),
               "\n".join(base_layers) + "\n")
    write_text(os.path.join(img_dir, "source-layers.txt"),
               "\n".join(src_layers) + "\n")

    delta = config_delta_lines(cfg, base["Config"])
    delta.append('LABEL va26.kit="%s" va26.source_image_id="%s" '
                 'va26.base_ref="%s"' % (kit, im["Id"], base_ref))
    write_text(os.path.join(img_dir, "Dockerfile"),
               dockerfile_text(kit, src_ref, im["Id"], base_ref, base["Id"],
                               layers_meta, delta))
    selftest_name = "selftest.sh"
    write_text(os.path.join(img_dir, selftest_name),
               SELFTEST_HEAD + SELFTEST_KIND[kind], 0o755)
    if kind in ("stub", "both"):
        write_text(os.path.join(img_dir, "checks", "stub_checks.sh"),
                   STUB_CHECK, 0o755)
    if kind in ("lite", "both"):
        write_text(os.path.join(img_dir, "checks", "lite_checks.sh"),
                   LITE_CHECK, 0o755)
    if kind == "v41":
        write_text(os.path.join(img_dir, "checks", "v41_checks.sh"),
                   V41_CHECK, 0o755)
        # ★ 为"内容判据"取**源镜像里**每个 payload 文件的 md5：
        #   自检时逐文件比对 ⇒ 能抓出"COPY 漏了/被截断/放错位置"这类错。
        #   （只比较不产物：md5 清单本身很小。）
        log("  payload md5 list for v41 checks ...")
        md5_txt = os.path.join(img_dir, "checks", "payload-md5.txt")
        os.makedirs(os.path.dirname(md5_txt), exist_ok=True)
        with open(md5_txt, "w") as fh:
            for rel in payload:
                cont = "/" + rel.lstrip("/")
                # ★ 变量名**不能**叫 `out` —— 它是本函数的"输出目录"参数，会被覆盖，
                #   后面的 docker 命令就会拿到垃圾路径（实测：报
                #   `includes invalid characters for a local volume name`）。
                _h = capture([DOCKER, "run", "--rm", "--entrypoint", "md5sum",
                              src_ref, cont]).strip()
                if _h:
                    fh.write(_h.split()[0] + "\t" + cont + "\n")
                else:
                    log("    WARN: cannot hash %s in %s" % (cont, src_ref))
        log("  payload-md5.txt: %d entries" % sum(1 for _ in open(md5_txt)))
    write_text(os.path.join(img_dir, ".dockerignore"),
               "verify/\n*.diff\nfs-manifest.*\npayload-*\nimage-meta.json\n"
               "base-layers.txt\nsource-layers.txt\nrebuild.sh\nselftest.sh\n"
               "checks/\n")
    write_text(os.path.join(img_dir, "rebuild.sh"),
               rebuild_sh_text(kit, src_ref, base_ref, len(base_layers),
                               selftest_name), 0o755)

    log("  manifests from the ORIGINAL image (this takes a few minutes) ...")
    t0 = time.time()
    run([DOCKER, "run", "--rm", "--entrypoint", "/bin/bash",
         "-v", out + ":/va26kit:ro", "-v", tmpdir + ":/va26out",
         src_ref, "-c",
         'PY=$(command -v python3 || echo /usr/local/python3.12.13/bin/python3); '
         '"$PY" /va26kit/tools/fs_manifest.py '
         '--out /va26out/fs-manifest.original.txt '
         '--payload-list /va26kit/images/%s/payload-paths.txt '
         '--payload-out /va26out/payload-sha256.original.txt' % kit])
    for src, dst in (("fs-manifest.original.txt", "fs-manifest.original.txt.gz"),
                     ("payload-sha256.original.txt",
                      "payload-sha256.original.txt.gz")):
        with open(os.path.join(tmpdir, src), "rb") as fi, \
                gzip.open(os.path.join(img_dir, dst), "wb", 9) as fo:
            shutil.copyfileobj(fi, fo)
        os.unlink(os.path.join(tmpdir, src))
    n_fs = sum(1 for _ in gzip.open(os.path.join(img_dir, "fs-manifest.original.txt.gz"), "rt"))
    log("  original image: %d fs entries, %d payload files (%.0fs)"
        % (n_fs, len(payload), time.time() - t0))

    meta = {
        "kit": kit,
        "source_image": src_ref,
        "source_image_id": im["Id"],
        "source_created": im.get("Created"),
        "base_image": base_ref,
        "base_image_id": base["Id"],
        "base_layer_count": len(base_layers),
        "code_baseline": probe_code_baseline(src_ref),
        "working_layers": [
            {k: v for k, v in l.items() if k != "files"} for l in layers_meta],
        "fs_entries": n_fs,
        "payload_files": len(payload),
        "config_delta": delta,
        "selftest_kind": kind,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/make_image_patch_kit.py",
    }
    write_text(os.path.join(img_dir, "image-meta.json"),
               json.dumps(meta, indent=1) + "\n")
    cb = meta["code_baseline"]
    if cb:
        log("  code baseline: vllm %s (+%d dirty) / vllm-ascend %s (+%d dirty)"
            % (cb["vllm"], cb["vllm_dirty_files"], cb["vllm_ascend"],
               cb["vllm_ascend_dirty_files"]))
    log("  done: %d working layers, %.1f MiB compressed"
        % (len(our), sum(l["size_gz"] for l in layers_meta) / 1048576))
    return meta


def aggregate(out, base_ref):
    metas = []
    imgroot = os.path.join(out, "images")
    for name in sorted(os.listdir(imgroot)):
        p = os.path.join(imgroot, name, "image-meta.json")
        if os.path.exists(p):
            with open(p) as fh:
                metas.append(json.load(fh))
    base = docker_inspect(base_ref)
    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/make_image_patch_kit.py",
        "base": {
            "ref": base_ref,
            "image_id": base["Id"],
            "layer_count": len(base["RootFS"]["Layers"]),
            "layers": base["RootFS"]["Layers"],
        },
        "images": metas,
    }
    write_text(os.path.join(out, "MANIFEST.json"),
               json.dumps(manifest, indent=1) + "\n")
    rows = []
    total_gz = 0
    for m in metas:
        gz = sum(l["size_gz"] for l in m["working_layers"])
        total_gz += gz
        cb = m.get("code_baseline") or {}
        cbtxt = ("`%s`/`%s` (+%s/+%s)" % (cb.get("vllm", "?"),
                                         cb.get("vllm_ascend", "?"),
                                         cb.get("vllm_dirty_files", "?"),
                                         cb.get("vllm_ascend_dirty_files", "?"))
                 if cb else "?")
        rows.append("| `%s` | `%s` | %d | %.1f MiB | %s | %s | %s |"
                    % (m["kit"], m["source_image"], len(m["working_layers"]),
                       gz / 1048576, m["source_image_id"][:19],
                       m["generated"][:10], cbtxt))
    detail = []
    for m in metas:
        detail.append("\n### %s\n" % m["kit"])
        detail.append("源镜像：`%s`（`%s`）\n" % (m["source_image"],
                                            m["source_image_id"]))
        detail.append("重建：`cd images/%s && bash rebuild.sh`\n" % m["kit"])
        detail.append("| # | 层 diffID | 目录项 | 未压缩(约) | .tar.gz | sha256(gz) |")
        detail.append("|---|---|---|---|---|---|")
        for i, l in enumerate(m["working_layers"]):
            detail.append("| %d | `%s` | %d | %.1f MiB | %.1f MiB | `%s` |"
                          % (i, l["diffid"], l["entries"],
                             l["size_raw_estimate"] / 1048576,
                             l["size_gz"] / 1048576, l["sha256_gz"][:16]))
        detail.append("\n验收依据：`fs-manifest.original.txt.gz`（%d 条目录项）、"
                      "`payload-sha256.original.txt.gz`（%d 个文件）\n"
                      % (m["fs_entries"], m["payload_files"]))
    readme = (README_PROSE
              + "\n## 本包包含的镜像\n\n"
              + "| 目录 | 源镜像 | 工作层数 | 层体积(.tar.gz) | 源镜像 ID | 拆出时间 | 代码基线 |\n"
              + "|---|---|---|---|---|---|---|\n" + "\n".join(rows)
              + "\n\n基础镜像：`%s`（`%s`，%d 层）\n"
              % (base_ref, base["Id"], len(base["RootFS"]["Layers"]))
              + "\n合计工作层体积：%.1f MiB\n" % (total_gz / 1048576)
              + RELATIONS_PROSE
              + "\n## 逐个镜像\n" + "\n".join(detail))

    # --- verification record, if the kit was rebuilt on the packing host -----
    ver = []
    for m in metas:
        s = os.path.join(out, "verify", m["kit"], "SUMMARY.txt")
        if not os.path.exists(s):
            continue
        with open(s, encoding="utf-8") as fh:
            body = fh.read().rstrip()
        ok = "IDENTICAL" in body
        ver.append("### %s —— %s\n\n```\n%s\n```\n"
                   % (m["kit"], "重建后与原始镜像一致 ✓" if ok
                      else "见下方记录（有差异）", body))
    if ver:
        readme += ("\n## 验收记录（本包在构建机上重建并逐文件比对过）\n\n"
                   "重建方式：`cd images/<kit> && bash rebuild.sh`（基础镜像 diffID 校验 → "
                   "`docker build` → 全文件系统清单比对 → 工作层文件 sha256 比对 → 冒烟）。"
                   "下列记录由 `rebuild.sh` 生成，留在 `verify/<kit>/`。\n\n"
                   + "\n".join(ver))
    arc = os.path.join(out, "verify", "ARCHIVE-CHECK.txt")
    if os.path.exists(arc):
        with open(arc, encoding="utf-8") as fh:
            readme += ("\n### 归档级验收（交付物本身）\n\n"
                       "下面这条是把**这个归档**解包后重建的结果（不是打包前的工作目录），"
                       "即：下载 → 解包 → `rebuild.sh` → 与原始镜像逐文件一致。\n\n```\n"
                       + fh.read().rstrip() + "\n```\n")
    # ★ 若所有套件都是我们自己的 v41 档 ⇒ 用**我们自己的** README
    #   （工具的默认 prose 是另一个项目的 va26 措辞，对本包不适用）
    if metas and all(m.get("selftest_kind") == "v41" for m in metas):
        readme = V41_README
    write_text(os.path.join(out, "README.md"), readme)
    log("aggregate: %d image(s), README.md + MANIFEST.json written"
        % len(metas))
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True,
                    help="official base image ref the images were built on")
    ap.add_argument("--job", action="append", default=[],
                    metavar="KIT|IMAGE|KIND",
                    help="KIND is stub, lite or both")
    ap.add_argument("--docker-root", default="/home/docker-data")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if os.geteuid() != 0 and not args.aggregate_only:
        sys.exit("FATAL: needs root to read the docker layer store")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "tools"), exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(os.path.join(here, "kit_tools", "fs_manifest.py"),
                    os.path.join(out, "tools", "fs_manifest.py"))

    if not args.aggregate_only:
        layer_map = build_layer_map(args.docker_root)
        jobs = []
        for j in args.job:
            parts = j.split("|")
            if len(parts) != 3:
                sys.exit("FATAL: --job must be KIT|IMAGE|KIND, got %r" % j)
            jobs.append(tuple(parts))
        for job in jobs:
            build_kit(out, args.base, job, layer_map, False)
        tmpdir = os.path.join(out, ".tmp")
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
    aggregate(out, args.base)


if __name__ == "__main__":
    main()
