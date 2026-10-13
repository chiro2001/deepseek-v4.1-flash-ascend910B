# 如何从源码重建本包的二进制

本包的 payload 是**构建产物**。本文给出把它**从源码重新造出来**的完整步骤 ——
这是"可复现"的另一半（另一半是 `package_release.sh` 的可复现打包 + `verify_consistency.sh` 的逐字节校验）。

源码在 [`experimental/fusion-3out/`](../../experimental/fusion-3out/)，
构建脚本在 [`.build/`](.build/)（本目录，随包发布）。

---

## 0. 依赖

| 项 | 要求 |
|---|---|
| 构建容器 | 一个装了 CANN 9.1 + vllm-ascend 源码树的容器（本文用 `dsv41-op-hcfuse`） |
| 源码树 | `/vllm-workspace/vllm-ascend/`（含 `csrc/`，且是**干净**的：没有历史实验留下的算子备份目录） |
| 磁盘 | ≥ 5 GB 空闲（`csrc/build` 会到百 MB 级） |
| 时间 | ops 编译 **≈9.5 分钟**；torch 扩展 **≈2 分钟** |

> ⚠️ **前置坑（踩过）**：`csrc/attention/` 下如果留着 `*.bak*` / `*.armA_staged` 之类的
> **历史备份目录**，CMake 的 GLOB 会把它们当算子扫进去，构建在 4 秒内失败并报
> `target 不存在`。构建前先 `find csrc -maxdepth 3 -name '*.bak*' -o -name '*.armA_staged'`
> 确认干净。

---

## 1. 部署算子源码

```bash
CT=<构建容器>
SRC=/vllm-workspace/vllm-ascend/csrc

# ① 把源码放进 csrc/attention/（目录名必须与算子 snake_case 名一致，CMake 自动发现）
docker cp experimental/fusion-3out/rms_norm_dynamic_quant_bf16 \
          $CT:$SRC/attention/rms_norm_dynamic_quant_bf16

# ② 把算子加进 A3 构建清单
docker cp .build/patch_build_aclnn.py $CT:/tmp/
docker exec $CT python3 /tmp/patch_build_aclnn.py      # 幂等
```

## 2. 编译算子 → 得到 OPP 安装包

```bash
docker exec $CT bash -lc "
  cd $SRC
  rm -rf build                                   # ★ 必须整删；部分删会 CMake 失败
  bash build.sh --pkg \
       --ops='rms_norm_dynamic_quant,rms_norm_dynamic_quant_bf16' \
       --soc=ascend910_93
"
# 期望：EXIT=0，产物 $SRC/build/cann-ops-transformer-custom_linux-aarch64.run
```

## 3. 装出 OPP 树（payload 的 A1）

```bash
docker exec $CT bash -lc "
  rm -rf /tmp/3out_stage
  bash $SRC/build/cann-ops-transformer-custom_linux-aarch64.run \
       --quiet --install-path=/tmp/3out_stage
"
# 期望：/tmp/3out_stage/vendors/custom_transformer/{op_api,op_impl,op_proto,...}
#       且 op_impl/.../kernel/ascend910_93/ 下有两个算子目录
```

## 4. 编译 torch 扩展（payload 的 A2）

```bash
# ① 加绑定（幂等；插入实现 + 注册 npu_rms_norm_dynamic_quant_bf16）
docker cp .build/add_binding.py $CT:/tmp/
docker exec $CT python3 /tmp/add_binding.py
docker exec $CT grep -c RMS_NORM_DYNAMIC_QUANT_BF16_BIND $SRC/torch_binding.cpp   # 期望 2

# ② 只编扩展、跳过 ops（ops 已在第 2 步编完）
docker exec $CT bash -lc "
  cd /vllm-workspace/vllm-ascend
  rm -rf build
  export VLLM_SKIP_OPS_BUILD=1        # .build/patch_build_aclnn.py 加的 guard
  python3 setup.py build_ext --inplace
"
# 期望：EXIT=0，产物 vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
```

> 扩展与 ops 是**解耦**的：扩展靠运行时 `dlopen` 找 aclnn 符号
> （`EXEC_NPU_CMD` 用 `GetOpApiFuncAddr` 动态查找），所以能跳过 ops 单独重编。

## 5. 组装 payload

```bash
# 从容器直接取（开发口径）
bash build_payload.sh --from-container $CT

# 或：打成产物包再组装（发布口径）
docker exec $CT bash -lc "cd /tmp/3out_stage && tar czf /tmp/art.tgz ."   # 加上 so/ 与 py/
bash build_payload.sh --artifacts <ar.tgz>
```

`build_payload.sh` 会做**内容抽查**：两个算子目录必须都在、两个 `.py` 必须带 `LNORM-FUSE` 标记。

## 6. 证明这次构建与发布包等价

```bash
# 装进一个测试容器
bash install.sh <test-container>

# 逐字节比对：容器 vs 本包 payload
bash verify_consistency.sh <test-container>
```

`verify_consistency.sh` 逐文件 sha256 比对，**任一不一致即 FAIL**。
这就是"你重建出来的和我发出去的，是同一批文件"的判据。

---

## 7. 已知的**不可复现**部分（诚实声明）

| 项 | 状态 |
|---|---|
| OPP 树的**文件内容** | ✅ 可复现（源码确定 ⇒ kernel 二进制确定，同一 CANN 版本下实测一致） |
| 两个算子的 **kernel 二进制名** | ❌ 带 32 位哈希（`RmsNormDynamicQuantBf16_<hash>.o`），哈希由编译输入决定；**同源码 + 同 CANN 版本得到同名**，换 CANN 版本会变 |
| torch 扩展 `.so` | ⚠️ 内容可复现，但**嵌入的构建时间戳/路径**未做 `-ffile-prefix-map` 归一 ⇒ **不同构建目录下 sha256 可能不同**；此时以 `verify_consistency.sh` 的**逐文件比对**为准，而不是比 `.so` 的 sha256 |
| `.run` 安装包本身 | ❌ 含时间戳，同输入两次构建 sha256 不同（所以本包不发 `.run`，只发**已安装树**） |

> 结论：**发布包的身份由 `PAYLOAD.sha256` 定义**（逐文件），
> 不用"归档 sha256"表达"内容相同"。归档 sha256 只保证**传输完整**。

---

## 8. 构建期的两个隐蔽陷阱（都会让产物"看起来正常但不对"）

1. **宏里的行续接符 `\` 前不能放 `//` 注释**
   C 预处理的**行拼接（阶段 2）先于注释处理（阶段 3）**，注释里的 `\` 仍然续行，
   会把下一行并进注释删掉。本算子踩过：`op.Process()` 被吞 ⇒ kernel 照常
   launch、日志无 ERROR、**输出全是未初始化内存**。
   已用最小 gcc 例子验证。排查手段：`cat -A` 看行尾，或用
   `output = torch.empty(...)` + 连续多次调用比对（结果每次不同 ⇒ 从未被写）。
2. **新增 UB buffer 必须同步进 tiling 的预算公式**
   `CheckUbNormalTiling()` 按每列字节数算 `rowStep`。kernel 里多 `InitBuffer`
   一个 buffer 而不改公式 ⇒ 超出 UB ⇒ `InitBuffer` **静默失败**、kernel 什么都不写。

两条的完整复盘见 `docs/MULTI-OUT-OP-IMPLEMENTATION-20261008.md` §8 / §9。
