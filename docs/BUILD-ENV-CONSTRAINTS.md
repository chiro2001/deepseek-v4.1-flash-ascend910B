# 构建环境约束（vllm-ascend csrc / opp 手工改库）

**日期**：2026-10-05　**来源**：metadata 线 + MoE 路由线（两条独立线各自踩到，已复现）
**适用**：在 `dsv41-op-hcfuse` 里改 `csrc/` 下的算子源码并产出可加载的库
**标注**：【实测】= 本仓真机复现过

---

## 0. 一句话

改算子源码后**不要跑 `cmake` 重新配置**；用 `ninja -t commands` 导出单点命令、
**手工编译 + 手工重链**，产出到独立路径 —— 这是本环境唯一安全的改库方式。

---

## 1. ⚠️ 不要重跑 cmake（最重要）【实测】

`csrc/moe/` 下有 **20+ 个 `.bak*` / `.bakDBG*` / `.bakFIX*` 备份目录也带 `CMakeLists.txt`**，
而 **当前 `build.ninja` 里它们出现 0 次**：

```bash
# 证据
ls /vllm-workspace/vllm-ascend/csrc/moe/ | grep bak | wc -l     # ≈20+
grep -c "bakA" /vllm-workspace/vllm-ascend/csrc/build/build.ninja   # 0
```

原因：`csrc/moe/CMakeLists.txt` 是 **`file(GLOB ...)` 遍历子目录**，凡有 `CMakeLists.txt` 的一律
`add_subdirectory` ⇒ **一旦重跑 cmake，这些垃圾目录会被收进构建**（编译报错或产出污染）。

**推论**：想加一个新算子目录（如本轮把 `moe_init_routing_v3` 放进 `csrc/moe/`），
**不能靠重配 cmake 让它进构建** —— 要么手工编译，要么先把 `.bak*` 清干净（**不推荐**，那会破坏别人的回退点）。

---

## 2. 安全改库流程：`ninja -t commands` + 手工编译 + 手工重链【实测】

### 2.1 取模板
```bash
B=/vllm-workspace/vllm-ascend/csrc/build
# 编译模板：找一个**同类型**的已有源文件（同目标、同编译选项）
ninja -C $B -t commands "<某个已有 .cpp.o 目标>" | tail -1 > cc_template.txt
# 链接模板：目标库名
ninja -C $B -t commands libcust_opmaster_rt2.0.so | tail -1 > ld_template.txt
```

### 2.2 替换三件套
从模板里替换：① 源文件路径 ② `-o <对象>` ③ `-MF/-MT <依赖文件>`
再按需追加 `-I<你自己的头目录>`。

### 2.3 手工重链（关键）
把新 `.o` **插进链接命令的对象列表**，并把 `-o <库>` 改成独立输出路径 —— **绝不覆盖原库**。

### 2.4 校验产物（做完必查）
```bash
strings -a <新库> | grep -c <你的算子名>     # 应从 0 变成 >0
strings -a <新库> | grep -c <原有算子名>     # 应保持不变（证明没丢东西）
```

---

## 3. 两个易踩的路径陷阱【实测】

| 陷阱 | 现象 | 正确做法 |
|---|---|---|
| **vendor 的 `op_tiling/liboptiling.so` 是符号链接** | 只替换它，运行时可能仍加载旧库 | 替换**真实目标** `op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so`（`readlink` 先确认） |
| **`~` 在 `docker exec ... bash -lc` 里展开成 `/root`** | 脚本"静默没执行"（`grep -c error` 得 0 会被误读成成功） | 容器内路径**一律用绝对路径** `/home/<user>/...` |

---

## 4. 版本漂移：容器内头 vs opensrc 头【实测】

从 `opensrc/ops-transformer-master` 取源码时，**include 路径与容器内的旧头布局不一致**：

| 项 | 容器内（旧） | opensrc（新） |
|---|---|---|
| tiling 基类头 | `tiling_base/tiling_base.h` | `op_host/tiling_base.h` |
| 模板注册 | `tiling_base/tiling_templates_registry.h` | `op_host/tiling_templates_registry.h` |
| `AiCoreParams` 字段 | `blockDim` | `numBlocks` |

**兼容手段**（按代价从低到高）：
1. **shim 转发头**：建 `<yourdir>/op_host/tiling_base.h` → `#include "tiling_base/tiling_base.h"`，
   加 `-I<yourdir>`（本轮用这条，一次成功）
2. **字段改名**：`numBlocks` → `blockDim`（两版结构体除名字外一致、且该字段**只写不读**时安全）
3. ⚠️ **不要混用两棵树**：`tiling_templates_registry.h` 两版差 **367 行**（真版本漂移），
   把 opensrc 的头整棵加进 `-I` 可能二义或行为不一致 —— **优先 shim 到容器内的头**

---

## 5. 构建缓存不会自动失效【实测】

* `cache/skcache/compile_outputs`（≈13 GB，含静态内核产物）**不会**因为换了 kernel/tiling 而失效
  ⇒ 服务侧可能**静默复用旧产物**。已加开关 `V41_OPP_CLEAR_SKCACHE=1`（会重编译，起服 +18 min）。
* 容器内另有 `/root/atc_data/kernel_cache`（ATC 内核缓存）—— 排查"改了为什么不生效"时应优先怀疑它。

---

## 6. 验证纪律（血泪）

1. **必须用"资源计数 / 结构指纹"验执行，不要用 `kernel_name`**
   —— 已被两次证明会给出假象。
2. **判"支持/不支持"必须有对照组**
   —— 例：判某个算子不支持某模式时，要同时跑一个"应当支持"的臂；否则分不清是"不支持"还是"构造错了"。
3. **只调一次的探针不可信**
   —— 例：`MoeInitRoutingV3` 在进程内**首次调用必失败**（`Cannot find binary`），第 1 次起成功。
   **任何单发探针都要"至少调用两次再判定"**。
4. **优先用 env 门控的硬判据**
   —— 例：`CAND_A_ABORT=1` 时让被测代码 `return GRAPH_FAILED`；
   若算子仍成功 ⇒ 该代码路径**肯定没被执行**。这比看日志/看 profiler 都可靠。

---

## 7. 交付物落点（本轮已产出）

| 文件 | 用途 |
|---|---|
| `~/tmp/moe/build_tiling.sh` / `link_tiling.sh` | 改一个**内置**算子的 tiling 并手工重链 |
| `~/tmp/moe/build_pathY.sh` / `link_pathY.sh` | 追加 proto + infershape 到 opsproto |
| `~/tmp/moe/make_vendor.sh` / `make_vendor_pathY.sh` | 在 vendor 副本上替换库并校验符号 |
| `~/tmp/metadata/tools/rebuild_aicpu.sh` | AICPU kernel 重编（**另一类坑**：`.so` 是 CUSTOM_COMMAND，`.o` 只是 order-only 依赖 ⇒ ninja 不重链） |
