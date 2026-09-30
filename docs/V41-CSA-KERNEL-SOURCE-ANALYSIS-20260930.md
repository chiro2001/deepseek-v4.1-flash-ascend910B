# SparseFlashMla（CSA 模板）源码级分析 —— DCP8 NaN / 非确定的根因定位

> 分析对象：vLLM-Ascend 随包发布的 AscendC 自定义算子 **SparseFlashMla** 的 **CSA 模板**
> （`custom_transformer` vendor），commit `e43cf1e9f5d9bead076853aa6bcacb671465de94`
> （`https://github.com/vllm-ascend/DSv4.1`，`vllm-ascend 0.1.dev5097+ge43cf1e9f`）。
> CANN 侧另有一份内置实现（`cann-9.1.0/opp/.../sparse_flash_mla/`，用 `scfa` 命名），
> 我们实际走的是 **custom_transformer 的 `csa`** 那份（错误信息里的
> `Aurora SparseFlashMla only compiles SWA and CSA templates` 即来自它）。
>
> 约定：【实测】= 真机/单卡跑出来的；【推断】= 源码 + 算术推出；【未确认】= 未验。

---

## 0. 结论摘要

| # | 发现 | 性质 |
|---|---|---|
| 1 | 内核的因果界 `cmpS2IdLimit` 由 `actS1Size`（**整段 Q 长度**）与 `actCmpS2Size`（**本 rank 的压缩长度**）推出，隐含 `cseq × ratio ≈ T`。DCP8 下该假设被破坏，界被**整体平移 −776** | 【实测+推断】**已确认** |
| 2 | 该界用来**丢弃**越界的键：`GetKeyGmOffset` 返回 −1 → `CopyInSingleKv` 直接 return，**不增加 `mte2Size`** | 【推断】**已确认（源码）** |
| 3 | 向量阶段只写了 `mte2Size` 行到 `kvMergeGm_`，而矩阵阶段读 `actualSingleProcessSInnerSize` 列 —— **两阶段之间没有"实际条数"的握手** | 【推断】**已确认（源码）** |
| 4 | ⇒ 未写到的行读到**上一次请求的残留/未初始化内存** → 产生 NaN 与非确定 | 【推断】**主因假设** |
| 5 | 预测"会丢弃键"的行区间 **[780, 892]** 与实测 NaN 行区间 **[800, 903]** 高度重叠 | 【实测】**强佐证** |
| 6 | 仍**未解释**：行 [893, 903] 也出现 NaN；索引改升序后仍非确定 | 【未确认】 |

---

## 1. 关键源码路径（文件与行号）

| 文件 | 关键位置 |
|---|---|
| `.../arch22/sparse_flash_mla_csa_kernel.h` | 43 `actS1Size` 声明；804 取整段 Q 长度；418–431 `GetSparseActualSeqLen`；433–460 `CountValidCmpSparseLen`（二分）；674/685 `cmpS2IdLimit` 赋值；828–830 `cmpMaskRight` |
| `.../arch22/sparse_flash_mla_csa_block_vector.h` | 525–535 `GetRealS2Idx`；538–546 `GetKeyGmOffset`（越界返回 −1）；572–591 `CopyInSingleKv`（丢弃键且不加计数）；594–633 `CopyInKv`；635–656 `CopyOutMrgeResult`；658–699 `ProcessVec0L`（打包循环） |
| `.../arch22/sparse_flash_mla_csa_block_cube.h` | 325/337/552/558/589 —— 矩阵阶段按 `actualSingleProcessSInnerSize(+Align)` 读取 `kvMergeGm_` |

---

## 2. 发现 1：因果界 `cmpS2IdLimit` 的坐标系假设

内核（`sparse_flash_mla_csa_kernel.h`）：

```cpp
int32_t actS1Size = 0;   // 43: “TND场景下当前Batch循环处理的S1轴的大小”
...
tempLoopInfo.actS1Size = GetActualSeqLenQ(tempLoopInfo.bIdx);   // 804: 整段 Q 长度
...
tempLoopInfo.s1EndIdx = Min(s1StartIdx + mBaseSize/gSize - 1, actS1Size - 1);  // 819
tempLoopInfo.cmpMaskRight = cmpMaskS2Size - tempLoopInfo.actS1Size;            // 830
// GetCmpMaskS2Size: return actualCmpS2Size * cmpRatio + residual;              // 402
...
int32_t thresHold = (tempLoopInfo.cmpMaskRight + tempLoopInfo.s1EndIdx + 1) / constInfo.cmpRatio;  // 426
int32_t bound = Min(actCmpS2Size, Min(sparseBlockCount * sparseBlockSize, Max(thresHold, 0)));
tempLoopInfo.actCmpS2Size = Min(bound, CountValidCmpSparseLen(bound));          // 427–431
...
info.cmpS2IdLimit = (tempLoopInfo.cmpMaskRight + tempLoopInfo.s1EndIdx + 1) / constInfo.cmpRatio;  // 674/685
```

化简（`ratio=1`、`residual=0`）：

```
cmpMaskRight = cseq − T
thresHold(t) = cseq − T + t + 1
```

⇒ 该式**只在 `cseq × ratio ≈ T` 时退化成 `thresHold(t) = t + 1`**（即"压缩 token g 对应未压缩 g·ratio"的全局线性映射）。

**DCP1（不切分）**：`cseq = 904`、`T = 904` ⇒ `cmpMaskRight = 0` ⇒ `thresHold = t+1`，被 `sparseBlockCount=512` 截顶 ⇒ **所有键都保留**。**这就是 DCP1 一直正确的原因。**

**DCP8（每 rank 1/8）**：`cseq = 128`、`T = 904` ⇒ `cmpMaskRight = −776` ⇒

```
thresHold(t) = t − 775
```

即界被**整体平移 −776**（776 = 904 − 128 = 本 rank 不持有的 token 数）。

**【实测】离线核算（`verify_s2idlimit.py`，跑真实 dump）**：

```
T=904 ratio=1 cseq(本rank行数)=128 actS1=904
cmpMaskS2Size = 128*1 = 128
cmpMaskRight  = 128 - 904 = -776
  t      s2IdLimit    n_valid   内核会用    索引>=界的个数   后果
  0      -775         1         0          0              ok
  240    -535         32        0          0              ok
  480    -295         64        0          0              ok
  720    -55          55        0          0              ok
  840    65           67        65         46             ★ 有键被丢弃
  903    128          63        63         0              ok
汇总：有键被静默丢弃的行数 = 112/904（12.4%），累计丢弃键数 = 1899
预测丢弃键的行区间 = [780, 892]
预测 actCmpS2Size=0（完全跳过 cmp）的行 = 776 个，区间 [0, 775]
```

> ⚠️ 这里同时暴露一个**语义问题**（不止 NaN）：对 `t ≤ 775` 的行，内核算出
> `thresHold ≤ 0` ⇒ **完全跳过压缩注意力**（只用 128 token 的 ori 滑窗）。
> 这对 DCP8 是错的——本 rank 的 128 行覆盖的是整条序列的 1/8，不是"序列开头"。

---

## 3. 发现 2 + 3：丢键之后，两阶段之间没有"实际条数"握手

```cpp
// block_vector.h:538  —— 越界即返回 -1
__aicore__ inline int64_t SMLAVectorBlock<SMLAT>::GetKeyGmOffset(
        int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit) {
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) { return -1; }
    ... // PA_BBND: 由 cmpBlockTableGm_ 取物理块
}

// block_vector.h:572  —— 拿到 -1 就**直接返回，不计入 mte2Size**
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyInSingleKv(
        int64_t &mte2Size, int64_t mte3Size, ..., int64_t keyBNBOffset, ...) {
    if (keyBNBOffset < 0) { return; }          // ← 键被静默丢弃
    ...
    mte2Size += validS2Count;                   // ← 只有成功拷贝才加计数
}

// block_vector.h:635  —— 只把**已拷贝**的行搬到 kvMergeGm_
__aicore__ inline void SMLAVectorBlock<SMLAT>::CopyOutMrgeResult(
        int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset, ...) {
    if (mte2Size <= mte3Size) { return; }
    dataCopyParams.blockCount = mte2Size - mte3Size;        // 实际条数
    DataCopyPad(kvMergeGm_[runInfo.cmpLoop % MERGE_CACHE_GM_BUF_NUM * 512 * 512 +
                           (s2GmStartOffset + mte3Size) * constInfo.headDim], ...);
}

// block_vector.h:658  —— 打包循环：**遇到第一个 -1 就 break**（假设有效项是连续前缀）
for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; ...) {
    GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
    if (unlikely(s2IdxArray0 < 0)) { CopyOutMrgeResult(...); ...; break; }
    ...
}
```

而**矩阵阶段**（`sparse_flash_mla_csa_block_cube.h`）读的是**期望长度**：

```cpp
uint32_t nSize = info.actualSingleProcessSInnerSize;                                     // 337
CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign); // 326
uint32_t kSize = info.actualSingleProcessSInnerSize;                                     // 589
```

**两个阶段共享 `kvMergeGm_`，但只传了"期望长度"，没有传"实际写入条数"。**
当存在被丢弃的键（发现 1 证明 DCP8 下必然发生）时：
`mte2Size < actualSingleProcessSInnerSize` ⇒ `kvMergeGm_` 尾部
`[mte2Size, actualSingleProcessSInnerSize)` **本次从未写过** ⇒ 读到残留/未初始化内容。

**【实测】区域比对（`region_match.py` + `nan_rows.py`，同一份真实 dump）**：

| | 行区间 |
|---|---|
| 公式**预测**会丢键 | **[780, 892]**（112 行） |
| **实测** `lse` 出现 NaN | **[800, 903]** |

⇒ 高度重叠。这是"丢键 ⇒ 未写洞 ⇒ 读残留 ⇒ NaN"链条的**强佐证**。

另注：`ProcessVec0L` 在 `else`（非 `HEAD_RATIO_ONE`）分支里把
`info.v0S2DealSize` **恒设为 512**，而矩阵阶段按
`actualSingleProcessSInnerSize` 读——两者口径也不一致（【推断】同一类握手缺失）。

---

## 4. 为什么这解释了我们全部的实测现象

| 实测现象 | 本分析的解释 |
|---|---|
| 输入全有限但输出有 NaN | 读的是 `kvMergeGm_` 里**从未写过的行** |
| 同一份输入两次调用结果不同、NaN 个数在 5xxx 波动 | 残留内容取决于**上一次请求**写进去的数据 |
| DCP1 完全正常 | `cseq×ratio = T` ⇒ 界正确 ⇒ **不丢键** ⇒ 缓冲完整写入 |
| DCP8 在 `T ≤ 450` 正常、`T ≥ 560` 失败 | 平移量 = `T − cseq×ratio` ≈ `T×(1−1/8)`，随 `T` 增大；`T` 越大丢键越多 |
| 索引值域跨 ≥3 页更容易触发 | 索引值越大越容易 `≥ s2IdLimit` ⇒ 丢键更多 |
| NaN 只出现在**靠后的行** | 平移后只有晚期的 `t` 才满足 `thresHold>0` 且 `n_valid>thresHold` |
| 块表全列指向同一页也触发 | 洞读到的物理页不同 ⇒ 残留数据的分布不同 |
| 单卡隔离（无 DCP/无服务栈）仍复现 | 这是内核内部行为，与上层无关 |

---

## 5. 尚未解释（诚实标注）

1. **行 [893, 903] 的 NaN**：这三个/十一个行按公式**不该丢键**，需第二条路径。
   候选：`v0S2DealSize=512` 与 `actualSingleProcessSInnerSize` 口径不一致、
   `s2LoopTimes` 多轮时的 `v0S2Start=512` 特例（kernel:686–690）。
2. **索引改成升序（`replay_idxvar.py` 的 `prefix`）后仍非确定**：按上述模型应无丢键。
   ⇒ 说明还有一条不依赖"索引值 ≥ 界"的路径【未确认】。
3. `CountValidCmpSparseLen` 的**二分**依赖"有效项是连续前缀"。
   我们侧已做稳定压缩（`remap_sparse_indices` 的 stable compaction）保证这一点，
   但这是个**脆弱契约**——任何非前缀布局都会让它算错。

---

## 6. 对算子团队的建议（按优先级）

1. **加"实际条数"握手**：向量阶段把 `mte2Size`（实际写入行数）传给矩阵阶段，
   矩阵阶段按它取 `min(实际, 期望)` 读；或对未写区域显式 `Duplicate` 填充
   （如 −inf 分数），从根上消除"读未初始化"。
2. **不要把"因果过滤"实现成静默丢键**：丢键应与掩码一致（masked → 明确的 −inf），
   而不是让两阶段的长度假设产生分歧。
3. **把因果界显式化为入参**：当前 `cmpMaskRight = cmpMaskS2Size − actS1Size`
   把"压缩长度与 Q 长度的关系"写死为全局线性映射；KV 分片场景（DCP）下
   需要一个显式的"本 rank 可见上界"输入。
4. `CountValidCmpSparseLen` 的二分建议加断言或改为线性/显式计数。

---

## 7. 复现与核对材料

| 材料 | 位置 |
|---|---|
| 生产 dump（2 rank）+ 全部脚本 | `cos://uploads-new/share/dsv41-dcp8-smla-nondeterminism-repro-v2-20260930.tar.zst`（88.91 MB，public-read） |
| 本文的离线核算脚本 | 包内 `verify_s2idlimit.py`、`region_match.py` |
| NaN 行分布 | 包内 `nan_rows.py`、`replay_nanpatt.py` |
| 索引数值变异 | 包内 `replay_idxvar.py`、`replay_fill.py` |
| 仓库内副本 | `experimental/v41-dcp/probes/` |

**单卡复现命令**（约 20 秒）：

```bash
docker run --rm --net=host --privileged --shm-size=8g \
  --device=/dev/davinciN --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v $(pwd):/d -e ASCEND_RT_VISIBLE_DEVICES=<chip> \
  quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3 \
  bash -lc "cd /d && python3 -u replay_dump.py /d/l20_T904_rank0.pt 12"
```

期望看到：`bit_identical=False`、`max|dlse|=nan`、`lse_nan_per_rep` 非零且各次不同。
