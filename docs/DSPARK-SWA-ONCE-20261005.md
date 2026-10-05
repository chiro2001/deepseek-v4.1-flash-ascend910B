# draft 的 SWA 索引每步被重复构造 K 次 —— 候选优化 `DSPARK_SWA_ONCE`（2026-10-05）

> 状态：**【待上卡验证】**（默认关，不影响现有交付行为）。
> 动机：`TAIL-OP-COUNT-20261005.md` 把"尾部小算子海"定为剩余最大结构性杠杆
> （s47 独占 **1.64 ms/步（6.4%）**、s35 metadata 1.19 ms/步），其中 **F2 = 位置/槽位链**
> 就是 draft 的 SWA 索引构建；本文发现它在**一个 decode 步里被重复算了 K（=5）次**。

## 0. 一句话

`llm_base_proposer.py` 的 draft 循环是 `for draft_index in range(num_speculative_tokens)`，
**每轮都重建一次 SWA 索引**；但该函数的输入（`block_table / seq_lens / query_start_loc /
num_actual_tokens`）逐轮只是被**重绑到常驻缓冲**、**数值完全相同**（`dcp_size==1`），
而且 device 分支本来就把 K 次结果**写进同一个 `dspark_swa_indices_buffer`**。
⇒ 其中 **4/5 是纯重复计算**，可以只算一次。

## 1. 证据链（全部为静态核对；标注【实测】/【推断】）

| # | 事实 | 依据 |
|---|---|---|
| 1 | draft 循环每步跑 `num_speculative_tokens`（=5）轮，每轮调一次 metadata builder | `llm_base_proposer.py:1138`（`for draft_index in range(self.num_speculative_tokens)`）【实测·代码】 |
| 2 | 该循环里逐轮改变的只有**缓冲地址**：`slot_mapping_group[draft_index]` / `seq_lens_group[draft_index]` / `query_start_loc_group[draft_index]`，**内容都是 `copy_` 同一份源** | `llm_base_proposer.py:1150-1160`【实测·代码】 |
| 3 | `draft_index>0` 时换 `block_table_tensor` 的唯一情形是 **`dcp_size > 1`**（`block_table_tensor_clone`） | `llm_base_proposer.py:1161-1164`【实测·代码】 |
| 4 | `build_for_drafting()` 里**真正**随 `draft_index` 变的只有 `get_cos_and_sin_dsa(..., draft_index=)` 与 `spec_slot_mapping[draft_index-1]`；SWA 索引那段的入参不含 `draft_index` | `draft/dsa_v1.py:1403-1429`【实测·代码】 |
| 5 | device 分支（`_device_metadata_enabled`）把结果写进**同一个** `self.dspark_swa_indices_buffer[: num_actual_tokens]` | `draft/dsa_v1.py:1550` + `:1224`（draft_index=0 走的 `build_req_metadata` 也用 `buffer=self.dspark_swa_indices_buffer`）【实测·代码】 |
| 6 | 该 device 分支**只在 `dcp_size == 1` 且非 PCP 时启用** ⇒ 与"内容相同"的适用条件正好一致 | `dspark_proposer.py:366-376`【实测·代码】 |
| 7 | 生产 profile 里这条链的算子计数与"每步 5 次"吻合：`FloorDiv 15-17`、`FloorMod 16`、`ClipByValueV2 13`、`GatherV3 18`、`Index 17`、`IndexCheck 19`（约 3-5 倍于单次构造） | `LEVERS-R5-20261004.md` §1 + `TAIL-OP-COUNT` §3.1【实测·profile】 |

【推断】收益量级：这条链在 s47 的 1.64 ms 里占约 **0.3-0.5 ms**，去掉 4/5 ⇒
**≈0.24-0.4 ms/步（1.0-1.6%）**。上卡后用 `ab_gate` 判定，不看单次跨 run 对比。

## 2. 改动（已入库，默认关）

| 文件 | 改动 |
|---|---|
| `patches/files/draft/dsa_v1.py` | 新增 `DSPARK_SWA_ONCE` / `DSPARK_SWA_ONCE_VERIFY` 两个开关；device 分支里 `draft_index > 0` 且 `dcp_size == 1` 时**跳过重算**、直接复用 buffer；新增 `_maybe_verify_swa_once()` 自校验 |
| `scripts/serve_a2.sh` | `-e DSPARK_SWA_ONCE` / `-e DSPARK_SWA_ONCE_VERIFY` 透传（默认 `0`/`0`，**不改现有交付行为**） |

设计约束（都写进了代码注释）：
* **只对 `dcp_size == 1` 开**（DCP>1 时 draft 步会换 block table，内容不再相同）；
* 复用**不改变 buffer 地址** ⇒ ACLGraph 捕获的 `data_ptr` 语义不受影响
  （这正是 `DSPARK_SWA_INDICES_RESIDENT` 当年踩过的坑）；
* `DSPARK_SWA_ONCE_VERIFY=1` 时，**前 5 个 replay 步**会额外重算一次并 `torch.equal` 逐位比对，
  往 stdout 打 `[SWA-ONCE] verify ok|MISMATCH`；**capture 期不做**（capture 区间禁止 D2H）。

## 3. 上卡验证步骤（等 a3-21 可用）

### 3.-1 跑之前必须先做两件事（否则会白跑一条臂）

**(a) 把这两个文件同步到 a3-21 的工作副本**（a3-21 的 `~/cedpd-repo` 在 `main` 上，
而本改动落在 `feat/v41-dcp8`）：

```bash
cd ~/cedpd-repo
git fetch origin feat/v41-dcp8
git checkout origin/feat/v41-dcp8 -- patches/files/draft/dsa_v1.py scripts/serve_a2.sh
git diff --stat HEAD -- patches/files/draft/dsa_v1.py scripts/serve_a2.sh   # 应看到两个文件被改
```

**(b) 起服后**在容器内**确认补丁真的在跑**（本项目"哑开关"教训的正向用法）：

```bash
docker exec dsv41-tp8k5 bash -lc \
  'grep -c "SWA-ONCE" /vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v1.py; \
   printf "DSPARK_SWA_ONCE=%s\n" "${DSPARK_SWA_ONCE:-unset}"'
# 期望：≥1 且 =1；若 grep 为 0 ⇒ 容器里是旧文件（挂载/同步问题），不要看性能数据
```

> 注：A3 交付走 `PATCH_MODE=mount` ⇒ **挂载的** `patches/files/draft/dsa_v1.py` 生效
> （`serve_a2.sh:1155-1158`）；镜像内 `/opt/dsv41/patches/draft/` 那份**只在 `PATCH_MODE=baked`
> 时**才会被 `cp` 进 live tree（`:1862-1872`）。当前发布镜像
> `local/dsv41-a3-tp8:20261005-1001` 构建于本补丁之前 ⇒ **凡是走 baked 模式的实例都拿不到它**。

### 3.0 ★ 上线前的四条风险自查（2026-10-05 静态完成，**全部通过**）

| # | 风险 | 核对结果 |
|---|---|---|
| 1 | **补丁会不会是死代码？** draft 循环里 `draft_index == 0` 走 `build_for_graph_capture`、只有 `>0` 才走 `build_for_drafting`；若 `use_compress=False` 则全部走 `build_for_graph_capture`，本补丁永不执行 | `use_compress = hasattr(hf_config, "compress_ratios")`（`llm_base_proposer.py:544`），而 V4.1-Flash 的 config **确有** `compress_ratios`（43 项，见 `a2/logs/129-…md` 与 `067-…md`）⇒ `use_compress=True` ⇒ **补丁在活路径上** |
| 2 | **replay 期 buffer 还会不会被刷新？** 若 K 次全跳过，buffer 会停在旧值 | 不会：每步 `draft_index == 0` 走 `build_for_graph_capture → build() → build_req_metadata()`，而那条路**同样**调用 `build_dspark_swa_indices(..., buffer=self.dspark_swa_indices_buffer)`（`:1224`），且该分支注明"**不**走 SAS metadata 缓存、每个 draft 步都要重建"⇒ 每步仍有且仅有 1 次正确写入 |
| 3 | **DCP>1 会不会误开？** 我在守卫里用过 `getattr(self, "dcp_size", 1)` —— 但 builder 上**没有** `dcp_size` 属性，getattr 恒返回 1 ⇒ 守卫形同虚设 | 已修：改为读 `enable_dspark_device_metadata()` 落的显式标记 `_swa_once_single_dcp`。该入口**唯一**调用点（`dspark_proposer.py:374`）条件里含 `dcp_size == 1 and not enable_pcp()` ⇒ 标记为真当且仅当前提成立 |
| 4 | **捕获期跳过写入会不会污染图？**（`dspark_proposer.py:709` 以 `draft_index=1` 直接调 `build_for_drafting` 做 capture） | 无害：`DSPARK_SWA_INDICES_RESIDENT` 当年的实验已证"**常驻化之后捕获期的内容不再重要**"（捕获期 seq_lens 取 0/6/1037/8192 全部 5/5）；且 replay 期每步由第 2 条路径刷新同一地址 |

**另外两个 `build_for_drafting` 调用点已确认不受影响**：
`llm_base_proposer.py:2795`（MTP 路径）与 `:3355`（DCP 路径）——前者不会启用 draft 侧 device metadata、
后者被 `dcp_size == 1` 门控排除，都到不了本快路径。

> 这四条正是"哑开关/静默无效"类事故的同一族：**先证明代码会被执行，再谈收益**。

```bash
# 0) 判据前置：先用 VERIFY 证明"复用的内容确实等于重算的内容"
DSPARK_SWA_ONCE=1 DSPARK_SWA_ONCE_VERIFY=1 bash ~/tmp/launch_armF.sh
grep -a "\[SWA-ONCE\] verify" ~/cedpd-repo/results/<RUN_ID>/serve.log | tail -5
#    期望：全部 ok（若出现 MISMATCH ⇒ 立即停用本候选，说明 dcp/block_table 假设不成立）

# 1) 单变量 A/B（同镜像、同启动器，只差这一个 env；≥3 轮交替）
bash tools/run_arm_suite.sh ~/tmp/launch_armF.sh  armF_swaoff 1 0 1
DSPARK_SWA_ONCE=1 bash tools/run_arm_suite.sh ~/tmp/launch_armF.sh armF_swaon 1 0 1
python3 tools/ab_gate.py armF_swaoff -- armF_swaon     # 负 = 更快

# 2) 正确性（必跑）：144K/1M 四针 + 并发 2 不同针
python3 tools/ced_pd_acceptance.py --base-url http://127.0.0.1:19210 --mode all --context-tokens 144000
```

**采纳门槛**（沿用本仓纪律）：`ab_gate` 中位为负且 ≥2/3 轮同向；144K 11/11；
`[SWA-ONCE] verify ok` 全部命中；三条件缺一不采纳。

### 3.1 ★ 可证伪指纹：改动到底有没有生效（照 `aic_mac_time` 的教训）

`KERNEL-CACHE-STALE` 那次踩过的坑是"以为生效、其实没生效"。本候选同样需要一个
**不依赖时间的计数指纹**——用同一份 profile 的算子计数对：

| 算子（`k6full` 交付口径 s47 每步计数） | 改动前 | **SWA-ONCE 后（预期）** |
|---|---:|---:|
| `FloorDiv` | 15–17 | **~3–4** |
| `FloorMod` | 16 | **~3** |
| `ClipByValueV2` | 13 | **~3** |
| `GatherV3` | 18 | **~4–5** |
| `Index` | 17 | **~3–4** |
| `IndexCheck` | 19 | **~4** |
| `SelectV2` | 25 | **~5–6** |

读法：这些算子在别的链里也有（不是专属），所以**不能看绝对值、要看"是否按 ~4/5 下降"**；
若计数完全不变 ⇒ 判"改动没生效"（先查 `DSPARK_SWA_ONCE` 是否真的进了容器
—— 本仓已有 5 个"哑开关"的教训）。

### 3.2 一个必须区分的点：本文只做"**步内**去重"

`build_req_metadata()` 里有一句注释："*the indices depend on the current step's block
table / sequence lengths, so they must be rebuilt whenever a DSpark draft step runs*" ——
它反对的是**跨步缓存**（上一步的索引拿到这一步用），**不是**步内去重。
本候选保留"每步至少重建一次"（由 `draft_index == 0` 的那次完成），
只是不再在同一份输入上重复 4 次 ⇒ **与该注释的约束不冲突**。

## 4. 顺带记录：同一循环里还有 6 个小算子/轮的"复制粘贴"

`llm_base_proposer.py:1150-1160` 每轮做 3 次 `copy_` + 3 次 `fill_`（把同一份
slot_mapping / seq_lens / query_start_loc 复制进 `*_group[draft_index]`），×5 轮 = **30 次小算子/步**。
它们的存在理由是"**地址跨 replay 稳定**"（graph 捕获 data_ptr），**不能直接删**；
但可以合并成"一次写、K 次设备内广播"或换用 `view`/别名——【未确认】是否与捕获语义冲突，
留作后续（本次不动，避免与 SWA-ONCE 混在一个单变量里）。
