# tiny 1+1 的 SWA-clip A/B 工具与已验证事实（2026-09-24）

本目录配套主分支 `feat/ced-pd-a3` 的 `[CED-SWA-CLIP]` 修复
（`experimental/ced/dsa_v41.py`，开关 `V41_CED_SWA_CLIP`，默认 1）。
根因分析见主分支 `docs/CED-D-1M-LAYOUT-BUG-20260924.md`。

## 1. 工具

| 文件 | 作用 |
|---|---|
| `tools/ced_swa_clip_matrix.py` | CPU 整数矩阵仿真。补 `ced_swa_clip_verify.py` 没覆盖的两块：**多请求同批**的逐行 rebase、以及 **N=100..2999 触发边界全扫**（判据 `N≥256 且 N%128≠0`）。 |
| `tools/ced_phy_watchdog.py` | a3-22 单卡实验的只读看门狗：非本项目进程 HBM 合计 > 36 GB 或整卡 HBM > 52 GB 时 `docker stop` **我们自己的**容器；PID 变化只记录。 |
| `tools/launch_ced_tiny_d.sh` | tiny 单卡 D 启动器（图模式臂：`GRAPH=1 EAGER=0` + prompt-tail eager + metadata inline + `MULTISTREAM=0 DSA_OVERLAP=0`，探针全开）；拒绝 chip0/chip1。 |
| `tools/ced_tiny_clip_ab.sh` | 单 arm 编排：起 D → 起 proxy → 发 filler → 重复目标请求 → 摘块形态/裁剪指纹 → 收容器。`CLIP=1` / `CLIP=0` 即两臂。 |
| `tools/ced_repeat_probe.py` | 逐轮落盘的重复探针；代理卡住时自动重启代理容器；每轮记录首 token、logprob、top-5。 |

## 2. 已验证事实

1. **离线判据（本目录产物）**
   - `ced_swa_clip_matrix.json`：真权重 1M 例 legacy 读列 `7965`（不在持有集
     `{7966,7967}`），clip 零越界；`N=100..2999` 全扫 **0 处不符**；
     多请求批次（N=4000 与 N=1019847 同批）逐行 rebase 无越界。
   - `ced_swa_clip_verify.txt`：`legacy 越界长度数=5（需 >0）`、
     `clip 越界长度数=0（需 =0）` ⇒ 通过。
2. **tiny 真机（a3-22 phy7，2026-09-24 21:38，修复已生效）**
   - D 以 `GRAPH=1 EAGER=0` 起，`/health=200`；
   - `N=2000` 的 replay 步打印
     `[CED-SWA-CLIP] layer=2/20 block_size=128 q_len=128 base_pages=[14]
     seqused_ori=[207] pages=2 seq_lens=[1999]`，
     与 `ced_swa_clip_matrix.py` 对 N=2000 的预测**逐项一致**；
   - `[CED-GRAPH] one-token prompt tail forced eager` 确认末 token 步仍走原路径。
   - 同布局重复 3 次：HTTP 200、`text` 与 `token_logprob` 完全相同
     （tiny dummy 权重 logits 近似并列，只作**同布局可复现性**证据）。
3. **未完成**：连续/碎片各 ≥10 次、两种碎片形态、`CLIP=1 vs 0` 的完整 A/B
   矩阵（按用户「先不动」指示停在起服务之前；a3-22 当前没有我们的容器）。

## 3. 恢复实验的步骤（不需要改代码）

```bash
# 0) 挑卡（只读）：AICore 持续 <5% 且 HBM 余量 ≥28 GB，且先做设备探针
docker run --rm --privileged --network host \
  --device /dev/davinciX --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=X -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info <image> \
  bash -lc 'python3 -c "import torch,torch_npu;print(torch.npu.device_count());torch.npu.set_device(0)"'
#   （phy5 曾被设备级 TsdOpen 失败挡住；探针 count 必须为 1）

# 1) 单 arm（重复两次，CLIP=1 / CLIP=0）
CLIP=1 DEVS=X FILLERS=3 REPEATS=40 bash tools/ced_tiny_clip_ab.sh arm_on
CLIP=0 DEVS=X FILLERS=3 REPEATS=40 bash tools/ced_tiny_clip_ab.sh arm_off

# 2) 看 results/<tag>/summary.json：
#    - 两臂 http_200 == requests
#    - 碎片臂 g0_descents_positive > 0（否则要先加大 FILLERS / 缩小 KV_CACHE_MEMORY_BYTES）
#    - CLIP=1 臂 unique_texts 单一且与 CLIP=0 一致；CLIP=0 臂允许出现分裂
```

## 4. 判据提醒（来自真权重线）

- tiny 的 dummy 权重 logits 近似并列，`text` 相同不代表数值逐位一致；
  要比就比 `first_token.logprob` 与 top-5 集合。
- 真权重线上 pass-pass 之间 digest 也有 6~8 量级差异，因此结论必须落在
  **失败率**（首 token 是否变 EOS / 是否分裂）而不是逐位相等。
