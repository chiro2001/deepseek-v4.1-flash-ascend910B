# 三形态目标：实证状态（2026-10-13，主目标已暂停）

用户已明确暂停三形态主目标；旧goal后来清除，新goal只追踪chip8–15算子核查与基线校准。
本文不是三形态完成收据。
当前新增任务仅是 physical chip8–15 的算子优化历史审计/必要补测及代码 push。
不得根据下面的旧启动记录恢复 CED/DCP 或宣称任何新服务在线。

A2 v4 的完整精度、地火三档和编程指标已刷新 README；A2 API 后来已停。
CED v4 / DCP v6 仍未完成新设备回归；共享 Engram 与单侧唯一 KV 池尚未实机达标。
暂停前最后现场快照是目标时钟 09:13:30，外部 `rain_fusion_att` 占 physical5；
这不是当前实时占用状态，算子任务会独立重新核查 physical8–15。

后续算子任务已在8–15重新起完整TP8 base2（不启CED/DCP），服务SID
`3cbd844d9f684c43867b4b85c9c78ebe`、入口18960；一次冻结地火2K/256×8已完成，
HTTP `(28.634374913,2.524752475,88.172082781)`、8/8 clean/usage/length，
prefill wall合计1.591584930s、TTFT mean0.321815077s。输入及输出文本SHA均匹配历史8/8，
主代理独立raw重评分通过；结果见 `../operator_stack/evidence/operator_revalidation_20261013/baseline2_client/`。
服务保留用于用户当前任务，当前在线性必须再看该任务的最新final snapshot，不能从旧记录推断。
三形态/DRAM结论仍未完成、不继承这个基线窗口的性能或质量结论。

## 历史有效检查点（按发生顺序保留，非当前在线状态）

A2 v4 已在线，SID `9d2ba214732b4eed84dc4f416f27c6ae`、实际 fp `61b841df…`。
同实例编程 8/8 题、128/128 用例；GSM100=100/100、Vision23=23/23、147456 四针=4/4。
主代理独立按冻结评分函数重评分并逐件核 long needle raw SHA/usage/stop；
详见 [strict_quality_audit.md](evidence/a2sim_tp8_v4/quality/strict_quality_audit.md)。
四针 prompt=147438/147436/147437/147435，均超过有效长度门145982。
客户端已串行结束；现交 workload_bench 地火2K→32K→147456，各8请求，不并发质量客户端。
地火新数据尚未终态；CEDv4仍只init成功后已停，DCPv6尚未部署，不能标全部交付完成。

此后地火2K正式8请求已终态exit0/无客户端PID，主代理独立复核全部压缩/解压SHA、
HTTP200/length/2048+256、同SID/fp、8rank真实delta、request tag与完成fence：
HTTP `(28.071336646, 2.524752475, 89.940586267)`，808 rounds / 2040 decode tokens；
TTFT mean/median `0.310031991/0.310232270 s`，prefill合计 `1.332813720 s`。
每rank graph808/eager8。server-fenced另为29.646913366ms/step，不混入HTTP三元组。
证据 [RESULT.md](evidence/a2sim_tp8_v4/client/dihuo_2k_a2sim_v4_20261013/RESULT.md)；
32K/147456仍按原放行序列串行测量。

32K此后已终态exit0/无客户端PID，主代理独立核8件gzip双SHA/size及原始usage：
每条32768/256/33024，HTTP200/length/clean；703 rounds/2040 decode tokens，
HTTP `(29.595011408,2.901849218,98.051971584)`；TTFTmean/median
`4.733176581/4.732731905 s`、prefill合计 `36.926545540 s`。
每rank graph703/eager8，server-fenced另为30.479052048ms/step。
代理消息曾误抄2K usage，root暂停下一档并核对原件后已明确恢复147456放行；
原始32K receipt未覆盖，不能以聊天误抄改写原件。CED新p5/d5 CPUplan/launcher已审核，
尚未execute或停A2；支持的完整npu-smi进程表确认8–15没有运行NPU进程。

147456此后也终态exit0/无客户端PID，8条usage精确147456/256/147712、HTTP200/length/clean；
主代理独立核8gzip双SHA/size、原始8rank计数/请求tag/fence与同SID/fp通过。
638 rounds/2040 decode，HTTP `(32.381891614,3.197492163,98.743217385)`，
TTFTmean/median `22.988100295/22.981017115 s`，prefill合计 `180.522022380 s`。
每rank graph638/eager8，server-fenced另为34.002878715ms/step。
**A2 v4本配置完整精度、编程和地火三档已通过。**root已明确授权site先prestop归档，
只SIGTERM项目A2 API/保留容器缓存证据，核资源释放、双容器source/OPP/ports/IPClen门后，
按已审核p5/d5脚本初始化CEDv4新P/D，暂不proxy/不generation；实际停服/ready状态待现场回执。

实际后续停服：只向A2 API342558发一次SIGTERM，API/EngineCore退出、18960释放、
16chips HealthOK且无运行NPU进程/HBM回2.9–3.1GB。run.exit0，但有runner/8worker
zombie，不称完全干净teardown；P容器仍Running/未重启。site继续归档并推进CED
双容器source/fullOPP/resource门，门全PASS后已授权p5/d5 init，尚无新ready/推理结果。
独立新wire wrapper保存真实P/D请求与P response transfer参数，原v4 proxy字节不改，
base SHA813c5a0e…/wrapper82873d5b…；18项带hook CPU控/顶层129通过。
它不读D/SSE或增发请求，I/O属D提交前TTFT；详见 [CED_WIRE_AUDIT.md](CED_WIRE_AUDIT.md)。
wrapper仅准备，proxy仍未放行；不改变已测A2数字。

**最新启动门：尚未启动新CED。**原p5/d5根目录中P侧是ro bind，现场可写门正确拦截；
root授权并独立重审新`jobs/p5`、`jobs/d5`配方，只改输出/cache/TMPDIR路径，原bad配方保留。
P jobs=host bind RW，D jobs=独立container overlay RW，不能从P host路径冒充读D证据；
proxy/wire也用P的`jobs/proxy5/wire`。新p5_jobs SHA23e5916c…、d5_jobs00331485…，
CPUfp P e7674b82…/D d192f5fb…，非未来actualfp；UUID+NUL96<107。
双容器29/source220/fullOPP树与config.ini再次全PASS，但临启动出现外部fyf_1009
physical5实际占用：PID3215293/rain_fusion_att、AICore100%。P/D scripts均未执行，
两jobs仍absent，目标端口无listener。site约60s只读复核，不动外部；bench CPU准备
在空闲8–15上先跑DCPv6的替代计划，尚未审核/execute，不能称新DCP在线。

外占后续迁至physical0、1，第三轮到physical9（8–15组），同一fyf_1009 cgroup；
不是仅原PID或chip5退出便恢复资源窗口。已向用户询问协调释放/预留16chips。
DCPv6 CPU配方已由root独立审查通过，scriptSHA b84f9f32…、CPUfp e1277a56…；
由于D组也受持续轮换占用，尚未授权启动任何形态。当前项目推理API均未在线，
SSH/两个项目容器仍在线，不能混为机器离线。wire副本P jobs/proxy5/ced_wire_audit.py
已核hash/--help，但wire输出目录absent，proxy未起。剩余CED/DCP设备验收及DRAM未完成。

## 最新CED进展（此前ready是历史状态）

最新A2推进：D4仅init/source门成功后受控停止，尚无推理。A2v3的8–15启动前
遇到外部xp_mk占12/14，未启动，门禁证据原样保留。现准备0–7的新A2v4，
仅派生actual async worker的1处物理guard并保留原snapshot/SDK/模型/图/计数。
generic v7包SHA827c8310…/388779B、顶层110CPU通过；36文件/source/plan/runner门通过，
root已审核放行，完整IPC86/107与0–7空闲/Health/归属门均过。A2v4 runner342556已
启动init-only，无生成；早期BI-META有transient skipped，待最终安装日志/实际8rank
来源/BI/图/Engram门，不先判pass。此后A2v4完成126静态核，8rank source/config门与
GET models/identity所有权门均过；ready SID9d2ba214732b4eed84dc4f416f27c6ae、actual fp
61b841df…，ready receipt已归档。root已放行新单code canary→通过后code8+bwrap，
site不并发生成，**尚无新正式精度/性能结论**，见PHYSICAL_WORKER_CANDIDATE.md。

P-v2+D-v3短d3 pair的首code canary失败，没有有效输出；实际8rank KV transfer与
128-token replay有日志，但_pool_kernel MTE/SMMU故障导致D退出，HCCL随后传播。
完整失败收据`evidence/ced_v3b_canary_failure_20261013/failure_diagnostics.md`，SHA=
`41b977a69bdd3479e564e018d84c3a54118fd809be7b7a59a6e0aac189d694d4`。
run.exit0只是runner收尾，不能记成功。正式code8/质量/地火未启动。
主代理已准备D-only四处masked残余地址clamp候选，原mask/ring/math不变；
6项CPU+plan负控、顶层100/100通过；未证明故障因果/未上设备。
新CED v4 source包179474B，SHA3916d72f…；两端新P4/D4测量窗口，仍best双流、
新model dsv41-ced-pd-v4，仅D选择component，详见`CED_COMPRESSOR_CANDIDATE.md`。
site已对旧proxy/P API计划内SIGTERM。P native teardown最后报corrupted size vs.
prev_size且exit134，证据独立保留，不记clean stop；旧服务端口及P/D NPU均已释放，
容器仍Running/restart0，仅余zombie。新外部matmul gate从NPU2/chip4/5转到NPU3/chip6
（非本项目），不动它，P4等0–7全部空闲。v4双容器29文件源门通过，预检收据SHA
`b245f7b1de1513a2aeef339f4fbc61bab6d2cf915feaee9a1dfb0e9894119ba4`。
D4在空闲8–15完成独立初始化，API1609471/EngineCore1610342、18991已listen；
8rank实际source ea3ead…/manifest accf27…门通过，SIDe10c11f1…、plan fp78e8e6fc…。
**未发生成请求/未起新proxy，不代表补丁推理/精度/性能成功。**因外部chip6持续占用，
主代理调整顺序：归档D4 ready/计划内停服后，先在8–15补A2v3长上下文与新基础精度，
CEDv4保留待16chip可用时恢复。A2v3 launcher/source门仍待审核，bench已准备但未访问API。
旧P累计failedcanary tag
已只读归档，不能与新D空窗口混用；下文旧pair ready身份不再是在线状态。

## TP8 P/D 同置，A3 模拟 A2

历史运行实例：a3-21 / `dsv41-real-stack-tp8-20261010-a321`，physical 8–15。
历史服务：`dsv41-delivery-a2sim-tp8-v2`，127.0.0.1:18960（已停，后来切至DCP）。
冻结源：`/work/src_dspark_general_quality_v11`；新适配器在
`/work/delivery_matrix_20261013_v2/delivery_matrix`，新 job 在
`/work/results/delivery_a2sim_tp8_v2`。

保留 BI/custom OPP、async、双多流、真实 Engram host 查询、DSpark5 草稿图；
DEVICE_INDEX=0、PREFIX=0、SEQ=1、BAT=8192、MAXLEN=147456。
此次是 A3 硬件模拟，不是真实 A2 硬件性能。

| 正式负载 | 输入/输出 | 请求 | 实际轮数 | 客户端 ms/step | A | 客户端 tok/s | TTFT 均值 | 服务端 prefill wall 合计 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 地火 | 2048/256 | 8 | 808 | 27.952854 | 2.524752 | 90.321815 | 0.316939 s | 1.376686 s |
| 中级编程 | 2048/自然结束≤1024 | 8 | 278 | 27.092052 | 4.985612 | 184.024876 | 0.313686 s | 1.388748 s |

两负载 clean 都为 8/8。编程输出 1394 tokens（decode 1386）；地火输出 2048
tokens（decode 2040）。客户端三元组以实际 final usage 去掉 prefill seed 后
的 tokens、真实 worker rounds 和客户端 decode wall 换算，自洽。

服务端完成 fence 包含后续 metrics RPC 的等待，另得 ms/step 29.563832
（地火）/32.423162（编程）；不得与上表客户端 tok/s 拼成三元组。
prefill wall 是 worker 首次调度到末 prompt bookkeeping，不是纯设备 kernel 时间。
当前 v2 只可信用于单 chunk 2K，长上下文已在未部署的新 worker 修正边界。

本配置精度门已完成：GSM100=100/100、Vision23=23/23；编程本机 bwrap
独立评分=8/8 题、128/128 用例。现场原编程 receipt 的 8 个 sandbox_error
仍原样保留，不覆盖旧证据。独立评分绑定原始 receipt/output SHA。
更正后的质量 receipt SHA=`0af8bd9f9d65a33686269c72da9ed19342aad68ed1f23feb6469ed29b9be27c4`，
本地质量证据见 `evidence/a2sim_tp8_v2/`；编程独立评分源在
`/home/chiro/tmp/dsv41-workload-bench/code_formal8_local_scoring.json`，
SHA=`d7db006a4279160468d7c9467435b185bc2c5c3d23f9608f613b826943f92797`。
原质量 receipt SHA=`39644ed4…` 仍保留；其中把 NPU 进程占用误当成
`actual_device_consumer_proof=true` 的字段已在新 receipt 明确改为 false，
不改变真实的 GSM/Vision 通过结论，不宣称逐层/cache 消费者追踪已完成。

原件（此时仍在远端，不能当成本地已取回）：

- `/home/l00886679/tmp/dsv41-workload-bench/code_formal8_20261013_v1.json`，1542385 B。
- `/home/l00886679/tmp/dsv41-workload-bench/dihuo_2k_20261013_v2.json`，2562505 B。
- `/home/l00886679/tmp/dsv41-workload-bench/code_canary_20261013_v2.json`，142038 B，schema PASS。
- 首个 canary 的 request-id 校验错误完整保留，未并入正式统计。

大原件不得通过 SSH 整文件传输。取证使用 COS 或派生单请求小文件，并保留
原件 SHA、旧/新路径映射。

诊断注意：旧 Engram `_RouteProbe` 使用字符串 bool，env=`0` 仍开启相位日志；
v2 原样保留此状态。新正式窗口会关闭该探针并重新测量，不借此修改旧数字。

## DCP8 集成

历史 DCP 源已取回并保留 SHA。`dcp_runtime_20261013_v3` 是 12 文件的候选：
保留 best worker/runner/context/model/platform，用 post-import 的 V41 interleave
窄补丁；合并 best QLI-NO-CAND 与 DCP cache spec/visibility；只为同节点
TP8=EP8=DCP8 放行 Engram guard。索引默认复制以保留完整 global top-k。
首轮 DCP v1 曾真机 ready，但唯一 canary 因首次 parallel draft 返回
`long_seq_args=None` 导致引擎退出；没有正式性能/质量数据。
选择性移植历史 DCP/DSpark 修复到 best 的两个 frozen draft 文件，保留
TARGETPOS/graph/BI/async 修复，CPU 坐标与 parallel 分支正负控 6/6。
第二轮 DCP v2 在初始化时被原 observer 的源码 SHA 白名单正确拦截，
仍未发 canary。两次失败原件与原因分别见 `evidence/dcp8_v1/` 和
`evidence/dcp8_v2/`；不得记为成功或用空响应计性能。

当前已为新 DCP candidate 加入受控的 observer 来源契约：校验原 best
源 SHA 与 derived 文件 manifest，只变新 worker 进程的确切 proposer
期望 SHA；ACL wrapper/NPUGraph 来源与真实 replay/输入地址检查保持。
该契约 4/4 CPU 正负控通过；下一独立 job `delivery_dcp8_v3` 已于 03:14 CST
正式启动，root 核验 container runner PID841623 / API PID841625 存活，run.exit
未产生；package=`/work/delivery_matrix_20261013_v5/delivery_matrix`、served-model=
`dsv41-delivery-dcp8-v3`、port18960。现已 ready，八 rank 的真实配置与
derived proposer provenance 均通过；ready receipt SHA=`eda6818226e3a4192d7b328b83602ae1c47fff993769e802c4496b815db09a2d`。
单条新 canary 已通过；正式编程、地火 2K/32K/144K与质量门均已完成。
该v3测试job已计划内SIGTERM停服，run.exit=0，容器保留；不再宣称18960在线。
所有原 frozen snapshot 字节未改。

| DCP8 v3 正式负载 | 请求 | 实际轮数 | HTTP 客户端 ms/step | A | HTTP decode tok/s | TTFT 均值 | worker prefill wall 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 地火 2048/256 | 8 | 771 | 54.735522 | 2.645914 | 48.339987 | 1.037855 s | 1.785922 s |
| 地火 32768/256 | 8 | 747 | 56.234841 | 2.730924 | 48.562842 | 16.338515 s | 129.867142 s |
| 地火 147456/256 | 8 | 696 | 58.336616 | 2.931034 | 50.243478 | 74.470565 s | 592.727552 s |
| 中级编程 2048/自然结束≤1024 | 8 | 259 | 53.574912 | 4.872587 | 90.949040 | 1.033510 s | 1.862085 s |

均 clean 8/8；编程输出在本机 bwrap 独立评分 8/8 题、128/128 用例通过，
远端无 bwrap 的 sandbox_error 原件保留。32K TTFT mean/median=
16.338515/16.325701 s。客户端三元组来自 manifest 的
`client_e2e_ms_per_round`、`A`、`decode_tok_s`，不是 `ms_step` 字段；后者
仍是包含 metrics RPC 延迟的 server fenced 口径（2K=63.604789、32K=57.475208、
144K=60.259262、编程=81.220745 ms/step）。不得将它与 HTTP rate 拼成三元组。

原件与分拆 manifest 位于 `/home/chiro/tmp/dsv41-workload-bench/evidence/`。
32K 原件留在远端 `/home/l00886679/tmp/dsv41-workload-bench/dihuo_32k_dcp8_v3.json`，
6117334 B，SHA=`56e6dd42a51b781acabb698320f34438b3a7c95e42ecdcd6b1d711f03958ae01`；
本机32K分拆单件最大770301 B。144K 原件
`/home/l00886679/tmp/dsv41-workload-bench/dihuo_147456_dcp8_v3.json`，16642155 B，
SHA=`08d8c954dda9e3beb7a890f5dd6202695e4922ba9ac22f66ddb0c3cd8c996cc4`，
本机已取回8条 gzip（最大653559 B），压缩/解压逐件SHA/size与manifest均已核验。
未用 SSH 传大原件。DCP GSM100完整结果99/100，0 empty/0 errors，
严格100/100门 **FAIL**；Vision23=23/23、mandatory negatives通过。
GSM唯一错误 `gsm8k-12`，gold13/pred12；原文在12年回本/13年盈利之间反复，
末尾有512预算截断迹象，但原evaluator未保存finish_reason/usage，不能确证
length finish。A2原同题gold13/pred13；没有修改门槛或覆盖原轮。
诊断见 `evidence/dcp8_v3/quality/gsm100_failure_diagnostic.md`。
GSM JSON SHA=`43548a28814db1ec6e0390bbb06627bed0cecc60cf8b3ab9a52405fd9721c7c2`；
Vision原件SHA=`de3e5596597bfbbf588c0aed8e76c14a757d687a954536ef77edd5023542f5bf`。
长针已完成runner正负控/长度校准（目标147456，full prompt147394–147397），
首个A的响应在host输出目录权限错误时未保存，不能算有效精度样本。
改用可写的host任务目录完整重跑四针，门槛与模型不变；正式4/4逐字正确，
usage prompt147435–147438，99.986–99.988%目标长度，completion7–8，均stop。
汇总SHA=`0149b95f64a2ae37ab7efb8e68affb064b378f97b94ff65f4058b262f332ac0e`。
同一道GSM题在同服务/同8shot/chat/512预算下仅重复诊断一次，13/13、stop、
349 completion tokens；不能用它覆盖GSM100正式99/100结果。
更正质量receipt（只修container64位ID，旧原件保留）：
`quality/strict_quality_receipt_corrected.json`，SHA=
`817953111fb09c977e01fc048dbe754c8f8e6508d3e11245787c5cf3cd5c631e`。
**当前不能标为 DCP8 整体精度达标。**

本机03:56左右只读复核：a3-21=192.168.45.21，项目容器仍 Running，
RestartCount=0，API 容器 PID841625 / 宿主 PID1479521，18960 listener正常。
客户端147456的PID1506117已exit0。现场代理曾误查本机daemon/混用PID namespace，
不是目标服务退出；远端旧Docker不支持 `docker context show`，不得用其失败导致
`&& docker ps`未执行的空输出推断容器消失。远端时钟比本机约慢8分钟。
后续真正计划内停服的前/后收据见 `PRESTOP_ARCHIVE_RECEIPT_20261013.md` 与
`POSTSTOP_ARCHIVE_RECEIPT_20261013.md`。SIGTERM后日志有shutdown EngineDeadError/
共享内存清理警告，明确保留；推理阶段已完成，不用shutdown日志改写质量结果。
24个旧worker zombie RSS/PSS=0；目标chips无运行进程，不删设备/不重启daemon。

独立DCP稳定性CPU候选已准备。v5统一gate并修改helper，但后续主代理源码审查
发现kernel/folded-pack两分支仍inline全量gather；v5未部署，原件保留且新门拒绝。
v6将所有5分支（8调用）接同一helper，kernel驻留pack地址不变；512B对齐、T分块并
复用一对collective workspace，gather帽128MiB（典型8192/H64/W640规划80MiB）。
v6 manifest SHA=`b0c1d36e44707c872707477677a5ef60a4d3d53f86c5df0d1092e47e41706a89`；
8项CPU正负控通过，**尚未部署/验精度/证明GSM失败原因**。v4/v5原件保留，
下一候选使用v6，见 `DCP_STABILITY_CANDIDATE.md`。

CED-native 候选 overlay 已从真实 SDK base 构建：admission→replay→prompt-tail
所有 patch 精确上下文检查通过，保留 best model/Engram。P native worker、
角色启动器、P→D 原样 SSE 代理与 D HTTP 护栏已准备，正在准备CED双服务，尚未验收。
新 proxy 对 TP8/12-group/replay128/peer 字段做请求边界校验，metrics 仅合并
相同 P/D SID 与 request-tag 窗口，拒绝缺 rank/重复 rank/跨请求证据。
新增 `--decode-multistream best|off`：默认保留双流，若长针精度失败，必须
保存失败并创建另一个双流关闭的候选；其 fingerprint 改变，不继承精度。
CED CPU 正负控35/35、delivery 顶层58/58、现有 pkg selfcheck 全通过。
只读核对确认固定 image SDK/ACL 与现容器一致，但原 image **缺 BI OPP**；
当前 OPP 是 writable layer 的214MiB/617文件。新 P 容器需复制并校验完整
安装树或安装已哈希的 A3 包，不得用910B包。来源 receipt 见
`evidence/ced_preflight/image-and-opp-readonly.md`（SHA=`337902580f50a5ace2a8f4ad49f583bbeb0e93c0a936a95fc548fc989990dc38`）。
新入口在 `--execute` 时真实校验 OPP version.info/三库，P/D ready 必须有
相同 OPP artifact fingerprint 与 overlay manifest SHA；此四文件门不冒充
完整安装树核验。原 observer 不 hash CED runner/DSA/connector，无需放宽。
进一步定向比对确认D另有树外 `opp/vendors/config.ini`（30B、priority=batch_invariant），
base image缺失；P须额外复制/只读bind该单文件，不能只挂vendor子树。SHA=
`7473a624041a7281607d1a988a826c67f14fb203dc36f161f8f33b9ff9a8b4a5`。
新增来源补充见 `evidence/ced_preflight/opp-activation-supplement-20261013.md`。
P前内存门PASS：停服后MemAvailable约1990.83GiB，PSS采样是权限受限下界，
没有dropcache。短暂SSH新连接失败后独立hostname/IP/Docker探测已恢复，goal从未暂停。
P0–7项目容器已创建：`dsv41-ced-p0-7-20261013-v2`，ID=
`16d3045321f8af0e07e7392a4feabe14603e684589947126f63f1b37b6c631bd`。
P已ready，BI-META/启用日志与8rank权重加载37.433GB/rank、KV3211560 tokens留证；
正式请求尚未开始。D复用原b1aa…8–15容器，不混入sharedEngram或DCP v5候选。
CED包v2已冻结并24/24远端字节核验，165969B，SHA=
`45e05870ccd73fadd904832a2f37ad63d5e9080127ae094f52efba7f61292513`。
**D-v2真实启动失败**：权重/KV/164静态kernel/graph capture完成后、API listen前，
`build_app`返回后添加HTTPmiddleware抛 `Cannot add middleware after an application has started`。
run.exit=1，完整48文件证据见 `evidence/ced_d8_v2_failed_startup_20261013/`，不覆盖。
实际原因是SageMaker middleware loader提前构建stack；ABI证据见
`evidence/ced_preflight/api-middleware-abi.md`（SHA=
`5f40547c9042f8c5847b83d5c6bd28198b2349ff9800115c573aee25135c8367`）。
主代理v3修复使用官方config.middleware构建前安装、并纳入runtime fingerprint；
prebuilt-stack CPU正负控验证了400拒绝/正常转发及旧时序抛错，CED38项通过。
新v3包166512B，SHA=`d4850d9e63285885767e54d1321f8eaec82d44b0b17cf8698b5e7df0a940d1d5`。
当前保留P-v2 ready，D-v3全新job重试（served_model与P相同）；core overlay/模型/BI配置不变，
D-v3首次长job路径计划fingerprint=`efd677133f6a012b8332112d3e41220d93b71a3de1e7aa3fa7bc654a8f838091`，
随后在ZMQ IPC初始化前因socket path>107字节退出；该指纹不可当成新ready实例。
同v3包已以全新短job root `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3`
重试，TMPDIR+UUID路径90/107字节；runner namespace PID1238305、EngineCore1239181，
8rank已组建。此后D短job已完成HTTP启动；实际identity/plan绑定的新fp为d877f65…，
不是旧长job的efd677…。D缺合同直连HTTP400且guard有拦截日志，之后engine/API仍活，
**中间件修复已实机确认，尚未证明KV transfer/精度/性能**。P producer按rank监听19090–19097，
19091已通过P TP1日志证明归属，不是D端口冲突；D consumer不bind该组，不杀listener或改端口。
两端同overlay/actual OPP且8rank身份均已核验，proxy18992身份/health/models均200；
pair SID=`3762dc9b083a4b33c045c877273d19a7cb5fc927dee2960b0181948d81424f41`，
pair runtime fp=`c70b47fc3e303f58e33e3b1fa0f8960565eb5eefdce3900fb5a394754c9aec8b`。
bench已获单编程canary放行；site不并发发请求，完整ready receipt待归档。
v1包原件保留，但以v2部署：其CED-only HTTP扩展在D直接拒绝非空错误合同，
避免错误请求进入引擎；原decode_guard字节未改。客户端下一版显式加入
HTTP/server两个ms_per_step字段、scope及旧字段别名；CED客户端已冻结，
最新客户端v2包46830B、SHA=`5de07966b80b1d4535fcdc7b3e5cfd1b1e97b8c81c2744b8507f424a8391b73b`，
增加P arm SID/tag/budget校验；本机33项/15subtests通过，远端无bwrap的评分失败不冒充
模型失败。CED prefill按串行两阶段 `max(P rank delta)+max(D-only rank delta)`
报告；排除 transfer 等待，非纯kernel。旧receipt不改写。
第二阶段验收边界见 `DRAM_DESIGN.md`。主代理已离线准备host-only shared mmap组件：
readonly MAP_SHARED直接绑定INT8/F32 checkpoint shard、构造不分配完整私有表、禁改写/活体remap；
只改actual best Engram的构造/加载3方法，路由/lookup数学函数AST不变，12项CPU正负控通过。
组件manifest SHA=`37ec7b7a9bb9a2812cabc35535784dc7a7b5c643270d2dd4f59e444a06e04e75`。
尚未接入CED基线或证明真实TP8/PSS共享；KV单侧卸载/内容唯一池仍待实现与设备验证。
已离线补充memfd内容唯一pool原语，8项CPU正负控通过；single-authority接口、
byte equality/hash碰撞保护、alias/pin生命周期/池满回滚，不代表已接NPU或实际容量收益。
范围与缺口见 `KV_CONTENT_POOL_CPU.md`。
新增`quality_wire_capture.py`在不修改冻结acc_eval/预算/评分的前提下旁路保存
每HTTPattempt的finish_reason/usage/消费wireSHA；4项CPU测试直接执行原complete
AST验证字节/文本透明，下一GSM正式轮使用，旧99/100不改写。
当前顶层delivery93/93、客户端21/21 CPU测试通过；CPU测试不代表实际NPU能力。

README已先写入新矩阵现有事实（A2/DCP实际指标、DCP严格FAIL、CED待测），历史数字单独标明。
三形态最终表仍待CED与A2长上下文、DCP修复回归；README是未提交改动，HEAD MANIFEST
尚未刷新，不宣称当前工作树已可作为校验和完整的发布包。

## 尚需完成

1. A2 模拟：已完成 2K/编程/基础精度；尚需无探针正式长上下文测量。
2. DCP8 与 CED-PD：起服、同负载、精度/长上下文/真实 draft graph 验收。
3. 用三形态实际数据刷新 README 与可复现配置。
4. CED 单侧 DRAM KV offload、跨 P/D host-only Engram mmap 共享、KV 唯一池。
5. 真正双向卸载/命中/容量/PSS/精度验收与最终 README 更新。
