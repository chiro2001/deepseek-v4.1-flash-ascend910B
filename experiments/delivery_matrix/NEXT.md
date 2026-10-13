# 接续工作检查点（三形态主目标已暂停）

## 当前任务与授权边界（2026-10-13）

用户已明确暂停三形态交付与 DRAM 接线主目标；旧goal后来被清除，并按用户要求重建为仅算子核查的goal。
用户随后要求只使用 a3-21 physical chip8–15 核查/补测既有算子优化，以及提交并 push 已有代码。
这不是恢复 CED、DCP 或完整交付矩阵的授权。现场必须重新核查宿主完整 NPU 进程表；
不操作外部容器，不由旧启动配方自动恢复服务。

README 已更新 A2 v4 完整实测数据与 CED/DCP 的失败/待验边界。
共享 Engram 接线候选和内容唯一 KV 池仍只有 CPU 验证，不能标为设备验收通过。
当前算子核查产物独立保存在 `../operator_stack/evidence/operator_revalidation_20261013/`。

最新算子窗口：首次base1启动失败原件保留；同参数仅换新jobroot的base2已就绪，
SID `3cbd844d9f684c43867b4b85c9c78ebe`，physical8–15，18960，服务按用户要求保持在线。
唯一1 warmup+8 measured的冻结地火2K/256窗口已经结束，不能再次执行旧client计划：
HTTP `(28.634374913,2.524752475,88.172082781)`，server-fenced `30.292077426`，
prefill合计`1.591584930s`，TTFT均值`0.321815077s`；8/8 clean/usage/length，
输入messages及输出文本SHA均8/8匹配历史A2v4。主代理已从原始8件重算counter/fence/三元组。
详情见 `../operator_stack/evidence/operator_revalidation_20261013/baseline2_client/`；
草稿图包装/metadata捕获及selector计数证据在同级 `draft_graph_evidence/`。
这些是当前基线，不是旧n=1候选在DSpark六行的兑现；slot/mask候选仍未接入/关闭。
任何后续起停/bench都先重新核实际PID/SID，不能根据此快照盲目重启或抢其他组。

## 历史检查点（以下 active/启动授权已被上面的暂停覆盖）

以下是暂停前的原始记录。工作区当时为cf16be3，README为本轮
tracked改动；`experiments/delivery_matrix/`为本轮新代码/证据，其他大量untracked
用户产物不动。README先有A2/DCP真实数字和FAIL/pending，MANIFEST尚未从新HEAD刷新。

## 现场下一步

- **本回合结束有效检查点：**A2全部实机验收/README已完成，所有目标推理API现在已停，
  SSH/项目P,D容器仍Running，勿称机器离线。fyf_1009的外部rain_fusion_att三轮迁到
  physical5→0→1→9，第三轮明确D8–15也占；原PID消失不代表轮换任务终态。
  root已async问用户协调释放/预留16chips，用户未回复。这是本次新fyf资源条件的
  **第一个受阻goal回合**；不能把3次poll算3个goal回合而提前update_goal blocked，
  不能pause/complete。后续若同条件在连续3个goal回合都无法实质推进，再按goal规则blocked。
  当前安全准备已穷尽：source/fullOPP/可写jobs/IPC/ports/RAM门与改正的CEDjobs配方已审核；
  新p5_jobs/d5_jobs均未执行，P,D未init。site已按root要求结束无限轮询，等后续goal只读复核
  或用户协调；不能自行启动单D/DCP绕过持续跨全16轮换占用，不能操作外部fyf等容器。
  DCPv6 CPUplan/recipe见evidence/dcp8_v6_cpu_prep_20261013/CPU_PLAN.md；root实读script SHA
  b84f9f32cbacaff3e47c5e62171a2dc1a46e7c373e5ab20fb5927ebf33a111ed，独立make_plan参数断言
  全PASS，文件验证函数仅把远端v6路径映射本机冻结原件（非SDK/device替身），CPUfp
  e1277a56ecf760c436cdcb6819c98763bf8b353b538f0634bba238fc1a592c90，不能作actualSID/fp。
  仍**未授权任何DCP execute**；下一真实窗口先36-file v7/D源220/fullOPP/write/IPC/8–15门。
  wirewrapper已单文件复制到P可写`.../jobs/proxy5/ced_wire_audit.py`，SHA82873d5b…/原proxy
  813c5a0e…/--help均PASS；这是源码准备目录，不是proxy已运行，wire子目录仍absent/18992拒绝。
  最后用户交付需准确说A2完整通过、CED/DCP待验/外占阻止、DRAM仅CPU原语和计划未接实机。
  当前HEADcf16be3/未提交，MANIFEST尚未从新HEAD刷新，不能宣称完整可发布包。

- **最新资源/配方，优先于下面旧转场授权：**A2已停、结果完整通过；CED新p5/d5还没起。
  旧p5根目录在P容器ro bind（实际可写门FAIL），原脚本没有执行。root明确改为P
  `/home/l00886679/tmp/dsv41-ced-runtime-20261013/jobs/p5`（host bind RW），D同前缀
  `jobs/d5`（D overlay RW/非P host同一backing）；原脚本保留，新p5_jobs/d5_jobs script已
  实读审核，仅env输出/native与cache两路径变化、kwargs完全不变，TMPDIR+UUID+NUL96。
  新脚本SHA P23e5916c2048ccc8a64948e88569b7e346b1f531c15b81644abc0ac1cfac443a /
  D00331485daab4ba7fca41ae3bbbcecea93954515a78469889b4e12c947198020。
  新CPUfp Pe7674b826f7c38a789ac1e5e0ee176eff2c1447b8f1b670a41710ed0c937f700 /
  Dd192f5fbd6d068d6079de6b8a2ae8ca552c723eb669a1d2e695f9ed1aa2cc3c9，OPP=null非actualfp。
  root已授权门PASS后执行这对新jobs脚本（不是旧parent脚本），暂不proxy/不gen。
  但最后现场门发现physical5实际外部PID3215293 rain_fusion_att（AICore100%，304MB），
  cgroup→Docker f7f117abc17e338d8051fc253d412dd60ae56f170f8fd3b1dd7c8451958b5419 /
  容器fyf_1009，非项目容器，不动它。两新jobs仍absent/all目标ports无listen。
  site已安排约60s只读两轮复核；如果自然释放且16全无冲突可按原授权init，不要为等待
  反复重启或清理。若持续，bench已followup CPU-only备选DCPv6：physical8–15、genericv7、
  dcp_runtime_20261013_v6 + draftv2、det-reduce1、index0/BI/async/best/spec5、PREFIX0/
  MAXLEN196608/KV4GiB，新model dsv41-delivery-dcp8-v6/18960、D新短job
  /home/l00886679/tmp/dsv41-dcp-v6-runtime/d6。**该DCP配方尚未root审核，不准execute/API**。
  CEDwire的未来audit-output必须是P可写`.../jobs/proxy5/wire`；wrapper source/hash不变。

- **实际停服/代理接管最新：**site已一次SIGTERM A2 API342558/host2898801，API/EngineCore
  退出、18960释放、16chipsHealthOK/无runningNPUprocess/HBM约2.9–3.1GB；run.exit0，
  但runner/8workerzombie，须留完整收据与RSS/PSS，不标clean、不杀僵尸/不reset容器。
  site一度仅回轻量状态后turn结束，root已followup明确继续CED双容器来源/fullOPP/
  resource/ports/job/IPClen门；**门全PASS已获授权p5/d5 init，不需再等待批准**，
  新actualSID/fp/16ranksource/clamp/BI/Engram/graph门后暂不proxy/不generation。
  root新独立wirewrapper ced_wire_audit.py（6870B），SHA
  82873d5b2c454d50580b80c39a46970f69e18c7b15118fa8f7dcec78d098c0d1，原proxy固定SHA
  813c5a0e1a8b7e7988ee9f02fd3a83d8fa38ede9a88713a9c65e5cf0cf99eb33。
  18新增CPU控/顶层129PASS；不改冻结v4package/P/D SDK/model，只记录P,D出站原body与
  P非stream响应，D/SSE不读取，不增发重试/header；I/O在D提交前属TTFT。
  site已获授权独立新wrapperdir复制核hash/--help，**未授权起proxy**。
  未来proxy命令见CED_WIRE_AUDIT.md，audit-output新proxy5/wire。worker_bench已完成
  CED冻结clientv2 CPU准备及新benchdir/model dsv41-ced-pd-v4，等待新pairroot放单canary。
  第二阶段内容唯一池方案/容量陷阱见UNIQUE_POOL_PLAN.md；这是计划不是实机接线。

- **最新：A2 v4完整验收已过。** code8/128cases、GSM100/100、Vision23/23、147456四针4/4；
  地火2K/32K/147456各8正式请求全部终态exit0/clean，root独立审计原件和8rank计数通过。
  最终147456 HTTP32.381891614/A3.197492163/98.743217385，TTFTmean22.988100295，
  prefill合计180.52202238；graph638/eager8每rank，实际usage147456/256/147712。
  README已齐A2v4三档+code及精度门。root已明确授权site：prestop归档后仅SIGTERM
  项目A2 API（container342558/host2898801以实时/proc匹配SID为准），容器/缓存/证据保留；
  核释放/Health、无外部实际占用、双容器29file/source220/config/fullOPP/config.ini/
  ports/IPClen门后按已审p5/d5初始化CEDv4新pair，可并行init；**不proxy/不generation**。
  每端新SID/actualfp/8rank来源与D-onlyclamp/BI/Engram/draft门出后root再放新canary。
  bench不再发A2请求，CPU准备CED冻结clientv2/新benchdir；不能继承旧failedpair身份。

- **最新性能/转场检查点：**A2v4地火2K与32K正式8请求已终态/归档，root独立审计
  压缩/解压SHA、8rank计数/usage/tag/fence通过。2K HTTP28.071336646/A2.524752475/
  89.940586267；32K HTTP29.595011408/A2.901849218/98.051971584，TTFTmean4.733176581，
  prefill合计36.92654554。代理32K进度消息误抄2Kusage后暂停下一档，root已核8原件实际
  prompt32768/completion256/total33024并明确恢复147456放行；不是实际错档或重跑。
  site已完成p5/d5 CPUrecipe/plan，root实读审查通过，均新短job且保留原3916d72v4包/模型
  dsv41-ced-pd-v4/best/Engram0/D-onlyclamp/TP8DCP1，不加DRAM/shared/DCP。
  p5_launch SHA bdd4b807108b2f05d711ac86737a11be5c7e07dfda75c6a3b36d67cab0d9bb5c，
  d5_launch SHA1f7277c1bbe42f20eb5039e9c024c0e73608c7fa0124e793c24a2156228efe99。
  CPUplan fp P457e5d32…/D1a6f0c80…不是未来execute actualfp；实际serve前覆盖placeholder
  输出/cache到p5/d5。TMPDIR+假定36字符UUID+NUL91<107。完整npu-smi进程表8–15无running
  NPU进程；外部全16cgroup映射仅权限，不当占用。**仍未授权execute/停A2**；待147456
  正式8请求终态/归档root审计后，明确放site唯一操作者停A2/归档释放，再新P/Dinit。

- **最新完整精度检查点：** A2 v4 的 code8=8/8题、128/128用例，GSM100=100/100、
  Vision23=23/23、147456四针=4/4。主代理按冻结原函数独立重评分，并核4raw请求/响应
  SHA、实际prompt147435–147438、stop和三identity同SID/fp；见strict_quality_audit.md。
  长针PID3186019已退出，site已归档全部小件；root明确放行bench冻结clientv2串行地火
  2K→32K→147456各8请求，site不并发生成。等终态/归档后更新README，再进行CEDv4新
  P/D job窗口16chips门与具体launcher审核；未经授权不提前停A2，不复用旧d4 SID。

- **最新：**D4已按计划停并留收据（仅init/source成功，没有推理）；外部matmul已退出，
  但之后xp_mk Qwen3 test_forward占physical12/14，A2v3在启动前被门阻拦，未执行。
  主代理改准备新A2v4在P容器physical0–7；研究确认P设备可见，但actual best worker
  8dc219…硬guard8–15，不能只改env。已派生仅1处literal到0–7，原SDK/model/async/
  图/计数与原snapshot不改，不伪造env。原相邻contract/privateAPI仍8–15，**不复用**；
  标准GSM/Vision与delivery客户端不依赖它。
  generic v7包`/home/chiro/tmp/dsv41-delivery-matrix/generic_delivery_20261013_v7.tgz`，
  388779B，SHA827c8310684b615a815c813600589e9c3da4ab482087587219c197a25fd35873。
  geometry component manifest df5461194e84c713352d85e3e92b03459c947c1541b3c045da1e676cb879c554；
  derivedworker7cabe528dc5df34d8851637deb1cd4ea2f13624ba5dbd427455d6d0fe5dc9ad0。
  顶层110CPU通过。service/prepare_launch新增--physical-chips/--physical-worker，
  P新model dsv41-delivery-a2sim-tp8-v4/18960/form a2-tp8-a3sim/maxlen196608/index0，
  双流/BI/async/真实DSpark5图保留；launcher先unset CED/DCP/旧geometry污染，A2明确DCP1。
  site已做v7三处36/36文件SHA门、P侧220/220source/config与geometry门；
  source_gate/plan.json SHA479c211a1dda8e1e3de3b155e3b3a83fd1c06968846d9ee414f1573814484686，
  CPU计划fp38063cf61a7167ca5b78753be16ae78a37d6a6324c1d55c65ca5b3ec9adce2ac；
  runner.sh SHA502aaabfd432208b9ee56a0d46225bde97dfb68bd43680b87d5de255da46f28c。
  root已实读审核并放行init-only，完整IPC86/107、18960归属/0–7空闲/Health/RAM门均过。
  A2v4 runner342556已启动，无generation；早期BI-META transient skipped RuntimeError，
  后多进程已installed=all/Enabling BI；还有causal_conv1d_update_npu import fallback warning。
  目前TP0–5出现，6/7尚待；实际service plan fp
  61b841dfcd20d7f2892934e1fbd8086ab409a36d99d5de9beea4dd12399e53f9，
  不同于CPU计划38063cf…；selection暂见SID9d2ba214…、engine/ASGI false。
  site待最终8rank实际BI/来源/graph与GET ownership门，不提前判pass。
  最新live poll runner342556/API342558/EngineCore343435与8workers存活，static compile
  7/126、KV526010≥MAXLEN196608；一次60s shm broadcast等待warning后编译继续，
  run.exit absent/18960未listen，**同job继续等，不因warning重启**。
  本机新增ready描述的geometry manifest错绑负控，111CPU通过；只加测试，v7部署包未改。
  此后site本轮实时复核all live/126静态核+engine/ASGI true、models/identity GET200，
  **A2v4已ready**，SID9d2ba214732b4eed84dc4f416f27c6ae、actualfp仍61b841df…。
  收据evidence/a2sim_tp8_v4/ready_receipt_20261013.md，SHA45254d97…。
  root已明确放bench新单codecanary→通过后串行code8+bwrap，失败立即停止/保留，不重试。
  site只读监督/准备质量资产，不能与bench并发生成。代码阶段完成再交GSM100/Vision23/
  147456四针；精度门过后再地火三档。此后新canary通过（冷值不计正式），code8正式
  已终态exit0且无client PID：HTTP三元组27.218877914/4.985611511/183.167415152，
  rounds278/clean8/8/TTFTmean .311842044/prefill合计1.42460445s；serverfenced32.571885432。
  本机bwrap8题128/128，remote8 sandbox_error原件保留；root已独立核8split/hash/输出/标签。
  scoreSHA9f232da2…、manifest6ec9d926…、大raw4c9196d1…(1419839B留远端)。
  root已放行site接管串行GSM100(原512)+wire旁路→Vision23→147456四针，不复用private合同。
  bench等待这些精度门，不Dihuo；client正式件repo复制待归档。见CODE8_RESULT.md。
  code8正式archive已repo client/code_formal8，收据SHA32a7d410…；所有split/manifest/
  本机score原样拷贝，远端大raw留原路径。
  GSM100当前已真实启动：A3-21 PID3182199 live，远端
  /home/l00886679/tmp/dsv41-a2sim-v4-quality/gsm100/，wire wrapper f5bc086…/原eval aad813…，
  model/SID/fp GET200与ready一致。最新25/100 checkpoint+25wire行，前20临时100%，
  不是最终质量结论；无并发客户端。继续poll同PID至终态，不因超时重发/重启。
  新job建议/home/l00886679/tmp/dsv41-a2sim-v4-runtime/a4，需可写/IPC<=107；
  新package/work/delivery_matrix_generic_20261013_v7/delivery_matrix。
  bench已被通知取消v3命令，待v4完整配置/ready后建新~/tmp/dsv41-a2sim-v4-bench。
  0–7重新读only free/Health/归属后才启动，不动任何外部容器。
  见PHYSICAL_WORKER_CANDIDATE.md；下面旧计划是追溯，不以v6或v3运行A2v4。
- **最新优先于下面旧启动历程：**P-v2+D-v3短d3的唯一canary真实失败，D API已退出，
  KV8rank已传输/128重放后_pool_kernel MTE/SMMU首发，HCCL随后传播；run.exit0不代表成功。
  失败目录`evidence/ced_v3b_canary_failure_20261013/`，diagnostics SHA41b977a6…。
  主代理四处masked残余projection负地址clamp已6项CPU/顶层100项通过，不宣称因果证明。
  新CED v4包179474B，SHA3916d72f0fb798dac9ffe3dab243abd4999e2de42e6f0cc83a7b3785ef940aab，
  主代理已授权site归档旧P测量后仅项目SIGTERM旧P/proxy，并新P4+D4（容器保留），
  新短job p4/d4、新model dsv41-ced-pd-v4，两端best双流；仅D加
  --compressor-overlay <v4package>/ced_compressor_20261013_v1。
  component manifest accf27fa871e62a78cb5e7aed3318c789f69feae14a5299db347795519e4e9f0；
  derivedsource ea3ead30e43d1e11e84cfb410a0a2faeec54e03ead01e53478c30e75dca564bc。
  原SDK0d4d78…导入门+8rank实际source描述必须匹配；新ready后再放新canary，不沿用旧SID。
  旧P只读metrics证实每rank已有1个failedcanary tag；不能把旧P和新D计数拼成pair。
  详见CED_COMPRESSOR_CANDIDATE.md。旧proxy已SIGTERM退出；旧P API收到SIGTERM后
  native teardown最后报corrupted size vs. prev_size，run.exit134，不能记clean stop。
  仅余RSS0 zombie、P/D服务NPU均释放/HealthOK、全部旧端口已释放，容器保留。
  **新外部占用：**m00933363_catlass_runtime_bz_a3_1的matmul gate曾占NPU2/chip4/5，
  后已自然释放，但迁到NPU3/chip6（同外部容器6c515…，PID2750473及profile子进程）。
  不动外部任务；P4须0–7全空闲，D4可在再次确认8–15空闲后独立初始化。
  旧P/proxy独立shutdown receipt SHA0852ecdc…已归档。
  v4双容器29文件源门已过；evidence/ced_v4_startup_20261013/package_preflight.md
  SHA b245f7b1de1513a2aeef339f4fbc61bab6d2cf915feaee9a1dfb0e9894119ba4。
  D4已init-only启动（未发generation/未起proxy），runner1609465、API1609471、
  EngineCore1610342，run.exit absent；plan fp
  78e8e6fccdf143e0bb05da9c3872651a47867439ef7952295cc5be0a5c801614，
  SIDe10c11f1fe9e4965a75a4fa20d9e8c11。此后D4已18991 listen/engine+ASGI true，
  8rank实际sourceSHA/manifest、graph=true/spec5/BI1/Engram index0均核验通过。
  **只有init/source gate成功，无generation/canary/精度/性能。**ready snapshot待site归档。
  P4仍因外部chip6占用hold，site约60s只读复核。用户已被async询问协调释放
  （当时占4/5，此后commentary已更正为6），尚无回复。
  为推进矩阵，主代理已授权调整顺序：site归档D4 ready后计划内SIGTERM D4 API，
  留停服/释放证据；用8–15先补A2 v3（model dsv41-delivery-a2sim-tp8-v3、18960、
  form a2-tp8-a3sim、device-index0/maxlen196608/route空/best BI async双流draft），
  新短job a3。先v6 source-only包每文件门+具体launcher/fingerprint回root审核，再起A2。
  CED v4包/launcher全部保留，待16chip可用再新窗口恢复，不继承旧测量。
  workload_bench已followup准备A2v3客户端，**不要API访问直到site/root明确ready**；
  新单canary→code8+bwrap→site GSM100/Vision23/147456四针→必要精度过后地火2/32/144K。
  新质量不能继承A2v2。local TMPDIR/home/chiro/tmp/dsv41-a2sim-v3-bench，
  remote/home/l00886679/tmp/dsv41-a2sim-v3-bench；未创建/使用/tmp。
  offload_research的10m占用监视turn已完成，需时复用。site仍唯一服务操作者。
- site_runtime唯一服务操作者；workload_bench客户端/独立评分；offload_research
  只读ABI资料。已有代理均gpt-6-luna[max]，等待用60s；主代理复杂实现。

## 历史启动历程（仅追溯，不表示当前在线）

- P容器`dsv41-ced-p0-7-20261013-v2`，CID16d30453…，18990已ready、physical0–7。
  image固定b69e5882…，整个vendors树（含30B config.ini）按全manifest复制/ro-bind。
- D复用旧`dsv41-real-stack-tp8-20261010-a321`/CIDb1aa15… physical8–15；旧DCP已
  SIGTERM exit0，有shutdown warnings/zombie证据，NPU资源释放，不重启daemon。
- D CED服务v2真实失败：Native build_app末尾SageMaker bootstrap预先封middleware栈，
  返回后app.middleware报错。完整证据保留。root v3改为config.middleware构建前
  安装同guard，HTTPmiddleware纳入指纹；实际SDK ABI确认，CPU prebuilt-stack正负控通过。
- 服务v3包 `/home/chiro/tmp/dsv41-delivery-matrix/ced_delivery_20261013_v3.tgz`，
  166512B，SHA d4850d9e63285885767e54d1321f8eaec82d44b0b17cf8698b5e7df0a940d1d5。
  P继续v2，D新v3，两端core overlay同7fc7e2…；served-model同`dsv41-ced-pd-v2`。
- D首次v3 job又因IPC path>107失败（不是middleware失败）。同包短root重试：
  `/home/l00886679/tmp/dsv41-ced-runtime-20261013/d3`，path90/107，runner1238305、
  EngineCore1239181。此新短job已HTTP ready且middleware负控400通过；不可继承旧长job指纹efd677…。
- P19091是producer TP1正常监听，P监听19090–19097；Dconsumer不bind该KV组。
  已通过P日志确认，不杀listener、不改19092。
- P/D identity/实际OPP/同core已核，proxy18992已ready；pair SID
  `3762dc9b083a4b33c045c877273d19a7cb5fc927dee2960b0181948d81424f41`，runtime fp
  `c70b47fc3e303f58e33e3b1fa0f8960565eb5eefdce3900fb5a394754c9aec8b`。
  D短job fp d877f65…，D guard缺合同HTTP400后当时仍在线；完整ready receipt已写
  evidence/ced_preflight/ced_v3_pair_ready_20261013.md，SHA9187ffdf…。
  随后的首canary已失败（见最新），预定序列仍为新的单codecanary验证真实KV transfer→
  code8+本机bwrap评分→site GSM100/Vision23/147456四针→必要精度通过后地火
  2K/32K/147456。所有请求串行；双流长针失败保留后另off候选重验，不能继承精度。
- CED客户端v2：46830B，SHA5de07966b80b1d4535fcdc7b3e5cfd1b1e97b8c81c2744b8507f424a8391b73b。
  P+D prefill按每请求max(P delta)+max(D-only delta)，不含transfer等待。
- 新GSM100可用quality_wire_capture.py旁路包原evaluator（原SHA aad813…）；
  wrapper6266B、SHA f5bc086e69fed7706933390cb4287460c3aa536a378f1d73b3700c7c3eeb985d。
  不改8shot/chat/512/抽题/评分/重试，额外逐HTTPattempt保存finish_reason/usage/wireSHA。
  4项CPU透明性正负控通过，尚未实际新GSM运行；sidecar不能代替题级strict结果。

## 已完成/未完成

A2 v2：地火2K与code8实际数据，GSM100/100、Vision23/23、code128/128；
缺无route探针且修正末chunk边界的32K/147456与长针，后续新job maxlen需≥147712。
DCP v3：code与三档地火都完成；GSM99/100严格FAIL、Vision23/23、长针4/4。
同题gsm12唯一诊断重复13/13、349 tokens stop，不覆盖整轮；原因未裁定。
质量corrected receipt81795311…（旧containerID手写错误原件保留），actual消费者证明false。
当前不能称三形态全部精度达标。

## 主代理离线候选（未部署，不混入CED基线）

1. DCP稳定性v6：all5gate统一且全部调用bounded helper（8处），128MiB gather帽、
   512B对齐、T分块且复用同一对workspace，kernel驻留pack地址保持。
   manifest b0c1d36e44707c872707477677a5ef60a4d3d53f86c5df0d1092e47e41706a89。
   8项CPU；CLI --dcp-det-reduce1拒旧v3/v5。v5的两个inline gather遗漏已发现，
   v4/v5原件保留但不部署。还需真实capture/
   内存/质量/性能；不宣称已证明GSM错误由归约造成。
   后续source-only包`/home/chiro/tmp/dsv41-delivery-matrix/dcp_delivery_20261013_v6_sources.tgz`，
   380546B，SHA859158d8055c4655a48c4d9c5a430f63842572a0a1ea91b9ae0d00ff890e5cb8；
   含修正后的worker/route-off，A2 maxlen默认196608，未部署。
2. shared_engram.py/build_shared_engram.py：actual best b422e989…的3个构造/加载
   方法窄改，INT8/F32 readonly MAP_SHARED；跳完整私有shard分配，不用ACL注册，
   阻止set_rows/live remap，路由/lookup AST原样。component manifest37ec7b7a…，
   12项CPU；未接CED overlay/真实TP8PSS或精度。
3. kv_content_pool.py：RAM memfd、内容byte去重+hash碰撞保护、alias/read lease/
   池满/源变动回滚、只读peer FD；8项CPU。只有single-authority/thread-safe原语，
   没IPC/coordinator、NPU DMA、per-ref mapping、P2quota或P/D单侧选择。不可标容量达标。

actual native offload ABI以a3-21 vLLM6e448d0/Ascend e43cf1e为主；13源码已小件
归档 `/home/chiro/tmp/dsv41-offload-research/ced-actual-offload-abi`，见证据actual-runtime-abi.md。
不要把graph_prep/ref或fed4fork当活动SDK；当前NPU adapter不传mapping、handler不消费
runs、没有共享mmap/P2接线，旧P-only prototype真实续算未通过，不能整文件倒灌。
接口清单与scope correction在evidence/dram_design/：单侧只selected侧TP8池/handler，
另一侧不读写DRAM KV池，仍Mooncake；P/D共享Engram独立。旧Pstore→Dread不采用。
MultiConnector首hit可被后续None覆盖、preempted优先分支、early-ACK/tracker注册竞态
需要显式selected-source与完成barrier；D-only hit仍须空recvACK释放P。

继续完成CED基线、A2长上下文和DCP严格精度回归，更新README完整表/复现；再部署并
验证单侧DRAM+共享Engram+唯一KV池，真实D2H/H2D/PSS/容量/精度都必须留证。
本机临时目录~/tmp/dsv41-任务名；SSH单文件<1MB，大原件留远端+分拆/gzip双SHA。
