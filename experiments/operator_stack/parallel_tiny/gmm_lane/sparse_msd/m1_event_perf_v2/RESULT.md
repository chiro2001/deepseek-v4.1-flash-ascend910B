# M1 event v2：数值通过，未获得统一收益

physical6唯一v2窗口退出0；pre-event R1_e0 raw smoke无Event且通过，然后9case完成3轮warmup＋12轮三路交错，共324测量样本，未剔除样本。实际接受SDK135/raw136（含smoke1）/candidate135/planner135；406次数值比较全部maxULP=0。271次raw/candidate canary与stream sync、135个planner frame oracle、406次args/descriptor seal和11次完整E48参数seal通过，运行后4个固定依赖SHA未变。cleanup=0，owner无保留，fresh physical6空闲、Health OK，容器全局env未变；NPU权限回0。

下表是每case每路12个operator设备/队列事件的中位，含candidate planner；单位μs。完整原始样本、paired delta、P05/P95及fresh-input区间见JSON/CSV。

| case | raw native | planner＋candidate | candidate/raw | paired更快轮数/12 |
|---|---:|---:|---:|---:|
| R1_e0 | 30.44 | 33.02 | 1.085 | 3 |
| R1_e47 | 39.29 | 33.04 | 0.841 | 9 |
| R3_e0 | 30.95 | 33.07 | 1.068 | 6 |
| R3_e47 | 39.50 | 32.80 | 0.830 | 10 |
| R3_three_slots | 37.33 | 45.24 | 1.212 | 3 |
| R6_e0 | 30.53 | 32.74 | 1.072 | 4 |
| R6_e47 | 39.64 | 33.10 | 0.835 | 10 |
| R6_mixed_counts | 38.09 | 38.96 | 1.023 | 5 |
| R6_six_slots | 46.72 | 52.08 | 1.115 | 0 |

candidate仅在e47的3/9 case中位更快，其余6/9更慢；fresh-input＋operator中位仅4/9更快。终态为`MIXED_M1_EVENT_RESULTS_NO_UNIFORM_GAIN`。原17数学体和tiling/固定objects均保持，v1第8参错误的CPU编译反例及失败日志继续冻结。

candidate对SDK的operator中位9/9更小；SDK host submit中位93.13–109.67μs，raw12.53–13.84μs，candidate15.40–20.18μs。设备事件含host提交时的队列空闲，该差距不能当作正式已入图GMM收益或纯math核差距。M1使用实际层0/rank0参数及synthetic activation，不能外推main M36/E48、draft M15/E16 K5几何，也不能外推完整TP8的ms/step、A或tok/s。

只读原始runtime compact：56803B，SHA `6b37d601cd8e64a1a14eb3e499f77289421972b080d67402732ffc2c350ae843`。`runtime_final_status.json`给出终态，`analyze_event_results.py`可从冻结evidence重算全部统计。
