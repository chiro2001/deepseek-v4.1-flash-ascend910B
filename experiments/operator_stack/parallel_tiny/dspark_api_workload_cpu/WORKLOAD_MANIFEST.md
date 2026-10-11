# 固定 2048-token prompt 清单

本清单仅为真实 tokenizer 的 CPU 准备；模型/HTTP/质量/性能收据未运行。warmup 两条独立保存，不进入 measured 八条。全部输出预算为256；未来v4性能统计须排除一个prefill seed，而不能直接把256当decode token。

| phase | name | task | prompt token | output budget | 原语料 byte 区间 | prompt SHA256 |
|---|---|---|---:|---:|---|---|
| warmup | warmup_0 | quote | 2048 | 256 | 6065–13576 | `cdb6e41086394dede6517dfbaecdc5efce6761c09475e256fa0e5d3c5bcdc486` |
| warmup | warmup_1 | qa | 2048 | 256 | 41292–48958 | `da4bc351581915d5c3ddb91a00431a37bf662af2e1dd1b9a0365bdac5ffea762` |
| measured | measured_0 | continue | 2048 | 256 | 77405–85266 | `74707c04476782348ee2185b63c760e780138d2c62449c6e1c3d4e77c9b7db7e` |
| measured | measured_1 | extract | 2048 | 256 | 112104–120375 | `bf951cac6d2ba85d23092f2d31d293bfc8cfbf5544ba045079390aace0686fc8` |
| measured | measured_2 | quote | 2048 | 256 | 149634–157764 | `9c0eea6579126ace4dc60e936168490931305d164e7e011c55ee2227cc59dda2` |
| measured | measured_3 | qa | 2048 | 256 | 183830–192178 | `b1b813ab7019d8674b21eba9100d053c9a90d415f75bc72b38c769fc48159b20` |
| measured | measured_4 | continue | 2048 | 256 | 219543–227175 | `945623399f29834b467a20e1162acfb7d3543e2f0ab2c78be9e3928b2c3a48d9` |
| measured | measured_5 | extract | 2048 | 256 | 253737–261059 | `320f9e262f3c7ffa74724985acc657b13d995e270ac94ad256a3876ddf6af719` |
| measured | measured_6 | quote | 2048 | 256 | 289466–296942 | `db5db12c9a474eee383045a137be59d5208aeb8cfe68fc4a5e50e4070d551009` |
| measured | measured_7 | qa | 2048 | 256 | 325827–334137 | `cdc3394bcaeb51bf2b3386082e8c4b27f7c5bedc1a4acea0b27bb5f6ec8734e8` |

全部10个prompt/原文区间互异且不重叠。以下为每条实际末尾问题；正文精确片段保存在同名 txt，token/text/source SHA 在 JSON 中。

**warmup_0（quote）**

请从上文中**逐字引用**开头第一句完整的原文（不要改写、不要总结，直接抄录）。
请将回答控制在约150字以内。

可见原文首句见证：一日，正当嗟悼之际，俄见一僧一道，远远而来，生得骨格不凡，丰神迥异。

**warmup_1（qa）**

请只根据上面这段文字回答：本段出现了哪些人物？他们各自做了什么？请分点作答，并引用原文依据。
请将回答控制在约150字以内。

可见原文首句见证：雨村听了，笑道：“可知我言不谬了。

**measured_0（continue）**

请接着上文，用相同的文风继续写下去。
请将回答控制在约150字以内。

可见原文首句见证：薛蟠素闻得都中乃第一繁华之地，正思一游，更趁此机会，一来送妹待选，二来望亲，三来亲自入部销算旧账，再计新支，──其实只为游览上国风光之意。

**measured_1（extract）**

任务：请统计上面这段文字里出现次数最多的十个词，逐一列出，并为每个词给出它在段内出现的原文片段（逐字引用 30 字以内）。
请将回答控制在约150字以内。

可见原文首句见证：周瑞家的听了，便已猜着几分来意。

**measured_2（quote）**

请从上文中**逐字引用**开头第一句完整的原文（不要改写、不要总结，直接抄录）。
请将回答控制在约150字以内。

可见原文首句见证：宝玉听了，忙下炕来，到了里间门前，只见吊着半旧的红绸软帘。

**measured_3（qa）**

请只根据上面这段文字回答：本段出现了哪些人物？他们各自做了什么？请分点作答，并引用原文依据。
请将回答控制在约150字以内。

可见原文首句见证：说起话儿来，金荣的母亲偏提起昨日贾家学房里的事，从头至尾，一五一十都和他小泵子说了。

**measured_4（continue）**

请接着上文，用相同的文风继续写下去。
请将回答控制在约150字以内。

可见原文首句见证：贾蔷又道：“如今要放你，我就担着不是。

**measured_5（extract）**

任务：请统计上面这段文字里出现次数最多的十个词，逐一列出，并为每个词给出它在段内出现的原文片段（逐字引用 30 字以内）。
请将回答控制在约150字以内。

可见原文首句见证：那时官客送殡的有镇国公牛清之孙现袭一等伯牛继宗，理国公柳彪之孙现袭一等子柳芳，齐国公陈翼之孙世袭三品威镇将军陈瑞文，治国公马魁之孙世袭三品威远将军马尚德，修国公侯晓明之孙世袭一等子侯孝康（缮国公诰命亡故，其孙石光珠守孝不得来）：这六家与荣宁二家，当日所称“八公”的便是。

**measured_6（quote）**

请从上文中**逐字引用**开头第一句完整的原文（不要改写、不要总结，直接抄录）。
请将回答控制在约150字以内。

可见原文首句见证：且说宝玉近因家中有这等大事，贾政不来问他的书，心中自是畅快。

**measured_7（qa）**

请只根据上面这段文字回答：本段出现了哪些人物？他们各自做了什么？请分点作答，并引用原文依据。
请将回答控制在约150字以内。

可见原文首句见证：谁想贾珍这边唱的是《丁郎认父》、《黄伯央大摆阴魂阵》，更有《孙行者大闹天宫》、《姜太公斩将封神》等类的戏文，倏尔神鬼乱出，忽又妖魔毕露。

真实来源：

- `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq/config.json` → `/home/l00886679/models/out/v41-w4a8-engram-dr-vision/config.json`；3394 B；SHA `40ebd329d3cb2d99d7176091afb580c182c21f48b88b63b550264a97e9c0d424`。
- `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq/tokenizer.json` → `/home/l00886679/projects/dsv41/models/out/v41-w4a8-dspark/tokenizer.json`；6367257 B；SHA `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`。
- `/home/l00886679/models/out/v41-w4a8-engram-dr-vision-qrot-mtpq/tokenizer_config.json` → `/home/l00886679/projects/dsv41/models/out/v41-w4a8-dspark/tokenizer_config.json`；801 B；SHA `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`。

- 既有正式 corpus `/work/client/data/hongloumeng.txt`，2468721 B，SHA `a7fc413bd6e3926482faddf2af9bfb4426e55d8e421481d60100c14160785578`；完整文件留远端。
- quote 问题 `/work/client/data/hlm_local/local_quote.txt`；SHA `bc0d64717ce8b069a1287adfc8da9df461149d113b6a1f4ef3d7e5bf895fce0f`；问题另追加“约150字”上限，原后缀未修改。
- qa 问题 `/work/client/data/hlm_local/local_qa.txt`；SHA `49ca197f896b06965af7625b62c8e6983ac820f201b662b05cf3979e32c9ea88`；问题另追加“约150字”上限，原后缀未修改。
- continue 问题 `/work/client/data/hlm_local/local_continue.txt`；SHA `ed8a9f2f9caec4937119347ae212e3e1de7c26623f7e8ebc070197d8503201e9`；问题另追加“约150字”上限，原后缀未修改。
- extract 问题 `/work/client/data/hlm_local/local_extract.txt`；SHA `36b509a9b3cd1950629526176658e59f9a13eb3172c652b64e088620b19df94b`；问题另追加“约150字”上限，原后缀未修改。
