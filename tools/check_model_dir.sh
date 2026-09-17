#!/usr/bin/env bash
# =============================================================================
# 模型目录自检 —— 起服前跑，把"配置 / 权重不匹配"提前抓出来
#
#   bash check_model_dir.sh /path/to/model-dir
#   （run_test.sh 会自动调用）
#
# 为什么需要：A2 上次起服失败报
#     WorkerProc failed to start ... model.py:486
#     AttributeError: 'NoneType' object has no attribute 'primes'
# 根因是模型目录 text_config.engram_layer_ids = []（没 engram 层），
# 但服务用了默认 enable_engram=true ⇒ EngramLayout.from_args() 返回 None
# ⇒ 到 worker 启动才炸（白等 4 分钟）。
#
# 退出码：0 = 通过；1 = 有致命问题；2 = 有警告但能跑
# =============================================================================
set -uo pipefail
MODEL=${1:-}
[ -n "$MODEL" ] || { echo "用法: bash check_model_dir.sh <模型目录>"; exit 1; }
MODEL="$MODEL" python3 - <<'PYEOF'
import json, os, sys

m = os.environ["MODEL"]
fatal = warn = 0

def o(s): print("  \033[32mOK\033[0m    %s" % s)
def w(s):
    global warn; warn += 1; print("  \033[33mWARN\033[0m  %s" % s)
def f(s):
    global fatal; fatal += 1; print("  \033[31mFATAL\033[0m %s" % s)
def h(s): print("         -> %s" % s)

print("[check] 模型目录: %s" % m)
if not os.path.isdir(m):
    print("[check] 目录不存在"); sys.exit(1)

# 0) 软链健康度 —— 量化产物是软链构造的（L5→L4→L3→L2→L1），必须先确认没断链。
#    否则起服会白等几分钟后在 worker 里报 `No such file or directory`。
_links = []
_broken = []
try:
    for _name in sorted(os.listdir(m)):
        _p = os.path.join(m, _name)
        if os.path.islink(_p):
            _links.append(_p)
            if not os.path.exists(_p):
                _broken.append((_p, os.readlink(_p)))
except OSError as _e:
    f("无法列出目录: %r" % (_e,))

if _broken:
    f("发现 %d 个悬空软链（起服必然失败）" % len(_broken))
    for _b, _t in _broken[:6]:
        h("%s -> %s" % (os.path.basename(_b), _t))
    h("修法：装配脚本必须写**绝对路径**软链，且源目录（L1）不能移动/删除。")
    h("自检工具：bash tools/model_mount_args.sh %s" % m)
elif _links:
    o("软链健康：%d 个软链全部可解析（起服时 serve_a2.sh 会自动逐层挂载）"
      % len(_links))

# 0b) 软链链条有多深？（只挂一层是常见错误，这里显式报出来）
try:
    _depths = {}
    for _p in _links[:200]:
        _hops, _cur = 0, _p
        while os.path.islink(_cur) and _hops < 16:
            _nxt = os.readlink(_cur)
            if not os.path.isabs(_nxt):
                _nxt = os.path.join(os.path.dirname(_cur), _nxt)
            _cur, _hops = os.path.normpath(_nxt), _hops + 1
        _depths[_hops] = _depths.get(_hops, 0) + 1
    if _depths:
        print("  info    软链跳数分布: %s"
              % ", ".join("%d 跳 × %d" % (k, v) for k, v in sorted(_depths.items())))
        if max(_depths) >= 2:
            h("链条 ≥2 跳 ⇒ 只挂 L5 一层会悬空；serve_a2.sh 已自动挂全部层级")
except Exception:
    pass

# 1) 必需文件
for name in ("config.json", "quant_model_weights.safetensors.index.json"):
    (o if os.path.isfile(os.path.join(m, name)) else f)("%s" % name)

# 2) config
cfg = {}
try:
    cfg = json.load(open(os.path.join(m, "config.json")))
    tc = cfg.get("text_config") or {}
    o("config.json 解析成功  model_type=%s  quant=%s"
      % (cfg.get("model_type"),
         (cfg.get("quantization_config") or {}).get("quant_method")
         or (cfg.get("quantization_config") or {}).get("method")))
except Exception as e:
    tc = {}
    f("config.json 解析失败: %r" % (e,))

layers = list(tc.get("engram_layer_ids") or [])
eng_files = [p for p in ("engram_extra.safetensors", "engram_int8")
             if os.path.exists(os.path.join(m, p))]
quarot = os.path.isfile(os.path.join(m, "optional", "quarot.safetensors"))

# 3) engram 一致性 —— 本次 A2 的失败点
print("  info    engram_layer_ids=%s  engram 权重条目=%s  optional/quarot=%s"
      % (layers, eng_files or "无", quarot))
if not layers and not eng_files:
    o("这是【无 Engram】的产物")
    w("起服时必须显式关 Engram：--additional-config 里加 \"enable_engram\": false")
    h("run_test.sh 会自动处理；手动起服请务必自己加，否则会报 'NoneType' has no attribute 'primes'")
elif layers and eng_files:
    o("Engram 配置与权重都在（%d 层）" % len(layers))
    if quarot:
        o("有 optional/quarot.safetensors")
    else:
        f("缺 optional/quarot.safetensors —— Engram gate 无法初始化")
        h("Engram 需要 quarot 旋转表；没有就只能关掉 Engram 起服")
else:
    f("Engram 配置与权重【不匹配】：配置 %d 层 / 权重条目 %s" % (len(layers), eng_files or "无"))
    h("(a) 关掉 Engram 起服（只能测主干性能）：enable_engram=false")
    h("(b) 或按 tools/derive/README.md 生成 Engram 派生产物")

# 4) vision
vision = os.path.isfile(os.path.join(m, "vision-00001-of-00001.safetensors"))
if vision:
    o("有 vision 分片")
    if os.path.isfile(os.path.join(m, "qrot_vision_report.json")):
        o("有 qrot_vision_report.json（qrot 修复版标记）")
    else:
        w("无法判定 vision 是否为 qrot 修复版；若未修复，视觉约 10/23 而非 23/23")
        h("用 tools/check_vision_rotation.py 确认；需修复则 tools/make_qrot_vision_product.py --src <此目录> --dst <新目录>")
else:
    w("没有 vision 分片 -> 视觉测试会跳过")

# 5) mtpq
import glob
mtpq = glob.glob(os.path.join(m, "mtpq-*.safetensors"))
if mtpq:
    o("有 mtpq 分片（%d 个，推荐配置）" % len(mtpq))
else:
    w("没有 mtpq 分片 -> 会用 BF16 draft，KV 少约 2.4 GB/rank")
    h("需要 mtpq 见 tools/derive/README.md")

# 6) 软链
broken = []
for e in os.listdir(m):
    p = os.path.join(m, e)
    if os.path.islink(p) and not os.path.exists(p):
        broken.append(e)
if broken:
    f("有 %d 个断掉的软链（拷目录时最常见的坑），前 5 个:" % len(broken))
    for b in broken[:5]:
        print("         %s" % b)
    h("用 rsync -L（跟随软链展开）或连同目标目录一起拷")
else:
    o("顶层软链完好")

# 汇总
print()
if fatal:
    print("[check] 结论：不可起服（%d 致命 / %d 警告）" % (fatal, warn)); sys.exit(1)
if warn:
    print("[check] 结论：可以起服，但 %d 个警告（见上）" % warn); sys.exit(2)
print("[check] 结论：全部通过")
PYEOF
exit $?
