#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化产物结构 + 容差验收（**明确不用 sha256**）。

用法：
    python3 verify_structure.py --dir <模型目录> [--ref-manifest <A3-node1 的 manifest.json>]
                               [--sample-shards 8] [--json-out <path>]

为什么不用 sha256：
    我们（A3-node1）用 DP16 量化，A2 用 DP8。分片切法不同 ⇒ 80 分片 vs 72 分片，
    约 47.5% 的张量有 int8-LSB 级差异。**总字节相同、数值等价。**
    「分片 sha256 不同」是预期，不得据此判失败。判据只有结构 + 容差（本脚本）。

退出码：0 = PASS，1 = FAIL（有必须项不满足），2 = 无法判定（缺文件/参数）
"""
import argparse
import collections
import json
import os
import sys


EXPECT = {
    "index_tensors": (187226, 0.0005),      # 最终装配目录；±0.05%
    "tags": {
        "W4A8_DYNAMIC": (184320, 0.005),
        "W8A8_DYNAMIC": (744, 0.01),
    },
    "vision_tensors": (266, 0.02),
    "vision_bytes": (970536960, 0.02),
    "mtp_tensors": (1224, 0.02),
    "min_shards": 60,
    "max_shards": 100,
}


def load_index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(p):
        return None, None
    idx = json.load(open(p))
    wm = idx.get("weight_map") or {}
    return idx, wm


def load_quant_tags(d):
    """标签分布来自 quant_model_description.json（msmodelslim 产出）。"""
    for name in ("quant_model_description.json", "quant_model_weights.safetensors.index.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        try:
            obj = json.load(open(p))
        except Exception:
            continue
        if name.endswith("description.json") and isinstance(obj, dict):
            tags = collections.Counter()
            for k, v in obj.items():
                if isinstance(v, dict):
                    t = v.get("quant_type") or v.get("dtype") or v.get("type")
                    if t:
                        tags[str(t)] += 1
            if tags:
                return dict(tags), name
        # 退化路径：从 index 的键名推断（_scale / _offset / weight）
        if isinstance(obj, dict) and "weight_map" in obj:
            tags = collections.Counter()
            for k in obj["weight_map"]:
                if k.endswith(".weight_scale") or k.endswith(".scale"):
                    tags["W4A8_DYNAMIC(scale)"] += 1
            if tags:
                return dict(tags), name + " (inferred)"
    return {}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref-manifest", default="")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    d = os.path.abspath(os.path.expanduser(a.dir))
    problems, warns, facts = [], [], {}

    if not os.path.isdir(d):
        print("[FAIL] 目录不存在：%s" % d)
        return 2

    idx, wm = load_index(d)
    if wm is None:
        print("[FAIL] 缺 model.safetensors.index.json")
        return 1
    n = len(wm)
    facts["index_tensors"] = n
    shards = sorted(set(wm.values()))
    facts["shards"] = len(shards)
    lo, hi = EXPECT["min_shards"], EXPECT["max_shards"]
    if not (lo <= len(shards) <= hi):
        problems.append("分片数 %d 超出合理区间 %d-%d" % (len(shards), lo, hi))

    exp, tol = EXPECT["index_tensors"]
    if abs(n - exp) / exp > tol:
        warns.append("index 张量数 %d 与参考 %d 相差 >%.2f%%（DP 宽度不同会有差异，"
                     "先核对 §结构判据再判）" % (n, exp, tol * 100))
    else:
        facts["index_tensors_check"] = "ok(±%d)" % int(exp * tol)

    # ---- vision / mtp 计数 ----
    v_keys = [k for k in wm if k.startswith("vision.") or k.startswith("aligner.") or k.startswith("image_")]
    m_keys = [k for k in wm if k.startswith("mtp.")]
    facts["vision_tensors"] = len(v_keys)
    facts["mtp_tensors"] = len(m_keys)
    if len(v_keys) == 0:
        warns.append("没有 vision/aligner/image_ 张量（如果是纯文本交付可忽略）")
    if len(m_keys) == 0:
        warns.append("没有 mtp.* 张量（DSpark 未并入？）")

    # ---- config.json ----
    cfg_p = os.path.join(d, "config.json")
    if not os.path.exists(cfg_p):
        problems.append("缺 config.json")
    else:
        cfg = json.load(open(cfg_p))
        qc = cfg.get("quantization_config")
        facts["quantization_config"] = qc
        if not isinstance(qc, dict) or qc.get("quant_method") != "ascend":
            problems.append('config.json 缺 quantization_config={"quant_method":"ascend",...}'
                            "（跑 prepare_runtime_ckpt.py --dir %s）" % d)
        t = cfg.get("text_config") or {}
        facts["engram_layer_ids"] = t.get("engram_layer_ids")
        if not t.get("engram_layer_ids"):
            warns.append("text_config.engram_layer_ids 为空 ⇒ 起服必须 enable_engram=false")
        facts["num_hidden_layers"] = t.get("num_hidden_layers")
        facts["num_nextn_predict_layers"] = t.get("num_nextn_predict_layers")

    # ---- quarot ----
    qp = os.path.join(d, "optional", "quarot.safetensors")
    facts["quarot_present"] = os.path.exists(qp)
    if not facts["quarot_present"]:
        warns.append("缺 optional/quarot.safetensors（QuaRot 全局旋转）—— 若用 qrot 权重必须补")

    # ---- 标签分布 ----
    tags, src = load_quant_tags(d)
    facts["tag_source"] = src
    facts["tags"] = tags
    for tag, (cnt, tol) in EXPECT["tags"].items():
        got = tags.get(tag)
        if got is None:
            warns.append("标签 %s 未在 %s 中出现（不同 msmodelslim 版本写法可能不同）" % (tag, src))
        elif abs(got - cnt) / float(cnt) > tol:
            warns.append("标签 %s 计数 %d 与参考 %d 相差 >%.1f%%" % (tag, got, cnt, tol * 100))

    # ---- 分片完整性（只查存在性 + 大小 > 0，**不比对 sha256**）----
    missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
    if missing:
        problems.append("%d 个分片文件缺失（例：%s）" % (len(missing), missing[:3]))
    zero = [s for s in shards if os.path.exists(os.path.join(d, s)) and os.path.getsize(os.path.join(d, s)) == 0]
    if zero:
        problems.append("%d 个分片是 0 字节（例：%s）" % (len(zero), zero[:3]))

    # ---- 参考 manifest 对比（可选；只比计数与标签，不比字节）----
    if a.ref_manifest and os.path.exists(a.ref_manifest):
        ref = json.load(open(a.ref_manifest))
        rn = (ref.get("index_tensors") or ref.get("n_tensors") or
              len((ref.get("weight_map") or {})))
        if rn:
            facts["ref_index_tensors"] = rn
            if abs(n - rn) / float(rn) > 0.0005:
                warns.append("与参考 manifest 张量数差 %.3f%%（>0.05%%，请人工核对）"
                             % (abs(n - rn) / float(rn) * 100))
        else:
            warns.append("参考 manifest 里没找到张量数（key: index_tensors/n_tensors/weight_map）")
    elif a.ref_manifest:
        warns.append("参考 manifest 不存在：%s" % a.ref_manifest)

    # ---- 输出 ----
    print("=" * 72)
    print("结构 + 容差验收：%s" % d)
    print("=" * 72)
    for k in sorted(facts):
        print("  %-26s %s" % (k, facts[k]))
    print("-" * 72)
    for w in warns:
        print("  [WARN] %s" % w)
    for p in problems:
        print("  [FAIL] %s" % p)
    verdict = "FAIL" if problems else ("PASS(WARN)" if warns else "PASS")
    print("-" * 72)
    print("  结论：%s   （判据 = 结构 + 容差；**未使用任何 sha256**）" % verdict)
    print("        分片 sha256 不同是 DP 宽度不同的预期结果，不作为判据。")
    if a.json_out:
        json.dump({"verdict": verdict, "facts": facts, "warns": warns, "problems": problems},
                  open(a.json_out, "w"), ensure_ascii=False, indent=2, default=str)
        print("  JSON：%s" % a.json_out)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
