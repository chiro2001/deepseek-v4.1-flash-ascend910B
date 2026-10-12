#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fusion_scan_rank0.py — 冻结 DSpark rank0 kernel_details.csv 的离线邻接融合候选扫描

只读：不触发任何 NPU 作业；不修改冻结文件；不做性能承诺。

口径（重要）：
  * 主分析对象 = Model ID 48（target 图），Step Id 5..14（10 步）。
  * 去重规则：Stream ID 为空/N/A 的通信行与具体 stream 行按
    (Step, Model, Start, Duration) 一一配对（本 CSV 上 926/926 全配对），
    配对后丢弃 N/A 行（它是同一设备 kernel 的另一个视图），保留带 stream 的行；
    N/A 行的可读算子名（hcom_allReduce__503_i_k）回贴到保留行。
  * 邻接：同一 (Step, Stream) 内按 (Start, Duration, 行号) 排序后的紧邻对/三元组；
    gap = 后一 kernel 的 Start − 前一 kernel 的 (Start+Duration)，单位 µs。
  * 收益口径：上界 = 参与融合算子的 Duration 之和（每步均值 ms）；
    保守 = 上界 × [0.35, 0.6]（依据：该图 kernel 均值 ~12.5µs、p50 5µs、
    每步 ~1900 个 kernel，融合主要回收小算子固定成本，不可能全量回收）。
  整图收益按“算子实例并集”去重后再给区间，不把 top 表直接相加。
"""
import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

CSV_DEFAULT = ("/work/results/formal_dspark_native_graph_profile_v1/profile/"
               "dspark_rank0_1869601_20261012002259638_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
TARGET_MODEL = "48"
STEPS = [str(i) for i in range(5, 15)]
CONSERVATIVE_LO, CONSERVATIVE_HI = 0.35, 0.6


def s(v):
    return (v or "").strip()


def fnum(v):
    try:
        return float(s(v))
    except (TypeError, ValueError):
        return None


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        for i, r in enumerate(rd):
            r["__rowid"] = i
            rows.append(r)
    return rows


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def median_or_none(xs):
    return statistics.median(xs) if xs else None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def gap_stats(gaps):
    return {
        "p10": pct(gaps, 0.10),
        "p50": median_or_none(gaps),
        "p90": pct(gaps, 0.90),
        "min": min(gaps) if gaps else None,
        "max": max(gaps) if gaps else None,
    }


# ---------------------------------------------------------------------------
# 1) 载入 + 去重
# ---------------------------------------------------------------------------
def dedup(rows):
    na_rows = [r for r in rows if s(r["Stream ID"]) in ("", "N/A")]
    concrete = [r for r in rows if s(r["Stream ID"]) not in ("", "N/A")]
    index = defaultdict(list)
    for r in concrete:
        index[(s(r["Step Id"]), s(r["Model ID"]), s(r["Start Time(us)"]), s(r["Duration(us)"]))].append(r)
    used = set()
    paired, unpaired = 0, []
    comm_name_of = {}
    pair_stream = Counter()
    for r in na_rows:
        key = (s(r["Step Id"]), s(r["Model ID"]), s(r["Start Time(us)"]), s(r["Duration(us)"]))
        cands = [c for c in index.get(key, []) if c["__rowid"] not in used]
        if cands:
            c = cands[0]
            used.add(c["__rowid"])
            paired += 1
            pair_stream[s(c["Stream ID"])] += 1
            comm_name_of[c["__rowid"]] = s(r["Name"])
        else:
            unpaired.append(r)
    return {
        "rows_total": len(rows),
        "rows_stream_na": len(na_rows),
        "rows_concrete": len(concrete),
        "paired_na_to_concrete": paired,
        "unpaired_na_rows": len(unpaired),
        "unpaired_na_examples": [
            {"step": s(r["Step Id"]), "model": s(r["Model ID"]), "name": s(r["Name"]),
             "type": s(r["Type"]), "start": s(r["Start Time(us)"]), "dur": s(r["Duration(us)"])}
            for r in unpaired[:10]
        ],
        "paired_by_stream": dict(pair_stream),
        "comm_name_backfilled": len(comm_name_of),
        "kept_rows_after_dedup": len(concrete),
        "comm_name_of": comm_name_of,
    }


# ---------------------------------------------------------------------------
# 2) 邻接对/三元组
# ---------------------------------------------------------------------------
def adjacency(rows, model, steps):
    by_stream = defaultdict(list)
    for r in rows:
        if s(r["Model ID"]) != model or s(r["Step Id"]) not in steps:
            continue
        by_stream[(s(r["Step Id"]), s(r["Stream ID"]))].append(r)
    pair_occ = defaultdict(list)   # (typeA,typeB) -> [(step, gap, durA, durB)]
    triple_occ = defaultdict(list)
    per_stream_boundaries = Counter()
    per_step_kernels = Counter()
    for (step, stream), lst in by_stream.items():
        lst = sorted(lst, key=lambda r: (fnum(r["Start Time(us)"]) or 0.0,
                                         fnum(r["Duration(us)"]) or 0.0, r["__rowid"]))
        n = len(lst)
        per_step_kernels[(step, stream)] = n
        for i in range(n - 1):
            a, b = lst[i], lst[i + 1]
            ta, tb = s(a["Type"]), s(b["Type"])
            pa = fnum(a["Start Time(us)"]) or 0.0
            pb = fnum(b["Start Time(us)"]) or 0.0
            da = fnum(a["Duration(us)"]) or 0.0
            db = fnum(b["Duration(us)"]) or 0.0
            gap = pb - (pa + da)
            pair_occ[(ta, tb)].append((step, gap, da, db, a["__rowid"], b["__rowid"], stream))
            per_stream_boundaries[(step, stream)] += 1
        for i in range(n - 2):
            a, b, c = lst[i], lst[i + 1], lst[i + 2]
            triple_occ[(s(a["Type"]), s(b["Type"]), s(c["Type"]))].append(
                (step, fnum(a["Duration(us)"]) or 0.0, fnum(b["Duration(us)"]) or 0.0,
                 fnum(c["Duration(us)"]) or 0.0, a["__rowid"], b["__rowid"], c["__rowid"]))
    return by_stream, pair_occ, triple_occ, per_stream_boundaries, per_step_kernels


def pair_table(pair_occ, steps, top=None, min_cnt=1):
    n_steps = len(steps)
    out = []
    for (ta, tb), occ in pair_occ.items():
        durs_a = sum(o[2] for o in occ)
        durs_b = sum(o[3] for o in occ)
        gaps = [o[1] for o in occ]
        rec = {
            "pair": [ta, tb],
            "cnt_total": len(occ),
            "cnt_per_step": round(len(occ) / n_steps, 2),
            "upper_per_step_ms": round((durs_a + durs_b) / n_steps / 1000.0, 4),
            "durA_per_step_ms": round(durs_a / n_steps / 1000.0, 4),
            "durB_per_step_ms": round(durs_b / n_steps / 1000.0, 4),
            "gap_us": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in gap_stats(gaps).items()},
            "streams": dict(Counter(o[6] for o in occ).most_common(4)),
            "steps_covered": len(set(o[0] for o in occ)),
        }
        if len(occ) >= min_cnt:
            out.append(rec)
    out.sort(key=lambda r: -r["upper_per_step_ms"])
    return out[:top] if top else out


def triple_table(triple_occ, steps, top=None):
    n_steps = len(steps)
    out = []
    for (ta, tb, tc), occ in triple_occ.items():
        da = sum(o[1] for o in occ)
        db = sum(o[2] for o in occ)
        dc = sum(o[3] for o in occ)
        out.append({
            "triple": [ta, tb, tc],
            "cnt_total": len(occ),
            "cnt_per_step": round(len(occ) / n_steps, 2),
            "upper_per_step_ms": round((da + db + dc) / n_steps / 1000.0, 4),
            "steps_covered": len(set(o[0] for o in occ)),
        })
    out.sort(key=lambda r: -r["upper_per_step_ms"])
    return out[:top] if top else out


# ---------------------------------------------------------------------------
# 3) allReduce 邻接明细
# ---------------------------------------------------------------------------
def comm_detail(rows, model, steps, comm_name_of):
    per_step = {}
    totals = {"ops": 0, "sum_us": 0.0}
    adj_pairs = []   # (step, gap, durA, durB)
    for step in steps:
        ops = []
        for r in rows:
            if s(r["Model ID"]) != model or s(r["Step Id"]) != step:
                continue
            if "hcom_" not in s(r["Type"]):
                continue
            ops.append(r)
        ops.sort(key=lambda r: fnum(r["Start Time(us)"]) or 0.0)
        seq = []
        for i, r in enumerate(ops):
            st = fnum(r["Start Time(us)"]) or 0.0
            dur = fnum(r["Duration(us)"]) or 0.0
            nxt = fnum(ops[i + 1]["Start Time(us)"]) if i + 1 < len(ops) else None
            gap = (nxt - (st + dur)) if nxt is not None else None
            seq.append({
                "comm_name": comm_name_of.get(r["__rowid"], s(r["Name"])),
                "type": s(r["Type"]), "dur_us": round(dur, 3),
                "start_us": round(st, 3),
                "gap_to_next_us": (round(gap, 3) if gap is not None else None),
            })
            totals["ops"] += 1
            totals["sum_us"] += dur
            if i + 1 < len(ops):
                adj_pairs.append((step, gap, dur, fnum(ops[i + 1]["Duration(us)"]) or 0.0))
        per_step[step] = {
            "n_ops": len(ops),
            "sum_ms": round(sum(o["dur_us"] for o in seq) / 1000.0, 4),
            "span_ms": (round((max((fnum(r["Start Time(us)"]) or 0.0) + (fnum(r["Duration(us)"]) or 0.0) for r in ops)
                               - min(fnum(r["Start Time(us)"]) or 0.0 for r in ops)) / 1000.0, 4) if ops else None),
            "gaps_us": [o["gap_to_next_us"] for o in seq if o["gap_to_next_us"] is not None],
            "seq": seq,
        }
    n_steps = len(steps)
    all_gaps = [o["gap_to_next_us"] for st in per_step.values() for o in st["seq"] if o["gap_to_next_us"] is not None]
    near = [g for g in all_gaps if g is not None and g <= 10.0]
    adj_near = [p for p in adj_pairs if p[1] is not None and p[1] <= 10.0]
    merge_upper_us = sum(min(p[2], p[3]) for p in adj_near)
    # 层内成对合并（图级重排，不要求邻接）：按序号两两配对 (0,1),(2,3)... 取 min(durA,durB)
    layer_pair_upper_us = 0.0
    layer_pairs_per_step = 0
    # 最优配对（上界）：升序排列后 (v2,v3),(v4,v5)... 配对，Σmin 取偶数位（0-based 奇号），
    # 等价于“除最小一个外，两两合并”的最大可能回收量
    optim_pair_upper_us = 0.0
    type_counter = Counter()
    for step in steps:
        ops = [r for r in rows if s(r["Model ID"]) == model and s(r["Step Id"]) == step
               and "hcom_" in s(r["Type"])]
        ops.sort(key=lambda r: fnum(r["Start Time(us)"]) or 0.0)
        for r in ops:
            type_counter[s(r["Type"])] += 1
        for i in range(0, len(ops) - 1, 2):
            da = fnum(ops[i]["Duration(us)"]) or 0.0
            db = fnum(ops[i + 1]["Duration(us)"]) or 0.0
            layer_pair_upper_us += min(da, db)
            layer_pairs_per_step += 1
        asc = sorted((fnum(r["Duration(us)"]) or 0.0) for r in ops)
        optim_pair_upper_us += sum(asc[1::2][:len(asc) // 2])
    layer_pairs_per_step = layer_pairs_per_step / n_steps
    return {
        "model": model,
        "ops_per_step": round(totals["ops"] / n_steps, 2),
        "comm_types_total": dict(type_counter),
        "sum_ms_per_step": round(totals["sum_us"] / n_steps / 1000.0, 4),
        "gap_us_stats": gap_stats(all_gaps),
        "gaps_le_10us": len(near), "gaps_total": len(all_gaps),
        "adjacent_pairs_le_10us_per_step": round(len(adj_near) / n_steps, 2),
        "adjacent_merge_upper_ms_per_step": round(merge_upper_us / n_steps / 1000.0, 4),
        "layerwise_coalesce_rule": "同层两个集合通信成对合并：合并后延迟~max(la,lb)，上界=Σ min(la,lb)",
        "layerwise_coalesce_pairs_per_step": round(layer_pairs_per_step, 2),
        "layerwise_coalesce_upper_ms_per_step": round(layer_pair_upper_us / n_steps / 1000.0, 4),
        "layerwise_coalesce_maxpairing_upper_ms_per_step": round(optim_pair_upper_us / n_steps / 1000.0, 4),
        "merge_upper_ms_per_step": round(optim_pair_upper_us / n_steps / 1000.0, 4),
        "per_step_compact": {
            k: {"n_ops": v["n_ops"], "sum_ms": v["sum_ms"], "span_ms": v["span_ms"],
                "gap_p50_us": median_or_none(v["gaps_us"]),
                "gap_min_us": min(v["gaps_us"]) if v["gaps_us"] else None,
                "gap_max_us": max(v["gaps_us"]) if v["gaps_us"] else None}
            for k, v in per_step.items()
        },
        "example_step_seq": per_step[steps[-1]]["seq"][:20],
        "example_step_seq_tail": per_step[steps[-1]]["seq"][-6:],
    }


# ---------------------------------------------------------------------------
# 4) 候选聚合
# ---------------------------------------------------------------------------
QUANT_CORE_TYPES = {"DynamicQuant", "RmsNormDynamicQuant", "QuantBatchMatmulV3",
                    "RmsNorm", "RmsNormCast", "Cast"}
QUANT_WIDE_EXTRA = {"MatMulV2", "MatMulV3", "GroupedMatmul", "GroupedMatmulSwigluQuantV2",
                    "Add", "Mul", "Neg"}
STRICT_QUANT_TYPES = {"DynamicQuant", "RmsNormDynamicQuant", "QuantBatchMatmulV3",
                      "GroupedMatmul", "GroupedMatmulSwigluQuantV2", "RmsNormCast"}
MOE_CHAIN_TYPES = {"MoeInitRoutingV3", "GroupedMatmulSwigluQuantV2", "GroupedMatmul", "Cast",
                   "MaskedFill", "Add", "Mul"}


def is_quant_core(pair):
    a, b = pair
    if not (a in QUANT_CORE_TYPES and b in QUANT_CORE_TYPES):
        return False
    return (a in STRICT_QUANT_TYPES) or (b in STRICT_QUANT_TYPES)


def is_quant_wide(pair):
    a, b = pair
    fam = QUANT_CORE_TYPES | QUANT_WIDE_EXTRA
    if not (a in fam and b in fam):
        return False
    if is_quant_core(pair):
        return True
    return (a in STRICT_QUANT_TYPES) or (b in STRICT_QUANT_TYPES)


def is_moe_chain(pair):
    a, b = pair
    if not (a in MOE_CHAIN_TYPES and b in MOE_CHAIN_TYPES):
        return False
    return ("GroupedMatmul" in a or "GroupedMatmul" in b or "MoeInitRouting" in a or "MoeInitRouting" in b)


def candidate_union(pair_occ, include_pred, steps, model="48"):
    """返回候选类的实例并集：上界=并集 Duration 之和/步。"""
    inst = {}
    pairs_used = Counter()
    for pair, occ in pair_occ.items():
        if not include_pred(pair):
            continue
        pairs_used[pair] += len(occ)
        for (step, gap, da, db, ra, rb, stream) in occ:
            inst[ra] = (step, da, stream, pair[0])
            inst[rb] = (step, db, stream, pair[1])
    n_steps = len(steps)
    upper_us = sum(v[1] for v in inst.values())
    by_type = Counter()
    for v in inst.values():
        by_type[v[3]] += v[1]
    return {
        "instances": len(inst),
        "kernels_per_step": round(len(inst) / n_steps, 1),
        "upper_ms_per_step": round(upper_us / n_steps / 1000.0, 4),
        "conservative_ms_per_step": [round(upper_us / n_steps / 1000.0 * CONSERVATIVE_LO, 3),
                                     round(upper_us / n_steps / 1000.0 * CONSERVATIVE_HI, 3)],
        "by_type_ms_per_step": {k: round(v / n_steps / 1000.0, 4) for k, v in by_type.most_common()},
        "pairs_used": {" -> ".join(k): v for k, v in pairs_used.most_common(12)},
    }


def greedy_disjoint(pair_occ, steps, top_k=12):
    """按 上界 排序的贪心互斥对集合（每个实例只进一个候选），用于交叉校验。"""
    n_steps = len(steps)
    ranked = sorted(((k, v) for k, v in pair_occ.items() if not any("hcom_" in t for t in k)),
                    key=lambda kv: -sum(o[2] + o[3] for o in kv[1]))
    used, chosen = set(), []
    for pair, occ in ranked:
        occ2 = [o for o in occ if o[4] not in used and o[5] not in used]
        if not occ2:
            continue
        for o in occ2:
            used.add(o[4])
            used.add(o[5])
        chosen.append({
            "pair": list(pair),
            "instances": 2 * len(occ2),
            "upper_ms_per_step": round(sum(o[2] + o[3] for o in occ2) / n_steps / 1000.0, 4),
        })
        if len(chosen) >= top_k:
            break
    total = sum(c["upper_ms_per_step"] for c in chosen)
    return {"rule": "排除集合通信对（单独按成对合并 min-rule 计），其余按上界降序贪心、实例互斥",
            "chosen": chosen, "union_upper_ms_per_step": round(total, 4),
            "conservative_ms_per_step": [round(total * CONSERVATIVE_LO, 3), round(total * CONSERVATIVE_HI, 3)]}


# ---------------------------------------------------------------------------
# 5) 源码开关检索
# ---------------------------------------------------------------------------
SCAN_PATTERNS = [
    ("quant_chain", r"fuse_norm_quant|AddRMSNormQuantFusionPass|norm_quant_fusion_pass|RmsNormDynamicQuant|rms_norm_dynamic_quant|DequantSwigluQuant|dequant_swiglu_quant|GroupedMatmulSwigluQuant|npu_quant_matmul|npu_dynamic_quant"),
    ("static_kernel", r"enable_static_kernel|static_kernel_compile|StaticKernel"),
    ("superkernel_or_combo", r"superkernel|super_kernel|SuperKernel|combo_kernels|combo_kernel|ComboKernel"),
    ("pass_registry", r"COMPILATION_PASS_KEY|graph_fusion_pass_manager|fuse_muls_add|MulsAddFusionPass|QKNormRopeFusionPass|allgather_chunk_noop|noop_elimination|fuse_qknorm_rope"),
    ("allreduce_or_mc2", r"enable_reduce_sample|reduce_sample|MatmulAllReduce|matmul_all_reduce|enable_fused_mc2|fused_mc2|MegaMoe|mc2_comm_alg|enable_mc2_hierarchy_comm"),
    ("hc_chain", r"HcPre|HcPost|hc_pre|hc_post"),
    ("sampling", r"rejection_sampler|RejectionSampler|TopKTopP|topk_topp|AscendSampler"),
    ("graph_paths", r"enable_npugraph_ex|torchair|npugraph_ex|static_kernel_compile|_process_kwargs_options"),
]
SCAN_ROOTS = [
    ("vllm_ascend", "/vllm-workspace/vllm-ascend/vllm_ascend"),
    ("vllm", "/vllm-workspace/vllm/vllm"),
]


def scan_sources(max_hits_per_pattern=22, timeout=120):
    results = {"roots": {}, "patterns": {}}
    for name, root in SCAN_ROOTS:
        info = {"path": root, "exists": os.path.isdir(root)}
        if info["exists"]:
            try:
                head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                                      capture_output=True, text=True, timeout=30)
                info["git_head"] = head.stdout.strip() or None
                when = subprocess.run(["git", "-C", root, "log", "-1", "--format=%cs"],
                                      capture_output=True, text=True, timeout=30)
                info["git_head_date"] = when.stdout.strip() or None
            except Exception as exc:  # pragma: no cover
                info["git_error"] = str(exc)
        results["roots"][name] = info

    for label, pat in SCAN_PATTERNS:
        entry = {"regex": pat, "hits": [], "total_lines": 0, "files": {}}
        for name, root in SCAN_ROOTS:
            if not os.path.isdir(root):
                continue
            try:
                proc = subprocess.run(
                    ["grep", "-rIn", "--include=*.py", "-E", pat, root],
                    capture_output=True, text=True, timeout=timeout)
            except Exception as exc:  # pragma: no cover
                entry["error"] = str(exc)
                continue
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            entry["total_lines"] += len(lines)
            for ln in lines:
                parts = ln.split(":", 2)
                if len(parts) < 3:
                    continue
                f, lineno, text = parts[0], parts[1], parts[2].strip()
                short = f.replace("/vllm-workspace/vllm-ascend/", "VA/").replace("/vllm-workspace/vllm/", "V/")
                entry["files"][short] = entry["files"].get(short, 0) + 1
                if len(entry["hits"]) < max_hits_per_pattern:
                    entry["hits"].append({"file": short, "line": int(lineno), "text": text[:150]})
        entry["files_top"] = dict(sorted(entry["files"].items(), key=lambda kv: -kv[1])[:12])
        results["patterns"][label] = entry
    return results


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-source-scan", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    dd = dedup(rows)
    comm_name_of = dd.pop("comm_name_of")
    kept = [r for r in rows if s(r["Stream ID"]) not in ("", "N/A")]

    by_stream, pair_occ, triple_occ, per_stream_boundaries, per_step_kernels = adjacency(kept, TARGET_MODEL, STEPS)

    # 每步/每步-流统计
    per_step = {}
    for step in STEPS:
        ks = [r for r in kept if s(r["Model ID"]) == TARGET_MODEL and s(r["Step Id"]) == step]
        durs = [fnum(r["Duration(us)"]) or 0.0 for r in ks]
        starts = [fnum(r["Start Time(us)"]) or 0.0 for r in ks]
        ends = [a + b for a, b in zip(starts, durs)]
        bystream = {}
        for st_name in sorted(set(s(r["Stream ID"]) for r in ks)):
            sks = [r for r in ks if s(r["Stream ID"]) == st_name]
            sd = [fnum(r["Duration(us)"]) or 0.0 for r in sks]
            ss = [fnum(r["Start Time(us)"]) or 0.0 for r in sks]
            se = [a + b for a, b in zip(ss, sd)]
            bystream[st_name] = {"kernels": len(sks), "sum_ms": round(sum(sd) / 1000.0, 4),
                                 "span_ms": round((max(se) - min(ss)) / 1000.0, 4)}
        per_step[step] = {
            "kernels": len(ks),
            "sum_ms": round(sum(durs) / 1000.0, 4),
            "span_ms": round((max(ends) - min(starts)) / 1000.0, 4),
            "streams": dict(Counter(s(r["Stream ID"]) for r in ks).most_common()),
            "per_stream": bystream,
            "boundaries": sum(v for (st, _), v in per_stream_boundaries.items() if st == step),
        }

    target_durs = sorted(fnum(r["Duration(us)"]) or 0.0
                         for r in kept if s(r["Model ID"]) == TARGET_MODEL and s(r["Step Id"]) in STEPS)
    dist = {
        "n": len(target_durs),
        "mean_us": round(sum(target_durs) / len(target_durs), 3),
        "p50_us": pct(target_durs, 0.5), "p90_us": pct(target_durs, 0.9), "p99_us": pct(target_durs, 0.99),
        "max_us": target_durs[-1],
        "n_lt_3us": sum(1 for d in target_durs if d < 3.0),
        "n_lt_5us": sum(1 for d in target_durs if d < 5.0),
        "sum_lt_5us_ms_per_step": round(sum(d for d in target_durs if d < 5.0) / len(STEPS) / 1000.0, 4),
    }

    pairs_top = pair_table(pair_occ, STEPS, top=40)
    triples_top = triple_table(triple_occ, STEPS, top=20)
    comm48 = comm_detail(kept, TARGET_MODEL, STEPS, comm_name_of)

    # 候选类（数据驱动 + v54 清单对齐）
    def is_hc(pair):
        return pair[0] in ("HcPre", "HcPost") or pair[1] in ("HcPre", "HcPost")

    cand_quant_core = candidate_union(pair_occ, is_quant_core, STEPS)
    cand_quant_wide = candidate_union(pair_occ, is_quant_wide, STEPS)
    cand_moe = candidate_union(pair_occ, is_moe_chain, STEPS)
    cand_hc = candidate_union(pair_occ, is_hc, STEPS)
    cand_comm = {
        "upper_ms_per_step_min_bound": comm48["layerwise_coalesce_maxpairing_upper_ms_per_step"],
        "observed_order_upper_ms_per_step": comm48["layerwise_coalesce_upper_ms_per_step"],
        "conservative_ms_per_step": [round(comm48["layerwise_coalesce_maxpairing_upper_ms_per_step"] * CONSERVATIVE_LO, 3),
                                     round(comm48["layerwise_coalesce_maxpairing_upper_ms_per_step"] * CONSERVATIVE_HI, 3)],
        "rule": comm48["layerwise_coalesce_rule"],
        "adjacency_finding": ("同流相邻 allReduce 对中 gap<=10us 的个数为 0（gap 中位 %.1fus）"
                             "⇒ 该收益必须靠图级调度成对合并，而非内核邻接融合"
                             % (comm48["gap_us_stats"]["p50"] or -1)),
        "v54_reference": "81x3.32ms -> ~1.7ms (省 ~1.6ms)",
    }

    inst_union = {}
    for pair, occ in pair_occ.items():
        if not (is_quant_core(pair) or is_hc(pair) or is_moe_chain(pair)):
            continue
        for (step, gap, da, db, ra, rb, stream) in occ:
            inst_union[ra] = (da, step)
            inst_union[rb] = (db, step)
    whole_upper_ms = sum(v[0] for v in inst_union.values()) / len(STEPS) / 1000.0
    whole = {
        "classes": ["quant_chain_core(RmsNorm/DynamicQuant/Cast/QBMM)", "HcPre/HcPost 链",
                    "MoE 段(MoeInitRoutingV3/GMMSwigluQuantV2/GMM/Cast)", "allReduce 成对合并(另计,不计入并集)"],
        "kernels_in_union_per_step": round(len(inst_union) / len(STEPS), 1),
        "upper_ms_per_step": round(whole_upper_ms, 4),
        "conservative_ms_per_step": [round(whole_upper_ms * CONSERVATIVE_LO, 3), round(whole_upper_ms * CONSERVATIVE_HI, 3)],
        "share_of_target_union": round(whole_upper_ms / (sum(v["sum_ms"] for v in per_step.values()) / len(STEPS)), 4),
        "with_comm_conservative_ms_per_step": [
            round((whole_upper_ms + comm48["layerwise_coalesce_maxpairing_upper_ms_per_step"]) * CONSERVATIVE_LO, 3),
            round((whole_upper_ms + comm48["layerwise_coalesce_maxpairing_upper_ms_per_step"]) * CONSERVATIVE_HI, 3)],
    }
    greedy = greedy_disjoint(pair_occ, STEPS, top_k=12)
    greedy["conservative_ms_per_step_with_comm"] = [
        round((greedy["union_upper_ms_per_step"] + comm48["merge_upper_ms_per_step"]) * CONSERVATIVE_LO, 3),
        round((greedy["union_upper_ms_per_step"] + comm48["merge_upper_ms_per_step"]) * CONSERVATIVE_HI, 3)]

    # 采样类算子在 target / draft / engram / eager 内是否存在
    # 注意：大小写敏感，避免 "Tensor" 中 "sorT" 命中小写 "sort" 这类误报
    sampling_pat = re.compile(r"Sample|Sampler|TopK|Topk|Sort|Sorted|Gumbel|Reject|Exponential|Random|RepairSampling")
    sampling = {}
    for model in (TARGET_MODEL, "47", "49", "4294967295"):
        hits = Counter()
        for r in kept:
            if s(r["Model ID"]) != model or s(r["Step Id"]) not in STEPS:
                continue
            nm, ty = s(r["Name"]), s(r["Type"])
            if sampling_pat.search(nm) or sampling_pat.search(ty):
                hits[(ty, nm)] += 1
        sampling[model] = {
            "matches": [{"type": k[0], "name": k[1], "rows_total_10_steps": v} for k, v in hits.most_common(12)],
            "rows_total": sum(hits.values()),
        }
    sampling["note"] = ("target 图内无采样/拒绝采样 kernel ⇒ v54 第 4 项(采样段)不在此 CSV 证据范围；"
                        "MoeGatingTopKHash 属路由 topk，不是采样器")

    # 图外 eager 段（≈ v46/v54 的 pre_target 窗口）类型聚合
    eager_types = {}
    for r in kept:
        if s(r["Model ID"]) != "4294967295" or s(r["Step Id"]) not in STEPS:
            continue
        ty = s(r["Type"])
        e = eager_types.setdefault(ty, {"n": 0, "dur_us": 0.0, "example": s(r["Name"])})
        e["n"] += 1
        e["dur_us"] += fnum(r["Duration(us)"]) or 0.0
    eager_top = {
        ty: {"cnt_per_step": round(e["n"] / len(STEPS), 2),
             "ms_per_step": round(e["dur_us"] / len(STEPS) / 1000.0, 4),
             "example": e["example"][:90]}
        for ty, e in sorted(eager_types.items(), key=lambda kv: -kv[1]["dur_us"])[:10]
    }

    # 按类型聚合（含样例形状，便于厂商对齐 kernel 语义）
    type_agg = {}
    for r in kept:
        if s(r["Model ID"]) != TARGET_MODEL or s(r["Step Id"]) not in STEPS:
            continue
        ty = s(r["Type"])
        e = type_agg.setdefault(ty, {"n": 0, "dur_us": 0.0, "example_name": s(r["Name"]),
                                     "in_shapes": s(r["Input Shapes"])[:110],
                                     "in_dtypes": s(r["Input Data Types"])[:70],
                                     "out_shapes": s(r["Output Shapes"])[:80]})
        e["n"] += 1
        e["dur_us"] += fnum(r["Duration(us)"]) or 0.0
    n_steps = len(STEPS)
    type_agg_out = {}
    for ty, e in sorted(type_agg.items(), key=lambda kv: -kv[1]["dur_us"])[:24]:
        type_agg_out[ty] = {
            "cnt_per_step": round(e["n"] / n_steps, 2),
            "ms_per_step": round(e["dur_us"] / n_steps / 1000.0, 4),
            "mean_us": round(e["dur_us"] / e["n"], 3),
            "example_name": e["example_name"][:120],
            "in_shapes": e["in_shapes"], "in_dtypes": e["in_dtypes"], "out_shapes": e["out_shapes"],
        }
    type_inventory = {
        ty: [round(e["n"] / n_steps, 2), round(e["dur_us"] / n_steps / 1000.0, 4)]
        for ty, e in sorted(type_agg.items(), key=lambda kv: -kv[1]["dur_us"])
    }

    # stream 140 上的边界 gap（发射开销的直接证据）
    gaps140 = [ (o[1], o[2], o[3]) for (ta, tb), occ in pair_occ.items() for o in occ if o[6] == "140" ]
    all140 = [g for g, _, _ in gaps140]
    small140 = [g for g, da, db in gaps140 if min(da, db) < 5.0]
    big140 = [g for g, da, db in gaps140 if min(da, db) >= 5.0]
    gap_evidence = {
        "stream140_boundaries_total": len(all140),
        "gap_us_all_p10_p50_p90": [pct(all140, 0.1), median_or_none(all140), pct(all140, 0.9)],
        "gap_us_argmin_dur_lt5us": [pct(small140, 0.1), median_or_none(small140), pct(small140, 0.9)],
        "gap_us_argmin_dur_ge5us": [pct(big140, 0.1), median_or_none(big140), pct(big140, 0.9)],
        "small_side_n": len(small140), "big_side_n": len(big140),
    }

    # 其他 model 摘要（47=draft / 49=engram / 4294967295=图外 eager=pre_target 窗口）
    others = {}
    for model in ("47", "49", "4294967295"):
        stats = {}
        for step in STEPS:
            ks = [r for r in kept if s(r["Model ID"]) == model and s(r["Step Id"]) == step]
            if not ks:
                continue
            durs = [fnum(r["Duration(us)"]) or 0.0 for r in ks]
            starts = [fnum(r["Start Time(us)"]) or 0.0 for r in ks]
            ends = [a + b for a, b in zip(starts, durs)]
            stats[step] = {"kernels": len(ks), "sum_ms": round(sum(durs) / 1000.0, 4),
                           "span_ms": round((max(ends) - min(starts)) / 1000.0, 4),
                           "streams": dict(Counter(s(r["Stream ID"]) for r in ks).most_common())}
        if stats:
            ks_all = list(stats.values())
            others[model] = {
                "per_step_example": stats[STEPS[-1]],
                "kernels_per_step_mean": round(sum(v["kernels"] for v in ks_all) / len(ks_all), 1),
                "sum_ms_per_step_mean": round(sum(v["sum_ms"] for v in ks_all) / len(ks_all), 4),
                "span_ms_per_step_mean": round(sum(v["span_ms"] for v in ks_all) / len(ks_all), 4),
            }

    out = {
        "meta": {
            "csv": args.csv,
            "csv_size_bytes": os.path.getsize(args.csv),
            "csv_sha256": sha256_file(args.csv),
            "target_model_id": TARGET_MODEL,
            "steps": STEPS,
            "generated_by": "fusion_scan_rank0.py",
            "scope": "offline-only; no NPU job launched; frozen files untouched",
        },
        "dedup": dd,
        "per_step": per_step,
        "kernel_duration_distribution": dist,
        "pairs_top": pairs_top,
        "triples_top": triples_top,
        "allreduce_comm_detail": comm48,
        "candidates": {"quant_chain_core": cand_quant_core, "quant_chain_wide": cand_quant_wide,
                       "moe_chain": cand_moe, "hc_chain": cand_hc, "comm_merge": cand_comm},
        "whole_graph": whole,
        "greedy_disjoint_crosscheck": greedy,
        "sampling_check": sampling,
        "type_aggregate_top24": type_agg_out,
        "type_inventory_all": type_inventory,
        "eager_pre_target_types_top10": eager_top,
        "stream_gap_evidence": gap_evidence,
        "other_models_summary": others,
    }
    if not args.skip_source_scan:
        out["source_scan"] = scan_sources()

    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(json.dumps({"written": args.out, "bytes": len(text)}, ensure_ascii=False))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
