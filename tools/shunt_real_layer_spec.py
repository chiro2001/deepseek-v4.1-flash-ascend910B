#!/usr/bin/env python3
"""从 trace 提取"一层"的真实算子规格：core/block/dur/输入输出 shape+dtype。
产出可直接用于实例化的 spec（JSON）。"""
import csv, sys, json, statistics
D=sys.argv[1]
rows=[]
with open(D+"/kernel_details.csv",newline="") as fh:
    for r in csv.DictReader(fh):
        if (r.get("Stream ID") or "").strip()!="109": continue
        try:
            st=float(r["Start Time(us)"]); du=float(r["Duration(us)"])
            aic=float(r.get("aicore_time(us)") or 0); aiv=float(r.get("aiv_time(us)") or 0)
        except: continue
        rows.append(dict(st=st,du=du,en=st+du,aic=aic,aiv=aiv,
                         name=(r.get("Name") or ""),core=(r.get("Accelerator Core") or "").strip(),
                         blk=(r.get("Block Num") or ""),mixblk=(r.get("Mix Block Num") or ""),
                         ins=(r.get("Input Shapes") or ""),outs=(r.get("Output Shapes") or ""),
                         indt=(r.get("Input Data Types") or ""),outdt=(r.get("Output Data Types") or "")))
rows.sort(key=lambda x:x["st"])
anch=sorted(x["st"] for x in rows if x["name"].startswith("HcPre")); PER=86
# 取一层：每层有 2 个 HcPre（注意 HcPre 每步 86 次 / 43 层）
S,E=anch[PER*5],anch[PER*5+2]
seq=[x for x in rows if x["st"]<E and x["en"]>S]
# 合并同一层内重复出现的算子（如多次 RmsNorm），保留首次
out=[]
for i,x in enumerate(seq):
    out.append(dict(i=i,name=x["name"].replace("aclnn","")[:52],core=x["core"],blk=x["blk"],
                    mixblk=x["mixblk"],dur_us=round(x["du"],2),
                    aic_us=round(x["aic"],2),aiv_us=round(x["aiv"],2),
                    ins=x["ins"],outs=x["outs"],indt=x["indt"],outdt=x["outdt"]))
print("一层算子数 %d，合计 %.1f µs"%(len(out),sum(x["dur_us"] for x in out)))
for x in out:
    print("%2d %-14s blk=%-3s %7.1fµs aic=%6.1f aiv=%6.1f  %s"%(x["i"],x["core"][:14],x["blk"],x["dur_us"],x["aic_us"],x["aiv_us"],x["name"][:44]))
    print("      in : %s"%x["ins"][:110])
    print("      out: %s | dt=%s"%(x["outs"][:70],x["indt"][:50]))
json.dump(out,open("/tmp/layer_spec.json","w"),indent=1,ensure_ascii=False)
print("\n-> /tmp/layer_spec.json")
