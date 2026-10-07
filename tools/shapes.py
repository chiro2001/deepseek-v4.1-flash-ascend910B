#!/usr/bin/env python3
import csv, sys
from collections import defaultdict
D=sys.argv[1]
want=("HcPre","RmsNorm","DynamicQuantV2","QuantMatmulWeightNz","InplacePartialRotaryMul",
      "SparseFlashMla","MatMulCommon_MatMulV2","HcPost","MoeGatingTopKHash","MoeInitRoutingV3",
      "GroupedMatmulSwigluQuant","GroupedMatmulWeightNz","MoeTokenUnpermute","MatMulV3Common",
      "ScatterNdUpdateSk","QuantLightningIndexerV2","InplaceCopy_Cast","_pool_kernel","AivKernel")
agg=defaultdict(lambda: defaultdict(int))
with open(D,newline="") as fh:
    for r in csv.DictReader(fh):
        nm=(r.get("Name") or ""); sid=(r.get("Stream ID") or "").strip()
        if sid!="109": continue
        for w in want:
            if w in nm:
                key=((r.get("Accelerator Core") or "").strip(),(r.get("Block Num") or ""),
                     (r.get("Input Shapes") or "")[:110],(r.get("Output Shapes") or "")[:90],
                     (r.get("Input Data Types") or "")[:60])
                agg[w][key]+=1
                break
for w in want:
    if w not in agg: continue
    print("###",w)
    for k,c in sorted(agg[w].items(), key=lambda x:-x[1])[:3]:
        print("   n=%-6d core=%-15s blk=%-4s"% (c,k[0],k[1]))
        print("      in : %s"%k[2])
        print("      out: %s"%k[3])
        print("      dt : %s"%k[4])
