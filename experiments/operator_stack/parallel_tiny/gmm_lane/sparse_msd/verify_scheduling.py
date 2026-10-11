"""Independent CPU tile-assignment oracle; validates AIC sparse/AIV dense agreement."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def planner(counts):
    rows, result = 0, []
    for expert, count in enumerate(counts):
        assert 0 <= count <= 6 and rows + count <= 6
        if count:
            result.append((expert, rows, count))
            rows += count
    assert len(result) <= 6
    return result


def native_core_tasks(counts, core, cores, base_m, single_n):
    tasks, pre_count, offset_m = [], 0, 0
    block_n = (5120 + single_n - 1) // single_n
    for expert in range(48):
        count = counts[expert]
        if count <= 0: continue
        m = 2 * count
        block_m = (m + base_m - 1) // base_m
        cur_count = pre_count + block_n * block_m
        block = core if core >= pre_count else core + cores
        while block < cur_count:
            tasks.append((expert, (block-pre_count)//block_n, (block-pre_count)%block_n,
                          m, offset_m, expert*2304*5120//2, expert*5120*8))
            block += cores
        pre_count = cur_count % cores
        offset_m += m
    return tasks


def sparse_core_tasks(descriptors, core, cores, base_m, single_n):
    tasks, pre_count = [], 0
    block_n = (5120 + single_n - 1) // single_n
    for expert, row_begin, count in descriptors:
        m, offset_m = 2*count, 2*row_begin
        block_m = (m + base_m - 1) // base_m
        cur_count = pre_count + block_n*block_m
        block = core if core >= pre_count else core+cores
        while block < cur_count:
            tasks.append((expert, (block-pre_count)//block_n, (block-pre_count)%block_n,
                          m, offset_m, expert*2304*5120//2, expert*5120*8))
            block += cores
        pre_count = cur_count % cores
    return tasks


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = random.Random(20261011)
    counts_cases = []
    for total in range(7):
        for _ in range(100):
            counts = [0]*48
            for row in range(total): counts[rng.randrange(48)] += 1
            counts_cases.append(counts)
    core_checks, task_checks = 0, 0
    for counts in counts_cases:
        descriptors = planner(counts)
        for base_m in [8,16,32,64,128]:
            for single_n in [128,256,512,1024]:
                for core in range(24):
                    native = native_core_tasks(counts,core,24,base_m,single_n)
                    sparse = sparse_core_tasks(descriptors,core,24,base_m,single_n)
                    assert native == sparse, (counts,core,base_m,single_n)
                    core_checks += 1
                    task_checks += len(native)
    result={"passed":True,"CPU_only":True,"NPU_launches":0,"seed":20261011,
            "route_distributions":len(counts_cases),"core_geometry_comparisons":core_checks,
            "exact_tile_pointer_event_order_comparisons":task_checks,
            "AIC_sparse_matches_AIV_native_dense_task_order":True,
            "totalExpertNum":48,"maximum_activeSlotCount":6,"capacity":6,
            "K":2304,"N":5120,"Cube_cores":24,"original_FP16_math_executed":False,
            "scope":"CPU group/core/tile and pointer ordering; native NPU math remains untested"}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
