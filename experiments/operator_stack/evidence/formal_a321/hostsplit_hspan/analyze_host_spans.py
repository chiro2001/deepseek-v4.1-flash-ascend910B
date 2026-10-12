"""Summarise the additive host-span probe receipts for one A/A job."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--job-root', type=Path, required=True)
    a = p.parse_args()
    job = a.job_root
    reqs = json.loads((job / 'requests.json').read_text())
    rows = [r for r in reqs if r.get('tag', '').startswith('pair-')]
    print(f"parsed {len(rows)} counted requests")
    for row in rows[:2]:
        tag = row['tag']
        counters_path = job / f'{tag}_counters.json'
        if not counters_path.exists():
            print(f"{tag}: no counters file")
            continue
        receipts = json.loads(counters_path.read_text())
        per_rank = []
        for receipt in receipts:
            spans = receipt.get('host_spans') or {}
            entry = {'rank': receipt['rank'], 'spans': spans.get('spans', {}),
                     'notes': spans.get('notes'), 'window_wall_s': spans.get('window_wall_s')}
            per_rank.append(entry)
        rank0 = per_rank[0]
        rounds = row.get('metrics', {}).get('rounds') if row.get('metrics') else None
        if rounds is None:
            rounds = row.get('rounds')
        print(f"\n== {tag}: rounds={rounds}")
        names = sorted({k for e in per_rank for k in e['spans']})
        for name in names:
            totals = [e['spans'][name]['total_s'] for e in per_rank if name in e['spans']]
            counts = [e['spans'][name]['count'] for e in per_rank if name in e['spans']]
            if not totals:
                continue
            med = sorted(totals)[len(totals) // 2]
            print(f"  {name:26s} ranks={len(totals)} median_total={med*1000:9.3f}ms "
                  f"per_round={med*1000/(rounds or 1):7.3f}ms calls/rank={counts[0]}")
        if rank0.get('window_wall_s'):
            print(f"  window_wall rank0 = {rank0['window_wall_s']*1000:.3f}ms")
        if rank0.get('notes'):
            print(f"  notes: {rank0['notes']}")


if __name__ == '__main__':
    main()
