"""Per-round alignment of the host-span call series with the counter round windows."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def union(intervals):
    total = 0
    end = None
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += right - left
        elif right > end:
            total += right - end
        end = max(right, end if end is not None else right)
    return total


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--job-root', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--rank', type=int, default=0)
    a = p.parse_args()
    receipts = json.loads((a.job_root / f'{a.tag}_counters.json').read_text())
    receipt = [r for r in receipts if r['rank'] == a.rank][0]
    spans = receipt['host_spans']
    calls = spans['calls']
    request = next(iter(receipt['requests'].values()))
    rounds = request['rounds']
    print(f"tag={a.tag} rank={a.rank} decode_rounds={len(rounds)} calls={len(calls)}")
    names = sorted({c[0] for c in calls})

    def intersect(start, end):
        return [c for c in calls if c[1] < end and c[2] > start]

    top = defaultdict(list)
    for name in names:
        top[name] = [c for c in calls if c[0] == name]

    print("\nper-round breakdown (ms):")
    header = f"{'round':>5} {'window':>8} {'exec':>8} {'prep':>6} {'csmeta':>7} {'samp':>8} {'bksync':>8} {'prop':>6} {'propose':>8} {'gap':>6}"
    print(header)
    agg = defaultdict(float)
    for index, row in enumerate(rounds):
        start, end = row['execute_start_ns'], row['sample_end_ns']
        window = (end - start) / 1e6

        def span_of(name):
            total = 0
            for c in calls:
                if c[0] != name:
                    continue
                left, right = max(c[1], start), min(c[2], end)
                if right > left:
                    total += right - left
            return total / 1e6

        exec_s = span_of('execute_model')
        prep = span_of('prepare_inputs')
        cs = span_of('calc_spec_metadata')
        samp = span_of('sample_tokens')
        bk = span_of('bookkeeping_sync')
        prop = span_of('propose_draft')
        dp = span_of('draft_propose')
        covered = union([(max(c[1], start), min(c[2], end)) for c in calls
                         if c[0] in ('execute_model', 'sample_tokens') and c[1] < end and c[2] > start]) / 1e6
        gap = window - covered
        print(f"{index:>5} {window:8.2f} {exec_s:8.2f} {prep:6.2f} {cs:7.2f} {samp:8.2f} {bk:8.2f} {prop:6.2f} {dp:8.2f} {gap:6.2f}")
        for key, value in (('window', window), ('exec', exec_s), ('prep', prep), ('csmeta', cs),
                           ('samp', samp), ('bksync', bk), ('prop', prop), ('propose', dp), ('gap', gap)):
            agg[key] += value
    count = len(rounds)
    print("\nmeans per round (ms):")
    for key in ('window', 'exec', 'prep', 'csmeta', 'samp', 'bksync', 'prop', 'propose', 'gap'):
        print(f"  {key:8s} {agg[key]/count:8.3f}")
    print("\nexec = prep + csmeta + [device target + launch wait]")
    print(f"  implied exec non-prep part: {(agg['exec']-agg['prep']-agg['csmeta'])/count:8.3f}")
    print(f"  implied sample non-nested part (sample - bksync - prop - propose): {(agg['samp']-agg['bksync']-agg['prop']-agg['propose'])/count:8.3f}")


if __name__ == '__main__':
    main()
