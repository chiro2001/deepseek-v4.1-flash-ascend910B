"""Cross-job (same harness/prompts/seeds) token and logprob comparison."""
import argparse
import json
import math
from pathlib import Path


def load(job):
    return json.loads((Path(job) / 'requests.json').read_text())


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    a = p.parse_args()
    left, right = load(a.left), load(a.right)
    by_tag = {r['tag']: r for r in left}
    pairs = 0
    token_equal = 0
    max_delta = 0.0
    for row in right:
        ref = by_tag.get(row['tag'])
        if ref is None:
            print(f"missing left tag {row['tag']}")
            continue
        pairs += 1
        same = ref['token_ids'] == row['token_ids']
        token_equal += same
        delta = None
        if ref.get('logprobs') and row.get('logprobs'):
            deltas = []
            for lstep, rstep in zip(ref['logprobs'], row['logprobs']):
                if set(lstep) != set(rstep):
                    deltas = None
                    break
                for key in lstep:
                    value = abs(lstep[key] - rstep[key])
                    if not math.isfinite(value):
                        deltas = None
                        break
                    deltas.append(value)
            if deltas is not None:
                delta = max(deltas)
                max_delta = max(max_delta, delta)
        print(f"{row['tag']:16s} tokens_equal={same} max_logprob_delta={delta} "
              f"ms_per_step={row.get('metrics', {}).get('ms_per_step') if row.get('metrics') else row.get('ms_per_step')}")
    print(f"\npairs={pairs} token_equal={token_equal}/{pairs} global_max_logprob_delta={max_delta}")


if __name__ == '__main__':
    main()
