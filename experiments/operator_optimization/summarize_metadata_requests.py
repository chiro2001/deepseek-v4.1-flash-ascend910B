"""Small coverage receipts; large actual routing/request payloads stay remote."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    requests = json.loads((args.root / 'requests.json').read_text())
    rows = []
    for request in requests:
        row = {key: request[key] for key in ['arm', 'tag', 'prompt_seed', 'route_shape', 'unique_experts',
                                             'metadata_coverage', 'blockmap_coverage'] if key in request}
        row['output_tokens'] = len(request['token_ids'])
        row['output_token_sha256'] = hashlib.sha256(json.dumps(request['token_ids'], separators=(',', ':')).encode()).hexdigest()
        if 'routes' in request:
            row['actual_routes_sha256'] = hashlib.sha256(json.dumps(request['routes'], separators=(',', ':')).encode()).hexdigest()
        rows.append(row)
    result = {'request_count': len(rows), 'requests': rows,
              'note': 'Routing arrays/step timings remain in the original remote requests.json; audit comparisons assert exact equality'}
    (args.root / 'coverage_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'request_count': len(rows), 'last_request': rows[-1]}, indent=2))


if __name__ == '__main__': main()
