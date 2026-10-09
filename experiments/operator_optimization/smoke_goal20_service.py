"""Verify the owned tiny model endpoint before exercising a completion."""
import argparse
import datetime
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:18971')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with urllib.request.urlopen(args.base_url + '/v1/models', timeout=10) as response:
        models = json.load(response)
    expected = 'dsv41-tiny-prof-20261009'
    assert [row['id'] for row in models['data']] == [expected], models
    with urllib.request.urlopen(args.base_url + '/health', timeout=10) as response:
        health = response.status
    assert health == 200, health
    body = {'model': expected, 'prompt': [100 + i % 97 for i in range(32)],
            'temperature': 0, 'max_tokens': 16, 'ignore_eos': True}
    request = urllib.request.Request(args.base_url + '/v1/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    assert result['model'] == expected and result['usage']['prompt_tokens'] == 32, result
    assert result['usage']['completion_tokens'] == 16 and result['choices'][0]['finish_reason'] == 'length', result
    receipt = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'model_id': expected, 'health': health, 'usage': result['usage'],
               'finish_reason': result['choices'][0]['finish_reason'], 'tiny_dummy': True,
               'note': 'Endpoint ownership and completion smoke; no cross-process throughput claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print('SERVICE_SMOKE', json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
