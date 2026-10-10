"""384/top6/48-local routing prefix mathematics and changing-input replay."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-chips', required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def route(x, ids, rank, mode):
        return torch_npu.npu_moe_init_routing_v2(x, ids, active_num=ids.numel(), expert_num=384,
            expert_tokens_num_type=mode, expert_tokens_num_flag=True,
            active_expert_range=[rank*48, (rank+1)*48], quant_mode=1)

    def verify(counts, prefix, ids, rank, tag):
        expected = torch.bincount(ids.cpu().reshape(-1).long(), minlength=384)[rank*48:(rank+1)*48]
        c, p = counts[2].cpu().long(), prefix[2].cpu().long()
        assert torch.equal(c, expected), (tag, rank, 'counts', c, expected)
        assert torch.equal(p, expected.cumsum(0)), (tag, rank, 'prefix', p, expected.cumsum(0))
        valid = int(expected.sum())
        # The inactive capacity tail is not written by either routing mode.
        assert torch.equal(counts[0][:valid].cpu(), prefix[0][:valid].cpu()), (tag, rank, 'quantized rows')
        assert torch.equal(counts[1].cpu(), prefix[1].cpu()), (tag, rank, 'expanded indices')
        assert torch.equal(counts[3][:valid].cpu(), prefix[3][:valid].cpu()), (tag, rank, 'dynamic scales')
        rows.append({'tag': tag, 'rank': rank, 'tokens': ids.shape[0], 'local_tokens': valid,
            'active_experts': int((expected > 0).sum()), 'mathematics_and_consumer_inputs_exact': True})

    for tokens in (1, 8, 17, 128):
        for case in ('spread', 'single-local', 'no-local'):
            for rank in range(8):
                torch.manual_seed(tokens*13 + rank)
                x = torch.randn(tokens, 5120, dtype=torch.bfloat16, device='npu')
                if case == 'spread':
                    ids = (torch.arange(tokens*6).reshape(tokens, 6)*37 + rank*11) % 384
                elif case == 'single-local':
                    ids = torch.tensor([rank*48, *[((rank+1)*48+i) % 384 for i in range(5)]]).repeat(tokens,1)
                else:
                    ids = (torch.arange(6) + ((rank+1) % 8)*48).repeat(tokens,1)
                ids = ids.to(device='npu', dtype=torch.int32)
                verify(route(x, ids, rank, 1), route(x, ids, rank, 0), ids, rank, case)
    rank = 3
    x = torch.randn(8, 5120, dtype=torch.bfloat16, device='npu')
    ids = (torch.arange(48).reshape(8,6) + 144).to(device='npu', dtype=torch.int32)
    for _ in range(3):
        route(x, ids, rank, 0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = route(x, ids, rank, 0)
    for iteration in range(6):
        new_ids = (torch.arange(48).reshape(8,6)*17 + iteration*51) % 384
        ids.copy_(new_ids.to(device='npu',dtype=torch.int32))
        x.copy_(torch.randn_like(x))
        graph.replay()
        verify(route(x, ids, rank, 1), captured, ids, rank, 'changing-input-replay-'+str(iteration))
    result = {'passed': True, 'cases': len(rows), 'independent_cases': 96, 'graph_replays': 6,
        'global_experts': 384, 'topk': 6, 'local_experts': 48, 'physical_chips': args.physical_chips,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows': rows, 'scope': 'routing only; no GMM arithmetic or E2E performance claim'}
    (out/'prefix_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FORMAL_PREFIX_PROBE_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
