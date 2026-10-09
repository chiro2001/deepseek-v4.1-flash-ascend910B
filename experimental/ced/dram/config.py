"""Build a compact, reviewable P-side composite KV connector configuration."""

from __future__ import annotations

import argparse
import json


MODULE = "vllm_ascend.distributed.kv_transfer.ced_dram.connector"


def make_config(kv_port: int, cpu_bytes: int, pending_blocks: int) -> dict:
    if not 1 <= kv_port <= 65528:
        raise ValueError("KV base port must leave room for eight TP ranks")
    if cpu_bytes <= 0 or pending_blocks < 8:
        raise ValueError("DRAM bytes must be positive and pending blocks >=8")
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_port": kv_port,
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeHybridConnector",
                    "kv_role": "kv_producer",
                    "kv_port": kv_port,
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 1, "tp_size": 8},
                        "decode": {"dp_size": 1, "tp_size": 8},
                    },
                },
                {
                    "kv_connector": "CEDOffloadingConnector",
                    "kv_connector_module_path": MODULE,
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": cpu_bytes,
                        "blocks_per_chunk": {"full": 8, "swa": 1},
                        "spec_name": "CEDNPUOffloadingSpec",
                        "spec_module_path": MODULE,
                        "offload_prompt_only": True,
                        "ced_max_pending_store_blocks": pending_blocks,
                    },
                },
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--kv-port", required=True, type=int)
    parser.add_argument("--cpu-bytes", required=True, type=int)
    parser.add_argument("--pending-blocks", default=16384, type=int)
    args = parser.parse_args()
    print(json.dumps(make_config(args.kv_port, args.cpu_bytes, args.pending_blocks), separators=(",", ":")))


if __name__ == "__main__":
    main()
