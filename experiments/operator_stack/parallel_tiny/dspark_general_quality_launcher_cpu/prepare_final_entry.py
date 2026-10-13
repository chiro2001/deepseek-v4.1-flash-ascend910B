"""Standalone one-command CPU entry, with delivery-pinned package loading."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(HERE))
    from cpu_entry import main as dispatch
    delivery = json.loads((HERE / 'delivery.json').read_text())
    dispatch(['--general-package', '/work/dspark_general_quality_api_cpu_20261011_v1',
              '--general-manifest-sha256', '5537b3ed8685e2c73c971f1eb73910ffa67be4dd499944b53ff9cb83c803ea9a',
              '--launcher-package', str(HERE), '--launcher-manifest-sha256', delivery['frozen_manifest_sha256'],
              '--action', 'prepare_final', '--', *sys.argv[1:]])


if __name__ == '__main__':
    main()
