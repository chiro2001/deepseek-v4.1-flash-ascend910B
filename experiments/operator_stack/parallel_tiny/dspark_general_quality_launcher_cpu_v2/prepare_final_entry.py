"""Standalone one-command CPU entry, with delivery-pinned package loading."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(HERE))
    from cpu_entry import main as dispatch
    delivery = json.loads((HERE / 'delivery.json').read_text())
    # [async arm v2] This delivery pins its own v2 sibling packages; the
    # general-package path is resolved by prepare_final from the deployment
    # root, so only its frozen manifest SHA is pinned here.
    dispatch(['--general-package', '/work/dspark_general_quality_api_cpu_20261012_v2',
              '--general-manifest-sha256', delivery['general_package_manifest_sha256'],
              '--launcher-package', str(HERE), '--launcher-manifest-sha256', delivery['frozen_manifest_sha256'],
              '--action', 'prepare_final', '--', *sys.argv[1:]])


if __name__ == '__main__':
    main()
