"""Standalone loader for separately delivered frozen CPU packages."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def bind(name, root, expected):
    manifest_path = root / 'frozen_files.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Pinned delivered package manifest differs: ' + name)
    manifest = json.loads(manifest_path.read_text())
    for relative, item in manifest['text_files'].items():
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Delivered package source differs: ' + relative)
    spec = importlib.util.spec_from_file_location(name, root / '__init__.py',
                                                 submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('general-package', 'launcher-package'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('general-manifest-sha256', 'launcher-manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--action', choices=('assets', 'export_candidate', 'prepare_snapshot', 'prepare_final', 'launch'), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if 'parallel_tiny' not in sys.modules:
        parent = types.ModuleType('parallel_tiny')
        parent.__path__ = []
        sys.modules['parallel_tiny'] = parent
    bind('parallel_tiny.dspark_general_quality_api_cpu', args.general_package, args.general_manifest_sha256)
    bind('parallel_tiny.dspark_general_quality_launcher_cpu', args.launcher_package, args.launcher_manifest_sha256)
    if args.arguments[:1] == ['--']:
        args.arguments = args.arguments[1:]
    module = importlib.import_module('parallel_tiny.dspark_general_quality_launcher_cpu.' + args.action)
    module.main(args.arguments)


if __name__ == '__main__':
    main()
