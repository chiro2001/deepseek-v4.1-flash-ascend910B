"""Env-gated import overlay for the DSpark draft-graph patch trio.

The frozen draft payload (``patches/files/draft/{dspark_proposer,llm_base_proposer,dsa_v1}.py``)
is published as whole-file replacements that the A2-era serve script bind-mounted over
``/vllm-workspace/vllm-ascend/vllm_ascend/{spec_decode/...,attention/...}``. Our long-running
A3 formal container cannot take new bind mounts, so this module installs a meta-path finder
that resolves exactly those three module names from ``draft_graph_files/`` instead of the
image copy. Everything else keeps importing from the image, byte-identically.

Activation is env-gated so eager runs stay byte-identical to the stock image:
the finder answers only while ``DSPARK_DRAFT_USE_CUDAGRAPH`` or ``DRAFT_GRAPH`` is ``1``,
which the frozen launch contract sets exactly when ``--draft-graph 1`` is requested.
A missing overlay file while the gate is active is a hard error (fail-closed, never a
silent fallback to the unpatched eager module).
"""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FILES_DIR = HERE / 'draft_graph_files'

OVERLAY = {
    'vllm_ascend.attention.dsa_v1': FILES_DIR / 'dsa_v1.py',
    'vllm_ascend.spec_decode.dspark_proposer': FILES_DIR / 'dspark_proposer.py',
    'vllm_ascend.spec_decode.llm_base_proposer': FILES_DIR / 'llm_base_proposer.py',
}


def overlay_active():
    return (os.environ.get('DSPARK_DRAFT_USE_CUDAGRAPH', '0') == '1'
            or os.environ.get('DRAFT_GRAPH', '0') == '1')


class DraftGraphOverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in OVERLAY or not overlay_active():
            return None
        source = OVERLAY[fullname]
        if not source.is_file():
            raise ImportError('draft-graph overlay payload is missing: ' + str(source))
        spec = importlib.util.spec_from_file_location(fullname, source)
        if spec is None:
            raise ImportError('draft-graph overlay cannot build a spec for ' + fullname)
        return spec


_FINDER = DraftGraphOverlayFinder()


def install_import_hook():
    if any(isinstance(finder, DraftGraphOverlayFinder) for finder in sys.meta_path):
        return
    # The finder must win over any cached stock module, so drop nothing here and
    # simply place it first. Already-imported stock modules are left alone; the
    # gate is only consulted at first import, which for these three modules
    # happens inside the engine/worker processes after the launch env is set.
    sys.meta_path.insert(0, _FINDER)


def overlay_provenance():
    """Small receipt helper for the counters/worker to record which files serve."""
    result = {}
    for name, source in OVERLAY.items():
        entry = {'path': str(source), 'present': source.is_file()}
        if source.is_file():
            import hashlib
            entry['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            entry['bytes'] = source.stat().st_size
        result[name] = entry
    result['active'] = overlay_active()
    return result
