"""Only immutable diagnostic snapshots opt into the reduced cache contract."""
import os

if os.getenv('STACK_TINY_PROFILE') == '1':
    try:
        from tiny_cache_plan import install_import_hook
        install_import_hook()
    except Exception:
        # Python otherwise prints a sitecustomize error and keeps running.
        # A failed topology hook must stop the diagnostic before allocation.
        import traceback
        traceback.print_exc()
        os._exit(2)

# [DSV41 draft-graph overlay] When the launch contract requests the draft graph
# (DSPARK_DRAFT_USE_CUDAGRAPH=1 / DRAFT_GRAPH=1), resolve the three published
# patch modules from the snapshot payload instead of the image copy. With the
# gate inactive this installs a finder that immediately answers None, so eager
# runs keep importing the image modules byte-identically.
try:
    import draft_graph_overlay as _draft_graph_overlay
    _draft_graph_overlay.install_import_hook()
    if _draft_graph_overlay.overlay_active():
        for _name, _spec in _draft_graph_overlay.OVERLAY.items():
            if not _spec.is_file():
                raise RuntimeError('draft-graph overlay payload missing for ' + _name + ': ' + str(_spec))
except Exception:
    import traceback
    traceback.print_exc()
    if (os.environ.get('DSPARK_DRAFT_USE_CUDAGRAPH', '0') == '1'
            or os.environ.get('DRAFT_GRAPH', '0') == '1'):
        # Fail-closed: never silently fall back to the stock eager module when
        # the caller explicitly asked for the captured draft graph.
        os._exit(2)
