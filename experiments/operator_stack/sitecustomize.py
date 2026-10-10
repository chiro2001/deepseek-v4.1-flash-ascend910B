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
