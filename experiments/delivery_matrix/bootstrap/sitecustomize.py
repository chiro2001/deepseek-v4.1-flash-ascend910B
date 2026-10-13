"""Install the deployment overlay, then execute the immutable best bootstrap."""
import os
from pathlib import Path
import runpy
import sys

try:
    if os.environ.get("DELIVERY_DCP_OVERLAY"):
        from delivery_matrix.dcp_overlay import install
        install(os.environ["DELIVERY_DCP_OVERLAY"])
        if not os.environ.get("DELIVERY_DCP_DRAFT"):
            raise RuntimeError("DCP8 with the best parallel draft requires its pinned first-pass fixes")
        from delivery_matrix.dcp_overlay import install_best_draft
        install_best_draft(os.environ["DELIVERY_DCP_DRAFT"])
    source = Path(os.environ["STACK_SRC_ROOT"]) / "stack/sitecustomize.py"
    runpy.run_path(str(source), run_name="_delivery_frozen_sitecustomize")
except BaseException:
    import traceback
    traceback.print_exc()
    # Python normally swallows sitecustomize errors. Fail before any model load.
    os._exit(2)
