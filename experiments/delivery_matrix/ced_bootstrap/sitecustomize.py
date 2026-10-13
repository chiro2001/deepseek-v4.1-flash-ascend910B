"""Role-scoped CED imports followed by the unchanged best BI/draft bootstrap."""
import os
from pathlib import Path
import runpy

try:
    if os.environ.get("DELIVERY_DCP_OVERLAY") or os.environ.get("V41_DCP") == "1":
        raise RuntimeError("CED-PD and DCP8 are separate delivery forms")
    from delivery_matrix.ced_overlay import install
    install(os.environ["DELIVERY_CED_OVERLAY"], os.environ["V41_CED_ROLE"])
    if os.environ.get("DELIVERY_CED_COMPRESSOR"):
        from delivery_matrix.ced_compressor_overlay import install as install_compressor
        install_compressor(os.environ["DELIVERY_CED_COMPRESSOR"], os.environ["V41_CED_ROLE"])
    if os.environ.get("DELIVERY_SHARED_ENGRAM"):
        from delivery_matrix.shared_engram_overlay import install as install_shared
        install_shared(os.environ["DELIVERY_SHARED_ENGRAM"], os.environ["V41_CED_ROLE"])
    elif os.environ.get("V41_ENGRAM_SHARED_HOST_MMAP") == "1":
        raise RuntimeError("A shared mmap request must not silently use the private Engram loader")
    runpy.run_path(str(Path(os.environ["STACK_SRC_ROOT"]) / "stack/sitecustomize.py"),
                   run_name="_delivery_ced_best_sitecustomize")
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(2)
