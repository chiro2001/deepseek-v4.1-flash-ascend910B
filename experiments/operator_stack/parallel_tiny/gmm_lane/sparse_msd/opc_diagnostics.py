"""Process-local OPC logger forwarding; no SDK edits or NPU launches.

OPC itself loads libruntime and performs read-only visible-device enumeration.
The caller must preserve empty ASCEND_RT_VISIBLE_DEVICES/ASCEND_VISIBLE_DEVICES.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/usr/local/Ascend/cann-9.1.0/python/site-packages/asc_opc_tool")
from asc_op_compile_base.common.utils import log


def output(message, *args, **kwargs):
    try:
        message = message % args if args else message
    except Exception:
        message = str(message) + " " + repr(args)
    print(message, flush=True)


for name in ["debug", "info", "warning", "warn", "error", "exception"]:
    if hasattr(log, name): setattr(log, name, output)
import opc
sys.argv = ["asc_opc"] + sys.argv[1:]
opc.main()
