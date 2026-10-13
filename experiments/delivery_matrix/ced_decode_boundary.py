"""CED-only extension of the preserved missing-params HTTP guard."""
import json
import sys

from fastapi.responses import JSONResponse

from .ced_proxy import validate_transfer
from .decode_guard import GENERATION_SUFFIXES, decode_guard, guard_enabled


async def ced_decode_guard(request, call_next):
    if guard_enabled() and request.method == "POST" and request.url.path.endswith(GENERATION_SUFFIXES):
        try:
            payload = json.loads(await request.body())
        except (ValueError, UnicodeError):
            payload = None  # Normal framework validation still owns invalid JSON.
        params = payload.get("kv_transfer_params") if isinstance(payload, dict) else None
        if params:
            try:
                validate_transfer(params)
            except ValueError as exc:
                print("[CED-DECODE-CONTRACT] rejected before engine: " + str(exc), file=sys.stderr, flush=True)
                return JSONResponse(status_code=400, content={"error": {
                    "type": "invalid_request_error", "code": "ced_invalid_transfer_contract",
                    "message": str(exc), "param": "kv_transfer_params"}})
    return await decode_guard(request, call_next)
