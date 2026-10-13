"""Strict route affinity for native DP; no proxy or independent TP replicas."""
import json
import os
import re

from .topology import Topology


# The native frontend is explicitly one process. Keys identify armed DP groups,
# not a global measurement window; different DP replicas may run concurrently.
WINDOWS = {}


class DPRouteMiddleware:
    def __init__(self, app):
        self.app = app
        self.topology = Topology.named(os.environ["DP_MATRIX_LAYOUT"])
        self.next_rank = 0

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") not in ("/v1/chat/completions", "/v1/completions")):
            return await self.app(scope, receive, send)
        values = {}
        for key, value in scope.get("headers", []):
            values.setdefault(key.lower(), []).append(value)
        ranks = values.get(b"x-data-parallel-rank", [])
        tags = values.get(b"x-dp-matrix-tag", [])
        error = None
        if len(ranks) > 1 or len(tags) > 1:
            error = "Duplicate DP routing/measurement headers are forbidden"
        elif ranks:
            raw = ranks[0]
            if not re.fullmatch(rb"[0-9]+", raw) or len(raw) > 2 or int(raw) >= self.topology.dp:
                error = "Invalid native data-parallel rank"
            else:
                dp_rank = int(raw)
        else:
            if WINDOWS or tags:
                error = "Measured generation requires an explicit native DP rank"
            else:
                dp_rank = self.next_rank
                self.next_rank = (self.next_rank + 1) % self.topology.dp
        if error is None:
            window = WINDOWS.get(dp_rank)
            if WINDOWS and window is None:
                error = "Unarmed DP generation is forbidden during a measured window"
            try:
                tag = tags[0].decode("utf-8") if tags else None
            except UnicodeDecodeError:
                tag, error = None, "Invalid measurement tag encoding"
            if (window is not None and tag != window["tag"]) or (window is None and tag is not None):
                error = "Generation tag differs from the armed DP replica"
            if window is not None and window.get("generation_started"):
                error = "Only one actual request is admitted to this DP counter window"
        if error is not None:
            data = json.dumps({"error": {"message": error, "type": "dp_route_contract"}}).encode()
            await send({"type": "http.response.start", "status": 400,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": data})
            return
        if window is not None:
            window["generation_started"] = True
            window["generation_complete"] = False
        routed = dict(scope)
        routed["headers"] = [(key, value) for key, value in scope.get("headers", [])
                             if key.lower() != b"x-data-parallel-rank"] + [
                                 (b"x-data-parallel-rank", str(dp_rank).encode())]

        async def response(message):
            if message["type"] == "http.response.start":
                if window is not None:
                    window["generation_http_status"] = message["status"]
                message = {**message, "headers": list(message.get("headers", [])) + [
                    (b"x-dp-matrix-rank", str(dp_rank).encode())]}
            await send(message)
            if (window is not None and message["type"] == "http.response.body"
                    and not message.get("more_body", False)):
                window["generation_complete"] = True

        return await self.app(routed, receive, response)
