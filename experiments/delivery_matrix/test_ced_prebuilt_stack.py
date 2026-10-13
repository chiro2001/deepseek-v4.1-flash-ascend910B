"""Native API prebuilt-stack regression; CPU ASGI fixture, no model imports."""
import ast
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from test_ced_proxy import transfer_contract

package = ModuleType("delivery_matrix")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules.setdefault("delivery_matrix", package)


class PrebuiltStackTests(unittest.IsolatedAsyncioTestCase):
    async def test_config_installs_decode_boundary_before_native_stack_is_sealed(self):
        source = Path(__file__).with_name("service.py").read_text()
        serve = next(node for node in ast.parse(source).body if isinstance(node, ast.AsyncFunctionDef) and node.name == "serve")
        assignment = next(node for node in ast.walk(serve) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Attribute) and target.attr == "middleware" for target in node.targets))
        config = NS()
        plan = {"http_middleware": ["delivery_matrix.ced_decode_boundary.ced_decode_guard"]}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "actual-service-config", "exec"), {"config": config, "plan": plan})
        self.assertEqual(config.middleware, plan["http_middleware"])
        app = FastAPI()
        calls = []
        @app.post("/v1/chat/completions")
        async def engine(payload: dict):
            calls.append(payload)
            return {"forwarded": True}
        # Model the official dotted-path loader and eager stack materialization.
        for path in config.middleware:
            module, function = path.rsplit(".", 1)
            app.middleware("http")(getattr(importlib.import_module(module), function))
        app.middleware_stack = app.build_middleware_stack()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://d") as client:
            with patch.dict(os.environ, {"V41_CED_ROLE": "decode", "V41_DECODE_API_GUARD": "1"}):
                bad = await client.post("/v1/chat/completions", json={"kv_transfer_params": {"do_remote_prefill": True}})
                good = await client.post("/v1/chat/completions", json={"kv_transfer_params": transfer_contract()})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(good.status_code, 200)
        self.assertEqual(len(calls), 1)

    async def test_old_post_build_middleware_injection_is_a_negative_control(self):
        app = FastAPI()
        app.middleware_stack = app.build_middleware_stack()
        async def guard(request, call_next): return await call_next(request)
        with self.assertRaisesRegex(RuntimeError, "after an application has started"):
            app.middleware("http")(guard)


if __name__ == "__main__":
    unittest.main()
