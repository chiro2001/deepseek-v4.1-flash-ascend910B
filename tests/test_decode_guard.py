"""离线单测：CED decode 侧请求边界护栏（patches/files/v41_decode_guard.py）。

跑法：python3 -m pytest tests/test_decode_guard.py -q

不依赖 a3-21、不依赖模型：用 FastAPI TestClient 起一个"只有护栏 + 一个
回声端点"的最小 app，按 vLLM ``api_server.build_app`` 的同一手法注册中间件
（``app.middleware("http")(decode_guard)``）。

为什么值得单独测：这条护栏是**事故（2026-09-27 00:01 decoder 被打挂）的
第一道防线**，而它的失效方式有两种，都很贵：
  * 拦不住 → 引擎会被一条普通请求打死（要重载 20 分钟）；
  * 拦多了 → 正常链路（P→D 转发，带 kv_transfer_params）被 400，服务等于挂。
所以两个方向都要有正例与负例。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "patches" / "files"))

import v41_decode_guard as g  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("V41_CED_ROLE", raising=False)
    monkeypatch.delenv("V41_DECODE_API_GUARD", raising=False)


def _make_client(calls: list) -> TestClient:
    app = FastAPI()
    app.middleware("http")(g.decode_guard)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.body()  # ← 护栏读过 body，这里必须还能读
        try:
            calls.append(json.loads(body))
        except Exception:  # 非 JSON：记录原始字节，证明"下游确实收到了"
            calls.append(body)
        return {"ok": True}

    @app.get("/health")
    async def health():
        calls.append("health")
        return {"status": "ok"}

    return TestClient(app)


def _chat_body(with_kv: bool) -> dict:
    body = {
        "model": "deepseek-v41-ced-pd",
        "messages": [{"role": "user", "content": "x" * 400}],  # >128 token 量级
        "max_tokens": 4,
        "stream": True,
    }
    if with_kv:
        body["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "p-engine",
            "remote_host": "127.0.0.1",
            "remote_port": 19090,
        }
    return body


def test_decode_role_rejects_request_without_kv_params(monkeypatch):
    """事故形状：decode 角色 + 无 kv_transfer_params → 400，且**不进引擎**。"""
    monkeypatch.setenv("V41_CED_ROLE", "decode")
    calls: list = []
    r = _make_client(calls).post("/v1/chat/completions", json=_chat_body(False))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "ced_decode_role_requires_kv_transfer_params"
    assert calls == [], "护栏必须在下游（引擎）之前拦下"
    assert g.guard_stats()["rejected"] >= 1


def test_decode_role_passes_forwarded_request(monkeypatch):
    """正常链路：P→D 转发带 kv_transfer_params → 放行，且 body 逐字节还在。"""
    monkeypatch.setenv("V41_CED_ROLE", "decode")
    calls: list = []
    body = _chat_body(True)
    r = _make_client(calls).post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    assert calls == [body], "护栏读过 body 后，下游必须还能读到同一份 body"


def test_decode_role_allows_health_and_gets(monkeypatch):
    monkeypatch.setenv("V41_CED_ROLE", "decode")
    calls: list = []
    r = _make_client(calls).get("/health")
    assert r.status_code == 200
    assert calls == ["health"], "就绪探针不能被护栏挡掉"


def test_decode_role_fails_open_on_non_json(monkeypatch):
    """解析不了就交给下游：护栏不替下游下结论（下游自己会报错）。"""
    monkeypatch.setenv("V41_CED_ROLE", "decode")
    calls: list = []
    r = _make_client(calls).post(
        "/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 200
    assert calls == [b"not json"], "非法 body 应原样透传给下游"


def test_prefill_role_is_untouched(monkeypatch):
    """同一份中间件挂在 P 上必须空转（P 本来就要收没有 kv_transfer_params 的请求）。"""
    monkeypatch.setenv("V41_CED_ROLE", "prefill")
    calls: list = []
    r = _make_client(calls).post("/v1/chat/completions", json=_chat_body(False))
    assert r.status_code == 200
    assert len(calls) == 1


def test_explicit_switch_off(monkeypatch):
    monkeypatch.setenv("V41_CED_ROLE", "decode")
    monkeypatch.setenv("V41_DECODE_API_GUARD", "0")
    calls: list = []
    r = _make_client(calls).post("/v1/chat/completions", json=_chat_body(False))
    assert r.status_code == 200
    assert len(calls) == 1


def test_empty_kv_params_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv("V41_CED_ROLE", "decode")
    calls: list = []
    body = _chat_body(False)
    body["kv_transfer_params"] = {}
    r = _make_client(calls).post("/v1/chat/completions", json=body)
    assert r.status_code == 400
    assert calls == []


def test_guard_enabled_matrix():
    import os

    os.environ["V41_CED_ROLE"] = "decode"
    os.environ.pop("V41_DECODE_API_GUARD", None)
    assert g.guard_enabled() is True
    os.environ["V41_DECODE_API_GUARD"] = "off"
    assert g.guard_enabled() is False
    os.environ["V41_CED_ROLE"] = "prefill"
    os.environ["V41_DECODE_API_GUARD"] = "1"
    assert g.guard_enabled() is False
    os.environ.pop("V41_CED_ROLE", None)
    os.environ.pop("V41_DECODE_API_GUARD", None)
