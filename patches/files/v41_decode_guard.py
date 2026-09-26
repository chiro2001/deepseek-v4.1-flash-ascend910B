"""[CED-DECODE-GUARD] decode 角色的**请求边界**护栏。

背景（2026-09-27 00:01 事故）
------------------------------------------------------------------------
CED 分离部署里，decode 实例只应该消费由 prefill 转发过来、带
``kv_transfer_params`` 的请求：它消费 P 算好的 KV，自己只做 ≤128 token 的
bounded replay。任何"直连 D 的普通生成请求"都会让 D 自己去 prefill：

* D 的固定 128-token replay 调度/注意力路径不支持这种请求；
* 旧行为是 worker 里 ``raise RuntimeError("CED decoder replay exceeded
  128 tokens")``。worker 异常不是"请求级错误"，而是让 EngineCore 直接退出
  （``EngineDeadError``）——**整个 D 实例死掉**，要重新加载 8 个 die 的权重
  才能恢复（实测 20 分钟量级）。

触发条件很宽：事故当时 18991 绑在 ``0.0.0.0``、没有 api-key，任何直连它、
prompt > 128 token 的请求都能打死实例（一条探针请求即命中）。

本模块把这件事提前到 **HTTP 层**：decode 角色上，没有
``kv_transfer_params`` 的生成请求直接 400，**永远进不了引擎**，也就不可能
触发 worker 里那条致命断言。

用法
------------------------------------------------------------------------
vLLM 自带扩展点（``vllm/entrypoints/openai/api_server.py`` 里
``for middleware in args.middleware``）：

    vllm serve ... --middleware v41_decode_guard.decode_guard

``scripts/serve_v2.sh`` 在 ``V41_CED_ROLE=decode`` 时自动加上这一项；
``scripts/serve_a2.sh`` 负责把本文件挂到容器里的
``/opt/dsv41/guards/v41_decode_guard.py``。模块自己再查一次
``V41_CED_ROLE``，所以即使被误挂到 P 上也只空转（两道保险）。

设计取舍
------------------------------------------------------------------------
* **只拦生成端点**（``/v1/chat/completions`` / ``/v1/completions`` /
  ``/v1/responses``）；``/health``、``/v1/models``、``/metrics`` 一律放行，
  否则就绪探针与监控会被自己挡掉。
* **JSON 解析失败一律放行**：这不是 fail-closed 的场合 —— 解析不了说明请求
  本身有问题，交给下游按正常流程报 422/400，护栏不替它下结论。
* ``kv_transfer_params`` 为 ``None`` / 空 dict 视同缺失：协议上 P 一定会带上
  ``do_remote_prefill`` / ``remote_block_ids`` 等字段（见
  ``examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py``）。
* 命中一次就写一行 ``[V41-DECODE-GUARD]`` 日志，**判据是可观测痕迹**，
  不是"env 传进去了"。
"""

from __future__ import annotations

import json
import os
import sys

from fastapi.responses import JSONResponse

__all__ = ["decode_guard", "guard_enabled", "guard_stats", "GENERATION_SUFFIXES"]

#: 需要护栏的生成端点（后缀匹配，兼容 /v1/... 前缀变化）。
GENERATION_SUFFIXES = ("/chat/completions", "/completions", "/responses")

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("", "0", "false", "no", "off")

_STATS = {"checked": 0, "rejected": 0, "passed_kv_params": 0, "passed_other": 0}


def _role() -> str:
    return os.environ.get("V41_CED_ROLE", "").strip().lower()


def guard_enabled() -> bool:
    """是否启用护栏：角色必须是 decode，且 ``V41_DECODE_API_GUARD`` 未显式关掉。"""
    if _role() != "decode":
        return False
    raw = os.environ.get("V41_DECODE_API_GUARD", "1").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return True  # 认不出的取值：保守地保持护栏开启


def guard_stats() -> dict:
    return dict(_STATS)


def _log(msg: str) -> None:
    # API server 的 stdout/stderr 都会进 serve.log；显式 flush，避免被缓冲吞掉。
    print(f"[V41-DECODE-GUARD] {msg}", file=sys.stderr, flush=True)


if guard_enabled():
    _log(
        "middleware loaded: 无 kv_transfer_params 的生成请求 → 400"
        "（不进引擎，参考 2026-09-27 decoder 被打挂的事故）"
    )


async def decode_guard(request, call_next):
    """Starlette/FastAPI 的 http 中间件（vLLM ``--middleware`` 入口）。"""
    if not guard_enabled():
        return await call_next(request)

    path = request.url.path
    if request.method != "POST" or not path.endswith(GENERATION_SUFFIXES):
        _STATS["passed_other"] += 1
        return await call_next(request)

    try:
        body = await request.body()
        payload = json.loads(body) if body else {}
    except Exception:  # noqa: BLE001 - 解析不了就不下结论，交给下游报错
        _STATS["passed_other"] += 1
        return await call_next(request)

    params = payload.get("kv_transfer_params") if isinstance(payload, dict) else None
    _STATS["checked"] += 1
    if params:
        _STATS["passed_kv_params"] += 1
        return await call_next(request)

    _STATS["rejected"] += 1
    _log(
        f"rejected {path}：没有 kv_transfer_params（本实例是 decode 半边，"
        f"只接受 prefill 转发的请求）；累计拦截 {_STATS['rejected']} 次"
    )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": (
                    "This instance is the decode half of a prefill/decode "
                    "disaggregated deployment and only accepts requests forwarded "
                    "by the prefill side (kv_transfer_params is required). "
                    "Send plain requests to the load-balance proxy instead, "
                    "e.g. http://127.0.0.1:18992/v1/chat/completions."
                ),
                "type": "invalid_request_error",
                "param": None,
                "code": "ced_decode_role_requires_kv_transfer_params",
            }
        },
    )
