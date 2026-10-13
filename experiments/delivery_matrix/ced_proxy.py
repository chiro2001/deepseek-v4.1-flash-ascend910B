"""Single-pair CED proxy with bound P/D identity and measured prefill sources.

The wire transfer contract matches the vendor proxy. Streaming chunks/usage
are forwarded unchanged; neither SSE events nor drafts are counted here.
"""
import argparse
import asyncio
import copy
from contextlib import asynccontextmanager
import hashlib
import json
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse


def pair_id(prefill, decode):
    return hashlib.sha256((prefill["service_instance_id"] + ":" + decode["service_instance_id"]).encode()).hexdigest()


def prefill_payload(payload):
    value = copy.deepcopy(payload)
    value.update(kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False,
                                    "remote_engine_id": None, "remote_block_ids": None,
                                    "remote_host": None, "remote_port": None},
                 stream=False, max_tokens=1, min_tokens=1)
    if "max_completion_tokens" in value:
        value["max_completion_tokens"] = 1
    value.pop("stream_options", None)
    return value


def validate_transfer(params):
    """Reject incomplete/mismatched CED wire contracts before D execution."""
    if not isinstance(params, dict):
        raise ValueError("Missing producer transfer dictionary")
    if (params.get("do_remote_prefill") is not True or params.get("do_remote_decode") is not False
            or params.get("ced_replay_tokens") != 128 or params.get("remote_ptp_size") != 8
            or params.get("ced_missing_swa_groups") != [7, 8, 9, 10, 11]):
        raise ValueError("Incompatible CED TP8/replay/cache-group contract")
    for key in ("remote_engine_id", "remote_request_id", "remote_host"):
        if not isinstance(params.get(key), str) or not params[key].strip():
            raise ValueError("Missing producer peer identity: " + key)
    for key in ("remote_port", "ced_prefix_tokens", "num_prompt_blocks"):
        if type(params.get(key)) is not int or params[key] <= 0:
            raise ValueError("Invalid producer numeric contract: " + key)
    if params["remote_port"] > 65528:
        raise ValueError("Producer port cannot cover eight TP ranks")
    groups = params.get("remote_block_ids")
    if (not isinstance(groups, list) or len(groups) != 12
            or any(not isinstance(group, list) or any(type(block) is not int or block < 0 for block in group)
                   for group in groups)
            or any(groups[index] for index in (7, 8, 9, 10, 11)) or not groups[0]):
        raise ValueError("Invalid producer block lists or uncomputed upper SWA data")


def create_app(prefill_url, decode_url, served_model):
    prefill_url, decode_url = prefill_url.rstrip("/"), decode_url.rstrip("/")
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=15), trust_env=False) as client:
            app.state.client = client
            yield
    app = FastAPI(lifespan=lifespan)

    async def backend_json(url, path, method="GET", payload=None):
        response = await app.state.client.request(method, url + path, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail={"backend": url, "path": path,
                                                        "status": response.status_code, "body": response.text[:3000]})
        return response.json()

    async def identity_pair():
        p, d = await asyncio.gather(backend_json(prefill_url, "/delivery/identity"),
                                   backend_json(decode_url, "/delivery/identity"))
        for value, role in ((p, "prefill"), (d, "decode")):
            if (value.get("served_model") != served_model or value.get("role") != role
                    or value.get("form") != "a3-ced-pd" or value.get("engine_initialized") is not True
                    or value.get("ASGI_startup_completed") is not True):
                raise HTTPException(status_code=503, detail="CED P/D identity is not the selected ready pair")
        manifest = p.get("ced_overlay_manifest_sha256")
        if (not isinstance(manifest, str) or len(manifest) != 64
                or manifest != d.get("ced_overlay_manifest_sha256")):
            raise HTTPException(status_code=503, detail="CED P/D must use the same pinned connector/replay source manifest")
        p_opp = (p.get("native_opp_receipt") or {}).get("artifact_fingerprint")
        d_opp = (d.get("native_opp_receipt") or {}).get("artifact_fingerprint")
        if not isinstance(p_opp, str) or len(p_opp) != 64 or p_opp != d_opp:
            raise HTTPException(status_code=503, detail="CED P/D must use the same actually verified A3 BI OPP")
        return p, d

    @app.get("/delivery/identity")
    async def identity():
        p, d = await identity_pair()
        result = copy.deepcopy(d)
        result.update(service_instance_id=pair_id(p, d),
                      runtime_fingerprint=hashlib.sha256((p["runtime_fingerprint"] + d["runtime_fingerprint"]).encode()).hexdigest(),
                      prefill_identity=p, decode_identity=d, topology="P8+D8; DCP=1")
        return result

    @app.post("/delivery/arm")
    async def arm(payload: dict):
        p, d = await identity_pair()
        # Every formal request belongs to exactly this one pair. No LB retries
        # or alternate backend can silently change the measurement window.
        d_ack = await backend_json(decode_url, "/delivery/arm", "POST", payload)
        p_ack = await backend_json(prefill_url, "/delivery/arm", "POST", payload)
        return {**d_ack, "service_instance_id": pair_id(p, d), "prefill_ack": p_ack}

    @app.get("/delivery/metrics")
    async def metrics():
        p_id, d_id = await identity_pair()
        p = await backend_json(prefill_url, "/delivery/metrics")
        d = await backend_json(decode_url, "/delivery/metrics")
        if (p.get("service_instance_id") != p_id["service_instance_id"]
                or d.get("service_instance_id") != d_id["service_instance_id"]):
            raise HTTPException(status_code=502, detail="CED worker metrics belong to a different P/D instance")
        p_ranks = {row["rank"]: row for row in p["ranks"]}
        if (len(p["ranks"]) != 8 or len(d["ranks"]) != 8 or set(p_ranks) != set(range(8))
                or {r["rank"] for r in d["ranks"]} != set(range(8))):
            raise HTTPException(status_code=502, detail="Incomplete actual CED TP8 worker measurements")
        for row in d["ranks"]:
            source = p_ranks[row["rank"]]
            p_tags = [entry.get("tag") for entry in source["requests"]]
            d_tags = [entry.get("tag") for entry in row["requests"]]
            if (p_tags != d_tags or any(not isinstance(tag, str) or not tag for tag in p_tags)
                    or len(set(p_tags)) != len(p_tags) or source.get("pending") or row.get("pending")
                    or source.get("errors") or row.get("errors")):
                raise HTTPException(status_code=502, detail="P and D must complete the same error-free request windows")
            row["cumulative"]["prefill_seconds"] += p_ranks[row["rank"]]["cumulative"]["prefill_seconds"]
            for entry in row["requests"]:
                entry["prefill_wall_scope"] = "P scheduled prefill + D bounded replay/tail CPU bookkeeping; transfer wait excluded"
        d.update(service_instance_id=pair_id(p_id, d_id), prefill_ranks=p["ranks"],
                 prefill_reporting="sum of independently observed P encoder and D replay/tail wall times")
        return d

    @app.get("/v1/models")
    async def models():
        p, d = await asyncio.gather(backend_json(prefill_url, "/v1/models"), backend_json(decode_url, "/v1/models"))
        if any([r.get("id") for r in value.get("data", [])] != [served_model] for value in (p, d)):
            raise HTTPException(status_code=503, detail="The two CED model catalogues do not match")
        return d

    @app.get("/health")
    async def health():
        await identity_pair()
        return {"ready": True, "served_model": served_model}

    @app.get("/metrics")
    async def prometheus():
        response = await app.state.client.get(decode_url + "/metrics")
        return Response(response.content, status_code=response.status_code,
                        media_type=response.headers.get("content-type", "text/plain"))

    @app.post("/tokenize")
    async def tokenize(request: Request):
        response = await app.state.client.post(prefill_url + "/tokenize", json=await request.json())
        return Response(response.content, status_code=response.status_code, media_type="application/json")

    async def generate(path, request):
        await identity_pair()
        payload = await request.json()
        if payload.get("model") != served_model:
            raise HTTPException(status_code=400, detail="A matching CED served-model is required")
        request_id = payload.get("request_id") or uuid.uuid4().hex
        headers = {"X-Request-Id": request_id}
        p = await app.state.client.post(prefill_url + path, json=prefill_payload(payload), headers=headers)
        if p.status_code != 200:
            return Response(p.content, status_code=p.status_code, media_type="application/json")
        params = p.json().get("kv_transfer_params")
        try:
            validate_transfer(params)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        decoder_payload = copy.deepcopy(payload)
        decoder_payload["kv_transfer_params"] = params
        if not payload.get("stream"):
            response = await app.state.client.post(decode_url + path, json=decoder_payload, headers=headers)
            return Response(response.content, status_code=response.status_code, media_type="application/json")
        decoder_request = app.state.client.build_request("POST", decode_url + path,
                                                        json=decoder_payload, headers=headers)
        response = await app.state.client.send(decoder_request, stream=True)
        if response.status_code != 200:
            content = await response.aread()
            await response.aclose()
            return Response(content, status_code=response.status_code, media_type="application/json")
        async def chunks():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
        return StreamingResponse(chunks(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await generate("/v1/chat/completions", request)

    @app.post("/v1/completions")
    async def completion(request: Request):
        return await generate("/v1/completions", request)
    return app


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--port", type=int, default=18992)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.prefill_url, args.decode_url, args.served_model), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
