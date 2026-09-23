#!/usr/bin/env python3
"""假 vLLM（只给自测用）：/health + /tokenize + /v1/chat/completions。

模式（环境变量）：
  MODE=clean   正确回答（从 prompt 里"检索"针）
  MODE=garbled 回答里塞 U+FFFD 与 NUL，并复读
  NOTOKENIZE=1 /tokenize 返回 404（测探针的近似回退）

用法：python3 _fake_vllm.py <port-file>
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = os.environ.get("MODE", "clean")
NOTOK = os.environ.get("NOTOKENIZE") == "1"
NEEDLE = re.compile(r"(ZQ7K-3341|VX2M-8890|HT4P-5527|RB9N-6014)")
CODE = "ZQ7K-3341-VX2M-8890-HT4P-5527-RB9N-6014-PLM3-7712-CDF8-2205"
HOME = "/work/out/checksum.txt"
# ★ 桩必须**回答被问到的那一条**（按问题里的"运维备忘 X"选）：
#   第一版只会返回 blob 里**第一个**针 ⇒ grow 模式第 2 轮起必然答错，
#   于是"干净回答"那一臂被判 7 项失败 —— 是**桩的缺陷**，不是探针的缺陷。
BY_KEY = {"A": "ZQ7K-3341", "B": "VX2M-8890", "C": "HT4P-5527", "D": "RB9N-6014"}
ASK = re.compile(r"运维备忘\s*([ABCD])")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._json({"status": "ok"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n).decode())
        except Exception:
            req = {}
        if self.path.startswith("/tokenize"):
            if NOTOK:
                return self._json({"error": "not supported"}, 404)
            p = req.get("prompt") or ""
            return self._json({"count": len(p), "tokens": [0] * len(p)})
        if self.path.startswith("/v1/chat/completions"):
            msgs = req.get("messages") or []
            blob = "\n".join(m.get("content") or "" for m in msgs)
            # ★ 真实模型是"**看问题决定**要不要调工具"，而不是"有 tools 就一定调"。
            #   第一版按"有 tools 就走工具分支"，于是"带 tools 问一个普通问题"被桩答成了
            #   工具调用（text 为空）⇒ 探针的 big-stream 臂被**误判成 FAIL**。
            #   判据（桩侧）：问题里明确要求写文件/调用工具时才走工具分支。
            _last_user = ""
            for _m in reversed(msgs):
                if _m.get("role") == "user":
                    _last_user = _m.get("content") or ""
                    break
            _want_tool = bool(req.get("tools")) and (
                "write_file" in _last_user or "调用" in _last_user
                or "写入文件" in _last_user)
            if _want_tool:
                content = (CODE[:-1] + "\ufffd") if MODE == "garbled" else CODE
                msg = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "c1", "type": "function",
                                       "function": {"name": "write_file",
                                                    "arguments": json.dumps(
                                                        {"path": HOME, "content": content},
                                                        ensure_ascii=False)}}]}
            else:
                last_user = ""
                for m_ in reversed(msgs):
                    if m_.get("role") == "user":
                        last_user = m_.get("content") or ""
                        break
                # ★ 取**最后一次**出现：`needle` 模式的最后一条 user 消息里，
                #   前面是塞满针的正文、后面才是问题（第一版取首个匹配 ⇒ 永远答 A，
                #   于是 needle 的 B/C/D 全"失败" —— 又是**桩**的缺陷）。
                hits = ASK.findall(last_user)
                if hits:
                    ans = BY_KEY[hits[-1]]
                else:
                    m = NEEDLE.search(blob)
                    ans = m.group(1) if m else "(未找到)"
                if MODE == "garbled":
                    # 复读片段要够长：repeat_loop 的判据是"同一 40 字窗口 ≥3 次"
                    ans = ans[:3] + "\ufffd\x00" + ans[3:] + ("重复片段" * 60)
                msg = {"role": "assistant", "content": ans}
            if req.get("stream"):
                # ★ 支持 SSE（否则"流式解析器坏了"会被误判成"模型坏了"）
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                def emit(delta, finish=None):
                    obj = {"choices": [{"index": 0, "delta": delta,
                                        "finish_reason": finish}]}
                    self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n")
                                     .encode())
                    self.wfile.flush()
                # 工具调用：**分片**发（模拟真实增量拼接）
                if msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    emit({"role": "assistant", "tool_calls": [
                        {"index": 0, "id": tc["id"], "type": "function",
                         "function": {"name": tc["function"]["name"], "arguments": ""}}]})
                    args = tc["function"]["arguments"]
                    step = max(1, len(args) // 4)
                    for i in range(0, len(args), step):
                        emit({"tool_calls": [{"index": 0,
                              "function": {"arguments": args[i:i + step]}}]})
                else:
                    txt = msg.get("content") or ""
                    # 内容也分片（含多字节字符边界）
                    step = max(1, len(txt) // 3)
                    for i in range(0, len(txt), step):
                        emit({"content": txt[i:i + step]})
                emit({}, finish="stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            return self._json({"choices": [{"message": msg, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 8}})
        self._json({"error": "not found"}, 404)


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
open(sys.argv[1], "w").write(str(srv.server_address[1]))
srv.serve_forever()
