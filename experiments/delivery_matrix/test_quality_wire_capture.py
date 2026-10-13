"""Byte transparency/metadata tests, including the exact pinned completion loop."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
import urllib.request

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_wire_cpu", HERE / "quality_wire_capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def request(stream=True):
    return urllib.request.Request("http://owned/v1/completions", data=json.dumps(
        {"model": "owned-model", "prompt": "unchanged prompt", "max_tokens": 512,
         "temperature": 0, "stream": stream}).encode())


def events(empty=False):
    first = {"id": "response-1", "choices": [{"index": 0, "text": "" if empty else "Answer: 13"}]}
    finish = {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}
    usage = {"choices": [], "usage": {"prompt_tokens": 2048, "completion_tokens": 349}}
    return b"\n".join(b"data: " + json.dumps(value).encode() for value in (first, finish, usage)) + b"\ndata: [DONE]\n"


class WireTests(unittest.TestCase):
    def test_stream_returns_identical_bytes_and_records_finish_usage_once(self):
        raw, records = events(), []
        wrapped = capture.WireCapture(lambda *a, **k: io.BytesIO(raw), records.append)(request())
        self.assertEqual(b"".join(wrapped), raw)
        wrapped.close()
        wrapped.close()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["consumed_wire_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(records[0]["finish_reasons"], [{"index": 0, "finish_reason": "stop"}])
        self.assertEqual(records[0]["usage"]["completion_tokens"], 349)
        self.assertTrue(records[0]["done_consumed"])

    def test_exact_original_completion_loop_is_unchanged_in_both_modes(self):
        source = (HERE.parents[1] / "tests/acc_eval.py").read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         "aad813de1b1af090c4186b0a443b7839339928f90b63e7c3a654d0e028b8e6e2")
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "complete")
        for serial in (True, False):
            for empty in (False, True):
                with self.subTest(serial=serial, empty=empty):
                    if serial:
                        raw = events(empty)
                    else:
                        raw = json.dumps({"choices": [{"text": "" if empty else "Answer: 13",
                                                       "finish_reason": "stop"}],
                                          "usage": {"completion_tokens": 349}}).encode()
                    records = []
                    scope = {"json": json, "SERIALIZE_PREFILL": serial, "EVAL_MODEL": "owned-model",
                             "PREFILL_LOCK": threading.Lock(),
                             "urllib": NS(request=NS(Request=urllib.request.Request,
                                 urlopen=capture.WireCapture(lambda *a, **k: io.BytesIO(raw), records.append)))}
                    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned-complete", "exec"), scope)
                    text = scope["complete"]("http://owned", "unchanged prompt", 512)
                    self.assertEqual(text, "" if empty else "Answer: 13")
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["max_tokens"], 512)
                    self.assertEqual(records[0]["usage"]["completion_tokens"], 349)
                    self.assertEqual(records[0]["finish_reasons"][0]["finish_reason"], "stop")

    def test_partial_stream_does_not_invent_completion_or_usage(self):
        records = []
        wrapped = capture.WireCapture(lambda *a, **k: io.BytesIO(events()), records.append)(request())
        next(iter(wrapped))
        wrapped.close()
        self.assertIsNone(records[0]["usage"])
        self.assertEqual(records[0]["finish_reasons"], [])
        self.assertFalse(records[0]["done_consumed"])

    def test_open_error_preserved_and_unrelated_request_not_observed(self):
        records = []
        def error(*a, **k): raise TimeoutError("same error")
        with self.assertRaisesRegex(TimeoutError, "same error"):
            capture.WireCapture(error, records.append)(request())
        self.assertEqual(records[0]["open_error_type"], "TimeoutError")
        marker = object()
        self.assertIs(capture.WireCapture(lambda *a, **k: marker, records.append)("http://owned/health"), marker)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
