"""CPU-only contract tests for pinned Dihuo prompt inputs and client ordering."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark
import frozen_dihuo_inputs as frozen


TASK_TMP = Path(os.environ.get("TMPDIR", Path.home() / "tmp/dsv41-operator-baseline-client"))
TASK_TMP.mkdir(parents=True, exist_ok=True)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class SyntheticArchive:
    """A tiny synthetic archive exercising the real validator without corpus text."""

    def __init__(self, root: Path):
        self.root = root
        self.requests = root / "requests"
        self.requests.mkdir(parents=True)
        self.prompts = []
        self.manifest = None
        self.manifest_path = root / "manifest.json"
        self._build()

    def _build(self):
        prompts = [
            f"[delivery-matrix-dihuo-{index + 1:02d}]\nSynthetic CPU-only prompt body {index:02d}."
            for index in range(8)
        ]
        prompt_hashes = [benchmark.sha256_text(text) for text in prompts]
        workload = {
            "tasks": ["dihuo"],
            "input_tokens_target": 2048,
            "dihuo_output_tokens": 256,
            "code_output_tokens": None,
            "prompt_sha256_by_task": {"dihuo": prompt_hashes},
            "code_unit_cases_sha256": {},
            "dihuo_corpus_sha256": "a" * 64,
            "dihuo_suffix_sha256": {"synthetic.txt": "b" * 64},
        }
        source = {
            "dihuo_source": {
                "corpus_sha256": "a" * 64,
                "suffix_sha256": {"synthetic.txt": "b" * 64},
            },
            "workload_manifest": workload,
            "workload_fingerprint": benchmark.canonical_json_sha256(workload),
        }
        artifacts = []
        self.prompts = prompts
        self.source = source
        for index, text in enumerate(prompts):
            name = f"dihuo_round00_request{index:02d}.json.gz"
            messages = [{"role": "user", "content": text}]
            metadata = {
                "sample_id": f"dihuo-{index + 1:02d}",
                "suffix_name": "synthetic.txt",
                "corpus_offset_chars": index,
                "corpus_slice_chars": len(text),
                "corpus_slice_stride_chars": 1,
                "tokenizer_count": 2048,
                "input_tokens_exact": True,
                "output_text": "metadata poison: old output",
                "quality_passed": True,
                "quality_score": 1.0,
                "ttft_s": 999.0,
                "request_wall_s": 999.0,
                "delivery_counter_delta": {"steps": 999},
            }
            request = {
                "task": "dihuo",
                "sample_id": f"dihuo-{index + 1:02d}",
                "round_index": 0,
                "request_index": index,
                "seed": index,
                "ignore_eos": True,
                "requested_max_tokens": 256,
                "prompt_text": text,
                "prompt_messages": messages,
                "prompt_sha256": benchmark.sha256_text(text),
                "prompt_messages_sha256": benchmark.canonical_json_sha256(messages),
                "source_metadata": metadata,
                "usage_tokens": {"prompt_tokens": 2048, "completion_tokens": 256, "total_tokens": 2304},
                "http_status": 200,
                "finish_reason": "length",
                "error": None,
                "sse_error_event": None,
                "clean": {"clean": True, "reasons": []},
                # Poison historical measurement fields: they must not enter a new run's samples.
                "output_text": "historical output must not be reused",
                "output_sha256": "c" * 64,
                "ttft_s": 999.0,
                "decode_s": 999.0,
                "A": 999.0,
                "ms_step": 999.0,
                "delivery_counter_delta": {"steps": 999, "A": 999.0},
                "delivery_metrics_before": {"historical": True},
                "delivery_metrics_after": {"historical": True},
            }
            document = {
                "schema_version": "delivery_matrix_request_receipt_v1",
                "source": source,
                "task": "dihuo",
                "round_index": 0,
                "request_index": index,
                "request": request,
            }
            self._write_artifact(index, document, artifacts)
        self.manifest = {
            "schema_version": "delivery_matrix_split_manifest_v1",
            "artifact_compression": "gzip",
            "artifact_count": 8,
            "artifacts": artifacts,
            "source": source,
        }
        self.write_manifest()

    def _write_artifact(self, index: int, document, artifacts=None):
        name = f"dihuo_round00_request{index:02d}.json.gz"
        raw = encode_json(document)
        compressed = gzip.compress(raw, mtime=0)
        (self.requests / name).write_bytes(compressed)
        row = {
            "path": name,
            "compression": "gzip",
            "size_bytes": len(compressed),
            "sha256": sha(compressed),
            "uncompressed_size_bytes": len(raw),
            "uncompressed_sha256": sha(raw),
            "prompt_sha256": document["request"]["prompt_sha256"],
        }
        if artifacts is None:
            self.manifest["artifacts"][index] = row
        else:
            artifacts.append(row)

    def read_document(self, index: int):
        path = self.requests / f"dihuo_round00_request{index:02d}.json.gz"
        return json.loads(gzip.decompress(path.read_bytes()))

    def rewrite_document(self, index: int, mutate):
        document = self.read_document(index)
        mutate(document)
        self._write_artifact(index, document)
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_bytes(encode_json(self.manifest))

    @property
    def pin(self):
        return sha(self.manifest_path.read_bytes())


class FrozenDihuoInputGateTests(unittest.TestCase):
    def archive(self, directory: str) -> SyntheticArchive:
        return SyntheticArchive(Path(directory) / "frozen")

    def load(self, archive: SyntheticArchive, *, input_tokens=2048, output_tokens=256, pin=None):
        return frozen.load_frozen_dihuo_inputs(
            archive.manifest_path,
            pin or archive.pin,
            input_tokens,
            output_tokens,
        )

    def test_valid_archive_is_ordered_unique_and_drops_historical_measurements(self):
        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            result = self.load(archive)
            samples = result["samples"]
            self.assertEqual(len(samples), 8)
            self.assertEqual([sample["sample_id"] for sample in samples], [f"dihuo-{i + 1:02d}" for i in range(8)])
            self.assertEqual([sample["prompt_text"] for sample in samples], archive.prompts)
            self.assertEqual(len({benchmark.sha256_text(sample["prompt_text"]) for sample in samples}), 8)
            self.assertEqual(result["proof"]["artifact_count"], 8)
            self.assertTrue(result["proof"]["all_inputs_verified_before_service_requests"])
            self.assertFalse(result["proof"]["historical_measurements_reused"])
            poison = {
                "output_text", "output_sha256", "ttft_s", "decode_s", "A", "ms_step",
                "delivery_counter_delta", "delivery_metrics_before", "delivery_metrics_after",
                "usage_tokens", "http_status", "finish_reason", "quality_passed", "quality_score",
                "request_wall_s",
            }
            expected_sample_keys = set(frozen.INPUT_METADATA_KEYS) | {
                "messages", "prompt_text", "tokenizer_count_scope", "input_tokens_exact_scope",
            }
            for sample in samples:
                self.assertTrue(poison.isdisjoint(sample))
                self.assertEqual(set(sample), expected_sample_keys)
                self.assertEqual(sample["messages"], [{"role": "user", "content": sample["prompt_text"]}])
                self.assertEqual(sample["tokenizer_count_scope"],
                                 "historical verified archive receipt; not counted by a tokenizer this run")
                self.assertEqual(sample["input_tokens_exact_scope"],
                                 "historical usage; current SSE exact-usage gate remains required")

    def test_manifest_pin_and_all_sha_layers_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                self.load(archive, pin="0" * 64)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            path = archive.requests / "dihuo_round00_request00.json.gz"
            content = bytearray(path.read_bytes())
            content[10] ^= 1
            path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "compressed SHA-256 mismatch"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            path = archive.requests / "dihuo_round00_request00.json.gz"
            raw = gzip.decompress(path.read_bytes()).replace(b"Synthetic CPU-only", b"Synthetic CPU-onlx", 1)
            compressed = gzip.compress(raw, mtime=0)
            path.write_bytes(compressed)
            archive.manifest["artifacts"][0].update(size_bytes=len(compressed), sha256=sha(compressed))
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "uncompressed SHA-256 mismatch"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.rewrite_document(
                0,
                lambda document: document["request"].update(prompt_messages_sha256="0" * 64),
            )
            with self.assertRaisesRegex(ValueError, "messages SHA-256 mismatch"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)

            def change_text(document):
                request = document["request"]
                changed = request["prompt_text"] + " changed"
                request["prompt_text"] = changed
                request["prompt_messages"] = [{"role": "user", "content": changed}]

            archive.rewrite_document(0, change_text)
            with self.assertRaisesRegex(ValueError, "prompt SHA-256 mismatch"):
                self.load(archive)

    def test_order_unique_budget_seed_ignore_eos_usage_and_marker_guards(self):
        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.manifest["artifacts"][0], archive.manifest["artifacts"][1] = (
                archive.manifest["artifacts"][1], archive.manifest["artifacts"][0]
            )
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "artifact order/path/compression mismatch"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.manifest["source"]["workload_manifest"]["prompt_sha256_by_task"]["dihuo"][1] = (
                archive.manifest["source"]["workload_manifest"]["prompt_sha256_by_task"]["dihuo"][0]
            )
            workload = archive.manifest["source"]["workload_manifest"]
            archive.manifest["source"]["workload_fingerprint"] = benchmark.canonical_json_sha256(workload)
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "eight distinct pinned prompt hashes"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            with self.assertRaisesRegex(ValueError, "historical workload differs"):
                self.load(archive, output_tokens=128)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            with self.assertRaisesRegex(ValueError, "historical workload differs"):
                self.load(archive, input_tokens=4096)

        negative_cases = (
            ("seed", lambda request: request.update(seed=99), "historical request generation contract mismatch"),
            ("ignore_eos", lambda request: request.update(ignore_eos=False), "historical request generation contract mismatch"),
            ("usage", lambda request: request["usage_tokens"].update(completion_tokens=255), "historical exact usage gate did not pass"),
        )
        for name, mutate, error in negative_cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
                archive = self.archive(directory)
                archive.rewrite_document(0, lambda document, mutate=mutate: mutate(document["request"]))
                with self.assertRaisesRegex(ValueError, error):
                    self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)

            def change_marker(document):
                request = document["request"]
                changed = "missing marker\n" + request["prompt_text"].split("\n", 1)[1]
                request["prompt_text"] = changed
                request["prompt_messages"] = [{"role": "user", "content": changed}]

            archive.rewrite_document(0, change_marker)
            with self.assertRaisesRegex(ValueError, "prompt/messages contract mismatch"):
                self.load(archive)

    def test_source_mismatch_and_request_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)

            def alter_source(document):
                document["source"]["dihuo_source"]["corpus_sha256"] = "f" * 64

            archive.rewrite_document(0, alter_source)
            with self.assertRaisesRegex(ValueError, "request receipt/source mismatch"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            request_path = archive.requests / "dihuo_round00_request00.json.gz"
            outside = Path(directory) / "outside.json.gz"
            outside.write_bytes(request_path.read_bytes())
            request_path.unlink()
            request_path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "artifact escapes the request directory"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            alias = archive.root / "manifest-alias.json"
            alias.symlink_to(archive.manifest_path)
            with self.assertRaisesRegex(ValueError, "manifest must not be a file symlink"):
                frozen.load_frozen_dihuo_inputs(alias, archive.pin, 2048, 256)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            moved = archive.root / "requests_actual"
            archive.requests.rename(moved)
            archive.requests.symlink_to(moved, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "requests directory escapes its manifest root"):
                self.load(archive)

    def test_manifest_compressed_and_raw_size_bounds_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.manifest["padding"] = "x" * (frozen.MAX_MANIFEST_BYTES + 1)
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "manifest size is outside"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.manifest["artifacts"][0]["size_bytes"] = frozen.MAX_COMPRESSED_BYTES + 1
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "artifact size is outside"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.manifest["artifacts"][0]["size_bytes"] += 1
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "compressed size mismatch"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.manifest["artifacts"][0]["uncompressed_size_bytes"] += 1
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "uncompressed size mismatch"):
                self.load(archive)

        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            archive.manifest["artifacts"][0]["uncompressed_size_bytes"] = frozen.MAX_REQUEST_BYTES + 1
            archive.write_manifest()
            with self.assertRaisesRegex(ValueError, "artifact size is outside"):
                self.load(archive)


class FrozenDihuoBenchmarkOrderTests(unittest.TestCase):
    model = "synthetic-model"
    sid = "d" * 32
    fingerprint = "e" * 64
    form = "synthetic-form"

    def archive(self, directory: str) -> SyntheticArchive:
        return SyntheticArchive(Path(directory) / "frozen")

    def args(self, archive: SyntheticArchive, **overrides):
        values = dict(
            base_url="http://service-must-be-mocked.invalid",
            model=self.model,
            expected_form=self.form,
            tasks="dihuo",
            input_tokens=2048,
            output_tokens=256,
            code_output_tokens=1024,
            canary_only=False,
            canary_output_tokens=128,
            rounds=1,
            warmups=1,
            request_timeout=1.0,
            tokenizer=None,
            corpus="/must-not-read/dihuo.txt",
            suffix_dir="/must-not-read/dihuo_local",
            frozen_dihuo_manifest=str(archive.manifest_path),
            frozen_dihuo_manifest_sha256=archive.pin,
            expected_service_instance_id=self.sid,
            expected_runtime_fingerprint=self.fingerprint,
            json_out="/unused/not-written.json",
            code_timeout=3.0,
            code_memory_mb=512,
            code_cpu_seconds=2,
            api_key=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def identity(self):
        return {
            "form": self.form,
            "service_instance_id": self.sid,
            "runtime_fingerprint": self.fingerprint,
            "prefill_service_instance_id": None,
            "decode_service_instance_id": None,
            "raw_identity": {"model_kwargs": {"max_model_len": 8192}},
        }

    def test_bad_manifest_pin_precedes_all_ownership_tokenizer_and_generation_calls(self):
        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            args = self.args(archive, frozen_dihuo_manifest_sha256="0" * 64)
            with (
                patch.object(benchmark, "verify_model_ownership") as model_get,
                patch.object(benchmark, "verify_service_identity") as identity_get,
                patch.object(benchmark, "make_token_counter") as tokenizer,
                patch.object(benchmark, "_run_warmup") as warmup,
                patch.object(benchmark, "_run_counter_canary") as canary,
                patch.object(benchmark, "_metrics_snapshot") as metrics,
                patch.object(benchmark, "_arm_delivery_window") as arm,
                patch.object(benchmark, "stream_chat_completion") as generation,
            ):
                with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                    benchmark.run(args)
            for mock in (model_get, identity_get, tokenizer, warmup, canary, metrics, arm, generation):
                mock.assert_not_called()

    def test_empty_frozen_pins_fail_closed_instead_of_using_corpus_or_tokenizer(self):
        cases = (
            ("empty path", {"frozen_dihuo_manifest": ""}),
            ("empty manifest hash", {"frozen_dihuo_manifest_sha256": ""}),
        )
        for label, overrides in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
                archive = self.archive(directory)
                with (
                    patch.object(benchmark, "verify_model_ownership") as model_get,
                    patch.object(benchmark, "verify_service_identity") as identity_get,
                    patch.object(benchmark, "make_token_counter") as tokenizer,
                    patch.object(benchmark, "load_dihuo_material") as corpus_read,
                    patch.object(benchmark, "prepare_dihuo_prompts") as prompt_build,
                    patch.object(benchmark, "_run_warmup") as warmup,
                    patch.object(benchmark, "_run_counter_canary") as canary,
                    patch.object(benchmark, "_metrics_snapshot") as metrics,
                    patch.object(benchmark, "_arm_delivery_window") as arm,
                    patch.object(benchmark, "stream_chat_completion") as generation,
                ):
                    with self.assertRaisesRegex(ValueError, "must not be empty"):
                        benchmark.run(self.args(archive, **overrides))
                for mock in (model_get, identity_get, tokenizer, corpus_read, prompt_build,
                             warmup, canary, metrics, arm, generation):
                    mock.assert_not_called()

        for label, overrides in (
            ("absent path with empty pin", {"frozen_dihuo_manifest": None,
                                            "frozen_dihuo_manifest_sha256": ""}),
            ("empty path with absent pin", {"frozen_dihuo_manifest": "",
                                             "frozen_dihuo_manifest_sha256": None}),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
                archive = self.archive(directory)
                with (
                    patch.object(benchmark, "verify_model_ownership") as model_get,
                    patch.object(benchmark, "verify_service_identity") as identity_get,
                    patch.object(benchmark, "make_token_counter") as tokenizer,
                    patch.object(benchmark, "load_dihuo_material") as corpus_read,
                    patch.object(benchmark, "prepare_dihuo_prompts") as prompt_build,
                    patch.object(benchmark, "_run_warmup") as warmup,
                    patch.object(benchmark, "_run_counter_canary") as canary,
                    patch.object(benchmark, "_metrics_snapshot") as metrics,
                    patch.object(benchmark, "_arm_delivery_window") as arm,
                    patch.object(benchmark, "stream_chat_completion") as generation,
                ):
                    with self.assertRaisesRegex(ValueError, "supplied together"):
                        benchmark.run(self.args(archive, **overrides))
                for mock in (model_get, identity_get, tokenizer, corpus_read, prompt_build,
                             warmup, canary, metrics, arm, generation):
                    mock.assert_not_called()

        for label, overrides in (
            ("empty expected sid", {"expected_service_instance_id": ""}),
            ("empty expected fingerprint", {"expected_runtime_fingerprint": ""}),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
                archive = self.archive(directory)
                with (
                    patch.object(benchmark, "verify_model_ownership") as model_get,
                    patch.object(benchmark, "verify_service_identity") as identity_get,
                    patch.object(benchmark, "make_token_counter") as tokenizer,
                    patch.object(benchmark, "load_dihuo_material") as corpus_read,
                    patch.object(benchmark, "prepare_dihuo_prompts") as prompt_build,
                    patch.object(benchmark, "_run_warmup") as warmup,
                    patch.object(benchmark, "_run_counter_canary") as canary,
                    patch.object(benchmark, "_metrics_snapshot") as metrics,
                    patch.object(benchmark, "_arm_delivery_window") as arm,
                    patch.object(benchmark, "stream_chat_completion") as generation,
                ):
                    with self.assertRaisesRegex(ValueError, "requires the current 32-hex service instance"):
                        benchmark.run(self.args(archive, **overrides))
                for mock in (model_get, identity_get, tokenizer, corpus_read, prompt_build,
                             warmup, canary, metrics, arm, generation):
                    mock.assert_not_called()

    def test_expected_sid_or_fingerprint_mismatch_stops_before_warmup_or_generation(self):
        cases = (
            ("sid", {"expected_service_instance_id": "0" * 32}),
            ("runtime-fingerprint", {"expected_runtime_fingerprint": "f" * 64}),
        )
        for label, overrides in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
                archive = self.archive(directory)
                with (
                    patch.object(benchmark, "verify_model_ownership", return_value={"verified": True}) as model_get,
                    patch.object(benchmark, "verify_service_identity", return_value=self.identity()) as identity_get,
                    patch.object(benchmark, "make_token_counter") as tokenizer,
                    patch.object(benchmark, "load_dihuo_material") as corpus_read,
                    patch.object(benchmark, "prepare_dihuo_prompts") as prompt_build,
                    patch.object(benchmark, "_run_warmup") as warmup,
                    patch.object(benchmark, "_run_counter_canary") as canary,
                    patch.object(benchmark, "_metrics_snapshot") as metrics,
                    patch.object(benchmark, "_arm_delivery_window") as arm,
                    patch.object(benchmark, "stream_chat_completion") as generation,
                ):
                    with self.assertRaisesRegex(ValueError, "does not match the pinned"):
                        benchmark.run(self.args(archive, **overrides))
                model_get.assert_called_once()
                identity_get.assert_called_once()
                for mock in (tokenizer, corpus_read, prompt_build, warmup, canary, metrics, arm, generation):
                    mock.assert_not_called()

    def test_valid_frozen_branch_avoids_corpus_and_tokenizer_and_keeps_serial_counter_protocol(self):
        with tempfile.TemporaryDirectory(dir=TASK_TMP) as directory:
            archive = self.archive(directory)
            events = []
            state = {"snapshots": 0, "arms": 0, "request_id": None, "index": None}

            def metrics_snapshot(*_args, **_kwargs):
                call = state["snapshots"]
                state["snapshots"] += 1
                if call % 2 == 0:
                    index = state["arms"]
                    events.append(("before", index))
                    return self.metric_snapshot(index, None)
                index = state["index"]
                events.append(("after", index))
                return self.metric_snapshot(index, state["request_id"])

            def arm_delivery(_base, max_tokens, tag, *_args, **_kwargs):
                index = state["arms"]
                state["arms"] += 1
                state["index"] = index
                state["request_id"] = tag
                events.append(("arm", index))
                return {
                    "ok": True,
                    "data": {
                        "schema": benchmark.COUNTER_SCHEMA.replace("delivery_runtime_metrics_v1", "delivery_counter_arm_v1"),
                        "service_instance_id": self.sid,
                        "tag": tag,
                        "max_tokens": max_tokens,
                        "ranks": [
                            {"rank": rank, "tag": tag, "max_tokens": max_tokens,
                             "armed": True, "async_mode": True}
                            for rank in range(8)
                        ],
                    },
                }

            def stream(*_positional, **kwargs):
                messages = _positional[2]
                max_tokens = _positional[3]
                request_id = kwargs["request_id"]
                seed = kwargs["seed"]
                if max_tokens == 1:
                    events.append(("warmup", 0))
                    text = "synthetic warmup"
                    usage = {"prompt_tokens": 2048, "completion_tokens": 1, "total_tokens": 2049}
                else:
                    index = state["index"]
                    events.append(("stream", index))
                    self.assertEqual(request_id, state["request_id"])
                    self.assertEqual(seed, index)
                    self.assertTrue(kwargs["ignore_eos"])
                    self.assertEqual(messages, [{"role": "user", "content": archive.prompts[index]}])
                    text = f"synthetic measured output {index}"
                    usage = {"prompt_tokens": 2048, "completion_tokens": 256, "total_tokens": 2304}
                return {
                    "output_text": text,
                    "output_sha256": benchmark.sha256_text(text),
                    "output_utf8_bytes": len(text.encode("utf-8")),
                    "usage_tokens": usage,
                    "finish_reason": "length",
                    "http_status": 200,
                    "error": None,
                    "sse_error_event": None,
                    "sse_json_events": 1,
                    "request_wall_s": 0.3,
                    "ttft_s": 0.05,
                    "decode_s": 0.2,
                    "decode_rate_completion_tokens": max(0, usage["completion_tokens"] - 1),
                    "decode_tok_s": max(0, usage["completion_tokens"] - 1) / 0.2,
                    "tpot_s": 0.2 / max(1, usage["completion_tokens"] - 1),
                }

            real_delta = benchmark.counter_metrics_delta

            def counted_delta(*args, **kwargs):
                result = real_delta(*args, **kwargs)
                events.append(("counter", state["index"]))
                return result

            with (
                patch.object(benchmark, "verify_model_ownership", return_value={"verified": True}),
                patch.object(benchmark, "verify_service_identity", return_value=self.identity()),
                patch.object(benchmark, "make_token_counter", side_effect=AssertionError("tokenizer must not run")) as tokenizer,
                patch.object(benchmark, "load_dihuo_material", side_effect=AssertionError("corpus must not be read")) as corpus_read,
                patch.object(benchmark, "prepare_dihuo_prompts", side_effect=AssertionError("prompts must not be rebuilt")) as prompt_build,
                patch.object(benchmark, "_metrics_snapshot", side_effect=metrics_snapshot),
                patch.object(benchmark, "_arm_delivery_window", side_effect=arm_delivery),
                patch.object(benchmark, "stream_chat_completion", side_effect=stream),
                patch.object(benchmark, "counter_metrics_delta", side_effect=counted_delta),
            ):
                evidence = benchmark.run(self.args(archive))

            tokenizer.assert_not_called()
            corpus_read.assert_not_called()
            prompt_build.assert_not_called()
            self.assertEqual(state["snapshots"], 16)
            self.assertEqual(state["arms"], 8)
            expected_events = [("warmup", 0)]
            for index in range(8):
                expected_events.extend([
                    ("before", index), ("arm", index), ("stream", index),
                    ("after", index), ("counter", index),
                ])
            self.assertEqual(events, expected_events)
            rows = evidence["groups"][0]["requests"]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["http_status"] == 200 and row["finish_reason"] == "length" for row in rows))
            self.assertTrue(all(row["usage_tokens"] == {
                "prompt_tokens": 2048, "completion_tokens": 256, "total_tokens": 2304,
            } for row in rows))
            self.assertTrue(all(row["delivery_counter_delta"]["available"] for row in rows))
            self.assertEqual([row["delivery_counter_delta"]["steps"] for row in rows], [2] * 8)
            self.assertTrue(evidence["groups"][0]["aggregate"]["valid_for_aggregation"])
            self.assertEqual(evidence["groups"][0]["aggregate"]["exact_input_length_requests"], 8)
            self.assertEqual(evidence["groups"][0]["aggregate"]["finish_reason_requests"], 8)
            self.assertEqual(evidence["summary_by_task"]["dihuo"]["steps"], 16)
            self.assertEqual(evidence["summary_by_task"]["dihuo"]["actual_outputs"], 8)
            self.assertEqual(evidence["config"]["frozen_input_gate"]["artifact_count"], 8)
            self.assertTrue(evidence["dihuo_source"]["corpus_file_read_this_run"] is False)

    def metric_snapshot(self, index: int, request_id: str | None):
        fingerprint = f"{index + 1:064x}"
        ranks = []
        for rank in range(8):
            cumulative = {
                "decode_rounds": 0 if request_id is None else 2,
                "decode_tokens": None,
                "decode_seconds": 0.0 if request_id is None else 0.2,
                "prefill_seconds": 0.0 if request_id is None else 0.3,
            }
            row = {
                "rank": rank,
                "cumulative": cumulative,
                "requests": [],
                "pending": False,
                "errors": [],
                "counter_scope": benchmark.COUNTER_SCOPE,
                "decode_tokens_scope": "HTTP final usage required for async scheduler trimming",
            }
            if request_id is not None:
                row["requests"] = [{
                    "request_id": f"engine-{index}",
                    "tag": request_id,
                    "prompt_tokens": 2048,
                    "seed_tokens": 1,
                    "decode_rounds": 2,
                    "async_mode": True,
                    "terminal_fence_observed": True,
                    "profile_excluded": True,
                    "terminal_tp_join_observed": True,
                    "decode_start_ns": 1_000_000_000,
                    "terminal_sample_end_ns": 1_100_000_000,
                    "fenced_end_ns": 1_200_000_000,
                    "window_fingerprint": fingerprint,
                    "prefill_wall_scope": "first scheduled prefill -> terminal prefill CPU bookkeeping",
                    "decode_wall_scope": "terminal prefill CPU bookkeeping -> worker completion fence; may include metrics RPC delay",
                    "raw_decode_tokens": None,
                    "last_round_tokens": None,
                }]
                row["last_window"] = {"tag": request_id, "window_fingerprint": fingerprint}
            ranks.append(row)
        return {
            "ok": True,
            "data": {
                "schema": benchmark.COUNTER_SCHEMA,
                "trusted": True,
                "counter_scope": benchmark.COUNTER_SCOPE,
                "served_model": self.model,
                "form": self.form,
                "service_instance_id": self.sid,
                "ranks": ranks,
            },
        }


if __name__ == "__main__":
    unittest.main()
