"""Actual upstream CLI, dataset dispatch and reducer; only I/O uses test doubles.

All responses and resulting timings belong to this unit fixture, not a model.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None,
    reason="run in pinned upstream CPU client environment",
)


def test_real_cli_uses_frozen_ids_and_observed_lengths(tmp_path, monkeypatch):
    from sglang.benchmark import serving as u
    import sglang.benchmark.datasets as datasets
    from scripts import run_upstream_round as driver
    from scripts.freeze_benchmark import upstream_payload
    from scripts.benchmark_evidence import input_hash

    payloads = [upstream_payload([i + 2] * 512, 256) for i in range(8)]
    manifest = {
        "runtime_context": {"endpoint": "http://127.0.0.1:30080"},
        "requests": {"short": payloads},
        "inputs": {"short": {"batch": input_hash(payloads)}},
        "seed": 20260907,
        "manifest_sha256": "f" * 64,
        "capture_adapter_sha256": hashlib.sha256(
            (Path(driver.__file__).parent / "upstream_capture.py").read_bytes()
        ).hexdigest(),
        "benchmark_module_sha256": hashlib.sha256(
            Path(u.__file__).read_bytes()
        ).hexdigest(),
    }
    events = []

    class Response:
        status = 200
        reason = "OK"

        def __init__(self, payload):
            self.payload = payload

        @property
        def content(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def __aiter__(self):
            async def parts():
                for text, finish in (("x", None), ("y", "length")):
                    item = {
                        "model": driver.MODEL_ID,
                        "choices": [{"text": text, "finish_reason": finish}],
                        "sglext": {"cached_tokens_details": {"device": 0, "host": 0}},
                    }
                    yield ("data: " + json.dumps(item) + "\n\n").encode()
                n = len(self.payload["prompt"])
                out = self.payload["max_tokens"]
                item = {
                    "model": driver.MODEL_ID,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": n,
                        "completion_tokens": out,
                        "total_tokens": n + out,
                    },
                }
                yield ("data: " + json.dumps(item) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

            return parts()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *, url, json, headers):
            events.append(("request", json["max_tokens"], list(json["prompt"])))
            return Response(json)

    class Tokenizer:
        chat_template = "test-only-template"

        def encode(self, text, **kwargs):
            return [1, 2]

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1, 2])

    monkeypatch.setattr(driver, "admit_tokenizer", lambda manifest, path: Tokenizer())
    monkeypatch.setattr(
        driver, "verify_http_epoch", lambda manifest: events.append(("epoch",))
    )
    monkeypatch.setattr(u, "_create_bench_client_session", lambda: Session())
    monkeypatch.setattr(u, "wait_for_endpoint", lambda *args, **kwargs: True)
    monkeypatch.setattr(u, "check_chat_template", lambda model: True)
    monkeypatch.setattr(u, "get_tokenizer", lambda path: Tokenizer())
    monkeypatch.setattr(
        u, "flush_server_cache", lambda *args: events.append(("flush",))
    )
    monkeypatch.setattr(
        u.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {}),
    )
    original = u.ASYNC_REQUEST_FUNCS["sglang-oai"]
    original_dataset = datasets.DATASET_MAPPING["random-ids"]
    try:
        target = tmp_path / "unit-only-short.jsonl"
        report = driver.run_round(
            manifest, "short", 0, "/test-only/tokenizer", target, allow_cold_flush=True
        )
        row = json.loads(target.read_text())
        assert report["requests"] == 8 and row["completed"] == 8
        assert (
            row["total_output_tokens"] == 2048
            and row["total_output_tokens_retokenized"] == 16
        )
        assert row["output_lens"] == [256] * 8 and row["input_lens"] == [512] * 8
        assert [event[1] for event in events if event[0] == "request"] == [32] + [
            256
        ] * 8
        assert events.index(("flush",)) == 2  # epoch, one warmup, flush
        assert events[-1] == ("epoch",)
        assert row["request_hashes"] == [input_hash(p) for p in payloads]
    finally:
        u.ASYNC_REQUEST_FUNCS["sglang-oai"] = original
        datasets.DATASET_MAPPING["random-ids"] = original_dataset
        for name in (
            "_qwen38_captured_oai",
            "_qwen38_capture_request",
            "_qwen38_capture_chunk",
            "_qwen38_capture_event",
            "_qwen38_capture_done",
        ):
            if hasattr(u, name):
                delattr(u, name)
