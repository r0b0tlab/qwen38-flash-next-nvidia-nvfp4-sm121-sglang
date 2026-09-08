"""Exercise the real pinned upstream function with CPU-only transport doubles."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None,
    reason="run this integration test in the pinned CPU client image",
)


def test_actual_upstream_parser_accepts_usage_only_events(tmp_path, monkeypatch):
    from sglang.benchmark import serving as upstream
    from scripts.upstream_capture import install_capture, MODEL_ID

    payload = {
        "model": MODEL_ID,
        "prompt": [1, 2, 3],
        "best_of": 1,
        "max_tokens": 2,
        "stream": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "return_cached_tokens_details": True,
    }
    include_usage = [True]

    class Response:
        status = 200
        reason = "OK"

        @property
        def content(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def __aiter__(self):
            async def chunks():
                event = {
                    "model": MODEL_ID,
                    "choices": [
                        {"text": "test-only output", "finish_reason": "length"}
                    ],
                    "sglext": {"cached_tokens_details": {"device": 0, "host": 0}},
                }
                yield ("data: " + json.dumps(event) + "\n\n").encode()
                if include_usage[0]:
                    event = {
                        "model": MODEL_ID,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                    yield ("data: " + json.dumps(event) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

            return chunks()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *, url, json, headers):
            assert json == payload
            return Response()

    monkeypatch.setattr(upstream, "_create_bench_client_session", lambda: Session())
    monkeypatch.setattr(
        upstream,
        "args",
        SimpleNamespace(
            disable_stream=False,
            disable_ignore_eos=False,
            return_logprob=False,
            top_logprobs_num=0,
            cache_report=True,
            header=None,
        ),
        raising=False,
    )
    original = upstream.ASYNC_REQUEST_FUNCS["sglang-oai"]
    sha = hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()
    records, patched_sha = install_capture(upstream, [payload], tmp_path, sha)
    request = upstream.RequestFuncInput(
        prompt=[1, 2, 3],
        api_url="http://test-only/v1/completions",
        prompt_len=3,
        output_len=2,
        model=MODEL_ID,
        lora_name=None,
        image_data=None,
        extra_request_body={
            key: payload[key]
            for key in (
                "temperature",
                "top_p",
                "ignore_eos",
                "stream_options",
                "return_cached_tokens_details",
            )
        },
    )
    try:
        output = asyncio.run(
            upstream.ASYNC_REQUEST_FUNCS["sglang-oai"](request_func_input=request)
        )
        assert (
            output.success is True and output.output_len == 2 and output.prompt_len == 3
        )
        assert records[-1]["observed_usage"]["completion_tokens"] == 2
        assert records[-1]["finish_reason"] == "length" and len(patched_sha) == 64
        include_usage[0] = False
        failed = asyncio.run(upstream.ASYNC_REQUEST_FUNCS["sglang-oai"](request))
        assert failed.success is False and failed.output_len == 0
        assert "missing_usage" in records[-1]["error"]
    finally:
        upstream.ASYNC_REQUEST_FUNCS["sglang-oai"] = original
        for name in (
            "_qwen38_captured_oai",
            "_qwen38_capture_request",
            "_qwen38_capture_chunk",
            "_qwen38_capture_event",
            "_qwen38_capture_done",
        ):
            delattr(upstream, name)
