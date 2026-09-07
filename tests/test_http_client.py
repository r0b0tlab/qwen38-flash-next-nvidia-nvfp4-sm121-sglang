"""Scaffold/contract tests for scripts/http_client.py against an in-process fake server.

Every response exercised here comes from tests/fake_sse_server.py — synthetic
test vectors, never real model output.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import http_client as hc  # noqa: E402
from fake_sse_server import FakeServer, MODEL_ID  # noqa: E402


@pytest.fixture(scope="module")
def fake():
    with FakeServer() as srv:
        yield srv


def client_for(srv: FakeServer, **kw) -> hc.OpenAICompatClient:
    return hc.OpenAICompatClient(srv.base, **kw)


# ---------------------------------------------------------------------------
# explicit-base / config policy
# ---------------------------------------------------------------------------


class TestExplicitBase:
    def test_missing_base_raises(self):
        with pytest.raises(hc.ConfigError):
            hc.OpenAICompatClient(None)

    def test_empty_base_raises(self):
        with pytest.raises(hc.ConfigError):
            hc.OpenAICompatClient("   ")

    def test_non_http_base_raises(self):
        with pytest.raises(hc.ConfigError):
            hc.OpenAICompatClient("ftp://example.invalid")

    def test_model_id_constant_exact(self):
        assert hc.MODEL_ID == "nvidia/Qwen3.8-Flash-Next-NVFP4"

    def test_transport_error_on_unreachable(self):
        c = hc.OpenAICompatClient("http://127.0.0.1:1")
        with pytest.raises(hc.TransportError):
            c.list_models(timeout=2.0)

    def test_timeout_classified_as_transport(self):
        c = hc.OpenAICompatClient("http://127.0.0.1:1")
        res = c.chat_stream([{"role": "user", "content": "x"}], max_tokens=4, timeout=1.0)
        assert res.error in ("transport", "timeout")


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


class TestHappyStream:
    def test_content_reasoning_usage_finish(self, fake):
        c = client_for(fake)
        res = c.chat_stream([{"role": "user", "content": "hi"}], max_tokens=16, timeout=10)
        assert res.ok, res.error_detail
        assert res.content == "Hello world!"
        assert res.reasoning == "thinking..."
        assert res.finish_reason == "stop"
        assert res.usage == {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}
        assert res.saw_done is True
        assert res.ttft_s is not None and res.ttft_s >= 0
        assert res.wall_s > 0
        assert res.e2e_output_tok_per_s == pytest.approx(7 / res.wall_s)
        assert res.model_reported == MODEL_ID

    def test_fragmented_utf8_stream(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/fragmented-utf8", {}, 10, capture_raw=False)
        assert res.ok
        assert "héllo 🌀 wörld — 你好" in res.content
        assert res.usage["completion_tokens"] == 7

    def test_first_token_is_not_role_event(self, fake):
        """Role-only and empty deltas must not start the token clock."""
        c = client_for(fake)
        res = c.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8, timeout=10)
        assert res.first_fragment_kind == "content"  # server sends role first
        assert res.ttft_s > 0 or res.ttft_s == 0

    def test_ttft_reasoning_counts(self, fake):
        """A reasoning-only-first stream still has a valid TTFT (reasoning)."""
        c = client_for(fake)
        res = c._stream_request(fake.url("reasoning-only").replace(fake.base, ""), {}, 10, capture_raw=False)
        assert res.ok
        assert res.content == ""
        assert res.reasoning == "only reasoning here"
        assert res.ttft_s is not None
        assert res.first_fragment_kind == "reasoning"

    def test_crlf_and_multiline_data(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/crlf-multiline", {}, 10, capture_raw=False)
        assert res.ok
        assert res.content == "abc"

    def test_timing_starts_at_request(self, fake):
        """Wall time must be at least the client-observed request duration and
        must NOT include admission time (admission is acquired by the caller
        before calling chat_stream)."""
        c = client_for(fake)
        gate = hc.ConcurrencyGate(1)
        with gate.slot():
            t0 = time.perf_counter()
            res = c.chat_stream([{"role": "user", "content": "hi"}], max_tokens=8, timeout=10)
            wall = time.perf_counter() - t0
        assert res.wall_s <= wall + 0.05  # clock covers the request, not admission


# ---------------------------------------------------------------------------
# failure modes — all preserved, none discarded
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_http_400_preserved(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/http-400", {}, 10, capture_raw=False)
        assert res.error == "http_status_400"
        assert res.http_status == 400
        assert "bad request" in res.raw_error_body

    def test_http_500_preserved(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/http-500", {}, 10, capture_raw=False)
        assert res.error == "http_status_500"
        assert res.http_status == 500
        assert "upstream exploded" in res.raw_error_body

    def test_partial_eof_missing_done(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/partial-eof", {}, 10, capture_raw=True)
        assert res.error == "missing_done"
        # completed generated content is preserved...
        assert "partial" in res.content
        # ...and the truncated unparseable event is kept as raw evidence
        assert any(isinstance(e, dict) and "_unparseable" in e for e in res.raw_events)
        assert not res.usage

    def test_missing_done_clean_truncation(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/stall-close", {}, 10, capture_raw=False)
        assert res.error == "missing_done"
        assert res.content == "stall"

    def test_invalid_usage_float(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/invalid-usage-float", {}, 10, capture_raw=False)
        assert res.error == "invalid_usage"
        assert res.usage is None

    def test_invalid_usage_bool(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/invalid-usage-bool", {}, 10, capture_raw=False)
        assert res.error == "invalid_usage"

    def test_invalid_usage_missing_field(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/invalid-usage-missing", {}, 10, capture_raw=False)
        assert res.error == "invalid_usage"
        assert "total_tokens" in res.error_detail

    def test_missing_usage_fails_claim(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/missing-usage", {}, 10, capture_raw=False)
        assert res.error == "missing_usage"
        valid, reason = hc.row_validity(res)
        assert valid is False and reason == "missing_usage"

    def test_finish_length_is_not_valid_pass(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/finish-length", {}, 10, capture_raw=False)
        assert res.ok
        assert res.finish_reason == "length"
        valid, reason = hc.row_validity(res)
        assert valid is False
        assert reason == "finish_length"
        # but the observed data is preserved for evidence
        assert res.content == "truncated outp"
        assert res.usage["completion_tokens"] == 4

    def test_role_only_never_counts_as_token(self, fake):
        c = client_for(fake)
        res = c._stream_request("/s/role-only", {}, 10, capture_raw=False)
        assert res.ok
        assert res.content == ""
        assert res.finish_reason == "stop"
        assert res.usage["completion_tokens"] == 6  # from server usage, not chunks

    def test_unparseable_json_event_flagged(self, fake):
        c = client_for(fake)
        res = hc.StreamResult()
        c._consume_event("not json {", res, time.perf_counter(), capture_raw=False)
        assert res.error == "invalid_usage" or res.error_detail.startswith("unparseable SSE JSON")


# ---------------------------------------------------------------------------
# non-stream responses
# ---------------------------------------------------------------------------


class TestNonStream:
    def test_chat_nonstream(self, fake):
        c = client_for(fake)
        res = c.chat_nonstream([{"role": "user", "content": "hi"}], max_tokens=8, timeout=10)
        assert res.ok
        assert res.text == "nonstream reply"
        assert res.finish_reason == "stop"
        assert res.usage["completion_tokens"] == 3

    def test_chat_nonstream_reasoning_field(self, fake):
        """reasoning_content / reasoning message fields are captured."""
        c = client_for(fake)
        res = c.chat_nonstream(
            [{"role": "user", "content": "hi"}],
            max_tokens=8,
            timeout=10,
            extra={"_reasoning": True},  # fake-server switch to emit reasoning_content
        )
        assert res.ok
        assert res.reasoning == "hmm"
        assert res.text == "nonstream reply"

    def test_completions_token_ids(self, fake):
        c = client_for(fake)
        res = c.completions_tokens([1, 2, 3, 4, 5], max_tokens=4, timeout=10)
        assert res.ok
        assert res.text == "AB"
        assert res.usage["prompt_tokens"] == 5  # echoed prompt length


# ---------------------------------------------------------------------------
# model identity
# ---------------------------------------------------------------------------


class TestModelIdentity:
    def test_verify_model_ok(self, fake):
        client_for(fake).verify_model(timeout=5)

    def test_verify_model_mismatch(self, fake):
        c = hc.OpenAICompatClient(fake.base + "/v1/models-wrong".replace("/v1/models-wrong", ""), )
        # point at a path-serving wrong model via list override
        class WrongClient(hc.OpenAICompatClient):
            def list_models(self, timeout=30.0):
                return ["some/other-model"]

        wc = WrongClient(fake.base)
        with pytest.raises(hc.ModelMismatchError):
            wc.verify_model(timeout=5)

    def test_no_silent_model_substitution(self, fake):
        """chat_stream always sends the frozen model id."""
        seen: dict = {}

        class Sniff(hc.OpenAICompatClient):
            def _open(self, method, path, body, timeout):
                seen["model"] = (body or {}).get("model")
                return super()._open(method, path, body, timeout)

        c = Sniff(fake.base)
        c.chat_stream([{"role": "user", "content": "x"}], max_tokens=4, timeout=10)
        assert seen["model"] == hc.MODEL_ID


# ---------------------------------------------------------------------------
# flush_cache owner admission
# ---------------------------------------------------------------------------


class TestFlushCache:
    def test_flush_without_admission_refused(self, fake):
        c = client_for(fake)
        with pytest.raises(hc.OwnerAdmissionError):
            c.flush_cache(admitted=False)

    def test_flush_with_admission_plaintext_ok(self, fake):
        c = client_for(fake)
        status, body = c.flush_cache(admitted=True, timeout=5)
        assert status == 200
        assert "flush" in body.lower()


# ---------------------------------------------------------------------------
# SSE decoder unit behavior
# ---------------------------------------------------------------------------


class TestSSEDecoder:
    def test_split_inside_utf8_codepoint(self):
        d = hc.SSEDecoder()
        raw = 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'.encode("utf-8")
        out = []
        for i in range(0, len(raw), 2):  # 2-byte slices cut 3-byte codepoints
            out += d.feed(raw[i : i + 2])
        out += d.close()
        assert json.loads(out[0])["choices"][0]["delta"]["content"] == "你好"

    def test_done_terminates(self):
        d = hc.SSEDecoder()
        events = d.feed(b"data: [DONE]\n\n")
        assert events == ["[DONE]"]

    def test_comment_only_event_ignored(self):
        d = hc.SSEDecoder()
        assert d.feed(b": ping\n\n") == []
        assert d.feed(b"data: x\n\n") == ["x"]

    def test_multiline_data_joined(self):
        d = hc.SSEDecoder()
        assert d.feed(b"data: a\ndata: b\n\n") == ["a\nb"]

    def test_usage_validator(self):
        assert hc.validate_usage_object({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}) == {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
        for bad in (
            {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
            {"prompt_tokens": 1.0, "completion_tokens": 2, "total_tokens": 3},
            {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 3},
            {"prompt_tokens": 1, "completion_tokens": 2},
            "17",
            None,
        ):
            with pytest.raises(hc.InvalidUsageError):
                hc.validate_usage_object(bad)


# ---------------------------------------------------------------------------
# request payload / thinking policy
# ---------------------------------------------------------------------------


class TestPayloadPolicy:
    def test_throughput_default_no_thinking(self):
        p = hc.build_chat_payload([{"role": "user", "content": "x"}], max_tokens=8)
        assert p["chat_template_kwargs"]["enable_thinking"] is False
        assert "reasoning_effort" not in p

    def test_thinking_low_effort(self):
        fields = hc.thinking_request_fields(True, "low")
        assert fields == {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"}

    def test_stream_options_include_usage(self):
        p = hc.build_chat_payload([], max_tokens=1, stream=True)
        assert p["stream_options"] == {"include_usage": True}
        p2 = hc.build_chat_payload([], max_tokens=1, stream=False)
        assert "stream_options" not in p2

    def test_row_validity_contract(self):
        ok = hc.StreamResult(content="answer", finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        v, reason = hc.row_validity(ok)
        assert v and reason is None
        v, reason = hc.row_validity(hc.StreamResult(error="http_status_500"))
        assert not v and reason == "http_status_500"
        v, reason = hc.row_validity(hc.StreamResult(content="", finish_reason="stop", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
        assert not v and reason == "empty_final_content"


# ---------------------------------------------------------------------------
# smoke CLI
# ---------------------------------------------------------------------------


class TestSmokeCLI:
    def test_cli_requires_base(self, capsys):
        with pytest.raises(SystemExit) as ei:
            hc.main(["--smoke-chat", "4"])
        assert ei.value.code == 2  # argparse error

    def test_cli_verify_model_and_smoke(self, fake):
        rc = hc.main(["--base", fake.base, "--verify-model", "--smoke-chat", "8"])
        assert rc == 0

    def test_cli_flush_requires_env_admission(self, fake):
        import os

        env_saved = os.environ.pop("QUAL_HARNESS_OWNER_ADMITTED", None)
        try:
            with pytest.raises(hc.OwnerAdmissionError):
                hc.main(["--base", fake.base, "--flush-cache"])
        finally:
            if env_saved is not None:
                os.environ["QUAL_HARNESS_OWNER_ADMITTED"] = env_saved
