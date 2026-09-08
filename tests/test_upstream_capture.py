"""Observed-wire accounting must replace upstream requested-count defaults."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/upstream_capture.py"
    assert path.exists(), "upstream capture adapter is not implemented"
    return importlib.import_module("scripts.upstream_capture")


def expected():
    return {
        "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "prompt": [1, 2, 3],
        "max_tokens": 2,
        "stream": True,
        "best_of": 1,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream_options": {"include_usage": True},
        "return_cached_tokens_details": True,
    }


def response(usage=True):
    event = {
        "model": expected()["model"],
        "choices": [{"text": "test-only", "finish_reason": "length"}],
        "sglext": {"cached_tokens_details": {"device": 0, "host": 0}},
    }
    if usage:
        event["usage"] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    return event


def test_usage_only_chunk_is_observed_and_validates_native_counts(tmp_path):
    m = module()
    record = m.CaptureRecord(expected(), 0, False, tmp_path)
    record.request(expected())
    record.chunk(b"data: test-only\n")
    record.event(response(False))
    record.event(
        {
            "model": expected()["model"],
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )
    record.done()
    output = SimpleNamespace(
        success=True, output_len=999, prompt_len=999, error="", cached_tokens=999
    )
    doc = record.finalize(output)
    assert output.success and output.output_len == 2 and output.prompt_len == 3
    assert doc["observed_usage"]["completion_tokens"] == 2 and doc["valid"] is True
    assert doc["finish_reason"] == "length"


def test_missing_usage_does_not_use_requested_output_len(tmp_path):
    m = module()
    record = m.CaptureRecord(expected(), 0, False, tmp_path)
    record.request(expected())
    record.chunk(b"data: test-only\n")
    record.event(response(False))
    record.done()
    output = SimpleNamespace(success=True, output_len=2, prompt_len=3, error="")
    doc = record.finalize(output)
    assert output.success is False and output.output_len == 0 and doc["valid"] is False
    assert "usage" in doc["error"]


def test_wrong_same_length_prompt_rejected_before_http(tmp_path):
    m = module()
    record = m.CaptureRecord(expected(), 0, False, tmp_path)
    wrong = {**expected(), "prompt": [1, 2, 99]}
    with pytest.raises(ValueError, match="request"):
        record.request(wrong)
    record.close()


@pytest.mark.parametrize(
    "fault", ["wrong_model", "wrong_usage", "bad_finish", "missing_done", "cache_hit"]
)
def test_incomplete_or_foreign_capture_never_passes(tmp_path, fault):
    m = module()
    record = m.CaptureRecord(expected(), 0, False, tmp_path)
    record.request(expected())
    record.chunk(b"data: test-only\n")
    event = response()
    if fault == "wrong_model":
        event["model"] = "other/model"
    if fault == "wrong_usage":
        event["usage"]["prompt_tokens"] = 4
    if fault == "bad_finish":
        event["choices"][0]["finish_reason"] = "stop"
    if fault == "cache_hit":
        event["sglext"]["cached_tokens_details"]["device"] = 1
    record.event(event)
    if fault != "missing_done":
        record.done()
    output = SimpleNamespace(success=True, output_len=2, prompt_len=3, error="")
    assert record.finalize(output)["valid"] is False
    assert output.success is False
