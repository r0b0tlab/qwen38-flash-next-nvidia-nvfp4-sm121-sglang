"""Pin native zero-elision semantics; replay real captured SSE without traffic."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import cache_reporting_contract as contract
from scripts.upstream_capture import CaptureRecord

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/cache_zero_elided.json").read_text())
WIRE = (ROOT / "tests/fixtures/cache_zero_elided.sse").read_bytes()


def verified_contract():
    return contract.from_runtime_lock(
        json.loads((ROOT / "locks/runtime.json").read_text())
    )


def replay(tmp_path, *, policy=None, fault=None, warmup=True):
    request = copy.deepcopy(FIXTURE["request"])
    if fault == "not_requested":
        request["return_cached_tokens_details"] = False
    record = CaptureRecord(request, 0, warmup, tmp_path, cache_contract=policy)
    record.request(request)
    record.chunk(WIRE)
    for line in WIRE.decode().splitlines():
        if not line.startswith("data: "):
            continue
        value = line[6:]
        if value == "[DONE]":
            if fault != "missing_done":
                record.done()
            continue
        event = json.loads(value)
        if fault == "missing_usage":
            event.pop("usage", None)
        if fault == "wrong_model":
            event["model"] = "other/model"
        record.event(event)
    if fault == "malformed_details":
        record.event(
            {"sglext": {"cached_tokens_details": {"device": False, "host": 0}}}
        )
    if fault == "cache_hit":
        record.event({"sglext": {"cached_tokens_details": {"device": 1, "host": 0}}})
    output = SimpleNamespace(success=True, error="", prompt_len=999, output_len=999)
    return record.finalize(output), output


def test_real_wire_identity_and_zero_elision(tmp_path):
    assert hashlib.sha256(WIRE).hexdigest() == FIXTURE["wire_sha256"]
    result, output = replay(tmp_path, policy=verified_contract())
    assert result["valid"] and output.success and result["cached_tokens"] == 0
    assert result["cache_details"] is None  # never fabricate an explicit response field
    assert result["cache_observation_source"] == "pinned_sglang_zero_elision"
    assert result["observed_usage"]["prompt_tokens"] == 512
    assert result["observed_usage"]["completion_tokens"] == 32


def test_unknown_emitter_still_refuses_missing_cache_observation(tmp_path):
    result, output = replay(tmp_path)
    assert not result["valid"] and "cache_observation_missing" in result["error"]


@pytest.mark.parametrize(
    "fault",
    [
        "not_requested",
        "missing_done",
        "missing_usage",
        "wrong_model",
        "malformed_details",
    ],
)
def test_bound_zero_does_not_mask_invalid_or_incomplete_response(tmp_path, fault):
    result, output = replay(tmp_path, policy=verified_contract(), fault=fault)
    assert not result["valid"] and not output.success


def test_nonzero_cold_cache_hit_remains_refused(tmp_path):
    result, output = replay(
        tmp_path, policy=verified_contract(), fault="cache_hit", warmup=False
    )
    assert not result["valid"] and result["cached_tokens"] == 1
    assert "cold_request_cache_hit" in result["error"]


def test_cache_contract_rejects_source_drift():
    lock = json.loads((ROOT / "locks/runtime.json").read_text())
    name = next(iter(contract.EXPECTED_SOURCE_HASHES))
    lock["sglang"]["python_files"][name] = "0" * 64
    with pytest.raises(ValueError):
        contract.from_runtime_lock(lock)


def test_cache_contract_rejects_unbound_boolean(tmp_path):
    with pytest.raises(ValueError):
        replay(tmp_path, policy=True)


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="requires the pinned CPU image"
)
def test_real_native_emitter_elides_only_zero_and_preserves_hits():
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )

    receiver = SimpleNamespace(enable_hicache_storage=lambda: False)
    req = SimpleNamespace(
        cached_tokens_device=0,
        cached_tokens_host=0,
        cached_tokens_storage=0,
        cached_tokens=0,
    )
    assert SchedulerOutputStreamer.get_cached_tokens_details(receiver, req) is None
    req.cached_tokens = 1
    assert SchedulerOutputStreamer.get_cached_tokens_details(receiver, req) == {
        "device": 1,
        "host": 0,
    }
    req.cached_tokens = 0
    req.cached_tokens_host = 3
    assert SchedulerOutputStreamer.get_cached_tokens_details(receiver, req) == {
        "device": 0,
        "host": 3,
    }
