import copy
import pytest
from scripts.native_profile_probe import idle, validate_native


def test_idle_waits_for_scheduler_gauge_to_settle(monkeypatch):
    import scripts.native_profile_probe as probe
    busy = b'sglang:num_running_reqs{x="0"} 1\nsglang:num_queue_reqs{x="0"} 0\n'
    quiet = b'sglang:num_running_reqs{x="0"} 0\nsglang:num_queue_reqs{x="0"} 0\n'
    snapshots = iter([busy, quiet])
    monkeypatch.setattr(probe, 'fetch', lambda base, path: next(snapshots))
    monkeypatch.setattr(probe.time, 'sleep', lambda seconds: None)
    idle('http://owned', timeout=1)


def test_idle_does_not_waive_a_busy_endpoint(monkeypatch):
    import scripts.native_profile_probe as probe
    busy = b'sglang:num_running_reqs{x="0"} 1\nsglang:num_queue_reqs{x="0"} 0\n'
    monkeypatch.setattr(probe, 'fetch', lambda base, path: busy)
    with pytest.raises(RuntimeError, match='not proven idle'):
        idle('http://owned', timeout=0)


def test_idle_requires_both_metrics(monkeypatch):
    import scripts.native_profile_probe as probe
    monkeypatch.setattr(probe, 'fetch', lambda base, path: b'sglang:num_running_reqs{x="0"} 0\n')
    with pytest.raises(RuntimeError, match='not proven idle'):
        idle('http://owned', timeout=0)


def sample():
    return {
        "prompt_tokens": 512,
        "completion_tokens": 256,
        "cached_tokens": 0,
        "finish_reason": {"type": "length", "length": 256},
    }


def test_valid_fixed_length_native_response():
    validate_native(sample(), 512, 256, True, True, True)


@pytest.mark.parametrize("field,value", [
    ("prompt_tokens", 511),
    ("completion_tokens", 255),
    ("cached_tokens", 1),
    ("cached_tokens", None),
    ("completion_tokens", True),
    ("finish_reason", {"type": "abort"}),
])
def test_bad_observed_evidence_fails(field, value):
    meta = copy.deepcopy(sample())
    meta[field] = value
    with pytest.raises(ValueError):
        validate_native(meta, 512, 256, True, True, True)


def test_missing_done_is_not_success():
    with pytest.raises(ValueError):
        validate_native(sample(), 512, 256, True, True, False)


def test_natural_prose_must_not_exhaust_budget():
    with pytest.raises(ValueError):
        validate_native(sample(), 512, 256, False, True, True)


def test_warmup_may_have_cache_hits_but_measurement_may_not():
    meta = sample()
    meta["cached_tokens"] = 128
    validate_native(meta, 512, 256, True, False, True)
    with pytest.raises(ValueError):
        validate_native(meta, 512, 256, True, True, True)
