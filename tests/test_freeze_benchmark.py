"""Frozen corpus producer identity and literal token-array tests (CPU only)."""

import importlib
from pathlib import Path

import pytest


def api():
    assert (
        Path(__file__).resolve().parents[1] / "scripts/freeze_benchmark.py"
    ).exists(), "manifest producer is not implemented"
    return importlib.import_module("scripts.freeze_benchmark")


class Tokenizer:
    all_special_ids = [0, 1]

    def get_vocab(self):
        return {str(i): i for i in range(2048)}


def test_frozen_arrays_repeat_exactly_without_special_tokens():
    f = api()
    a = f.frozen_token_rows(Tokenizer(), 512, 20260907)
    b = f.frozen_token_rows(Tokenizer(), 512, 20260907)
    assert a == b and len(a) == 8
    assert all(
        len(row) == 512 and all(type(x) is int and x not in (0, 1) for x in row)
        for row in a
    )
    assert len({row[0] for row in a}) == 8


def test_upstream_payload_has_observed_usage_and_fixed_length_controls():
    p = api().upstream_payload([2, 3, 4], 256)
    assert (
        p["prompt"] == [2, 3, 4] and p["ignore_eos"] is True and p["max_tokens"] == 256
    )
    assert p["stream_options"] == {"include_usage": True}
    assert p["return_cached_tokens_details"] is True


def test_bad_token_and_budget_values_are_not_normalized():
    f = api()
    with pytest.raises(ValueError):
        f.upstream_payload([True, 2], 256)
    with pytest.raises(ValueError):
        f.upstream_payload([2, 3], 256.0)
