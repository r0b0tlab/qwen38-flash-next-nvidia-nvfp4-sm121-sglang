"""Client tokenizer admission and mutation refusal before benchmark traffic."""

from types import SimpleNamespace
import pytest
from scripts import run_upstream_round as r
from scripts.benchmark_evidence import input_hash


def test_tokenizer_is_bound_to_manifest_assets(monkeypatch):
    assert hasattr(r, "admit_tokenizer"), (
        "runner does not validate its actual tokenizer"
    )
    from scripts import niah

    tok = SimpleNamespace(
        _qwen38_tokenizer_assets={"config.json": "a" * 64}, _qwen38_model_sha="b" * 40
    )
    monkeypatch.setattr(niah, "load_real_tokenizer", lambda path: tok)
    manifest = {
        "model_sha": "b" * 40,
        "tokenizer_assets": tok._qwen38_tokenizer_assets,
        "tokenizer": input_hash(tok._qwen38_tokenizer_assets),
    }
    assert r.admit_tokenizer(manifest, "/test-only") == tok
    manifest["model_sha"] = "c" * 40
    with pytest.raises(ValueError):
        r.admit_tokenizer(manifest, "/test-only")
