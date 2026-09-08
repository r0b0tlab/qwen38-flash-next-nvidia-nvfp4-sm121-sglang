"""Prose library-path freeze, binding and exclusivity regressions."""

import json
from types import SimpleNamespace

import pytest
from scripts import bench_real as b


def test_library_freezes_inputs_before_client_creation(tmp_path, monkeypatch):
    out = tmp_path / "prose.jsonl"
    seen = []

    class Client:
        def __init__(self, base):
            assert out.with_suffix(".jsonl.corpus.json").exists()
            seen.append(True)

        def verify_model(self):
            pass

        def chat_stream(self, *a, **kw):
            return SimpleNamespace(
                ok=True,
                error=None,
                error_detail="",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 500,
                    "completion_tokens": 200,
                    "total_tokens": 700,
                },
                wall_s=10.0,
                ttft_s=1.0,
                e2e_output_tok_per_s=20.0,
                first_fragment_kind="content",
                content="test-only fixture prose " * 30,
                raw_events=[{"test_only": True}],
                model_reported=b.MODEL_ID,
            )

    monkeypatch.setattr(b, "OpenAICompatClient", Client)
    result = b.run_bench("http://unit.test", out, repeats=1, warmup=0)
    assert seen == [True] and result["ok"] is True
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(len(row["input_sha256"]) == 64 for row in rows)
    assert len(out.with_suffix(".jsonl.raw.jsonl").read_text().splitlines()) == 3
    assert result["promotion_bound"] is False


def test_library_refuses_existing_output_before_client(tmp_path, monkeypatch):
    out = tmp_path / "prose.jsonl"
    out.write_text("sentinel")

    def refuse(*a, **kw):
        raise AssertionError("client was constructed")

    monkeypatch.setattr(b, "OpenAICompatClient", refuse)
    with pytest.raises((FileExistsError, ValueError, SystemExit)):
        b.run_bench("http://unit.test", out)
    assert out.read_text() == "sentinel"
