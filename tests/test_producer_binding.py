"""Producer binding validates actual request values before any HTTP call."""

import json
from types import SimpleNamespace

import pytest
from scripts import bench_real as b, compare as c
from scripts import benchmark_evidence as evidence


def manifest(requests):
    doc = {
        "schema": "qwen38fn.promotion.v1",
        "model_id": b.MODEL_ID,
        "model_sha": "a" * 40,
        "image_id": "sha256:" + "b" * 64,
        "source_tree": "c" * 40,
        "tokenizer": "d" * 64,
        "input_token_sha": "e" * 64,
        "context": 32768,
        "total_pool": 32768,
        "concurrency": 1,
        "sampling": {"temperature": 0, "top_p": 1},
        "lever": "none",
        "inputs": {
            "prose": {
                key: evidence.input_hash(value) for key, value in requests.items()
            }
        },
    }
    doc["manifest_sha256"] = c.manifest_fingerprint(doc)
    return doc


def test_actual_request_change_is_rejected_before_client(tmp_path, monkeypatch):
    doc = manifest(b.prose_payloads())
    first = next(iter(doc["inputs"]["prose"]))
    doc["inputs"]["prose"][first] = "f" * 64
    doc["manifest_sha256"] = c.manifest_fingerprint(doc)

    def forbid(*a, **kw):
        raise AssertionError("constructed a client before binding")

    monkeypatch.setattr(b, "OpenAICompatClient", forbid)
    with pytest.raises(ValueError, match="request plan"):
        b.run_bench("http://unit.test", tmp_path / "rows.jsonl", promotion_manifest=doc)


def test_bound_prose_rows_link_to_the_pretraffic_manifest(tmp_path, monkeypatch):
    doc = manifest(b.prose_payloads())

    class Client:
        def __init__(self, base):
            pass

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
    out = tmp_path / "rows.jsonl"
    result = b.run_bench(
        "http://unit.test", out, repeats=1, warmup=0, promotion_manifest=doc
    )
    assert result["promotion_bound"] is True
    for row in map(json.loads, out.read_text().splitlines()):
        assert row["_evidence"]["manifest_sha256"] == doc["manifest_sha256"]
        assert row["_evidence"]["input_sha256"] == row["input_sha256"]
        assert row["input_sha256"] == doc["inputs"]["prose"][row["case"]]
