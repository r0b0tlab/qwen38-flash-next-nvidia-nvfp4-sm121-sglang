"""Vision producer semantics and byte evidence (CPU transport fixtures only)."""

import json
from types import SimpleNamespace

import pytest
from scripts import vision_bench as v


def one_case():
    return {
        "case_id": "red_square_on_blue@512",
        "kind": "image",
        "prompt": v.mvf.PROMPTS["red_square_on_blue"],
        "png": b"test-only PNG bytes",
    }


class Client:
    content = "red blue"
    reported = v.MODEL_ID
    calls = 0

    def __init__(self, base):
        pass

    def verify_model(self):
        pass

    def chat_stream(self, messages, **kwargs):
        type(self).calls += 1
        return SimpleNamespace(
            ok=True,
            error=None,
            error_detail="",
            finish_reason="stop",
            usage={"prompt_tokens": 128, "completion_tokens": 8, "total_tokens": 136},
            ttft_s=1.0,
            wall_s=2.0,
            e2e_output_tok_per_s=4.0,
            content=self.content,
            reasoning="",
            model_reported=self.reported,
            first_fragment_kind="content",
            raw_events=[{"test_only": True}],
        )


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv(v.ENV_ENABLE, "1")
    monkeypatch.setattr(v, "build_cases", lambda _: [one_case()])
    monkeypatch.setattr(v, "OpenAICompatClient", Client)
    monkeypatch.setattr(Client, "content", "red blue")
    monkeypatch.setattr(Client, "reported", v.MODEL_ID)
    monkeypatch.setattr(Client, "calls", 0)


def test_warmups_and_actual_payload_hashes_are_retained(tmp_path, setup):
    out = tmp_path / "vision.jsonl"
    result = v.run_vision_benchmark(
        "http://unit.test", tmp_path, out, variant="AR", warmup=1
    )
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 2 and [r["warmup"] for r in rows] == [True, False]
    assert result["measured_rows"] == 1 and result["ok"] is True
    assert all(len(r["input_sha256"]) == 64 and r["media_sha256"] for r in rows)
    assert rows[0]["input_sha256"] != rows[1]["input_sha256"]  # different output budget
    assert out.with_suffix(".jsonl.inputs.json").exists()
    assert len(out.with_suffix(".jsonl.raw.jsonl").read_text().splitlines()) == 2


def test_semantic_failure_is_not_transport_success(tmp_path, setup, monkeypatch):
    monkeypatch.setattr(Client, "content", "blue red")
    out = tmp_path / "vision.jsonl"
    result = v.run_vision_benchmark(
        "http://unit.test", tmp_path, out, variant="NEXTN", warmup=0
    )
    assert result["ok"] is False and result["semantic_errors"]
    row = json.loads(out.read_text())
    assert row["ok"] is True and row["valid"] is False


def test_raw_written_before_semantic_assessment(tmp_path, setup, monkeypatch):
    out = tmp_path / "vision.jsonl"
    seen = []

    def assess(case_id, text):
        raw = out.with_suffix(".jsonl.raw.jsonl")
        assert raw.exists() and json.loads(raw.read_text())["raw_response"]
        seen.append(True)
        return True

    monkeypatch.setattr(v, "semantic_answer", assess)
    v.run_vision_benchmark("http://unit.test", tmp_path, out, variant="AR", warmup=0)
    assert seen == [True]


def test_wrong_reported_model_refused(tmp_path, setup, monkeypatch):
    monkeypatch.setattr(Client, "reported", "wrong/model")
    result = v.run_vision_benchmark(
        "http://unit.test", tmp_path, tmp_path / "rows.jsonl", variant="AR", warmup=0
    )
    assert result["ok"] is False


def test_library_output_is_exclusive_before_client_requests(tmp_path, setup):
    out = tmp_path / "vision.jsonl"
    out.write_text("sentinel")
    with pytest.raises((ValueError, FileExistsError, SystemExit)):
        v.run_vision_benchmark(
            "http://unit.test", tmp_path, out, variant="AR", warmup=0
        )
    assert Client.calls == 0 and out.read_text() == "sentinel"


@pytest.mark.parametrize(
    "case,correct,wrong",
    [
        ("circles_3@512", "3", "5"),
        ("ocr_text@1536", "R7K9", "r7k9"),
        ("left_green_right_yellow@512", "left", "right"),
        ("colors_first_red@video", "red yellow", "yellow red"),
        ("multi_square_pair@512", "red blue", "blue red"),
    ],
)
def test_semantic_matrix(case, correct, wrong):
    assert v.semantic_answer(case, correct) is True
    assert v.semantic_answer(case, wrong) is False
    assert v.semantic_answer(case, "<think>" + correct) is False
