"""Tests for scripts/vision_bench.py.

Two layers:
- Endpoint tests (``TestEndpointVision``): they SEND REAL REQUESTS and are
  DESELECTED on ordinary CPU CI. They run only when BOTH
  ``QUAL_HARNESS_VISION_ENABLED=1`` and ``QUAL_HARNESS_VISION_BASE`` are set
  (parent runs these against the owner-admitted endpoint). Fixture generation
  passing is never claimed as a model-vision pass.
- Scaffold tests: payload construction, env-gating, exclusive output, raw
  evidence preservation, token-accounting recording — against the in-process
  fake server (synthetic vectors only, never model evidence).
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_vision_fixtures as mvf  # noqa: E402
import vision_bench as vb  # noqa: E402
from fake_sse_server import FakeServer  # noqa: E402

pytest.importorskip("PIL")


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    out = tmp_path_factory.mktemp("vision-bench-fixtures")
    mvf.generate_all(out)
    return out


# ---------------------------------------------------------------------------
# env gating — fail closed when not explicit
# ---------------------------------------------------------------------------


class TestEnvGating:
    def test_refuses_without_env(self, fixtures, tmp_path, monkeypatch):
        monkeypatch.delenv(vb.ENV_ENABLE, raising=False)
        with pytest.raises(SystemExit, match="QUAL_HARNESS_VISION_ENABLED"):
            vb.run_vision_benchmark(
                "http://127.0.0.1:1", fixtures, tmp_path / "out.jsonl", variant="AR"
            )

    def test_cli_refuses_existing_output(self, tmp_path, monkeypatch):
        out = tmp_path / "exists.jsonl"
        out.write_text("seed\n")
        rc = vb.main(
            [
                "--base",
                "http://127.0.0.1:1",
                "--fixtures",
                str(tmp_path),
                "--output",
                str(out),
                "--variant",
                "AR",
            ]
        )
        assert rc == 2

    def test_cli_requires_base(self):
        with pytest.raises(SystemExit):
            vb.main(["--output", "/tmp/x.jsonl", "--variant", "AR"])


# ---------------------------------------------------------------------------
# payload construction — answer-blind prompts, data URLs, exact model id
# ---------------------------------------------------------------------------


class TestPayloads:
    def test_image_message_uses_data_url_and_prompt_only(self, fixtures):
        raw = (fixtures / "images" / "red_square_on_blue.png").read_bytes()
        msg = vb.image_message(mvf.PROMPTS["red_square_on_blue"], raw)
        assert msg["content"][0]["type"] == "image_url"
        url = msg["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        base64.b64decode(url.split(",", 1)[1])  # round-trips
        text = msg["content"][1]["text"]
        assert text == mvf.PROMPTS["red_square_on_blue"]
        assert "red_square_on_blue" not in text  # no filename leak
        assert "blue" not in text.lower()  # no answer leak

    def test_video_message_uses_data_url(self, fixtures):
        raw = (fixtures / "videos" / "colors_first_red.mp4").read_bytes()
        msg = vb.video_message(mvf.PROMPTS["colors_first_red"], raw)
        assert msg["content"][0]["type"] == "video_url"
        url = msg["content"][0]["video_url"]["url"]
        assert url.startswith("data:video/mp4;base64,")
        base64.b64decode(url.split(",", 1)[1])

    def test_multi_image_message(self, fixtures):
        a = (fixtures / "images" / "red_square_on_blue.png").read_bytes()
        b = (fixtures / "images" / "blue_square_on_red.png").read_bytes()
        msg = vb.multi_image_message("Which colors?", [a, b])
        types = [c["type"] for c in msg["content"]]
        assert types == ["image_url", "image_url", "text"]

    def test_resize_and_aspect_variants(self, fixtures):
        raw = (fixtures / "images" / "circles_3.png").read_bytes()
        import io

        from PIL import Image

        for t in (1024, 1536):
            out = vb.resize_png(raw, t)
            assert Image.open(io.BytesIO(out)).size == (t, t)
        wide = Image.open(io.BytesIO(vb.aspect_png(raw, "wide_1024x512")))
        tall = Image.open(io.BytesIO(vb.aspect_png(raw, "tall_512x1024")))
        assert wide.size == (1024, 512) and tall.size == (512, 1024)

    def test_build_cases_matrix(self, fixtures):
        cases = vb.build_cases(fixtures)
        ids = [c["case_id"] for c in cases]
        # 8 images x (512 + 1024 + 1536 + wide + tall) + 1 multi + 2 videos
        assert len(ids) == 8 * 5 + 1 + 2 == 43
        assert "multi_square_pair@512" in ids
        assert "colors_first_red@video" in ids
        assert "ocr_text@1536" in ids
        assert "red_bar_tall_blue_short@tall_512x1024" in ids
        # every case's prompt is exactly the fixture prompt (answer-blind)
        for c in cases:
            if c["case_id"].startswith("multi_"):
                continue
            base = c["case_id"].split("@")[0]
            assert c["prompt"] == mvf.PROMPTS[base]


# ---------------------------------------------------------------------------
# scaffold run against the fake server (transport path only)
# ---------------------------------------------------------------------------


class TestScaffoldRun:
    def test_run_preserves_rows_and_raw_evidence(self, fixtures, tmp_path, monkeypatch):
        monkeypatch.setenv(vb.ENV_ENABLE, "1")
        with FakeServer() as srv:
            out = tmp_path / "vision.jsonl"
            summary = vb.run_vision_benchmark(
                srv.base,
                fixtures,
                out,
                variant="AR",
                repeats=1,
                warmup=1,
                timeout_s=20,
                no_resize_matrix=True,
            )
            assert summary["transport_errors"] == []
            assert (
                summary["semantic_errors"] and summary["ok"] is False
            )  # generic fake text is not visual evidence
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            assert summary["rows"] == len(rows) > 0
            for row in rows:
                assert row["variant"] == "AR"
                assert row["model"] == vb.MODEL_ID
                assert row["raw_response"], (
                    "raw events must be persisted before assertion"
                )
                assert row["usage"]["prompt_tokens"] >= 0
                assert row["ttft_s"] is not None
                assert row["finish_reason"] == "stop"

    def test_run_records_errors_without_discarding(
        self, fixtures, tmp_path, monkeypatch
    ):
        """A failing case is recorded per-row and surfaced in errors."""
        monkeypatch.setenv(vb.ENV_ENABLE, "1")
        with FakeServer() as srv:
            out = tmp_path / "vision2.jsonl"
            # Transport succeeds, but generic fake text must fail visual semantics.
            summary = vb.run_vision_benchmark(
                srv.base,
                fixtures,
                out,
                variant="NEXTN",
                repeats=1,
                warmup=0,
                timeout_s=20,
                no_resize_matrix=True,
            )
            assert summary["transport_errors"] == []
            assert summary["semantic_errors"] and summary["ok"] is False
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            assert all(r["variant"] == "NEXTN" and not r["valid"] for r in rows)


# ---------------------------------------------------------------------------
# endpoint vision tests — parent-only, deselected on CPU CI
# ---------------------------------------------------------------------------

_qualifies = os.environ.get(vb.ENV_ENABLE) == "1" and bool(
    os.environ.get("QUAL_HARNESS_VISION_BASE")
)

reason = (
    "endpoint vision tests send real requests; set QUAL_HARNESS_VISION_ENABLED=1 "
    "and QUAL_HARNESS_VISION_BASE to run them"
)


@pytest.mark.skipif(not _qualifies, reason=reason)
class TestEndpointVision:
    """Real-endpoint assertions: actual final answer + finish stop + preserved raw.

    These exercise the real vision path against the owner-provided endpoint.
    """

    def test_square_counterfactual(self, tmp_path):
        base = os.environ["QUAL_HARNESS_VISION_BASE"]
        fixtures = Path(os.environ.get("QUAL_HARNESS_VISION_FIXTURES", "fixtures"))
        if not (fixtures / "manifest.json").exists():
            mvf.generate_all(fixtures)
        out = tmp_path / "endpoint.jsonl"
        summary = vb.run_vision_benchmark(
            base,
            fixtures,
            out,
            variant="AR",
            repeats=1,
            warmup=1,
            timeout_s=120,
            no_resize_matrix=True,
        )
        assert summary["errors"] == []
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        row = next(r for r in rows if r["case_id"] == "red_square_on_blue@512")
        assert row["finish_reason"] == "stop"
        assert row["usage"]["completion_tokens"] > 0
        assert "red" in row["final_text"].lower()
        row2 = next(r for r in rows if r["case_id"] == "blue_square_on_red@512")
        assert "blue" in row2["final_text"].lower()

    def test_token_accounting_recorded_not_proof(self, tmp_path):
        """prompt_tokens must grow with image conditioning; but growth alone is
        never treated as proof of vision (final text is)."""
        base = os.environ["QUAL_HARNESS_VISION_BASE"]
        fixtures = Path(os.environ.get("QUAL_HARNESS_VISION_FIXTURES", "fixtures"))
        if not (fixtures / "manifest.json").exists():
            mvf.generate_all(fixtures)
        out = tmp_path / "acct.jsonl"
        summary = vb.run_vision_benchmark(
            base,
            fixtures,
            out,
            variant="AR",
            repeats=1,
            warmup=0,
            timeout_s=120,
            no_resize_matrix=True,
        )
        assert summary["errors"] == []
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        by_id = {r["case_id"]: r for r in rows}
        assert by_id["ocr_text@512"]["usage"]["prompt_tokens"] > 0
        # image token accounting recorded — proof stays the final text
        assert isinstance(
            by_id["colors_first_red@video"]["usage"]["prompt_tokens"], int
        )
