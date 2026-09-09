"""Driver tests against the synthetic fake SSE server (never real model output)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import perf_w4a16_method as perf  # noqa: E402
from fake_sse_server import FakeServer, MODEL_ID  # noqa: E402
from http_client import OpenAICompatClient  # noqa: E402


@pytest.fixture(scope="module")
def fake():
    with FakeServer() as srv:
        srv.httpd.chat_scenario_default = "long-answer"
        yield srv


def test_run_row_reports_aggregate_rate(fake):
    client = OpenAICompatClient(fake.base, model=MODEL_ID)
    row = perf.run_row(client, "synthetic", 512, 30.0)
    assert row["valid"] is True
    assert row["finish_reason"] == "stop"
    assert row["completion_tokens"] > 0
    assert row["aggregate_output_tokens_per_second"] == pytest.approx(
        row["completion_tokens"] / row["wall_s"]
    )


def test_quick_suite_produces_finite_summary(fake, tmp_path):
    out = tmp_path / "perf.json"
    rc = perf.main(
        [
            "--base",
            fake.base,
            "--output",
            str(out),
            "--quick",
            "--concurrency",
            "1",
            "2",
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["status"] == "PASS"
    assert report["summary"]["errors"] == 0
    assert report["summary"]["dedicated_c1_median"] > 0
    assert set(report["summary"]["ladder_medians"]) == {"1", "2"}
    assert report["method"]["thinking"] == "off"


def test_transport_failure_marks_row_invalid(fake):
    # wrong base: connection refused -> transport error, never a fake success
    client = OpenAICompatClient("http://127.0.0.1:1", model=MODEL_ID)
    row = perf.run_row(client, "synthetic", 16, 1.0)
    assert row["valid"] is False
    assert row["aggregate_output_tokens_per_second"] is None
    assert row["error"] in {"transport", "timeout"}
