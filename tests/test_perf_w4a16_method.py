"""Driver tests against the synthetic fake SSE server (never real model output)."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


# ---------------------------------------------------------------------------
# ladder admission proof (occupancy sampling)
# ---------------------------------------------------------------------------


def test_parse_metrics_sums_label_sets():
    text = "\n".join(
        [
            "# HELP sglang:num_running_reqs running",
            'sglang:num_running_reqs{model_name="x"} 2.0',
            'sglang:num_running_reqs{model_name="y"} 1.0',
            "sglang:num_queue_reqs 3.0",
            "sglang:other_metric 99",
        ]
    )
    assert perf.parse_metrics(text) == {"running": 3.0, "queue": 3.0}


def test_parse_metrics_empty():
    assert perf.parse_metrics("") == {}


class _MetricsHandler(BaseHTTPRequestHandler):
    body = "sglang:num_running_reqs 1.0\n"

    def do_GET(self):  # noqa: N802 - stdlib signature
        data = self.body.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # noqa: A002
        pass


def test_sampler_proves_true_vs_queued_concurrency():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MetricsHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        _MetricsHandler.body = "sglang:num_running_reqs 4.0\n"
        sampler = perf.OccupancySampler(base, 4, interval=0.05)
        with sampler:
            time.sleep(0.2)
        report = sampler.report()
        assert report["true_concurrency"] is True
        assert report["max_running_observed"] == 4.0
        assert report["samples"] > 0

        _MetricsHandler.body = (
            "sglang:num_running_reqs 2.0\nsglang:num_queue_reqs 2.0\n"
        )
        queued = perf.OccupancySampler(base, 4, interval=0.05)
        with queued:
            time.sleep(0.2)
        report = queued.report()
        assert report["true_concurrency"] is False
        assert report["max_queue_observed"] == 2.0
    finally:
        srv.shutdown()
        srv.server_close()
