"""--retries: only infra verdicts are retried; every attempt is a persisted row.

r02 evidence: the 258,044-token two-key case died at wall 590s with
"Remote end closed connection without response" (infra), recorded fail-closed
with no recovery path. A bounded, infra-only retry keeps every attempt as an
explicit row (attempt=N) so nothing is silently discarded; scored model
verdicts are never retried.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import niah as nz  # noqa: E402

CASE = None


def _case():
    global CASE
    if CASE is None:
        CASE = nz.build_case(
            nz.FakeTokenizer(),
            case_id="retry_t",
            n_prompt_tokens=600,
            depths=(0.5,),
            codes=("ZEPHYR-4821",),
            query=nz.QUERY_TEMPLATE.format(n=1),
        )
    return CASE


def _ok_result(case):
    return types.SimpleNamespace(
        error=None,
        error_detail="",
        http_status=200,
        finish_reason="stop",
        usage={
            "prompt_tokens": case.n_tokens,
            "completion_tokens": 8,
            "total_tokens": case.n_tokens + 8,
        },
        wall_s=0.01,
        text="ZEPHYR-4821",
        reasoning="",
        raw_body="{}",
    )


def _infra_result():
    return types.SimpleNamespace(
        error="transport",
        error_detail="transport: Remote end closed connection without response",
        http_status=None,
        finish_reason=None,
        usage=None,
        wall_s=590.0,
        text="",
        reasoning="",
        raw_body="",
    )


class FlakyClient:
    """Fails the first N calls with a transport error, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def completions_tokens(self, ids, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            return _infra_result()
        return _ok_result(_case())


class ScoredMissClient:
    """Never errors at the transport layer; always a wrong answer."""

    def __init__(self):
        self.calls = 0

    def completions_tokens(self, ids, **kw):
        self.calls += 1
        res = _ok_result(_case())
        res.text = "WRONG-CODE"
        return res


def _rows(out_path):
    return [json.loads(ln) for ln in Path(out_path).read_text().splitlines()]


def test_retry_recovers_after_transport_drop(tmp_path):
    out = tmp_path / "rows.jsonl"
    client = FlakyClient(fail_times=1)
    summary = nz._execute_cases(client, [_case()], out, retries=1)
    rows = _rows(out)
    assert len(rows) == 2
    assert [r["attempt"] for r in rows] == [1, 2]
    assert rows[0]["verdict"] == "infra_error"
    assert rows[1]["verdict"] == "pass"
    assert summary["verdicts"][_case().case_id] == "pass"
    assert client.calls == 2


def test_retries_zero_preserves_single_infra_row(tmp_path):
    out = tmp_path / "rows.jsonl"
    client = FlakyClient(fail_times=1)
    nz._execute_cases(client, [_case()], out, retries=0)
    rows = _rows(out)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "infra_error"
    assert rows[0]["attempt"] == 1
    assert client.calls == 1


def test_scored_verdict_never_retried(tmp_path):
    out = tmp_path / "rows.jsonl"
    client = ScoredMissClient()
    nz._execute_cases(client, [_case()], out, retries=3)
    rows = _rows(out)
    assert len(rows) == 1  # a wrong answer is a model failure, not infra
    assert rows[0]["verdict"] == "needle_miss"
    assert client.calls == 1


def test_retries_exhausted_records_infra_error(tmp_path):
    out = tmp_path / "rows.jsonl"
    client = FlakyClient(fail_times=5)
    summary = nz._execute_cases(client, [_case()], out, retries=2)
    rows = _rows(out)
    assert [r["attempt"] for r in rows] == [1, 2, 3]
    assert all(r["verdict"] == "infra_error" for r in rows)
    assert summary["verdicts"][_case().case_id] == "infra_error"
    assert client.calls == 3
