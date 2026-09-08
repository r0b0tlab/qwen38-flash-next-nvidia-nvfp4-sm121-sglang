"""Reducer may not accept requested-count defaults as observed wire evidence."""

import json
import pytest
from scripts import compare as c
from tests.test_compare import _upstream


def wire_row():
    row = json.loads(_upstream(11).splitlines()[0])
    row.update(
        usage_source="observed_oai_sse",
        finish_source="observed_oai_sse",
        finish_reasons=["length"] * 8,
        observed_usage=[
            {"prompt_tokens": 512, "completion_tokens": 256, "total_tokens": 768}
            for _ in range(8)
        ],
        cached_tokens=[0] * 8,
        request_hashes=[f"{i:064x}" for i in range(8)],
    )
    return row


@pytest.mark.parametrize(
    "fault",
    ["missing_usage", "missing_finish", "invented_usage", "bool_usage", "cache_hit"],
)
def test_requested_only_or_bad_wire_evidence_is_rejected(fault):
    row = wire_row()
    if fault == "missing_usage":
        row.pop("usage_source")
    if fault == "missing_finish":
        row.pop("finish_reasons")
    if fault == "invented_usage":
        row["observed_usage"][0]["completion_tokens"] = 2
    if fault == "bool_usage":
        row["observed_usage"][0]["prompt_tokens"] = True
    if fault == "cache_hit":
        row["cached_tokens"][0] = 1
    with pytest.raises(c.Reject):
        c._upstream_rate(row, "unit-only", "short")
