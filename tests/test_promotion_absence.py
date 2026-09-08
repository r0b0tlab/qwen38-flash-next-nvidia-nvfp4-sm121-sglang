"""Missing lanes and unpaired vision may not produce promotion PASS."""

import importlib.util
from pathlib import Path
import pytest
from scripts import compare as c

spec = importlib.util.spec_from_file_location(
    "old_compare_test_helpers", Path(__file__).with_name("test_compare.py")
)
assert spec and spec.loader
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def params():
    return dict(
        base_rows=c.measured_prose_rows(h._mk_rows(10), "b"),
        cand_rows=c.measured_prose_rows(h._mk_rows(11), "c"),
        base_manifest=h._manifest(),
        cand_manifest=h._manifest("nextn"),
    )


def test_prose_only_can_never_promote():
    assert c.compare(**params())["verdict"] == "NOT_OPTIMIZED"


def test_short_tag_never_bypasses_row_validity():
    row = h._mk_rows(10)[0]
    row.update(case="short", valid=False, finish_reason="length")
    with pytest.raises(c.Reject):
        c.measured_prose_rows([row], "probe")


def test_vision_pairing_requires_same_case_and_input():
    base = [
        {
            "case_id": f"c{i}",
            "repeat": 0,
            "input_sha256": "a" * 64,
            "ttft_s": 1,
            "wall_s": 2,
            "warmup": False,
            "error": None,
            "valid": True,
            "finish_reason": "stop",
        }
        for i in range(3)
    ]
    candidate = [{**r, "case_id": "other" + r["case_id"]} for r in base]
    with pytest.raises(c.Reject):
        c.vision_ttft_gate(base, candidate)
    candidate = [{**r, "input_sha256": "b" * 64} for r in base]
    with pytest.raises(c.Reject):
        c.vision_ttft_gate(base, candidate)


def test_vision_pairing_rejects_count_drift_and_duplicates():
    base = [
        {
            "case_id": f"c{i}",
            "repeat": 0,
            "input_sha256": "a" * 64,
            "ttft_s": 1,
            "wall_s": 2,
            "warmup": False,
            "error": None,
            "valid": True,
            "finish_reason": "stop",
        }
        for i in range(3)
    ]
    with pytest.raises(c.Reject):
        c.vision_ttft_gate(base, base + [base[0]])
