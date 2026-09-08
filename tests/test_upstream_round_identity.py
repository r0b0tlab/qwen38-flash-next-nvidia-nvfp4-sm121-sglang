"""Five upstream rounds must be distinct measurements, not five copies."""

import copy
import pytest
from scripts import compare as c
from tests.test_promotion_complete import pair


@pytest.mark.parametrize(
    "lane", ["base_upstream", "cand_upstream", "base_medium", "cand_medium"]
)
@pytest.mark.parametrize(
    "fault", ["duplicate", "duplicate_id", "relabel_copy", "unordered", "boolean"]
)
def test_duplicate_or_misidentified_rounds_do_not_promote(lane, fault):
    data = pair()
    rows = data[lane]
    # Current older fixtures lack native result identities; populate the
    # intended complete contract without treating these as model evidence.
    for i, item in enumerate(rows):
        item["row"].setdefault("repeat", i)
        item["row"].setdefault("native_result_sha256", f"{i + 1:064x}")
    if fault == "duplicate":
        data[lane] = [copy.deepcopy(rows[0]) for _ in range(5)]
    if fault == "duplicate_id":
        rows[-1]["row"]["repeat"] = 3
    if fault == "relabel_copy":
        data[lane] = [copy.deepcopy(rows[0]) for _ in range(5)]
        for i, item in enumerate(data[lane]):
            item["row"]["repeat"] = i
    if fault == "unordered":
        data[lane] = list(reversed(rows))
    if fault == "boolean":
        rows[0]["row"]["repeat"] = False
    try:
        result = c.compare(**data)
    except c.Reject:
        return
    assert result["verdict"] != "PASS", (lane, fault)
