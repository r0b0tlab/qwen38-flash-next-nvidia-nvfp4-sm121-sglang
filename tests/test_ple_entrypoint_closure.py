"""Controller closure of the packaged PLE preparation contract."""

import json
import sys

import pytest

from runtime import entrypoint as ep
from tests.test_ple_entrypoint import CORE_MODULE, SOURCES, SHA, TABLE_SIZE, make_plan


def result(mode="reuse"):
    return {
        "receipt_path": ep.ple_dir(SHA) + "/prepared.json",
        "receipt_sha256": "a" * 64,
        "mode": mode,
        "copied_bytes": 0 if mode == "reuse" else TABLE_SIZE,
        "elapsed_seconds": 0.25,
    }


@pytest.mark.parametrize("mode", ["reuse", "materialized"])
def test_exact_core_api_result_is_accepted(mode):
    value = result(mode)
    assert ep.validate_prepared_result(value, ep.ple_dir(SHA), TABLE_SIZE) == value


def test_uncontracted_digest_alias_is_rejected():
    value = result("materialized")
    value["digest"] = value.pop("receipt_sha256")
    with pytest.raises(ep.PLEPreparationError):
        ep.validate_prepared_result(value, ep.ple_dir(SHA), TABLE_SIZE)


def test_core_plan_validator_runs_before_preparation(tmp_path, monkeypatch):
    path = tmp_path / "ple.json"
    path.write_text(json.dumps(make_plan()))
    monkeypatch.setattr(ep, "PLE_PLAN_PATH", str(path))
    module = type(sys)(CORE_MODULE)
    calls = []

    def validate(plan):
        calls.append("validate")
        raise ValueError("rejected by core plan schema")

    def prepare(*args):
        calls.append("prepare")
        return result()

    module.validate_plan = validate
    module.prepare_cache = prepare
    monkeypatch.setitem(sys.modules, CORE_MODULE, module)
    with pytest.raises(ep.PLEPreparationError):
        ep.prepare_ple_cache(SOURCES)
    assert calls == ["validate"]


@pytest.mark.parametrize(
    "files",
    [
        None,
        {},
        [None],
        ["file"],
        [{"path": ep.PLE_SOURCE_NAME, "size": 53717551730.0, "sha256": "35" * 32}],
    ],
)
def test_invalid_source_inventory_is_refused(files):
    sources = {"model": {**SOURCES["model"], "files": files}}
    with pytest.raises(ep.PLEPreparationError):
        ep.bind_ple_plan(make_plan(), sources)


def test_extreme_elapsed_integer_is_refused_without_overflow():
    value = result()
    value["elapsed_seconds"] = 10**400
    with pytest.raises(ep.PLEPreparationError):
        ep.validate_prepared_result(value, ep.ple_dir(SHA), TABLE_SIZE)
