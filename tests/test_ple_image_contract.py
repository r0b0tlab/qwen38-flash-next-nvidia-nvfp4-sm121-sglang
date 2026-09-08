"""The built image must attest the exact PLE plan and source binding."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ple_image_audit", ROOT / "docker/verify_environment.py"
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture(tmp_path):
    (tmp_path / "locks").mkdir()
    raw = (ROOT / "locks/ple.json").read_bytes()
    (tmp_path / "locks/ple.json").write_bytes(raw)
    sources = json.loads((ROOT / "locks/sources.json").read_text())
    (tmp_path / "locks/sources.json").write_text(json.dumps(sources))
    runtime = {"ple_plan_sha256": hashlib.sha256(raw).hexdigest()}
    calls = []
    # Pure contract double here; actual installed-core validation is a build gate.
    core = SimpleNamespace(
        _load_strict_json=lambda data, what: json.loads(data),
        validate_plan=lambda plan: calls.append("validate") or plan,
        plan_sha256=lambda plan: hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    return runtime, sources, core, calls


def test_image_audit_attests_plan_and_invokes_installed_validator(tmp_path):
    runtime, sources, core, calls = fixture(tmp_path)
    result = AUDIT.verify_ple_plan(tmp_path, runtime, core)
    assert result["file_sha256"] == runtime["ple_plan_sha256"]
    assert (
        result["table_sha256"]
        == json.loads((ROOT / "locks/ple.json").read_text())["table"]["sha256"]
    )
    assert calls == ["validate"]


def test_image_audit_rejects_plan_drift_before_core(tmp_path):
    runtime, sources, core, calls = fixture(tmp_path)
    path = tmp_path / "locks/ple.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        AUDIT.verify_ple_plan(tmp_path, runtime, core)
    assert calls == []


def test_image_audit_rejects_source_binding_drift(tmp_path):
    runtime, sources, core, calls = fixture(tmp_path)
    sources["model"]["sha"] = "0" * 40
    (tmp_path / "locks/sources.json").write_text(json.dumps(sources))
    with pytest.raises(ValueError):
        AUDIT.verify_ple_plan(tmp_path, runtime, core)


def test_runtime_gate_calls_plan_audit_not_just_defines_it():
    import ast

    tree = ast.parse((ROOT / "docker/verify_environment.py").read_text())
    verify = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify"
    )
    calls = [
        n.func.id
        for n in ast.walk(verify)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "verify_ple_plan" in calls
